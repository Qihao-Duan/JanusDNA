import math
from functools import partial

import torch
import torch.nn.functional as F
import torchmetrics.functional as tm_f
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from torchmetrics.classification import MulticlassRecall, MulticlassPrecision

from torchmetrics import Metric


class CorrectAggregatedMetric(Metric):
    """This is needed to calculate some metrics b/c small batch sizes cause aggregation via a simple
        average to be off, as some classes might not be present in batch but will get penalized with a 0."""
    def __init__(self, class_idx: int, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.class_idx = torch.tensor(class_idx)
        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _update(self, numerator, denominator, preds, y) -> tuple:
        raise NotImplemented

    def update(self, logits: torch.Tensor, y: torch.Tensor):
        # update metric states
        preds = torch.argmax(logits, dim=-1)
        logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)
        assert preds.shape == y.shape, f"preds shape {preds.shape} != y shape {y.shape}"
        self.numerator, self.denominator = self._update(self.numerator, self.denominator, preds, y)

    def compute(self):
        # compute final result
        value = self.numerator.float() / self.denominator if self.denominator > 0 else torch.tensor(0.0)
        return value

    def reset(self):
        self.numerator = torch.tensor(0.0)
        self.denominator = torch.tensor(0.0)

class AccuracyPerClass(CorrectAggregatedMetric):
    """Calculate per class accuracy, i.e. P(y_hat = class_idx AND y = class_idx OR y_hat != class_idx AND y != class_idx)
    """
    def _update(self, numerator, denominator, preds, y) -> tuple:
        # Filter down to the class of interest
        class_idx = self.class_idx
        relevant_idxs = (y == class_idx)
        numerator += (preds[relevant_idxs] == class_idx).sum()
        denominator += relevant_idxs.sum()
        relevant_idxs = (y != class_idx)
        numerator += (preds[relevant_idxs] != class_idx).sum()
        denominator += relevant_idxs.sum()
        return numerator, denominator

class PrecisionPerClass(CorrectAggregatedMetric):
    """Calculate per class precision, i.e. P(y_hat = y | y_hat = class_idx)
    """
    def _update(self, numerator, denominator, preds, y) -> tuple:
        # Filter down to the class of interest
        class_idx = self.class_idx
        relevant_idxs = (preds == class_idx)
        numerator += (preds[relevant_idxs] == y[relevant_idxs]).sum()
        denominator += relevant_idxs.sum()
        return numerator, denominator


class RecallPerClass(CorrectAggregatedMetric):
    """Calculate per class recall, i.e. P(y_hat = y | y = class_idx)
    """
    def _update(self, numerator, denominator, preds, y) -> tuple:
        # Filter down to the class of interest
        class_idx = self.class_idx
        relevant_idxs = (y == class_idx)
        numerator += (preds[relevant_idxs] == y[relevant_idxs]).sum()
        denominator += relevant_idxs.sum()
        return numerator, denominator


def mcc(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return matthews_corrcoef(y.cpu().numpy(), y_hat.cpu().numpy())


def last_k_ppl(logits, y, seq_len=1024, k=None):
    '''
    Calculate perplexity for last k tokens in a sequence.

    logits: (batch_size * seq_len, vocab_size), note, already flattened
    y: (batch_size * seq_len), note, already flattened
    seq_len: int, length of each sequence in the batch
    k: if None, use all tokens in sequence
    
    returns: (batch_size,)  ppl for each sequence in the batch
    '''

    if k is None:
        k = 0  # use the entire sequence

    # need to reshape logits and y to be (batch_size, seq_len, vocab_size) and (batch_size, seq_len)
    # respectively
    # breakpoint()
    logits = logits.view(-1, seq_len, logits.shape[-1])
    y = y.view(-1, seq_len)

    # only use the last k values of seq dim in logits and y
    logits = logits[:, -k:, :]
    y = y[:, -k:]

    # reshape to flatten the batch and seq_len dimensions
    logits = logits.reshape(-1, logits.shape[-1])
    y = y.reshape(-1)
    # get avg and put on cpu
    return F.cross_entropy(logits, y, reduction='none').view(y.shape[0], -1).mean().exp().cpu()


def _student_t_map(mu, sigma, nu):
    sigma = F.softplus(sigma)
    nu = 2.0 + F.softplus(nu)
    return mu.squeeze(axis=-1), sigma.squeeze(axis=-1), nu.squeeze(axis=-1)

def student_t_loss(outs, y):
    mu, sigma, nu = outs[..., 0], outs[..., 1], outs[..., 2]
    mu, sigma, nu = _student_t_map(mu, sigma, nu)
    y = y.squeeze(axis=-1)

    nup1_half = (nu + 1.0) / 2.0
    part1 = 1.0 / nu * torch.square((y - mu) / sigma)
    Z = (
        torch.lgamma(nup1_half)
        - torch.lgamma(nu / 2.0)
        - 0.5 * torch.log(math.pi * nu)
        - torch.log(sigma)
    )

    ll = Z - nup1_half * torch.log1p(part1)
    return -ll.mean()

def gaussian_ll_loss(outs, y):
    mu, sigma = outs[..., 0], outs[..., 1]
    y = y.squeeze(axis=-1)
    sigma = F.softplus(sigma)
    ll = -1.0 * (
        torch.log(sigma)
        + 0.5 * math.log(2 * math.pi)
        + 0.5 * torch.square((y - mu) / sigma)
    )
    return -ll.mean()

def binary_cross_entropy(logits, y):
    # BCE loss requires squeezing last dimension of logits so it has the same shape as y
    # requires y to be float, since it's overloaded to represent a probability
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1), y.float())


def binary_accuracy(logits, y):
    return torch.eq(logits.squeeze(-1) >= 0, y).float().mean()

def padded_cross_entropy(logits, y, pad_mask, pad_value=-1):
    """Will ignore the pad value in label (eg, -1)
    
    logits: (batch_size, seq_len, vocab_size)
    y: (batch_size, seq_len)
    pad_mask: (batch_size, seq_len)
    
    """

    # need to apply pad mask to y
    y_pad = y + pad_mask * pad_value

    logits = logits.view(-1, logits.shape[-1])
    y_pad = y_pad.view(-1)
    return F.cross_entropy(logits, y_pad, ignore_index=pad_value)


def cross_entropy(logits, y, ignore_index=-100):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return F.cross_entropy(logits, y, ignore_index=ignore_index)


def soft_cross_entropy(logits, y, label_smoothing=0.0):
    logits = logits.view(-1, logits.shape[-1])
    # target is now 2d (no target flattening)
    return F.cross_entropy(logits, y, label_smoothing=label_smoothing)


def accuracy(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    preds = torch.argmax(logits, dim=-1)
    if y.numel() > logits.shape[0]:
        # Mixup leads to this case: use argmax class
        y = y.argmax(dim=-1)
    y = y.view(-1)
    return torch.eq(preds, y).float().mean()


def accuracy_ignore_index(logits, y, ignore_index=-100):
    num_classes = logits.shape[-1]
    preds = torch.argmax(logits, dim=-1)
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    accuracy = tm_f.classification.accuracy(preds, y, 'multiclass', num_classes=num_classes, ignore_index=ignore_index, average='micro')
    return accuracy


def accuracy_at_k(logits, y, k=1):
    logits = logits.view(-1, logits.shape[-1])
    if y.numel() > logits.shape[0]:
        # Mixup leads to this case: use argmax class
        y = y.argmax(dim=-1)
    y = y.view(-1)
    return torch.topk(logits, k, dim=-1)[1].eq(y.unsqueeze(-1)).any(dim=-1).float().mean()


def f1_binary(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="binary")


def f1_macro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="macro")


def f1_micro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    y_hat = torch.argmax(logits, dim=-1)
    return f1_score(y.cpu().numpy(), y_hat.cpu().numpy(), average="micro")


def roc_auc_macro(logits, y):
    logits = logits.view(
        -1, logits.shape[-1]
    ).detach()  # KS: had to add detach to eval while training
    y = y.view(-1)
    return roc_auc_score(
        y.detach().cpu().numpy(), F.softmax(logits, dim=-1).detach().cpu().numpy()[:, 1], average="macro"
    )


def roc_auc_micro(logits, y):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return roc_auc_score(
        y.detach().cpu().numpy(), F.softmax(logits, dim=-1).detach().cpu().numpy()[:, 1], average="micro"
    )


def mse(outs, y, len_batch=None):
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    # outs = outs.squeeze(-1)
    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)
    if len_batch is None:
        return F.mse_loss(outs, y)
    else:
        # Computes the loss of the first `lens` items in the batches
        # TODO document the use case of this
        mask = torch.zeros_like(outs, dtype=torch.bool)
        for i, l in enumerate(len_batch):
            mask[i, :l, :] = 1
        outs_masked = torch.masked_select(outs, mask)
        y_masked = torch.masked_select(y, mask)
        return F.mse_loss(outs_masked, y_masked)

def forecast_rmse(outs, y, len_batch=None):
    # TODO: generalize, currently for Monash dataset
    return torch.sqrt(F.mse_loss(outs, y, reduction='none').mean(1)).mean()

def mae(outs, y, len_batch=None):
    # assert outs.shape[:-1] == y.shape and outs.shape[-1] == 1
    # outs = outs.squeeze(-1)
    if len(y.shape) < len(outs.shape):
        assert outs.shape[-1] == 1
        outs = outs.squeeze(-1)
    if len_batch is None:
        return F.l1_loss(outs, y)
    else:
        # Computes the loss of the first `lens` items in the batches
        mask = torch.zeros_like(outs, dtype=torch.bool)
        for i, l in enumerate(len_batch):
            mask[i, :l, :] = 1
        outs_masked = torch.masked_select(outs, mask)
        y_masked = torch.masked_select(y, mask)
        return F.l1_loss(outs_masked, y_masked)


# Metrics that can depend on the loss
def loss(x, y, loss_fn):
    """ This metric may be useful because the training loss may add extra regularization (e.g. weight decay implemented as L2 penalty), while adding this as a metric skips the additional losses """
    return loss_fn(x, y)


def bpb(x, y, loss_fn):
    """ bits per byte (image density estimation, speech generation, char LM) """
    return loss_fn(x, y) / math.log(2)


def ppl(x, y, loss_fn):
    return torch.exp(loss_fn(x, y))


def eqtl_loss(preds, z):
    # BCE loss requires squeezing last dimension of logits so it has the same shape as y
    # requires y to be float, since it's overloaded to represent a probability
    # loss_seq = torch.sum(-torch.log(prob.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1))) / (y.size(0) * y.size(1)) # ! original version
    print("pred probality of this batch for loss calculation: ", preds.gather(dim=-1, index=z.unsqueeze(-1)).squeeze(-1))
    # loss_label = torch.sum(-torch.log(preds.gather(dim=-1, index=z.unsqueeze(-1)).squeeze(-1))) / z.size(0)
    loss_label = cross_entropy(preds, z)
    # loss = loss_seq + loss_label
    print("loss for this batch: ", loss_label)

    return loss_label


def ms_janusfusion_lastTokenAcc(preds, y, ignore_index=0):
    outcome = []
    preds = preds.view(-1, 131072, preds.shape[-1])
    y = y.view(-1, 131072)
    # print("preds shape: ", preds.shape)
    # print("y shape: ", y.shape)
    last_tokens = torch.argmax(preds, dim=-1)
    
    for batch in range(preds.shape[0]):
        correct = (last_tokens[batch, -1] == y[batch, -1]).float()
        outcome.append(correct.item())
    
    return sum(outcome) / len(outcome) # accuracy
    

    


# should have a better way to do this
output_metric_fns = {
    "binary_cross_entropy": binary_cross_entropy,
    "cross_entropy": cross_entropy,
    "padded_cross_entropy": padded_cross_entropy,
    "binary_accuracy": binary_accuracy,
    # "precision": MulticlassPrecision,
    # "precision_species": partial(MulticlassPrecision, task='multiclass', average=None),
    "precision_species": partial(MulticlassPrecision, average=None),
    # "recall_species": partial(MulticlassRecall, task='multiclass', average=None),
    "recall_species": partial(MulticlassRecall, average=None),
    # "precision_class": partial(MulticlassPrecision, average=None),
    "precision_per_class": PrecisionPerClass,
    "recall": MulticlassRecall,
    "recall_per_class": RecallPerClass,
    "accuracy": accuracy,
    "accuracy_per_class": AccuracyPerClass,
    "accuracy_ignore_index": accuracy_ignore_index,
    'accuracy@3': partial(accuracy_at_k, k=3),
    'accuracy@5': partial(accuracy_at_k, k=5),
    'accuracy@10': partial(accuracy_at_k, k=10),
    "eval_loss": loss,
    "mcc": mcc,
    "mse": mse,
    "mae": mae,
    "forecast_rmse": forecast_rmse,
    "f1_binary": f1_binary,
    "f1_macro": f1_macro,
    "f1_micro": f1_micro,
    "roc_auc_macro": roc_auc_macro,
    "roc_auc_micro": roc_auc_micro,
    "soft_cross_entropy": soft_cross_entropy,  # only for pytorch 1.10+
    "student_t": student_t_loss,
    "gaussian_ll": gaussian_ll_loss,
    "eqtl_loss": eqtl_loss,
    "last_token_acc": ms_janusfusion_lastTokenAcc,
}

loss_metric_fns = {
    "loss": loss,
    "bpb": bpb,
    "ppl": ppl,
}
metric_fns = {**output_metric_fns, **loss_metric_fns}  # TODO py3.9

