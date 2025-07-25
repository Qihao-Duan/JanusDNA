import inspect
from typing import List

import torch
import torch.nn as nn
from einops import rearrange

import src.models.nn.utils as U
import src.tasks.metrics as M
import torchmetrics as tm
from src.models.nn.adaptive_softmax import AdaptiveEmbedding, ProjectedAdaptiveLogSoftmax
from src.tasks.torchmetrics import torchmetric_fns as tm_mine
from src.utils.config import to_list, instantiate
from torchmetrics import MetricCollection


class BaseTask:
    """ Abstract class that takes care of:
    - loss function
    - arbitrary metrics
    - forward pass
    - (optional) encoder module that interfaces with dataset (inputs) and model
    - (optional) decoder module that interfaces with dataset (targets) and model
    """
    encoder = None
    decoder = None

    def __init__(self, dataset=None, model=None, loss=None, loss_val=None, metrics=None, torchmetrics=None):
        """ This class is allowed to grab attributes directly off a constructed dataset and model object """
        self.dataset = dataset
        self.model = model
        if metrics is None:
            metrics = []
        self.metric_names = to_list(metrics)

        if torchmetrics is None:
            torchmetrics = []
        self.torchmetric_names = to_list(torchmetrics)
        self._tracked_torchmetrics = {}

        # The decoder might pass through arguments that the loss needs (e.g. sequence lengths)
        # but might also pass through extraneous arguments (e.g. sampling rate)
        # Wrap loss and metrics so that they accept kwargs and

        # Create loss function
        self.loss = instantiate(M.output_metric_fns, loss, partial=True)
        self.loss = U.discard_kwargs(self.loss)
        if loss_val is not None:
            self.loss_val = instantiate(M.output_metric_fns, loss_val, partial=True)
            self.loss_val = U.discard_kwargs(self.loss_val)
        torchmetrics = MetricCollection(self._init_torchmetrics()) # init trochmetrics regarding names
        self.train_torchmetrics = torchmetrics.clone(prefix='train/')
        self.val_torchmetrics = torchmetrics.clone(prefix='val/')
        self.test_torchmetrics = torchmetrics.clone(prefix='test/')

    def _init_torchmetrics(self):
        """
        Instantiate torchmetrics.
        """
        tracked_torchmetrics = {}

        for name in self.torchmetric_names:
            if name in tm_mine:
                tracked_torchmetrics[name] = tm_mine[name]()
            elif name in ['AUROC', 'StatScores', 'Precision', 'Recall', 'F1', 'F1Score']:
                tracked_torchmetrics[name] = getattr(tm, name)(
                    average='macro', num_classes=self.dataset.d_output, compute_on_step=False
                )
            elif name in ['MultilabelAUROC', 'MultilabelAveragePrecision']:
                tracked_torchmetrics[name] = getattr(tm, name)(
                    average='macro', num_labels=self.dataset.d_output
                )
            elif '@' in name:
                k = int(name.split('@')[1])
                mname = name.split('@')[0]
                tracked_torchmetrics[name] = getattr(tm, mname)(
                    average='macro', num_classes=self.dataset.d_output, compute_on_step=False, top_k=k
                )
            else:
                tracked_torchmetrics[name] = getattr(tm, name)(compute_on_step=False)

        return tracked_torchmetrics

    def _reset_torchmetrics(self, prefix=None):
        """
        Reset torchmetrics for a prefix
        associated with a particular dataloader (e.g. train, val, test).

        Generally do this at the start of an epoch.
        """
        all_prefixes = [prefix] if prefix is not None else self._tracked_torchmetrics

        for prefix in all_prefixes:
            if prefix in self._tracked_torchmetrics:
                self._tracked_torchmetrics[prefix].reset()

    def get_torchmetrics(self, prefix):
        """
        Compute torchmetrics for a prefix associated with
        a particular dataloader (e.g. train, val, test).

        Generally do this at the end of an epoch.
        """
        return {name: self._tracked_torchmetrics[prefix][name].compute() for name in self.torchmetric_names}

    def torchmetrics(self, x, y, prefix, loss=None):
        """
        Update torchmetrics with new x, y .
        Prefix corresponds to a particular dataloader (e.g. train, val, test).

        Generally call this every batch.
        """
        if prefix not in self._tracked_torchmetrics:
            self._init_torchmetrics(prefix)
        self._tracked_torchmetrics[prefix](x, y, loss=loss)

        # for name in self.torchmetric_names:
        #     if name.startswith('Accuracy'):
        #         if len(x.shape) > 2:
        #             # Multi-dimensional, multi-class
        #             self._tracked_torchmetrics[prefix][name].update(x.transpose(1, 2), y.squeeze())
        #             continue
        #     self._tracked_torchmetrics[prefix][name].update(x, y)

    def get_torchmetrics(self, prefix):
        return self._tracked_torchmetrics[prefix]

    def metrics(self, x, y, **kwargs):
        """
        Metrics are just functions
        output metrics are a function of output and target
        loss metrics are a function of loss (e.g. perplexity)
        """
        output_metrics = {
            name: U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)
            for name in self.metric_names if name in M.output_metric_fns
        }
        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }
        return {**output_metrics, **loss_metrics}

    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch  # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        # w can model-specific constructions, such as key_padding_mask for transformers or state for RNNs
        x, w = encoder(x, **z)
        model_name = model.__class__.__name__
        if "Janus" in model_name:
            x, moe_loss, state = model(x, **w, state=_state)
        else:
            x, state = model(x, **w, state=_state)
            moe_loss = 0
        self._state = state
        x, w = decoder(x, state=state, **z)
        return x, y, w, moe_loss # x, input 1; y, target; z, extra input;


class LongrangeTask:
    """ Abstract class that takes care of:
    - loss function
    - arbitrary metrics
    - forward pass
    - (optional) encoder module that interfaces with dataset (inputs) and model
    - (optional) decoder module that interfaces with dataset (targets) and model
    """
    encoder = None
    decoder = None

    def __init__(self, dataset=None, model=None, loss=None, loss_val=None, metrics=None, torchmetrics=None):
        """ This class is allowed to grab attributes directly off a constructed dataset and model object """
        self.dataset = dataset
        self.model = model
        if metrics is None:
            metrics = []
        self.metric_names = to_list(metrics)

        if torchmetrics is None:
            torchmetrics = []
        self.torchmetric_names = to_list(torchmetrics)
        self._tracked_torchmetrics = {}

        # The decoder might pass through arguments that the loss needs (e.g. sequence lengths)
        # but might also pass through extraneous arguments (e.g. sampling rate)
        # Wrap loss and metrics so that they accept kwargs and

        # Create loss function
        self.loss = instantiate(M.output_metric_fns, loss, partial=True)
        self.loss = U.discard_kwargs(self.loss)
        if loss_val is not None:
            self.loss_val = instantiate(M.output_metric_fns, loss_val, partial=True)
            self.loss_val = U.discard_kwargs(self.loss_val)
        torchmetrics = MetricCollection(self._init_torchmetrics()) # init trochmetrics regarding names
        self.train_torchmetrics = torchmetrics.clone(prefix='train/')
        self.val_torchmetrics = torchmetrics.clone(prefix='val/')
        self.test_torchmetrics = torchmetrics.clone(prefix='test/')

    def _init_torchmetrics(self):
        """
        Instantiate torchmetrics.
        """
        tracked_torchmetrics = {}

        for name in self.torchmetric_names:
            if name in tm_mine:
                tracked_torchmetrics[name] = tm_mine[name]()
            elif name in ['AUROC', 'StatScores', 'Precision', 'Recall', 'F1', 'F1Score']:
                tracked_torchmetrics[name] = getattr(tm, name)(
                    average='macro', num_classes=self.dataset.d_output, compute_on_step=False
                )
            elif name in ['MultilabelAUROC', 'MultilabelAveragePrecision']:
                tracked_torchmetrics[name] = getattr(tm, name)(
                    average='macro', num_labels=self.dataset.d_output
                )
            elif '@' in name:
                k = int(name.split('@')[1])
                mname = name.split('@')[0]
                tracked_torchmetrics[name] = getattr(tm, mname)(
                    average='macro', num_classes=self.dataset.d_output, compute_on_step=False, top_k=k
                )
            else:
                tracked_torchmetrics[name] = getattr(tm, name)(compute_on_step=False)

        return tracked_torchmetrics

    def _reset_torchmetrics(self, prefix=None):
        """
        Reset torchmetrics for a prefix
        associated with a particular dataloader (e.g. train, val, test).

        Generally do this at the start of an epoch.
        """
        all_prefixes = [prefix] if prefix is not None else self._tracked_torchmetrics

        for prefix in all_prefixes:
            if prefix in self._tracked_torchmetrics:
                self._tracked_torchmetrics[prefix].reset()

    def get_torchmetrics(self, prefix):
        """
        Compute torchmetrics for a prefix associated with
        a particular dataloader (e.g. train, val, test).

        Generally do this at the end of an epoch.
        """
        return {name: self._tracked_torchmetrics[prefix][name].compute() for name in self.torchmetric_names}

    def torchmetrics(self, x, y, prefix, loss=None):
        """
        Update torchmetrics with new x, y .
        Prefix corresponds to a particular dataloader (e.g. train, val, test).

        Generally call this every batch.
        """
        if prefix not in self._tracked_torchmetrics:
            self._init_torchmetrics(prefix)
        self._tracked_torchmetrics[prefix](x, y, loss=loss)

    def get_torchmetrics(self, prefix):
        return self._tracked_torchmetrics[prefix]

    def metrics(self, x, y, **kwargs):
        """
        Metrics are just functions
        output metrics are a function of output and target
        loss metrics are a function of loss (e.g. perplexity)
        """
        output_metrics = {
            name: U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)
            for name in self.metric_names if name in M.output_metric_fns
        }
        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }
        return {**output_metrics, **loss_metrics}

    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, target = batch  # z holds extra dataloader info such as resolution
        x, _ = encoder(x)
        y, _ = encoder(y)
        
        # w can model-specific constructions, such as key_padding_mask for transformers or state for RNNs
        x, moe_loss_x, state = model(x, state=_state)
        y, moe_loss_y, state = model(y, state=_state)
        moe_loss = moe_loss_x + moe_loss_y
        # concat
        # todo: here is only pure sequences but not tissue type.
        refalt = torch.cat((x, y), dim=-1)
        self._state = state
        refalt, w = decoder(refalt, state=state)
        return refalt, target, w, moe_loss # x, input 1; y, target; z, extra input;


class Scalar(nn.Module):
    def __init__(self, c=1):
        super().__init__()
        self.c = c

    def forward(self, x):
        return x * self.c


class LMTask(BaseTask):
    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""
        # z holds arguments such as sequence length
        x, y, *z = batch  # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]
        # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x, w = encoder(x, **z)
        # Needed for Mamba (open-source repo version)
        model_name = model.__class__.__name__
        if "state" in inspect.signature(model.forward).parameters.keys():
            x, state = model(x, **w, state=_state)
        else:
            if "Janus" in model_name:
                x = model(x, **w)
                state = None
            else:
                x = model(x, **w)
                state = None
        self._state = state
        x, w = decoder(x, state=state, **z)

        if hasattr(x, 'logits'):
            logits = x.logits
        if hasattr(x, 'aux_loss'):
            moe_loss = x.aux_loss
        else:
            moe_loss = 0
            
        # logits = rearrange(logits, '... C -> (...) C')
        if isinstance(logits, tuple) or isinstance(logits, list):
            logits = [rearrange(l, "... C -> (...) C") for l in logits]
        else:
            logits = rearrange(logits, "... C -> (...) C")
            
        y = rearrange(y, '... -> (...)')

        return logits, y, w, moe_loss


class MultiClass(BaseTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.continual_metrics = {}
        for name in self.metric_names:
            if name.endswith('_per_class'):
                for spec_idx, spec in enumerate(self.dataset.species):
                    self.continual_metrics[name + '_' + spec] = M.output_metric_fns[name](spec_idx)
            elif name in ['precision_species', 'recall_species']:
                self.continual_metrics[name] = M.output_metric_fns[name](num_classes=len(self.dataset.species))

    def metrics(self, x, y, **kwargs):
        output_metrics = {}
        for name in self.metric_names:
            if name in M.output_metric_fns:
                if name.endswith('_per_class'):
                    for spec_idx, spec in enumerate(self.dataset.species):
                        self.continual_metrics[name + '_' + spec] = self.continual_metrics[name + '_' + spec].to(
                            x.device)
                        self.continual_metrics[name + '_' + spec].update(x, y)
                        output_metrics[name + '_' + spec] = self.continual_metrics[name + '_' + spec].compute()
                elif name in ['precision_species', 'recall_species']:
                    self.continual_metrics[name] = self.continual_metrics[name].to(x.device)
                    metrics = self.continual_metrics[name](x, y)
                    for spec_idx, spec in enumerate(self.dataset.species):
                        output_metrics[name[:-7] + spec] = metrics[spec_idx]
                else:
                    output_metrics[name] = U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)

        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }

        return {**output_metrics, **loss_metrics}

    def _reset_torchmetrics(self, prefix=None):
        super()._reset_torchmetrics(prefix)
        for name in self.metric_names:
            if name.endswith('_per_class'):
                for spec_idx, spec in enumerate(self.dataset.species):
                    self.continual_metrics[name + '_' + spec].reset()


class HG38Task(LMTask):

    def __init__(self, dataset=None, model=None, loss=None, loss_val=None, metrics=None, torchmetrics=None,
                 last_k_ppl=None, per_token_ppl=None):
        """ Extending LMTask to add custom metrics for HG38 task 
        
        last_k_ppl: config for custom ppl, with hparams to pass with it

        per_token_ppl: config for per token ppl calc, with list of k (ppls) to track

        """
        self.dataset = dataset
        self.model = model
        if metrics is None:
            metrics = []
        self.metric_names = to_list(metrics)
        self.last_k_ppl = last_k_ppl
        self.per_token_ppl = per_token_ppl

        if torchmetrics is None:
            torchmetrics = []
        self.torchmetric_names = to_list(torchmetrics)
        self._tracked_torchmetrics = {}

        # The decoder might pass through arguments that the loss needs (e.g. sequence lengths)
        # but might also pass through extraneous arguments (e.g. sampling rate)
        # Wrap loss and metrics so that they accept kwargs and

        # Create loss function
        self.loss = instantiate(M.output_metric_fns, loss, partial=True)
        self.loss = U.discard_kwargs(self.loss)
        if loss_val is not None:
            self.loss_val = instantiate(M.output_metric_fns, loss_val, partial=True)
            self.loss_val = U.discard_kwargs(self.loss_val)
        torchmetrics = MetricCollection(self._init_torchmetrics())
        self.train_torchmetrics = torchmetrics.clone(prefix='train/')
        self.val_torchmetrics = torchmetrics.clone(prefix='val/')
        self.test_torchmetrics = torchmetrics.clone(prefix='test/')

        # Create custom metrics for last k ppl
        # last_k_ppl is a list of dicts (configs), so loop through them
        if self.last_k_ppl is not None:
            self.custom_ppl_dict = {}
            for k in self.last_k_ppl:
                key_name = "last_" + str(k) + "_ppl"
                # create config
                custom_ppl_config = {"_name_": "last_k_ppl", "k": k, "seq_len": self.dataset.max_length}
                k_ppl_fn = instantiate(M.output_metric_fns, custom_ppl_config, partial=True)
                k_ppl_fn = U.discard_kwargs(k_ppl_fn)
                self.custom_ppl_dict[key_name] = k_ppl_fn

        # Create custom metric for per token ppl
        if self.per_token_ppl is not None:
            per_token_ppl_config = {"_name_": "per_token_ppl", "ks": self.per_token_ppl["ks"],
                                    "seq_len": self.dataset.max_length}
            per_token_fn = instantiate(M.output_metric_fns, per_token_ppl_config, partial=True)
            per_token_fn = U.discard_kwargs(per_token_fn)
            self.per_token_fn = per_token_fn

    def metrics(self, x, y, **kwargs):
        """
        Need to modify metrics to include custom metrics
        """

        output_metrics = {
            name: U.discard_kwargs(M.output_metric_fns[name])(x, y, **kwargs)
            for name in self.metric_names if name in M.output_metric_fns
        }
        loss_metrics = {
            name: U.discard_kwargs(M.loss_metric_fns[name])(x, y, self.loss, **kwargs)
            for name in self.metric_names if name in M.loss_metric_fns
        }

        # loop through all custom ppls and add them to output_metrics
        if self.last_k_ppl is not None:
            for key_name, k_ppl_fn in self.custom_ppl_dict.items():
                output_metrics[key_name] = k_ppl_fn(x, y, **kwargs)

        # loop through all custom ppls and add them to output_metrics
        if self.per_token_ppl is not None:
            # returns k ppl values, (averaged over batch)
            per_k_ppl = self.per_token_fn(x, y, **kwargs)

            # loop over ks to log metric
            for ind, k in enumerate(self.per_token_ppl["ks"]):
                key_name = "ppl_at_{}".format(k)
                output_metrics[key_name] = per_k_ppl[ind]  # should be in order

        return {**output_metrics, **loss_metrics}


class AdaptiveLMTask(BaseTask):
    def __init__(
            self,
            div_val,
            cutoffs: List[int],
            tie_weights: bool,
            tie_projs: List[bool],
            init_scale=1.0,
            bias_scale=0.0,
            dropemb=0.0,
            dropsoft=0.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        n_tokens = self.dataset.n_tokens
        d_model = self.model.d_model
        d_output = self.model.d_output

        encoder = AdaptiveEmbedding(
            n_tokens,
            d_model,
            d_model,
            cutoffs=cutoffs,
            div_val=div_val,
            init_scale=init_scale,
            dropout=dropemb,
        )

        if tie_weights:
            assert d_model == d_output
            emb_layers = [i.weight for i in encoder.emb_layers]
        else:
            emb_layers = None

        # Construct decoder/loss
        emb_projs = encoder.emb_projs
        loss = ProjectedAdaptiveLogSoftmax(
            n_tokens, d_output, d_output,
            cutoffs, div_val=div_val,
            tie_projs=tie_projs,
            out_projs=emb_projs,
            out_layers_weights=emb_layers,
            bias_scale=bias_scale,
            dropout=dropsoft,
        )

        self.encoder = encoder
        self.loss = loss


class MaskedMultiClass(MultiClass):

    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""

        # z holds arguments such as sequence length
        x, y, *z = batch # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        x, w = encoder(x) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
        x, state = model(x)
        self._state = state
        x, w = decoder(x, state=state, **z)
        return x, y, w


class VariantEffectPrediction(MultiClass):

    def forward(self, batch, encoder, model, _state):
        """Passes a batch through the encoder, backbone, and decoder"""

        # z holds arguments such as sequence length
        x, y, *z = batch # z holds extra dataloader info such as resolution
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        x = model(x) # [B, L, d]

        assert x.size(1) == y.size(1)
        return x, y



class EnhancerPromoterTask(MultiClass):

    def forward(self, batch, encoder, model, _state):
        """Passes a batch through the encoder, backbone, and decoder"""

        # z holds arguments such as sequence length
        x, y = batch # z holds extra dataloader info such as resolution
        print(x.shape)
        print(x.dtype)
        # prob, preds = model(x) # [B, L, d]
        # print(z.shape)
        # print(z.dtype)
        preds = model(x)

        #assert preds.size(1) == z.size(1)

        return preds, y
    
    
class EQTLTask(MultiClass):

    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""

        # z holds arguments such as sequence length
        x, y, z = batch # z holds extra dataloader info such as resolution
        
        y, _ = encoder(y)
        model_name = model.__class__.__name__
        if "Janus" in model_name:
            # print("data for this round in task: ", x, y, z)
            # print(y.shape, y.dtype)
            # test_input = torch.ones((1, 1024), device=y.device, dtype=y.dtype)
            preds, moe_loss, state = model(y, state=_state)
        else:
            preds, state = model(y, state=_state)
            moe_loss = 0
        self._state = state
        # print("pred_before_decoder: \n", preds)
        preds, w = decoder(preds, state=state)
        # print("pred_after_decoder: \n", preds)
        
        return preds, z, w, moe_loss
    
class ContactMapPredictionTask(MultiClass):

    def forward(self, batch, encoder, model, decoder, _state):
        """Passes a batch through the encoder, backbone, and decoder"""

        # z holds arguments such as sequence length
        x, y = batch # z holds extra dataloader info such as resolution
        model_name = model.__class__.__name__
        if "Janus" in model_name:
            preds, moe_loss, state = model(x, state=_state)
        else:
            preds, state = model(x, state=_state)
            moe_loss = 0
        self._state = state
        
        preds, w = decoder(preds, state=state)

        assert preds.size(1) == y.size(1)
        return preds, y, w, moe_loss


registry = {
    'base': BaseTask,
    'multiclass': MultiClass,
    'lm': LMTask,
    'hg38': HG38Task,
    'lrb': LongrangeTask,
    "masked_multiclass": MaskedMultiClass,
    "contact_map_prediction": ContactMapPredictionTask,
    "variant_effect_prediction": VariantEffectPrediction,
    "eqtl": EQTLTask,
    "enhancer_promoter": EnhancerPromoterTask
}
