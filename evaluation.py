import copy
import os
import random
import time
import json
from functools import partial, wraps
from typing import Callable, List, Sequence

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from pytorch_lightning.strategies.ddp import DDPStrategy
from tqdm.auto import tqdm
from pytorch_lightning.strategies.ddp import DDPStrategy
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train
from src.dataloaders import SequenceDataset  # TODO make registry
from src.tasks import decoders, encoders, tasks
from src.utils import registry
from src.utils.optim_groups import add_optimizer_hooks
# from src.dataloaders.datasets.akita_dataset import get_dataloader
# from src.dataloaders.datasets.enformer_dataset import get_dataloader
# from src.dataloaders.datasets.eqtl_dataset import get_dataloader
# from src.dataloaders.datasets.enhancer_promoter_dataset import get_dataloader


log = src.utils.train.get_logger(__name__)

# Turn on TensorFloat32 (speeds up large model training substantially)
import torch.backends
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)




# Lots of annoying hacks to get WandbLogger to continuously retry on failure
class DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx) -> "DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        pass


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""

    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment


class CustomWandbLogger(WandbLogger):

    def __init__(self, *args, **kwargs):
        """Modified logger that insists on a wandb.init() call and catches wandb's error if thrown."""

        super().__init__(*args, **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
        .. code-block:: python
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                while True:
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        break
                    except Exception as e:
                        print("wandb Exception:\n", e)
                        t = random.randint(30, 60)
                        print(f"Sleeping for {t} seconds")
                        time.sleep(t)

                # define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment


class SequenceLightningModule(pl.LightningModule):
    def __init__(self, config):
        # Disable profiling executor. This reduces memory and increases speed.
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()
        # Passing in config expands it one level, so can access by self.hparams.train instead of self.hparams.config.train
        self.save_hyperparameters(config, logger=False)

        # Dataset arguments
        self.dataset = SequenceDataset.registry[self.hparams.dataset._name_](
            **self.hparams.dataset
        )

        # Check hparams
        self._check_config()

        # PL has some bugs, so add hooks and make sure they're only called once
        self._has_setup = False

        self.setup()  ## Added by KS

    def setup(self, stage=None):
        if not self.hparams.train.disable_dataset:
            self.dataset.setup()

        # We need to set up the model in setup() because for some reason when training with DDP, one GPU uses much more memory than the others
        # In order to not overwrite the model multiple times during different stages, we need this hack
        # TODO PL 1.5 seems to have an option to skip hooks to avoid this
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5410#issuecomment-762257024
        if self._has_setup:
            return
        else:
            self._has_setup = True

        # Convenience feature: if model specifies encoder, combine it with main encoder
        encoder_cfg = utils.to_list(self.hparams.encoder) + utils.to_list(
            self.hparams.model.pop("encoder", None)
        )
        decoder_cfg = utils.to_list(
            self.hparams.model.pop("decoder", None)
        ) + utils.to_list(self.hparams.decoder)

        # Instantiate model
        config_path = self.hparams.model.pop("config_path", None)
        print("1_config_path:", config_path, "\n")
        if config_path is not None:
            print("-- load model config from pre-trained model dir -- ")
            with open(config_path) as f:
                model_config_from_file = json.load(f)
                print(self.hparams.model._name_)
                if "caduceus" in self.hparams.model._name_ or "hyena" in self.hparams.model._name_:
                    print(f"current_model: {self.hparams.model._name_} -- config_path: {config_path} \n")
                    pretrained_model_config_from_file = model_config_from_file
                else:    
                    pretrained_model_config_path = model_config_from_file["config_path"]
                    print(f"current_model: {self.hparams.model._name_} -- pretrained_config_path: {pretrained_model_config_path} \n")
                    with open(pretrained_model_config_path) as pretrained_config_f:
                        pretrained_model_config_from_file = json.load(pretrained_config_f)
            self.hparams.model.update(pretrained_model_config_from_file)
            print(f"model config has been loaded as:\n {pretrained_model_config_from_file}")
            # if hasattr(self.hparams.model, "config_path"):
            #     pretrained_config_path = self.hparams.model.pop("config_path")
            #     print(f"pretrained model config path is {pretrained_config_path}")
            # else:
            #     print("pretrained model config path is not specified")
            # Check if dropout_layer_norm is compiled
            try:
                from flash_attn.ops.layer_norm import dropout_add_layer_norm
            except ImportError:
                if self.hparams.model.get("fused_dropout_add_ln", None) is not None:
                    self.hparams.model.update({"fused_dropout_add_ln": False})
        # # Instantiate model
        # self.model = utils.instantiate(registry.model, self.hparams.model)
        # Instantiate the config class if using hydra's _target_ paradigm for the config
        if self.hparams.model.get("config", None) is not None and self.hparams.model.config.get("_target_", None) is not None:
            print("load config and initialize model regarding this config")
            model_hparams = OmegaConf.to_container(self.hparams.model, resolve=True)
            model_hparams["config"] = hydra.utils.instantiate(model_hparams["config"])
            self.model = utils.instantiate(registry.model, model_hparams)
            print(self.model)
        else:
            self.model = utils.instantiate(registry.model, self.hparams.model)
            
        if (name := self.hparams.train.post_init_hook['_name_']) is not None:
            kwargs = self.hparams.train.post_init_hook.copy()
            del kwargs['_name_']
            for module in self.modules():
                if hasattr(module, name):
                    getattr(module, name)(**kwargs)

        # Instantiate the task
        self.task = utils.instantiate(
            tasks.registry, self.hparams.task, dataset=self.dataset, model=self.model
        )

        # Create encoders and decoders
        encoder = encoders.instantiate(
            encoder_cfg, dataset=self.dataset, model=self.model
        )
        decoder = decoders.instantiate(
            decoder_cfg, model=self.model, dataset=self.dataset
        )

        # Extract the modules so they show up in the top level parameter count
        self.encoder = U.PassthroughSequential(self.task.encoder, encoder)
        self.decoder = U.PassthroughSequential(decoder, self.task.decoder)
        print("encoder:", self.encoder)
        print("decoder:", self.decoder)
        self.loss = self.task.loss
        self.loss_val = self.task.loss
        if hasattr(self.task, 'loss_val'):
            self.loss_val = self.task.loss_val
        self.metrics = self.task.metrics
        self.train_torchmetrics = self.task.train_torchmetrics
        self.val_torchmetrics = self.task.val_torchmetrics
        self.test_torchmetrics = self.task.test_torchmetrics
        self.fw = open(self.hparams.train.eval_log_path, "w", encoding="utf-8")  # cell_type = "HFF"
        
    def close(self):
        if not self.fw.closed:
            self.fw.close()
            print("fw has been closed")

    def load_state_dict(self, state_dict, strict=False):
        if self.hparams.train.pretrained_model_state_hook['_name_'] is not None:
            model_state_hook = utils.instantiate(
                registry.model_state_hook,
                self.hparams.train.pretrained_model_state_hook.copy(),
                partial=True,
            )
            state_dict = model_state_hook(self.model, state_dict)

            print("Custom load_state_dict function is running.")

        # strict==True will require all modules to match
        # strict==False can allow encoder/decoder to be loaded from scratch too
        return super().load_state_dict(state_dict, strict=strict)

    def _check_config(self):
        assert self.hparams.train.state.mode in [None, "none", "null", "reset", "bptt", "tbptt"]
        assert (
            (n := self.hparams.train.state.n_context) is None
            or isinstance(n, int)
            and n >= 0
        )
        assert (
            (n := self.hparams.train.state.n_context_eval) is None
            or isinstance(n, int)
            and n >= 0
        )

    def _initialize_state(self):
        """Called at model setup and start of epoch to completely reset state"""
        self._state = None
        self._memory_chunks = []

    def _reset_state(self, batch, device=None):
        """Called to construct default_state when necessary, e.g. during BPTT"""
        device = device or batch[0].device
        self._state = self.model.default_state(*batch[0].shape[:1], device=device)

    def _detach_state(self, state):
        if isinstance(state, torch.Tensor):
            return state.detach()
        elif isinstance(state, tuple):
            return tuple(self._detach_state(s) for s in state)
        elif isinstance(state, list):
            return [self._detach_state(s) for s in state]
        elif isinstance(state, dict):
            return {k: self._detach_state(v) for k, v in state.items()}
        elif state is None:
            return None
        else:
            raise NotImplementedError

    def _process_state(self, batch, batch_idx, train=True):
        """Handle logic for state context."""
        # Number of context steps
        key = "n_context" if train else "n_context_eval"
        n_context = self.hparams.train.state.get(key)

        # Don't need to do anything if 0 context steps. Make sure there is no state
        if n_context == 0 and self.hparams.train.state.mode not in ['tbptt']:
            self._initialize_state()
            return

        # Reset state if needed
        if self.hparams.train.state.mode == "reset":
            if batch_idx % (n_context + 1) == 0:
                self._reset_state(batch)

        # Pass through memory chunks
        elif self.hparams.train.state.mode == "bptt":
            self._reset_state(batch)
            with torch.no_grad():  # should be unnecessary because individual modules should handle this
                for _batch in self._memory_chunks:
                    self.forward(_batch)
            # Prepare for next step
            self._memory_chunks.append(batch)
            self._memory_chunks = self._memory_chunks[-n_context:]

        elif self.hparams.train.state.mode == 'tbptt':
            _, _, z = batch
            reset = z["reset"]
            if reset:
                self._reset_state(batch)
            else:
                self._state = self._detach_state(self._state)

    # def forward(self, batch):
    #     """Passes a batch through the encoder, backbone, and decoder"""
    #     # z holds arguments such as sequence length
    #     x, y, *z = batch # z holds extra dataloader info such as resolution
    #     if len(z) == 0:
    #         z = {}
    #     else:
    #         assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
    #         z = z[0]

    #     x, w = self.encoder(x, **z) # w can model-specific constructions such as key_padding_mask for transformers or state for RNNs
    #     x, state = self.model(x, **w, state=self._state)
    #     self._state = state
    #     x, w = self.decoder(x, state=state, **z)
    #     return x, y, w

    def forward(self, batch):
        return self.task.forward(batch, self.encoder, self.model, self.decoder, self._state)

    def step(self, x_t):
        x_t, *_ = self.encoder(x_t) # Potential edge case for encoders that expect (B, L, H)?
        x_t, state = self.model.step(x_t, state=self._state)
        self._state = state
        # x_t = x_t[:, None, ...] # Dummy length
        # x_t, *_ = self.decoder(x_t, state=state)
        # x_t = x_t[:, 0, ...]
        x_t, *_ = self.decoder.step(x_t, state=state)
        return x_t

    def _shared_step(self, batch, batch_idx, prefix="train"):

        self._process_state(batch, batch_idx, train=(prefix == "train"))
        # x, y, z = batch # z holds extra dataloader info such as resolution
        # print("alt_seq: \n", y[:, :10])
        # print("alt_seq shape: \n", y.shape)
        # print("target: \n", z)
        x, y, w, moe_loss = self.forward(batch)
        # print("x: \n", x)
        # print("y: \n", y)

        # Loss
        if prefix == 'train':
            loss = self.loss(x, y)
        else:
            loss = self.loss_val(x, y)

        prob = torch.softmax(x, dim=-1)
        # print("prob: \n", prob)
        # print("prob shape: \n", prob.shape)
        if prob.dim() == 1:
            print(str(prob[1].item()) + " " + str(y.item()) + "\n") # TODO: here is only working for auroc
            self.fw.write(str(prob[1].item()) + " " + str(y.item()) + "\n")
            self.fw.flush()
        else:
            for i in range(prob.size(0)):
                print(str(prob[i][1].item()) + " " + str(y[i].item()) + "\n") # TODO: here is only working for auroc
                self.fw.write(str(prob[i][1].item()) + " " + str(y[i].item()) + "\n")
                self.fw.flush()
            
        # Metrics
        metrics = self.metrics(x, y)
        metrics["loss"] = loss
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Calculate torchmetrics
        # torchmetrics = getattr(self, f'{prefix}_torchmetrics')
        # torchmetrics(x, y, loss=loss)
        
        log_on_step = 'eval' in self.hparams and self.hparams.eval.get('log_on_step', False) and prefix == 'train'

        self.log_dict(
            metrics,
            on_step=log_on_step,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # log the whole dict, otherwise lightning takes the mean to reduce it
        # https://pytorch-lightning.readthedocs.io/en/stable/visualize/logging_advanced.html#enable-metrics-for-distributed-training
        # self.log_dict(
        #     torchmetrics,
        #     on_step=log_on_step,
        #     on_epoch=True,
        #     prog_bar=True,
        #     add_dataloader_idx=False,
        #     sync_dist=True,
        # )

        return loss

    def on_train_epoch_start(self):
        # Reset training torchmetrics
        self.task._reset_torchmetrics("train")

    # def on_train_epoch_end(self, outputs):
    #     # Log training torchmetrics
    #     super().on_train_epoch_end(outputs)

    def on_validation_epoch_start(self):
        # Reset all validation torchmetrics
        for name in self.val_loader_names:
            self.task._reset_torchmetrics(name)

    # def on_validation_epoch_end(self, outputs):
    #     # Log all validation torchmetrics
    #     super().on_validation_epoch_end(outputs)

    def on_test_epoch_start(self):
        # Reset all test torchmetrics
        for name in self.test_loader_names:
            self.task._reset_torchmetrics(name)

    # def on_test_epoch_end(self, outputs):
    #     # Log all test torchmetrics
    #     super().on_test_epoch_end(outputs)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._shared_step(batch, batch_idx, prefix="train")

        # Log the loss explicitly so it shows up in WandB
        # Note that this currently runs into a bug in the progress bar with ddp (as of 1.4.6)
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/9142ƒ
        # We additionally log the epochs under 'trainer' to get a consistent prefix with 'global_step'
        loss_epoch = {"trainer/loss": loss, "trainer/epoch": self.current_epoch}

        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # Log any extra info that the models want to expose (e.g. output norms)
        metrics = {}
        for module in list(self.modules())[1:]:
            if hasattr(module, "metrics"):
                metrics.update(module.metrics)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ema = (
            self.val_loader_names[dataloader_idx].endswith("/ema")
            and self.optimizers().optimizer.stepped
        )  # There's a bit of an annoying edge case with the first (0-th) epoch; it has to be excluded due to the initial sanity check
        if ema:
            self.optimizers().swap_ema()
        loss = self._shared_step(
            batch, batch_idx, prefix=self.val_loader_names[dataloader_idx]
        )
        if ema:
            self.optimizers().swap_ema()

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.model.eval()
        with torch.no_grad():
            loss = self._shared_step(
                batch, batch_idx, prefix=self.test_loader_names[dataloader_idx]
            )
            return loss


    def configure_optimizers(self):
        # Set zero weight decay for some params
        if 'optimizer_param_grouping' in self.hparams.train:
            add_optimizer_hooks(self.model, **self.hparams.train.optimizer_param_grouping)

        # Normal parameters
        all_params = list(self.parameters())
        params = [p for p in all_params if not hasattr(p, "_optim")]

        optimizer = utils.instantiate(registry.optimizer, self.hparams.optimizer, params)

        del self.hparams.optimizer._name_

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
        hps = [
            # dict(s) for s in set(frozenset(hp.items()) for hp in hps)
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
            # dict(s) for s in dict.fromkeys(frozenset(hp.items()) for hp in hps)
        ]  # Unique dicts
        print("Hyperparameter groups", hps)
        for hp in hps:
            params = [p for p in all_params if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **self.hparams.optimizer, **hp}
            )

        ### Layer Decay ###

        if self.hparams.train.layer_decay['_name_'] is not None:
            get_num_layer = utils.instantiate(
                registry.layer_decay,
                self.hparams.train.layer_decay['_name_'],
                partial=True,
            )

            # Go through all parameters and get num layer
            layer_wise_groups = {}
            num_max_layers = 0
            for name, p in self.named_parameters():
                # Get layer id for each parameter in the model
                layer_id = get_num_layer(name)

                # Add to layer wise group
                if layer_id not in layer_wise_groups:
                    layer_wise_groups[layer_id] = {
                        'params': [],
                        'lr': None,
                        'weight_decay': self.hparams.optimizer.weight_decay
                    }
                layer_wise_groups[layer_id]['params'].append(p)

                if layer_id > num_max_layers: num_max_layers = layer_id

            # Update lr for each layer
            for layer_id, group in layer_wise_groups.items():
                group['lr'] = self.hparams.optimizer.lr * (self.hparams.train.layer_decay.decay ** (num_max_layers - layer_id))

            # Reset the torch optimizer's param groups
            optimizer.param_groups = []
            for layer_id, group in layer_wise_groups.items():
                optimizer.add_param_group(group)

        # Print optimizer info for debugging
        keys = set([k for hp in hps for k in hp.keys()])  # Special hparams
        utils.train.log_optimizer(log, optimizer, keys)
        # Configure scheduler
        if "scheduler" not in self.hparams:
            return optimizer
        lr_scheduler = utils.instantiate(
            registry.scheduler, self.hparams.scheduler, optimizer
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,  # 'epoch' or 'step'
            "monitor": self.hparams.train.monitor,
            "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
        }
        # See documentation for how to configure the return
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        return [optimizer], [scheduler]

    def train_dataloader(self):
        # return get_dataloader("data_long_range_dna/enhancer_promoter_interaction/CRISPRi_EPI", "train")
        return self.dataset.train_dataloader(**self.hparams.loader)

    def _eval_dataloaders_names(self, loaders, prefix):
        """Process loaders into a list of names and loaders"""
        if utils.is_dict(loaders):
            return [
                f"{prefix}/{k}" if k is not None else prefix for k in loaders.keys()
            ], list(loaders.values())
        elif utils.is_list(loaders):
            return [f"{prefix}/{i}" for i in range(len(loaders))], loaders
        else:
            return [prefix], [loaders]

    def _eval_dataloaders(self):

        # Return all val + test loaders
        # val_loaders = get_dataloader("Whole_Blood", "valid")
        # test_loaders = get_dataloader("Whole_Blood", "test")
        
        # test_loaders = get_dataloader("hyena-dna/data/Enformer/mm10.ml.fa",
        #                       "mouse", "test")
        # val_loaders = get_dataloader("data_long_range_dna/enhancer_promoter_interaction/CRISPRi_EPI", "valid")
        # test_loaders = get_dataloader("data_long_range_dna/enhancer_promoter_interaction/CRISPRi_EPI", "test")
        val_loaders = self.dataset.val_dataloader(**self.hparams.loader)
        test_loaders = self.dataset.test_dataloader(**self.hparams.loader)

        val_loader_names, val_loaders = self._eval_dataloaders_names(val_loaders, "val")
        test_loader_names, test_loaders = self._eval_dataloaders_names(
            test_loaders, "test"
        )

        # Duplicate datasets for ema
        if self.hparams.train.ema > 0.0:
            val_loader_names += [name + "/ema" for name in val_loader_names]
            val_loaders = val_loaders + val_loaders
            test_loader_names += [name + "/ema" for name in test_loader_names]
            test_loaders = test_loaders + test_loaders

        # adding option to only have val loader at eval (eg if test is duplicate)
        if self.hparams.train.get("remove_test_loader_in_eval", False):
            return val_loader_names, val_loaders
        # adding option to only have test loader at eval
        elif self.hparams.train.get("remove_val_loader_in_eval", True):
            return test_loader_names, test_loaders
        # default behavior is to add test loaders in eval
        else:
            return val_loader_names + test_loader_names, val_loaders + test_loaders

    def val_dataloader(self):
        val_loader_names, val_loaders = self._eval_dataloaders()
        self.val_loader_names = val_loader_names
        return val_loaders

    def test_dataloader(self):
        test_loader_names, test_loaders = self._eval_dataloaders()
        self.test_loader_names = ["final/" + name for name in test_loader_names]
        return test_loaders


### pytorch-lightning utils and entrypoint ###

def create_trainer(config, **kwargs):
    callbacks: List[pl.Callback] = []
    logger = None

    # WandB Logging
    if config.get("wandb") is not None:
        # Pass in wandb.init(config=) argument to get the nice 'x.y.0.z' hparams logged
        # Can pass in config_exclude_keys='wandb' to remove certain groups
        import wandb

        logger = CustomWandbLogger(
            config=utils.to_dict(config, recursive=True),
            settings=wandb.Settings(start_method="fork"),
            **config.wandb,
        )

    # Lightning callbacks
    if "callbacks" in config:
        for _name_, callback in config.callbacks.items():
            if config.get("wandb") is None and _name_ in ["learning_rate_monitor"]:
                continue
            log.info(f"Instantiating callback <{registry.callbacks[_name_]}>")
            callback._name_ = _name_
            callbacks.append(utils.instantiate(registry.callbacks, callback))

    # Add ProgressiveResizing callback
    if config.callbacks.get("progressive_resizing", None) is not None:
        num_stages = len(config.callbacks.progressive_resizing.stage_params)
        print(f"Progressive Resizing: {num_stages} stages")
        for i, e in enumerate(config.callbacks.progressive_resizing.stage_params):
            # Stage params are resolution and epochs, pretty print
            print(f"\tStage {i}: {e['resolution']} @ {e['epochs']} epochs")

    # Configure ddp automatically
    n_devices = config.trainer.get('devices', 1)
    if isinstance(n_devices, Sequence):  # trainer.devices could be [1, 3] for example
        n_devices = len(n_devices)
    if n_devices > 1 and config.trainer.get('strategy', None) is None:
        config.trainer.strategy = dict(
            _target_='pytorch_lightning.strategies.DDPStrategy',
            find_unused_parameters=False,
            gradient_as_bucket_view=True,  # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
        )

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    # special processing for seqlen warmup scheduler (reload)
    if config.callbacks.get("seqlen_warmup_reload", None) is not None:
        # we need to instantiate manually instead of with hydra, since it expects a dict instead of a hydra config for the accumulate_grad_batches
        # so we convert everything to dicts (from hydra configs)
        trainer_config_dict = dict(config.trainer)
        epochs_cume = 0  # track cumulative epochs
        accumulate_grad_schedule = {}  # contains the accumulate_grad_batches schedule to init the trainer

        for stage in config.callbacks.seqlen_warmup_reload.stage_params:
            batch_size = stage['batch_size']  # curr batch size at this stage
            grad_accum_factor = config.train.global_batch_size // batch_size  # grad accum factor for this stage
            accumulate_grad_schedule[epochs_cume] = grad_accum_factor  # set the grad accum factor for this stage
            epochs_cume += stage['epochs']  # increment epochs_cume for next stage
        trainer_config_dict['accumulate_grad_batches'] = accumulate_grad_schedule  # set the accumulate_grad_batches schedule
        trainer_config_dict.pop('_target_')  # only hydra uses this to instantiate
        # Set DDPStrategy to work with pl.Trainer
        config.trainer.pop('strategy')
        trainer_config_dict['strategy'] = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
        trainer = pl.Trainer(**trainer_config_dict, callbacks=callbacks, logger=logger)
    else:
        trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)    

    return trainer


def train(config):
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)
    trainer = create_trainer(config)
    model = SequenceLightningModule(config)

    # Load pretrained_model if specified #!! should be commented out
    # if config.train.get("pretrained_model_path", None) is not None:
    #     # PTL style.  Note, method returns a new model object, and need to pass config.
    #     model = SequenceLightningModule.load_from_checkpoint(
    #         config.train.pretrained_model_path,
    #         config=config,
    #         # strict=True ##!!!!!here is important!!!!!
    #         strict=False,
    #     )
    #     print(" ---------- just load the backbone from ", config.train.pretrained_model_path)
        
        # for name, param in model.named_parameters():
        #     print(f"参数 {name}: {param.data[:5]}")  # 打印前5个值
    # trainer.validate(model)

    # Run initial validation epoch (useful for debugging, finetuning)
    # if config.train.validate_at_start:
    #     print("Running validation before training")
    #     trainer.validate(model)

    # if config.train.ckpt is not None:
    #     trainer.fit(model, ckpt_path=config.train.ckpt)
    # else:
    #     trainer.fit(model)

    # if config.train.test:
    #     # load weight from fine-tuned ckpt
    #     # best_val_ckpt = "/qihao/work/caduceus/outputs/downstream/longrange_benchmark/crab_grad_seqlen-1k_d_model-24_n_layer-3_lr-8e-3_mambaBCnorm_Xavier_bf16-mixed_weighttie_nomoeloss_attntie/seed-1/checkpoints/val/loss.ckpt"
    #     # print(f"test with loaded weight from {best_val_ckpt}")
    #     # trainer.test(model, ckpt_path=best_val_ckpt)
    #     trainer.test(model)
        #!!should be left
    if config.train.test:
        best_val_ckpt = config.train.pretrained_model_path
        # Update config so we do not load just the backbone
        config.train.pretrained_model_state_hook.update({"_name_": None})
        # Remove validation loader
        config.train.update({"remove_val_loader_in_eval": True})
        config.train.update({"remove_test_loader_in_eval": False})
        ckpt = torch.load(best_val_ckpt)
        log.info(f"Loaded best validation checkpoint from epoch {ckpt['epoch']}")
        trainer.validate(model, ckpt_path=best_val_ckpt)

    model.close()


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):

    # Process config:
    # - register evaluation resolver
    # - filter out keys used only for interpolation
    # - optional hooks, including disabling python warnings or debug friendly configuration
    config = utils.train.process_config(config)

    # Pretty print config using Rich library
    utils.train.print_config(config, resolve=True)

    train(config)


if __name__ == "__main__":
    main()
