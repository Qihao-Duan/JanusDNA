"""Dataloaders for genomics datasets, including pretraining and downstream tasks.

    - Adapted from:
        https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
    - Adapted from:
        https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
"""
import os
import copy
from typing import Any, List, Union

import torch
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader

import numpy as np
import random

from caduceus.tokenization_caduceus import CaduceusTokenizer
import src.utils.train
from src.dataloaders.base import SequenceDataset, default_data_path
from src.dataloaders.datasets.genomic_bench_dataset import GenomicBenchmarkDataset
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.dataloaders.datasets.hg38_dataset import HG38Dataset
from src.dataloaders.datasets.nucleotide_transformer_dataset import NucleotideTransformerDataset
from src.dataloaders.fault_tolerant_sampler import FaultTolerantDistributedSampler
from src.dataloaders.fault_tolerant_sampler import RandomFaultTolerantSampler

from src.dataloaders.datasets.DNAlong_benchmark import AkitaDataset, BasenjiDataSet, EPIseqDataSet, EQTLseqDataSet

logger = src.utils.train.get_logger(__name__)


class HG38(SequenceDataset):
    """
    Base class, other dataloaders can inherit from this class.

    You must implement the following functions:
        - __init__
        - setup

    You can then use (already have access to) the following functions:
        - train_dataloader
        - val_dataloader
        - test_dataloader

    """
    _name_ = "hg38"  # this name is how the dataset config finds the right dataloader

    def __init__(self, bed_file, fasta_file, tokenizer_name=None, dataset_config_name=None, max_length=1024, d_output=2,
                 rc_aug=False,
                 max_length_val=None, max_length_test=None, val_ratio=0.0005, val_split_seed=2357,
                 add_eos=False, detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, shuffle=False,
                 num_workers=1,
                 fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                 mlm=False, mlm_probability=0.15,
                 compare_AR_MS=False,
                 *args, **kwargs):
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug  # reverse compliment augmentation
        self.max_length = max_length
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.bed_file = bed_file
        self.fasta_file = fasta_file
        
        self.compare_AR_MS = compare_AR_MS

        # handle if file paths are None (default paths)
        if self.bed_file is None:
            self.bed_file = default_data_path / self._name_ / "human-sequences.bed"
        if self.fasta_file is None:
            self.fasta_file = default_data_path / self._name_ / "hg38.ml.fa"

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

        self.mlm = mlm
        self.mlm_probability = mlm_probability

        # To be instantiated in `setup`
        self.tokenizer = None
        self.vocab_size = 0

    def setup(self, stage=None):
        """Set up the tokenizer and init the datasets."""
        # TODO instantiate with registry

        if self.tokenizer_name == "char":
            logger.info("**Using Char-level tokenizer**")
            # self.tokenizer = CharacterTokenizer(
            #     characters=["A", "C", "G", "T", "N"],
            #     model_max_length=self.max_length,
            #     add_special_tokens=False,
            # )
            self.tokenizer = CaduceusTokenizer(
                model_max_length=self.max_length,
                # add_special_tokens=False ##!
            )
        else:
            raise NotImplementedError(f"Tokenizer {self.tokenizer_name} not implemented.")

        self.vocab_size = len(self.tokenizer)

        self.init_datasets()  # creates the datasets.  You can also just create this inside the setup() here.

    def init_datasets(self):
        """Init the datasets (separate from the tokenizer)"""

        # delete old datasets to free memory
        if hasattr(self, "dataset_train"):
            self.dataset_train.fasta.seqs.close()
            del self.dataset_train.fasta.seqs

        # delete old datasets to free memory
        if hasattr(self, "dataset_test"):
            self.dataset_test.fasta.seqs.close()
            del self.dataset_test.fasta.seqs

        # Create all splits: torch datasets
        self.dataset_train, self.dataset_val, self.dataset_test = [
            HG38Dataset(split=split,
                        bed_file=self.bed_file,
                        fasta_file=self.fasta_file,
                        max_length=max_len,
                        tokenizer=self.tokenizer,  # pass the tokenize wrapper
                        tokenizer_name=self.tokenizer_name,
                        add_eos=self.add_eos,
                        return_seq_indices=False,
                        rc_aug=self.rc_aug,
                        return_augs=False,
                        mlm=self.mlm,
                        mlm_probability=self.mlm_probability, )
            for split, max_len in
            zip(["train", "valid", "test"], [self.max_length, self.max_length_val, self.max_length_test])
        ]

        return

    def train_dataloader(self, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            # TD [2022-12-26]: We need the distributed_sampler_kwargs in case of model parallel:
            # In that case the number of replicas and the data parallel rank are more complicated.
            # distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            # manully set these values
            num_replicas = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            sampler = (FaultTolerantDistributedSampler(
                self.dataset_train,
                num_replicas=num_replicas,
                rank=rank
                # **distributed_sampler_kwargs
            ) if self.ddp else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    "epoch": self.fast_forward_epochs,
                    "counter": self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        print(f"shuffle: {shuffle}!!!!!")
        print(f"sampler: {sampler}!!!!!")
        loader = self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                   shuffle=shuffle, sampler=sampler, **kwargs)
        return loader

    def val_dataloader(self, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        kwargs["drop_last"] = False
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval, **kwargs)

    def test_dataloader(self, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        kwargs["drop_last"] = False
        # TODO: Should have separate train and eval loaders
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval, **kwargs)

    @staticmethod
    def _data_loader(dataset: Dataset, batch_size: int, shuffle: bool = False, sampler=None, **kwargs) -> DataLoader:
        
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32 + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            worker_init_fn=seed_worker,
            **kwargs,
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"]
            # TD [2022-08-07] ["epoch_loop.batch_progress"]["total"]["completed"] is 1 iteration
            # behind, so we're using the optimizer"s progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"][
                "completed"]
        # At this point the train loader hasn't been constructed yet


class GenomicBenchmark(HG38):
    _name_ = "genomic_benchmark"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(
            self, dataset_name, train_val_split_seed,
            dest_path=None, tokenizer_name="char", d_output=None, rc_aug=False,
            conjoin_train=False, conjoin_test=False,
            max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
            padding_side="left", val_ratio=0.0005, val_split_seed=2357, add_eos=False,
            detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
            shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
            fast_forward_epochs=None, fast_forward_batches=None, *args, **kwargs
    ):

        self.dataset_name = dataset_name
        self.train_val_split_seed = train_val_split_seed
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == "char":
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=["A", "C", "G", "T", "N"],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                # add_special_tokens=False,
                padding_side=self.padding_side,
            )

        # Create all splits: torch datasets (only train/test in this benchmark, val created below)
        self.dataset_train, self.dataset_test = [
            GenomicBenchmarkDataset(
                split=split,
                max_length=max_len,
                dataset_name=self.dataset_name,
                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                tokenizer_name=self.tokenizer_name,
                use_padding=self.use_padding,
                d_output=self.d_output,
                add_eos=self.add_eos,
                dest_path=self.dest_path,
                rc_aug=self.rc_aug,
                conjoin_train=self.conjoin_train,
                conjoin_test=self.conjoin_test,
                return_augs=False
            )
            for split, max_len in zip(["train", "test"], [self.max_length, self.max_length_val])
        ]

        val_data, train_data = torch.utils.data.random_split(
            list(zip(self.dataset_train.all_seqs, self.dataset_train.all_labels)),
            lengths=[0.1, 0.9],
            generator=torch.Generator().manual_seed(self.train_val_split_seed)
        )
        self.dataset_val = copy.deepcopy(self.dataset_train)
        self.dataset_train.all_seqs = [train_data[i][0] for i in range(len(train_data))]
        self.dataset_train.all_labels = [train_data[i][1] for i in range(len(train_data))]

        self.dataset_val.all_seqs = [val_data[i][0] for i in range(len(val_data))]
        self.dataset_val.all_labels = [val_data[i][1] for i in range(len(val_data))]
        self.dataset_val.split = "val"


class NucleotideTransformer(HG38):
    _name_ = "nucleotide_transformer"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, train_val_split_seed,
                 tokenizer_name="char", d_output=None, rc_aug=False,
                 conjoin_train=False, conjoin_test=False,
                 max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                 padding_side="left", val_ratio=0.0005, val_split_seed=2357, add_eos=False,
                 detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=True, shuffle_eval=None, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, *args, **kwargs):

        self.dataset_name = dataset_name
        self.train_val_split_seed = train_val_split_seed
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.shuffle_eval = shuffle_eval if shuffle_eval is not None else shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == "char":
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=["A", "C", "G", "T", "N"],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                # add_special_tokens=False,
                padding_side=self.padding_side,
            )

        # Create all splits: torch datasets (only train/test in this benchmark)
        # self.dataset_train, self.dataset_val = [
        self.dataset_train, self.dataset_test = [
            NucleotideTransformerDataset(
                split=split,
                max_length=max_len,
                tokenizer=self.tokenizer,  # pass the tokenize wrapper
                dataset_name=self.dataset_name,
                tokenizer_name=self.tokenizer_name,
                use_padding=self.use_padding,
                d_output=self.d_output,
                add_eos=self.add_eos,
                rc_aug=self.rc_aug,
                conjoin_train=self.conjoin_train,
                conjoin_test=self.conjoin_test,
                return_augs=False
            )
            for split, max_len in zip(["train", "test"], [self.max_length, self.max_length_val])
        ]

        ds_train_val_split = self.dataset_train.seqs.train_test_split(
            test_size=0.1,
            seed=self.train_val_split_seed
        )
        self.dataset_val = copy.deepcopy(self.dataset_train)
        self.dataset_train.seqs = ds_train_val_split["train"]

        self.dataset_val.split = "val"
        self.dataset_val.seqs = ds_train_val_split["test"]


""" ----------------------- DNALong Benchmark ---------------------------"""


# todo
# contact_map_prediction
class AkitaBenchmark(HG38):
    _name_ = "akita_benchmark"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                 max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                 padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False,
                 detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, 
                 
                 cell_type='HFF',
                 conjoin_train=False, conjoin_test=False,
                 *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.cell_type = cell_type
        
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        
        SUPPORTED_CELL_TYPE = ["HFF", "H1hESC", "GM12878", "IMR90", "HCT116"]
        assert self.cell_type in SUPPORTED_CELL_TYPE, "Error: cell type {self.cell_type} is not supported for Contact Map Prediction Benchmark"

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_
            
        self.train_data_path = self.dest_path + '/contact_map_prediction/targets/train-*.tfr'
        self.valid_data_path = self.dest_path + '/contact_map_prediction/targets/valid-*.tfr'
        self.test_data_path = self.dest_path + '/contact_map_prediction/targets/test-*.tfr'

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                # add_special_tokens=False,
                padding_side=self.padding_side,
            )

        self.dataset_train = AkitaDataset(self.train_data_path, self.cell_type, self.tokenizer, conjoin_train=self.conjoin_train, conjoin_test=self.conjoin_test, add_eos=self.add_eos)
        self.dataset_val = AkitaDataset(self.valid_data_path, self.cell_type, self.tokenizer, conjoin_train=self.conjoin_train, conjoin_test=self.conjoin_test, add_eos=self.add_eos)
        self.dataset_test = AkitaDataset(self.test_data_path, self.cell_type, self.tokenizer, conjoin_train=self.conjoin_train, conjoin_test=self.conjoin_test, add_eos=self.add_eos)
        
        
    def train_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The train dataloader """
        print("train dataloader")
        return torch.utils.data.DataLoader(self.dataset_train, num_workers=0, batch_size=self.batch_size)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return torch.utils.data.DataLoader(self.dataset_val, num_workers=0, batch_size=self.batch_size)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return torch.utils.data.DataLoader(self.dataset_test, num_workers=0, batch_size=self.batch_size)


# regulatory_sequence_activity
# ! seems to only have mouse, should support human also?!

class EnformerBenchmark(HG38):
    _name_ = "enformer_benchmark"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                 max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                 padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False,
                 detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, 
                 organism="human",
                 *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.organism = organism

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

  
        self.data_path = self.dest_path + 'regulatory_sequence_activity_prediction/'
        
        self.SEQUENCE_LENGTH = 196_608
        self.BIN_SIZE = 128
        self.TARGET_LENGTH = 896
        
        assert self.organism == "human" or self.organism == "mouse", "Error: {self.organism} is not supported by 'regulatory_sequence_activity' mission"
        self.fasta_path = self.data_path + 'regulatory_sequence_activity_prediction/human/seqs/hg38.ml.fa' if self.organism == "human" else self.data_path + 'regulatory_sequence_activity_prediction/mouse/seqs/mm10.ml.fa'

        
        
        

    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                # add_special_tokens=False,
                padding_side=self.padding_side,
            )

        self.dataset_train = BasenjiDataSet(self.data_path, self.organism, "train", self.SEQUENCE_LENGTH, self.fasta_path)

        self.dataset_val = BasenjiDataSet(self.data_path, self.organism, "valid", self.SEQUENCE_LENGTH, self.fasta_path)

        self.dataset_test = BasenjiDataSet(self.data_path, self.organism, "test", self.SEQUENCE_LENGTH, self.fasta_path)
        



# eqtl_prediction
class EQTLBenchmark(HG38):
    _name_ = "eqtl_benchmark"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                 max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                 padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False,
                 detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, 
                 
                 cell_type="Adipose_Subcutaneous",
                 conjoin_train=False, conjoin_test=False,
                 *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.cell_type = cell_type
        
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

        supported_cell_type = ['Adipose_Subcutaneous', 'Artery_Tibial', 'Cells_Cultured_fibroblasts', 'Muscle_Skeletal', 'Nerve_Tibial', 'Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg', 'Thyroid', 'Whole_Blood']
        assert self.cell_type  in supported_cell_type, "Error: cell type {self.cell_type} is not supported"


    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CaduceusTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                # add_special_tokens=False,
                padding_side=self.padding_side,
            )

        task_root_path = os.path.join(self.dest_path, "eQTL")
        eQTL_config_file = os.path.join(task_root_path, "config", "gtex_hg38.%s.config" % (self.cell_type))
        self.dataset_train = EQTLseqDataSet(self.tokenizer, self.tokenizer_name, task_root_path, eQTL_config_file, 'train', self.add_eos, self.conjoin_train, self.conjoin_test)
        self.dataset_val = EQTLseqDataSet(self.tokenizer, self.tokenizer_name, task_root_path, eQTL_config_file, 'valid', self.add_eos, self.conjoin_train, self.conjoin_test)
        self.dataset_test = EQTLseqDataSet(self.tokenizer, self.tokenizer_name, task_root_path, eQTL_config_file, 'test', self.add_eos, self.conjoin_train, self.conjoin_test)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The train dataloader """
        print("train dataloader here")
        return torch.utils.data.DataLoader(self.dataset_train, num_workers=0, batch_size=self.batch_size)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return torch.utils.data.DataLoader(self.dataset_val, num_workers=0, batch_size=self.batch_size_eval)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return torch.utils.data.DataLoader(self.dataset_test, num_workers=0, batch_size=self.batch_size_eval)


# enhancer_target_gene_prediction
class EnhancerPromoterBenchmark(HG38):
    _name_ = "enhancer_promoter_benchmark"
    l_output = 0  # need to set this for decoder to work correctly

    def __init__(self, dataset_name, dest_path=None, tokenizer_name='char', d_output=None, rc_aug=False,
                 max_length=1024, use_padding=True, max_length_val=None, max_length_test=None,
                 padding_side='left', return_mask=False, val_ratio=0.0005, val_split_seed=2357, add_eos=False,
                 detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=True, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None, 
                 
                 conjoin_train=False, conjoin_test=False,
                 *args, **kwargs):

        self.dataset_name = dataset_name
        self.dest_path = dest_path
        self.tokenizer_name = tokenizer_name
        self.d_output = d_output
        self.rc_aug = rc_aug
        self.max_length = max_length
        self.use_padding = use_padding
        self.max_length_val = max_length_val if max_length_val is not None else max_length
        self.max_length_test = max_length_test if max_length_test is not None else max_length
        self.padding_side = padding_side
        self.return_mask = return_mask
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test

        if self.dest_path is None:
            self.dest_path = default_data_path / self._name_

        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant


    def setup(self, stage=None):
        # TODO instantiate with registry

        if self.tokenizer_name == 'char':
            print("**Using Char-level tokenizer**")
            self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_length + 2,  # add 2 since default adds eos/eos tokens, crop later
                # add_special_tokens=False,
                padding_side=self.padding_side,
            )
            
        ETGP_config_file = os.path.join(self.dest_path, "enhancer_target_gene", "config", "CRISPRi_EPI_K562_hg19.config")
        task_root_path = os.path.join(self.dest_path, "enhancer_target_gene")
            
        self.dataset_train = EPIseqDataSet(task_root_path, ETGP_config_file, 'train', tokenizer=self.tokenizer, add_eos=self.add_eos, conjoin_train=self.conjoin_train, conjoin_test=self.conjoin_test)
        self.dataset_val = EPIseqDataSet(task_root_path, ETGP_config_file, 'valid', tokenizer=self.tokenizer, add_eos=self.add_eos, conjoin_train=self.conjoin_train, conjoin_test=self.conjoin_test)
        self.dataset_test = EPIseqDataSet(task_root_path, ETGP_config_file, 'test', tokenizer=self.tokenizer, add_eos=self.add_eos, conjoin_train=self.conjoin_train, conjoin_test=self.conjoin_test)
        


    def train_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The train dataloader """
        print("train dataloader")
        return torch.utils.data.DataLoader(self.dataset_train, num_workers=0, batch_size=self.batch_size)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return torch.utils.data.DataLoader(self.dataset_val, num_workers=0, batch_size=self.batch_size)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader, it's a dummy loader just to make the trainer happy, we don't use it."""
        return torch.utils.data.DataLoader(self.dataset_test, num_workers=0, batch_size=self.batch_size)
