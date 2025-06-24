import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import torch
import sys
import os
import json
from collections import namedtuple
import logging
import random
import selene_sdk
from abc import ABCMeta
from selene_sdk.targets import Target, GenomicFeatures
from selene_sdk.samplers import Sampler
from selene_sdk.utils import get_indices_and_probabilities
# from selene_sdk.samplers import RandomPositionsSampler
from selene_sdk.samplers.dataloader import SamplerDataLoader
import numpy as np
import pandas as pd
import pyfaidx
import functools
from kipoiseq import Interval
import os
from natsort import natsorted
import glob
from tqdm import tqdm
import tabix
import pyBigWig
import kipoiseq
from torch.utils.data import Dataset, DataLoader

from src.dataloaders.utils.rc import coin_flip, string_reverse_complement

def load_data(root='./DNALongBench/data', task_name = 'regulatory_sequence_activity', organism = 'human', cell_type='HFF', batch_size=16, sequence_length=196608):
    if task_name == 'regulatory_sequence_activity':
        assert organism == "human" or organism == "mouse"
        human_fasta_path = root + 'regulatory_sequence_activity_prediction/human/seqs/hg38.ml.fa'
        mouse_fasta_path = root + 'regulatory_sequence_activity_prediction/mouse/seqs/mm10.ml.fa'
        data_path = root + 'regulatory_sequence_activity_prediction/'
        
        # SEQUENCE_LENGTH = 196608
        # BIN_SIZE = 128
        # TARGET_LENGTH = 896

        fasta_path = human_fasta_path if organism == "human" else mouse_fasta_path

        train_dataset = BasenjiDataSet(data_path, organism, 'train', sequence_length, fasta_path)
        valid_dataset = BasenjiDataSet(data_path, organism, 'valid', sequence_length, fasta_path)
        test_dataset = BasenjiDataSet(data_path, organism, 'test', sequence_length, fasta_path)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
        
    elif task_name == 'contact_map_prediction':
        train_data_path = root + 'contact_map_prediction/targets/train-*.tfr'
        valid_data_path = root + 'contact_map_prediction/targets/valid-*.tfr'
        test_data_path = root + 'contact_map_prediction/targets/test-*.tfr'
        SEQUENCE_LENGTH = 1048576
        TARGET_LENGTH = 99681

        train_dataset = AkitaDataset(train_data_path, cell_type = cell_type, target_length=TARGET_LENGTH)
        valid_dataset = AkitaDataset(valid_data_path, cell_type = cell_type, target_length=TARGET_LENGTH)
        test_dataset = AkitaDataset(test_data_path, cell_type = cell_type, target_length=TARGET_LENGTH)
     
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
        
    elif task_name == 'transcription_initiation_signal_prediction':
        
        genome, noblacklist_genome = get_genomes(root+"transcription_initiation_signal_prediction/seqs/Homo_sapiens.GRCh38.dna.primary_assembly.fa")
        tfeature = GenomicSignalFeatures([root+"transcription_initiation_signal_prediction/targets/agg.plus.bw.bedgraph.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.encodecage.plus.v2.bedgraph.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.encoderampage.plus.v2.bedgraph.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.plus.grocap.bedgraph.sorted.merged.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.plus.allprocap.bedgraph.sorted.merged.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.minus.allprocap.bedgraph.sorted.merged.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.minus.grocap.bedgraph.sorted.merged.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.encoderampage.minus.v2.bedgraph.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.encodecage.minus.v2.bedgraph.bw",
        root+"transcription_initiation_signal_prediction/targets/agg.minus.bw.bedgraph.bw"],
                                       ['cage_plus','encodecage_plus','encoderampage_plus', 'grocap_plus','procap_plus','procap_minus','grocap_minus'
        ,'encoderampage_minus', 'encodecage_minus','cage_minus'],
                                       (100000,),
                                       [root+"transcription_initiation_signal_prediction/targets/blacklists/fantom.blacklist8.plus.bed.gz",root+"transcription_initiation_signal_prediction/targets/blacklists/fantom.blacklist8.minus.bed.gz"],
                                       [0,9], [1,8], [0.61357, 0.61357])


        
        sampler = RandomPositionsSampler(
                        reference_sequence = genome,
                        target= tfeature,
                        features = [''],
                        test_holdout=['chr8', 'chr9'],
                        validation_holdout= ['chr10'],
                        sequence_length= 100000,
                        center_bin_to_predict= 100000,
                        position_resolution=1,
                        random_shift=0,
                        random_strand=False
        )
        sampler.mode="train"
        train_loader = SamplerDataLoader(sampler, num_workers=1, batch_size=16, seed=3)

        validseq = noblacklist_genome.get_encoding_from_coords("chr10", 0, 114364328)
        validcage = tfeature.get_feature_data("chr10", 0, 114364328)
        class ValidDataset(Dataset):
            def __init__(self, seq, cage, window_size=100000, step_size=50000):
                """
                seq: (N, 4) numpy array, the one-hot-encoded genomic sequence
                cage: (10, N) numpy array, the target features
                window_size: int, size of the sliding window
                step_size: int, step size for the sliding window
                """
                self.seq = seq
                self.cage = cage
                self.window_size = window_size
                self.step_size = step_size
                self.num_windows = (seq.shape[0] - window_size) // step_size + 1

            def __len__(self):
                return self.num_windows

            def __getitem__(self, idx):
                """
                Returns a tuple (input_sequence, target_features) for the idx-th window.
                """
                start = idx * self.step_size
                end = start + self.window_size
                input_seq = self.seq[start:end, :]  # Shape: (window_size, 4)
                target_cage = self.cage[:, start:end]  # Shape: (10, window_size)
                return input_seq, target_cage


        # Create the validation dataset
        valid_dataset = ValidDataset(validseq, validcage, window_size=100000, step_size=50000)

        # Create the validation DataLoader
        valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)

        chr8_seq = genome.get_encoding_from_coords('chr8',0, 145138636) 
        chr8_cage = tfeature.get_feature_data('chr8',0, 145138636)

        chr9_seq = genome.get_encoding_from_coords('chr9',0, 138394717) 
        chr9_cage = tfeature.get_feature_data('chr9',0, 138394717)
        class TestDataset(Dataset):
            def __init__(self, seq, cage, window_size=100000, step_size=50000):
                """
                seq: (N, 4) numpy array, the one-hot-encoded genomic sequence
                cage: (10, N) numpy array, the target features
                window_size: int, size of the sliding window
                step_size: int, step size for the sliding window
                """
                self.seq = seq
                self.cage = cage
                self.window_size = window_size
                self.step_size = step_size
                self.num_windows = (seq.shape[0] - window_size) // step_size + 1

            def __len__(self):
                return self.num_windows

            def __getitem__(self, idx):
                """
                Returns a tuple (input_sequence, target_features) for the idx-th window.
                """
                start = idx * self.step_size
                end = start + self.window_size
                input_seq = self.seq[start:end, :]  # Shape: (window_size, 4)
                target_cage = self.cage[:, start:end]  # Shape: (10, window_size)
                return input_seq, target_cage


        # Combine the data for chr8 and chr9 into a single dataset
        class CombinedTestDataset(Dataset):
            def __init__(self, seqs, cages, window_size=100000, step_size=50000):
                """
                seqs: List of numpy arrays, each with shape (N, 4).
                cages: List of numpy arrays, each with shape (10, N).
                window_size: int, size of the sliding window.
                step_size: int, step size for the sliding window.
                """
                self.datasets = []
                for seq, cage in zip(seqs, cages):
                    self.datasets.append(TestDataset(seq, cage, window_size, step_size))

                # Calculate the cumulative number of windows for indexing
                self.cumulative_sizes = np.cumsum([len(dataset) for dataset in self.datasets])

            def __len__(self):
                return self.cumulative_sizes[-1]

            def __getitem__(self, idx):
                """
                Identifies the appropriate dataset and retrieves the corresponding item.
                """
                # Determine which dataset the idx belongs to
                for i, size in enumerate(self.cumulative_sizes):
                    if idx < size:
                        dataset_idx = i
                        if i > 0:
                            idx -= self.cumulative_sizes[i - 1]
                        break

                # Get the item from the appropriate dataset
                return self.datasets[dataset_idx][idx]


        # Combine the data into the test dataset
        test_dataset = CombinedTestDataset([chr8_seq, chr9_seq], [chr8_cage, chr9_cage], window_size=100000, step_size=50000)

        # Create the test DataLoader
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

        return train_loader, valid_loader, test_loader
    
    elif task_name == 'enhancer_target_gene_prediction':
        ETGP_config_file = os.path.join(root, "enhancer_target_gene", "config", "CRISPRi_EPI_K562_hg19.config")
        task_root_path = os.path.join(root, "enhancer_target_gene")
        train_dataset = EPIseqDataSet(task_root_path, ETGP_config_file, 'train')
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=batch_size)
        valid_dataset = EPIseqDataSet(task_root_path, ETGP_config_file, 'valid')
        valid_loader = torch.utils.data.DataLoader(valid_dataset, num_workers=0, batch_size=batch_size)
        test_dataset = EPIseqDataSet(task_root_path, ETGP_config_file, 'test')
        test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=batch_size)

    elif task_name == 'eqtl_prediction':
        supported_cell_type = ['Adipose_Subcutaneous', 'Artery_Tibial', 'Cells_Cultured_fibroblasts', 'Muscle_Skeletal', 'Nerve_Tibial', 'Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg', 'Thyroid', 'Whole_Blood']
        if cell_type not in supported_cell_type:
            print(f"Error: cell type {cell_type} is not supported", file = sys.stderr)
            print("Supported cell types are: ", supported_cell_type, file = sys.stderr)
        eQTL_config_file = os.path.join(root, "eQTL", "config", "gtex_hg38.%s.config" % (cell_type))
        task_root_path = os.path.join(root, "eQTL")
        train_dataset = EQTLseqDataSet(task_root_path, eQTL_config_file, 'train')
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=batch_size)
        valid_dataset = EQTLseqDataSet(task_root_path, eQTL_config_file, 'valid')
        valid_loader = torch.utils.data.DataLoader(valid_dataset, num_workers=0, batch_size=batch_size)
        test_dataset = EQTLseqDataSet(task_root_path, eQTL_config_file, 'test')
        test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=batch_size)
    
        
    return train_loader, valid_loader, test_loader



class AkitaDataset(torch.utils.data.IterableDataset):
    def __init__(self, tfr_pattern, cell_type, tokenizer, add_eos=False, target_length=99681, tokenizer_name='char',
                 conjoin_train=False, conjoin_test=False):
        super(AkitaDataset).__init__()
        self.dataset = self.read_tfr(tfr_pattern)
        self.cell_type = cell_type
        target_ind_dict = {'HFF': 0, 'H1hESC': 1, 'GM12878': 2, 'IMR90': 3, 'HCT116': 4}
        self.target_ind = target_ind_dict[self.cell_type]
        self.target_length=target_length
        
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.add_eos = add_eos
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test


    def onehot_to_char(self, onehot):
        """
        Convert a one-hot encoded DNA sequence to a string of characters.

        Parameters:
        onehot (numpy.ndarray): A 2D array where each row is a one-hot encoded nucleotide.

        Returns:
        str: The corresponding DNA sequence as a string of characters.
        """
        mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        return ''.join(mapping[row.argmax()] for row in onehot)
        

    
    # Adapted from https://github.com/calico/basenji/blob/498d0b1cd02e1ba11658d273ef257a07f5a69657/bin/tfr_qc.py#L98
    def file_to_records(self, filename):
        return tf.data.TFRecordDataset(filename, compression_type='ZLIB')
    def parse_proto(self, example_protos):
        features = {
            'sequence': tf.io.FixedLenFeature([], tf.string),
            'target': tf.io.FixedLenFeature([], tf.string)
          }
        parsed_features = tf.io.parse_example(example_protos, features=features)
        seq = tf.io.decode_raw(parsed_features['sequence'], tf.uint8)
        targets = tf.io.decode_raw(parsed_features['target'], tf.float16)
        return seq, targets

    def read_tfr(self, tfr_pattern):
        tfr_files = natsorted(glob.glob(tfr_pattern))
        if tfr_files:
            dataset = tf.data.Dataset.list_files(tf.constant(tfr_files), shuffle=False)
        else:
            print('Cannot order TFRecords %s' % tfr_pattern, file=sys.stderr)
            dataset = tf.data.Dataset.list_files(tfr_pattern)
        dataset = dataset.flat_map(self.file_to_records)
        # with tf.device('/cpu:0'):  
        #     dataset = dataset.map(self.parse_proto)
        #     dataset = dataset.batch(1)
        dataset = dataset.map(self.parse_proto)
        dataset = dataset.batch(1)
        return dataset
    def __iter__(self):
        for seq_raw, targets_raw in self.dataset:
            seq = seq_raw.numpy().reshape(-1,4).astype('int8')
            # print("Shape of Sequence:", seq.shape)
            seq_char = self.onehot_to_char(seq)
            seq_ids = self.tokenizer(
                seq_char,
                add_special_tokens=False,
                padding="do_not_pad",
                max_length=1048576,
                truncation=True,
            )["input_ids"]
            
            if self.conjoin_train or (self.conjoin_test and self.subset != "train"):
                seq_rc_char = string_reverse_complement(seq_ids)
                seq_rc_ids = self.tokenizer(
                    seq_rc_char,
                    add_special_tokens=False,
                    padding="do_not_pad",
                    max_length=1048576,
                    truncation=True,
                )["input_ids"]

                # need to handle eos here
                if self.add_eos:
                    # append list seems to be faster than append tensor
                    seq_rc_ids.append(self.tokenizer.sep_token_id)
                seq_ids_tensor = torch.stack((torch.LongTensor(seq_ids), torch.LongTensor(seq_rc_ids)), dim=1)
                # print("conjoin_test rc chain!")
            else:
                # convert to tensor
                seq_ids_tensor = torch.LongTensor(seq_ids) 
            
            targets = targets_raw.numpy().reshape(self.target_length,-1).astype('float16')
            # yield {
            #       "sequence": seq,
            #       "target": targets[:, self.target_ind],
                
            #   }
            yield seq_ids_tensor, targets[:, self.target_ind]


class FastaStringExtractor:

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


class BasenjiDataSet(torch.utils.data.IterableDataset):
    @staticmethod
    def get_organism_path(data_path, organism):
        return os.path.join(data_path, organism)

    @classmethod
    def get_metadata(cls, data_path, organism):
        # Keys:
        # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
        # pool_width, crop_bp, target_length
        path = os.path.join(cls.get_organism_path(data_path, organism), 'statistics.json')
        with tf.io.gfile.GFile(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def one_hot_encode(sequence):
        return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

    @classmethod
    def get_tfrecord_files(cls, data_path, organism, subset):
        # Sort the values by int(*).
        return sorted(tf.io.gfile.glob(os.path.join(
            cls.get_organism_path(data_path, organism), 'targets', f'{subset}-*.tfr'
        )), key=lambda x: int(x.split('-')[-1].split('.')[0]))

    @property
    def num_channels(self):
        metadata = self.get_metadata(self.data_path, self.organism)
        return metadata['num_targets']

    @staticmethod
    def deserialize(serialized_example, metadata):
        """Deserialize bytes stored in TFRecordFile."""
        # Deserialization
        feature_map = {
            'sequence': tf.io.FixedLenFeature([], tf.string),  # Ignore this, resize our own bigger one
            'target': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_example(serialized_example, feature_map)
        sequence = tf.io.decode_raw(example['sequence'], tf.bool)
        sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
        sequence = tf.cast(sequence, tf.float32)

        target = tf.io.decode_raw(example['target'], tf.float16)
        target = tf.reshape(target,
                            (metadata['target_length'], metadata['num_targets']))
        target = tf.cast(target, tf.float32)

        return {'sequence_old': sequence,
                'target': target}

    @classmethod
    def get_dataset(cls, data_path, organism, subset, num_threads=8):
        metadata = cls.get_metadata(data_path, organism)
        dataset = tf.data.TFRecordDataset(cls.get_tfrecord_files(data_path, organism, subset),
                                          compression_type='ZLIB',
                                          num_parallel_reads=num_threads).map(
            functools.partial(cls.deserialize, metadata=metadata)
        )
        return dataset

    def __init__(self, data_path, organism: str, subset: str, seq_len: int, fasta_path: str, n_to_test: int = -1):
        assert subset in {"train", "valid", "test"}
        assert organism in {"human", "mouse"}
        self.data_path = data_path
        self.organism = organism
        self.subset = subset
        self.base_dir = self.get_organism_path(data_path, organism)
        self.seq_len = seq_len
        self.fasta_reader = FastaStringExtractor(fasta_path)
        self.n_to_test = n_to_test
        with tf.io.gfile.GFile(f"{self.base_dir}/sequences.bed", 'r') as f:
            region_df = pd.read_csv(f, sep="\t", header=None)
            region_df.columns = ['chrom', 'start', 'end', 'subset']
            self.region_df = region_df.query('subset==@subset').reset_index(drop=True)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is None, "Only support single process loading"
        # If num_threads > 1, the following will actually shuffle the inputs! luckily we catch this with the sequence comparison
        basenji_iterator = self.get_dataset(self.data_path, self.organism, self.subset, num_threads=1).as_numpy_iterator()
        for i, records in enumerate(basenji_iterator):
            loc_row = self.region_df.iloc[i]
            target_interval = Interval(loc_row['chrom'], loc_row['start'], loc_row['end'])
            sequence_one_hot = self.one_hot_encode(self.fasta_reader.extract(target_interval.resize(self.seq_len)))
            if self.n_to_test >= 0 and i < self.n_to_test:
                old_sequence_onehot = records["sequence_old"]
                if old_sequence_onehot.shape[0] > sequence_one_hot.shape[0]:
                    diff = old_sequence_onehot.shape[0] - sequence_one_hot.shape[0]
                    trim = diff // 2
                    np.testing.assert_equal(old_sequence_onehot[trim:(-trim)], sequence_one_hot)
                elif sequence_one_hot.shape[0] > old_sequence_onehot.shape[0]:
                    diff = sequence_one_hot.shape[0] - old_sequence_onehot.shape[0]
                    trim = diff // 2
                    np.testing.assert_equal(old_sequence_onehot, sequence_one_hot[trim:(-trim)])
                else:
                    np.testing.assert_equal(old_sequence_onehot, sequence_one_hot)
            # yield {
            #     "sequence": sequence_one_hot,
            #     "target": records["target"],
            # }
            yield sequence_one_hot, records["target"]




# class EPIseqDataSet(torch.utils.data.IterableDataset):

#     def __init__(self, root_path, config_file, subset, tokenizer, tokenizer_name='char', add_eos=False, conjoin_train=False, conjoin_test=False):
#         super(EPIseqDataSet).__init__()
#         #########################
#         # load config
#         #########################
#         self.config = parse_config(config_file)
#         self.tokenizer = tokenizer
#         self.tokenizer_name = tokenizer_name
#         self.add_eos = add_eos
#         self.conjoin_train = conjoin_train
#         self.conjoin_test = conjoin_test
#         # init
#         self.organism = 'Unknown'
#         if self.config.get('organism', None) is not None:
#             self.organism = self.config['organism']
#         self.subset = subset
#         if self.subset is None: 
#             if self.config.get('subset', None) is not None:
#                 self.subset = self.config['subset']
#             else:
#                 print("subset is required either in the config file or passed as a parameter", file = sys.stderr)
#         try:
#             assert self.subset in ['train', 'valid', 'test']
#         except AssertionError:
#             print("subset must be either train, valid, or test", file = sys.stderr)
#         # verbose level
#         self.verbose = 0
#         if self.config.get('verbose', None) is not None:
#             self.verbose = self.config['verbose']
#         # show config
#         if self.verbose > 0:
#             print_config(self.config)
#         print("> load config done")

#         ############################
#         # load fasta file
#         ############################
#         if self.config.get('genome_fa', None) is None:
#             print('genome_fa is required in the config file', file = sys.stderr)
#         self.fasta_reader = FastaStringExtractor(os.path.join(root_path, self.config['genome_fa']))
#         print("> init fasta extractor done")

#         ############################
#         # load dataset, ie EPI data
#         ############################
#         if self.config.get('seq_len_cutoff', None) is None:
#             print('seq_len_cutoff is required in the config file', file = sys.stderr)
#             print('will use default value 450000', file = sys.stderr)
#             self.config['seq_len_cutoff'] = 450000
#         if self.config.get('enhancer_tabix_file', None) is None:
#             print('enhancer_tabix_file is required in the config file', file = sys.stderr)
#         self.dataset = parse_EPI(root_path, self.config['EPI_file'], self.config['enhancer_tabix_file'], self.fasta_reader, self.subset, self.config['seq_len_cutoff'], self.config['tss_flank_upstream'], self.config['tss_flank_downstream'], self.config['region_flank_upstream'], self.config['region_flank_downstream'])
#         if self.verbose > 0:
#             print(f'Loaded {len(self.dataset)} EPI records')

#     def __iter__(self):
#         # iterate over the data
#         target2int = {'positive':1, 'negative':0}
#         for index, record in enumerate(self.dataset):
#             # convert the sequence to one-hot encoding
#             # sequence = one_hot_encode(record[0])
#             # sequence = np.argmax(sequence, axis=-1)
#             # target = np.array(target2int[record[1]]).astype('long')
#             # #yield {
#             # #    'x': sequence,
#             # #    'y': target 
#             # #}
#             # yield sequence, target
#             seq_ids = self.tokenizer(
#                 record[0],
#                 add_special_tokens=False,
#                 padding="do_not_pad",
#                 max_length=self.config['seq_len_cutoff'],
#                 truncation=True,
#             )["input_ids"]
            
#             if self.conjoin_train or (self.conjoin_test and self.subset != "train"):
#                 # flip the sequence
#                 seq_rc = string_reverse_complement(record[0])
#                 seq_rc_ids = self.tokenizer(
#                     seq_rc,
#                     add_special_tokens=False,
#                     padding="do_not_pad",
#                     max_length=self.config['seq_len_cutoff'],
#                     truncation=True,
#                 )["input_ids"]
#                 # need to handle eos here
#                 if self.add_eos:
#                     # append list seems to be faster than append tensor
#                     seq_rc_ids.append(self.tokenizer.sep_token_id)
#                 seq_ids_tensor = torch.stack((torch.LongTensor(seq_ids), torch.LongTensor(seq_rc_ids)), dim=1)
#             else:
#                 seq_ids_tensor = torch.LongTensor(seq_ids)
                
#             target = np.array(target2int[record[1]]).astype('long')
            
#             yield seq_ids_tensor, target


class EPIseqDataSet(Dataset):
    def __init__(self, root_path, config_file, subset, tokenizer, tokenizer_name='char', add_eos=False, conjoin_train=False, conjoin_test=False):
        super(EPIseqDataSet, self).__init__()
        #########################
        # load config
        #########################
        self.config = parse_config(config_file)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.add_eos = add_eos
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.organism = self.config.get('organism', 'Unknown')
        self.subset = subset
        if self.subset is None:
            self.subset = self.config.get('subset', None)
            if self.subset is None:
                print("subset is required either in the config file or passed as a parameter", file=sys.stderr)
                sys.exit(1)
        assert self.subset in ['train', 'valid', 'test'], "subset must be either train, valid, or test"
        self.verbose = self.config.get('verbose', 0)
        if self.verbose > 0:
            print_config(self.config)
        print("> load config done")

        ############################
        # load fasta file
        ############################
        if self.config.get('genome_fa', None) is None:
            print('genome_fa is required in the config file', file=sys.stderr)
            sys.exit(1)
        self.fasta_reader = FastaStringExtractor(os.path.join(root_path, self.config['genome_fa']))
        print("> init fasta extractor done")

        ############################
        # load dataset, i.e., EPI data
        ############################
        self.config['seq_len_cutoff'] = self.config.get('seq_len_cutoff', 450000)
        if self.config.get('enhancer_tabix_file', None) is None:
            print('enhancer_tabix_file is required in the config file', file=sys.stderr)
            sys.exit(1)
        self.dataset = parse_EPI(
            root_path,
            self.config['EPI_file'],
            self.config['enhancer_tabix_file'],
            self.fasta_reader,
            self.subset,
            self.config['seq_len_cutoff'],
            self.config['tss_flank_upstream'],
            self.config['tss_flank_downstream'],
            self.config['region_flank_upstream'],
            self.config['region_flank_downstream']
        )
        self.dataset_size = len(self.dataset)
        print(f'Loaded {self.dataset_size} EPI records')

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        record = self.dataset[index]
        target2int = {'positive': 1, 'negative': 0}

        seq_ids = self.tokenizer(
            record[0],
            add_special_tokens=False,
            padding="do_not_pad",
            max_length=self.config['seq_len_cutoff'],
            truncation=True,
        )["input_ids"]

        if self.conjoin_train or (self.conjoin_test and self.subset != "train"):
            seq_rc = string_reverse_complement(record[0])
            seq_rc_ids = self.tokenizer(
                seq_rc,
                add_special_tokens=False,
                padding="do_not_pad",
                max_length=self.config['seq_len_cutoff'],
                truncation=True,
            )["input_ids"]
            if self.add_eos:
                seq_rc_ids.append(self.tokenizer.sep_token_id)
            seq_ids_tensor = torch.stack((torch.LongTensor(seq_ids), torch.LongTensor(seq_rc_ids)), dim=1)
        else:
            seq_ids_tensor = torch.LongTensor(seq_ids)

        target = np.array(target2int[record[1]]).astype('long')
        return seq_ids_tensor, target


def parse_EPI(root_path, EPI_file, blacklist_tabix_file, fasta_reader, subset_group_name, seq_len_cutoff = 450000, tss_flank_upstream = 3000, tss_flank_downstream = 3000, region_flank_upstream = 500, region_flank_downstream = 500):
    """
    this function will parse the following file format to reresent an enhancer-promoter interaction entry
        > gene_chrom  gene_start gene_end region_chrom region_start region_end gene_id gene_strand region_id target subset
        > other columns may exists but will be ignored
    then, the function will search for any enhancers between the tested region and the gene tss the sequence in these enhancers will be masked N. This is crucial because we want to ensure that these sequences don’t influence the model training process. 
    """
    # check subset
    if subset_group_name not in ['train', 'valid', 'test']:
        print(f'Error: unknown subset group {subset_group_name}')
    # load records as dataframe
    df = pd.read_csv(os.path.join(root_path, EPI_file), sep='\t', header=0)
    # load the tabix
    blacklist_tabix = tabix.open(os.path.join(root_path, blacklist_tabix_file))
    # parse the records and convert it into a list of DNA sequence and target (ie, true or false EPI)
    dataset = []
    print("> Start parsing EPI records to build the dataset %s" % (subset_group_name))
    N_total = df.shape[0]
    N_pass = 0
    N_skip_for_chrom_diff = 0 # skip if enhancer and gene tss are in different chromosomes
    N_skip_for_len = 0 # skip if the distance between enhancer and gene tss is too long
    N_skip_for_strand = 0 # skip if gene strand is not + or - or unknown
    N_skip_for_short = 0 # skip if the distance between enhancer and gene tss is 0, ie, they are overlapping
    for index, row in tqdm(df.iterrows(), total = N_total):
        gene_chrom = row['gene_chrom']
        gene_start = row['gene_start']
        gene_end = row['gene_end']
        region_chrom = row['region_chrom']
        region_start = row['region_start']
        region_end = row['region_end']
        gene_id = row['gene_id']
        gene_strand = row['gene_strand']
        region_id = row['region_id']
        target = row['target']
        subset_group = row['subset']
        # skip if in different subset group
        if subset_group != subset_group_name:
            continue
        # skip if in different chromosomes
        if gene_chrom != region_chrom:
            N_skip_for_chrom_diff += 1
            continue
        # get tss
        if gene_strand == '+':
            tss_start = gene_start # tss start from gene start
        elif gene_strand == '-':
            tss_start = gene_end - 1
        else:
            print(f'Warning: unknown strand {gene_strand}')
            N_skip_for_strand += 1
            continue
        tss_end = tss_start + 1
        # add flanking sequence
        tss_start = tss_start - tss_flank_upstream
        tss_end = tss_end + tss_flank_downstream
        region_start = region_start - region_flank_upstream
        region_end = region_end + region_flank_downstream
        # calculate the distance between tss and region
        distance = max(0, max(tss_start, region_start) - min(tss_end, region_end))
        # check if the sequence length is too long
        if distance > seq_len_cutoff:
            N_skip_for_len += 1
            continue
        # get the interverl between the region and tss
        sequence_start = min(tss_start, region_start)
        sequence_end = max(tss_end, region_end)
        # get the sequence
        region_interval = Interval(region_chrom, sequence_start, sequence_end)
        region_seq = fasta_reader.extract(region_interval)
        # mask any enhancer sequence
        if distance > 0:
            if tss_start > region_end:
                query_start = region_end
                query_end = tss_start
            else:
                query_start = tss_end
                query_end = region_start
            query_result = blacklist_tabix.query(region_chrom, query_start, query_end)
            for overlap in query_result:
                o_start = int(overlap[1])
                o_end = int(overlap[2])
                # mask the correspoinding DNA sequence as 'N'
                rev_start = o_start - sequence_start
                rev_end = o_end - sequence_start
                if rev_start > 0 and rev_start < len(region_seq) and rev_end > 0 and rev_end < len(region_seq):
                    region_seq = region_seq[:rev_start] + 'N' * (rev_end - rev_start) + region_seq[rev_end:]
        # debug
        #region_seq =  region_seq[:20000] + region_seq[-20000:]
        # flip (reverse complement) the sequence if gene is on the downstream of region (ie, genomic coordinate is larger)
        if gene_start > region_end:
            region_seq = rcDNA(region_seq) 
        # padding N to the sequence if the sequence
        # debug
        #if len(region_seq) <= 40000:
        #    region_seq = region_seq + 'N' * (40000 - len(region_seq))
        #else:
        #    region_seq = region_seq[:40000]
        if len(region_seq) <= seq_len_cutoff:
            region_seq = region_seq + 'N' * (seq_len_cutoff - len(region_seq))
        else:
            region_seq = region_seq[:seq_len_cutoff]
        N_pass +=1
        # append to the dataset
        dataset.append((region_seq, target))
    # report the statistics
    print("# Finish parsing EPI records")
    print("# Total records: ", N_total)
    print("# Skipped records due to different chromosomes: ", N_skip_for_chrom_diff)
    print("# Skipped records due to distance cutoff: ", N_skip_for_len)
    print("# Skipped records due to unknown strand: ", N_skip_for_strand)
    print("# Select records %s with subset %s " % (N_pass, subset_group_name))
    return dataset


def rcDNA(seq):
    """Get reverse complemented of a DNA sequence"""
    diccomseq = {'A':'T', 'T':'A', 'G':'C', 'C':'G', 'N':'N'}
    return "".join(diccomseq[nu.upper()] for nu in seq[::-1])


def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def parse_config(config_file):
    """
    load config as a dictionary
    """
    config_table = {}
    with open(config_file, 'r') as fin:
        for line in fin:
            row = line.strip().split()
            key = row[0]
            value = row[1]
            data_type = row[2]
            if data_type == 'int':
                value = int(value)
            elif data_type == 'float':
                value = float(value)
            elif data_type == 'bool':
                value = bool(value)
            elif data_type == 'str':
                value = str(value)
            elif data_type == 'list':
                value_list = value.split(',')
                # get the data type of the list
                new_value_list = []
                for value in value_list:
                    if value.isnumeric():
                        value = int(value)
                    elif value.replace('.','',1).isdigit():
                        value = float(value)
                    else:
                        value = str(value)
                    new_value_list.append(value)
                value = new_value_list
            else:
                value = str(value)
            #
            config_table[key] = value
    return config_table


def print_config(config_table):
    """
    print config
    """
    for key, value in config_table.items():
        print(f'{key}: {value}')


# class EQTLseqDataSet(torch.utils.data.IterableDataset):

#     def __init__(self, tokenizer, tokenizer_name, root_path, config_file, subset, add_eos=False, conjoin_train=False, conjoin_test=False):
#         super(EQTLseqDataSet).__init__()
#         #########################
#         # load config
#         #########################
#         self.config = parse_config(config_file)
#         # init
#         self.organism = 'Unknown'
#         if self.config.get('organism', None) is not None:
#             self.organism = self.config['organism']
#         self.subset = subset
#         if self.subset is None: 
#             if self.config.get('subset', None) is not None:
#                 self.subset = self.config['subset']
#             else:
#                 print("subset is required either in the config file or passed as a parameter", file = sys.stderr)
#         try:
#             assert self.subset in ['train', 'valid', 'test']
#         except AssertionError:
#             print("subset must be either train, valid, or test", file = sys.stderr)
#         # verbose level
#         self.verbose = 0
#         if self.config.get('verbose', None) is not None:
#             self.verbose = self.config['verbose']
#         # show config
#         if self.verbose > 0:
#             print_config(self.config)
#         print("> load config done")

#         self.tokenizer = tokenizer
#         self.tokenizer_name = tokenizer_name
        
#         self.conjoin_train = conjoin_train
#         self.conjoin_test = conjoin_test
        
#         self.add_eos = add_eos
#         ############################
#         # load fasta file
#         ############################
#         if self.config.get('genome_fa', None) is None:
#             print('genome_fa is required in the config file', file = sys.stderr)
#         self.fasta_reader = FastaStringExtractor(os.path.join(root_path, self.config['genome_fa']))
#         print("> init fasta extractor done")

#         ############################
#         # load dataset, ie eQTL data
#         ############################
#         if self.config.get('seq_len_cutoff', None) is None:
#             print('seq_len_cutoff is required in the config file', file = sys.stderr)
#             print('will use default value 450000', file = sys.stderr)
#             self.config['seq_len_cutoff'] = 450000
#         if self.config.get('eQTL_tabix_file', None) is None:
#             print('eQTL_tabix_file is required in the config file', file = sys.stderr)
#         self.dataset = parse_eQTL(root_path, self.config['eQTL_file'], self.config['eQTL_tabix_file'], self.fasta_reader, self.subset, self.config['seq_len_cutoff'], self.config['tss_flank_upstream'], self.config['tss_flank_downstream'], self.config['region_flank_upstream'], self.config['region_flank_downstream'])
#         self.dataset_size = len(self.dataset)
#         # if self.verbose > 0:
#         print(f'Loaded {len(self.dataset)} eQTL records')
            
#     def __len__(self):
#         return self.dataset_size

#     def __iter__(self):
#         # iterate over the data
#         target2int = {'positive':1, 'negative':0}
#         for index, record in enumerate(self.dataset):
#             # # convert the sequence to one-hot encoding
#             # # print(f"record: \n {record[0]} \n {record[1]} \n {record[2]}")
#             # sequence_ref = one_hot_encode(record[0])
#             # sequence_alt = one_hot_encode(record[1])
#             # sequence_ref = np.argmax(sequence_ref, axis=-1)
#             # sequence_alt = np.argmax(sequence_alt, axis=-1)
#             # sequence_ref_alt = np.concatenate([sequence_ref, sequence_alt], axis = 0)
#             # target = np.array(target2int[record[2]]).astype('long')
#             # # yield {
#             # #     'x_ref': sequence_ref,
#             # #     'x_alt': sequence_alt,
#             # #     'y': target 
#             # # }
#             # sequence_ref = self.tokenizer(record[0])
#             # sequence_alt = self.tokenizer(record[1])
            
#             # !seems to be useless
#             # sequence_ref_ids = self.tokenizer(
#             #     record[0],
#             #     add_special_tokens=False,
#             #     padding="do_not_pad",
#             #     max_length=self.config['seq_len_cutoff'],
#             #     truncation=True,
#             # )["input_ids"]
            
#             sequence_alt_ids = self.tokenizer(
#                 record[1],
#                 add_special_tokens=False,
#                 padding="do_not_pad",
#                 max_length=self.config['seq_len_cutoff'],
#                 truncation=True,
#             )["input_ids"]
            
#             if self.conjoin_train or (self.conjoin_test and self.subset != "train"):
#                 sequence_alt_rc = string_reverse_complement(record[1])
#                 sequence_alt_rc_ids = self.tokenizer(
#                     sequence_alt_rc,
#                     add_special_tokens=False,
#                     padding="do_not_pad",
#                     max_length=self.config['seq_len_cutoff'],
#                     truncation=True,
#                 )["input_ids"]

#                 # need to handle eos here
#                 if self.add_eos:
#                     # append list seems to be faster than append tensor
#                     sequence_alt_rc_ids.append(self.tokenizer.sep_token_id)
#                 seq_alt_ids = torch.stack((torch.LongTensor(sequence_alt_ids), torch.LongTensor(sequence_alt_rc_ids)), dim=1)
#                 # print("conjoin_test rc chain!")
#             else:
#                 # convert to tensor
#                 seq_alt_ids = torch.LongTensor(sequence_alt_ids)
            
#             # sequence_ref_ids = torch.LongTensor(sequence_ref_ids)
#             # sequence_alt_ids = torch.LongTensor(sequence_alt_ids)
            
            
#             # target = torch.LongTensor(target2int[record[2]]) 
#             target = np.array(target2int[record[2]]).astype('long')
            
#             # print("data for this round: ", sequence_ref_ids, sequence_alt_ids, target)
#             yield torch.LongTensor(0), seq_alt_ids, target

class EQTLseqDataSet(Dataset):
    def __init__(self, tokenizer, tokenizer_name, root_path, config_file, subset, add_eos=False, conjoin_train=False, conjoin_test=False):
        super(EQTLseqDataSet, self).__init__()
        #########################
        # load config
        #########################
        self.config = parse_config(config_file)
        self.organism = self.config.get('organism', 'Unknown')
        self.subset = subset
        if self.subset is None:
            self.subset = self.config.get('subset', None)
            if self.subset is None:
                print("subset is required either in the config file or passed as a parameter", file=sys.stderr)
                sys.exit(1)
        assert self.subset in ['train', 'valid', 'test'], "subset must be either train, valid, or test"
        self.verbose = self.config.get('verbose', 0)
        if self.verbose > 0:
            print_config(self.config)
        print("> load config done")

        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.add_eos = add_eos

        ############################
        # load fasta file
        ############################
        if self.config.get('genome_fa', None) is None:
            print('genome_fa is required in the config file', file=sys.stderr)
            sys.exit(1)
        self.fasta_reader = FastaStringExtractor(os.path.join(root_path, self.config['genome_fa']))
        print("> init fasta extractor done")

        ############################
        # load dataset, i.e., eQTL data
        ############################
        self.config['seq_len_cutoff'] = self.config.get('seq_len_cutoff', 450000)
        if self.config.get('eQTL_tabix_file', None) is None:
            print('eQTL_tabix_file is required in the config file', file=sys.stderr)
            sys.exit(1)
        self.dataset = parse_eQTL(
            root_path,
            self.config['eQTL_file'],
            self.config['eQTL_tabix_file'],
            self.fasta_reader,
            self.subset,
            self.config['seq_len_cutoff'],
            self.config['tss_flank_upstream'],
            self.config['tss_flank_downstream'],
            self.config['region_flank_upstream'],
            self.config['region_flank_downstream']
        )
        self.dataset_size = len(self.dataset)
        print(f'Loaded {self.dataset_size} eQTL records')

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        record = self.dataset[index]
        target2int = {'positive': 1, 'negative': 0}

        # Tokenize alt sequence
        sequence_alt_ids = self.tokenizer(
            record[1],
            add_special_tokens=False,
            padding="do_not_pad",
            max_length=self.config['seq_len_cutoff'],
            truncation=True,
        )["input_ids"]

        if self.conjoin_train or (self.conjoin_test and self.subset != "train"):
            sequence_alt_rc = string_reverse_complement(record[1])
            sequence_alt_rc_ids = self.tokenizer(
                sequence_alt_rc,
                add_special_tokens=False,
                padding="do_not_pad",
                max_length=self.config['seq_len_cutoff'],
                truncation=True,
            )["input_ids"]
            if self.add_eos:
                sequence_alt_rc_ids.append(self.tokenizer.sep_token_id)
            seq_alt_ids = torch.stack((torch.LongTensor(sequence_alt_ids), torch.LongTensor(sequence_alt_rc_ids)), dim=1)
        else:
            seq_alt_ids = torch.LongTensor(sequence_alt_ids)

        target = np.array(target2int[record[2]]).astype('long')
        return torch.LongTensor([0]), seq_alt_ids, target


def parse_eQTL(root_path, eQTL_file, blacklist_tabix_file, fasta_reader, subset_group_name, seq_len_cutoff = 450000, tss_flank_upstream = 3000, tss_flank_downstream = 3000, region_flank_upstream = 500, region_flank_downstream = 500):
    """
    this function will parse the following file format to reresent an variant-gene interaction entry
        > gene_chrom  gene_start gene_end region_chrom region_start region_end gene_id gene_strand region_id target subset allele1 allele2
        > other columns may exists but will be ignored
    then, the function will search for any variants between the tested region and the gene tss the sequence in these variants will be masked N. This is crucial because we want to ensure that these sequences don’t influence the model training process. 
    """
    """
    length: the length, is at least should depended on the tss (gene) length where the variant located and the variant length. So the minimum length has been fixed.
    """
    # check subset
    if subset_group_name not in ['train', 'valid', 'test']:
        print(f'Error: unknown subset group {subset_group_name}')
    # load records as dataframe
    df = pd.read_csv(os.path.join(root_path, eQTL_file), sep='\t', header=0)
    # load the tabix
    blacklist_tabix = tabix.open(os.path.join(root_path, blacklist_tabix_file))
    # parse the records and convert it into a list of DNA sequence and target (ie, true or false eQTL)
    dataset = []
    print("> Start parsing eQTL records to build the dataset %s" % (subset_group_name))
    N_total = df.shape[0]
    N_pass = 0
    N_skip_for_chrom_diff = 0 # skip if variant and gene tss are in different chromosomes
    N_skip_for_len = 0 # skip if the distance between variant and gene tss is too long
    N_skip_for_strand = 0 # skip if gene strand is not + or - or unknown
    for index, row in tqdm(df.iterrows(), total = N_total):
        gene_chrom = row['gene_chrom']
        gene_start = row['gene_start']
        gene_end = row['gene_end']
        region_chrom = row['region_chrom']
        region_start = row['region_start']
        region_end = row['region_end']
        gene_id = row['gene_id']
        gene_strand = row['gene_strand']
        region_id = row['region_id']
        target = row['target']
        subset_group = row['subset']
        allele1 = row['allele1']
        allele2 = row['allele2']
        # skip if in different subset group
        if subset_group != subset_group_name:
            continue
        # skip if in different chromosomes
        if gene_chrom != region_chrom:
            N_skip_for_chrom_diff += 1
            continue
        # get tss
        if gene_strand == '+':
            tss_start = gene_start
        elif gene_strand == '-':
            tss_start = gene_end - 1
        else:
            print(f'Warning: unknown strand {gene_strand}')
            N_skip_for_strand += 1
            continue
        tss_end = tss_start + 1
        # add flanking sequence
        tss_start = tss_start - tss_flank_upstream
        tss_end = tss_end + tss_flank_downstream
        variant_start = region_start
        variant_end = region_end
        region_start = region_start - region_flank_upstream
        region_end = region_end + region_flank_downstream
        # calculate the distance between tss and region
        distance = max(0, max(tss_start, region_start) - min(tss_end, region_end))
        # check if the sequence length is too long
        if distance > seq_len_cutoff:
            N_skip_for_len += 1
            continue
        # get the interverl between the region and tss
        sequence_start = min(tss_start, region_start)
        sequence_end = max(tss_end, region_end)
        # get the sequence
        region_interval = Interval(region_chrom, sequence_start, sequence_end)
        region_seq = fasta_reader.extract(region_interval)
        # check if the variant sequence matches the reference sequence
        variant_rev_start = variant_start - sequence_start
        variant_rev_end = variant_end - sequence_start
        assert region_seq[variant_rev_start:variant_rev_end] == row['allele1']
        # mask any eQTL sequence
        if distance > 0:
            if tss_start > region_end:
                query_start = region_end
                query_end = tss_start
            else:
                query_start = tss_end
                query_end = region_start
            query_result = blacklist_tabix.query(region_chrom, query_start, query_end)
            for overlap in query_result:
                o_start = int(overlap[1])
                o_end = int(overlap[2])
                # mask the correspoinding DNA sequence as 'N'
                rev_start = o_start - sequence_start
                rev_end = o_end - sequence_start
                if rev_start > 0 and rev_start < len(region_seq) and rev_end > 0 and rev_end < len(region_seq):
                    region_seq = region_seq[:rev_start] + 'N' * (rev_end - rev_start) + region_seq[rev_end:]
        # create variant sequence 
        region_seq_var =  region_seq[:variant_rev_start] + row['allele2'] + region_seq[variant_rev_end:]
        # flip (reverse complement) the sequence if gene is on the downstream of region (ie, genomic coordinate is larger)
        if gene_start > region_end:
            region_seq = rcDNA(region_seq) 
            region_seq_var = rcDNA(region_seq_var)
        # padding N to the sequence if the sequence
        if len(region_seq) <= seq_len_cutoff:
            region_seq = region_seq + 'N' * (seq_len_cutoff - len(region_seq))
        else:
            region_seq = region_seq[:seq_len_cutoff]
        if len(region_seq_var) <= seq_len_cutoff:
            region_seq_var = region_seq_var + 'N' * (seq_len_cutoff - len(region_seq_var))
        else:
            region_seq_var = region_seq_var[:seq_len_cutoff]
        N_pass +=1
        # append to the dataset
        dataset.append((region_seq, region_seq_var, target))
    # report the statistics
    print("# Finish parsing eQTL records")
    print("# Total records: ", N_total)
    print("# Skipped records due to different chromosomes: ", N_skip_for_chrom_diff)
    print("# Skipped records due to distance cutoff: ", N_skip_for_len)
    print("# Skipped records due to unknown strand: ", N_skip_for_strand)
    print("# Select records %s with subset %s " % (N_pass, subset_group_name))
    return dataset







class OnlineSampler(Sampler, metaclass=ABCMeta):
    """
    A sampler in which training/validation/test data is constructed
    from random sampling of the dataset for each batch passed to the
    model. This form of sampling may alleviate the problem of loading an
    extremely large dataset into memory when developing a new model.

    Parameters
    ----------
    reference_sequence : selene_sdk.sequences.Sequence
        A reference sequence from which to create examples.
    target : selene_sdk.targets.Target or str
        A `selene_sdk.targets.Target` object to provide the targets that
        we would like to predict, or a str to provide path to tabix-indexed,
        compressed BED file (`*.bed.gz`) of genomic coordinates mapped to
        the genomic features we want to predict. Using str as target will
        be deprecated in the future. Please consider using a GenomicFeatures
        object instead.
    features : list(str)
        List of distinct features that we aim to predict.
    seed : int, optional
        Default is 436. Sets the random seed for sampling.
    validation_holdout : list(str) or float, optional
        Default is `['chr6', 'chr7']`. Holdout can be regional or
        proportional. If regional, expects a list (e.g. `['X', 'Y']`).
        Regions must match those specified in the first column of the
        tabix-indexed BED file. If proportional, specify a percentage
        between (0.0, 1.0). Typically 0.10 or 0.20.
    test_holdout : list(str) or float, optional
        Default is `['chr8', 'chr9']`. See documentation for
        `validation_holdout` for additional information.
    sequence_length : int, optional
        Default is 1000. Model is trained on sequences of `sequence_length`
        where genomic features are annotated to the center regions of
        these sequences.
    center_bin_to_predict : int, optional
        Default is 200. Query the tabix-indexed file for a region of
        length `center_bin_to_predict`.
    feature_thresholds : float [0.0, 1.0], optional
        Default is 0.5. The `feature_threshold` to pass to the
        `GenomicFeatures` object. Use str target and feature_thresholds
        is deprecated and will be removed in the future. Please consider 
        passing GenomicFeatures object directly to target instead.
    mode : {'train', 'validate', 'test'}, optional
        Default is `'train'`. The mode to run the sampler in.
    save_datasets : list(str), optional
        Default is `[]` the empty list. The list of modes for which we should
        save the sampled data to file (e.g. `["test", "validate"]`).
    output_dir : str or None, optional
        Default is None. The path to the directory where we should
        save sampled examples for a mode. If `save_datasets` is
        a non-empty list, `output_dir` must be specified. If
        the path in `output_dir` does not exist it will be created
        automatically.

    Attributes
    ----------
    reference_sequence : selene_sdk.sequences.Sequence
        The reference sequence that examples are created from.
    target : selene_sdk.targets.Target
        The `selene_sdk.targets.Target` object holding the features that we
        would like to predict.
    validation_holdout : list(str) or float
        The samples to hold out for validating model performance. These
        can be "regional" or "proportional". If regional, this is a list
        of region names (e.g. `['chrX', 'chrY']`). These regions must
        match those specified in the first column of the tabix-indexed
        BED file. If proportional, this is the fraction of total samples
        that will be held out.
    test_holdout : list(str) or float
        The samples to hold out for testing model performance. See the
        documentation for `validation_holdout` for more details.
    sequence_length : int
        The length of the sequences to  train the model on.
    bin_radius : int
        From the center of the sequence, the radius in which to detect
        a feature annotation in order to include it as a sample's label.
    surrounding_sequence_radius : int
        The length of sequence falling outside of the feature detection
        bin (i.e. `bin_radius`) center, but still within the
        `sequence_length`.
    modes : list(str)
        The list of modes that the sampler can be run in.
    mode : str
        The current mode that the sampler is running in. Must be one of
        the modes listed in `modes`.

    Raises
    ------
    ValueError
            If `mode` is not a valid mode.
    ValueError
        If the parities of `sequence_length` and `center_bin_to_predict`
        are not the same.
    ValueError
        If `sequence_length` is smaller than `center_bin_to_predict` is.
    ValueError
        If the types of `validation_holdout` and `test_holdout` are not
        the same.

    """
    STRAND_SIDES = ('+', '-')
    """
    Defines the strands that features can be sampled from.
    """

    def __init__(self,
                 reference_sequence,
                 target,
                 features,
                 seed=436,
                 validation_holdout=['chr6', 'chr7'],
                 test_holdout=['chr8', 'chr9'],
                 sequence_length=1001,
                 center_bin_to_predict=201,
                 feature_thresholds=0.5,
                 mode="train",
                 save_datasets=[],
                 output_dir=None):

        """
        Creates a new `OnlineSampler` object.
        """
        super(OnlineSampler, self).__init__(
            features,
            save_datasets=save_datasets,
            output_dir=output_dir)

        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed + 1)

        if (sequence_length + center_bin_to_predict) % 2 != 0:
            raise ValueError(
                "Sequence length of {0} with a center bin length of {1} "
                "is invalid. These 2 inputs should both be odd or both be "
                "even.".format(
                    sequence_length, center_bin_to_predict))

        surrounding_sequence_length = sequence_length - center_bin_to_predict
        if surrounding_sequence_length < 0:
            raise ValueError(
                "Sequence length of {0} is less than the center bin "
                "length of {1}.".format(
                    sequence_length, center_bin_to_predict))

        # specifying a test holdout partition is optional
        if test_holdout:
            self.modes.append("test")
            if isinstance(validation_holdout, (list,)) and \
                    isinstance(test_holdout, (list,)):
                self.validation_holdout = [
                    str(c) for c in validation_holdout]
                self.test_holdout = [str(c) for c in test_holdout]
                self._holdout_type = "chromosome"
            elif isinstance(validation_holdout, float) and \
                    isinstance(test_holdout, float):
                self.validation_holdout = validation_holdout
                self.test_holdout = test_holdout
                self._holdout_type = "proportion"
            else:
                raise ValueError(
                    "Validation holdout and test holdout must have the "
                    "same type (list or float) but validation was "
                    "type {0} and test was type {1}".format(
                        type(validation_holdout), type(test_holdout)))
        else:
            self.test_holdout = None
            if isinstance(validation_holdout, (list,)):
                self.validation_holdout = [
                    str(c) for c in validation_holdout]
            else:
                self.validation_holdout = validation_holdout

        if mode not in self.modes:
            raise ValueError(
                "Mode must be one of {0}. Input was '{1}'.".format(
                    self.modes, mode))
        self.mode = mode

        self.surrounding_sequence_radius = int(
            surrounding_sequence_length / 2)
        self.sequence_length = sequence_length
        self.bin_radius = int(center_bin_to_predict / 2)
        self._start_radius = self.bin_radius
        if center_bin_to_predict % 2 == 0:
            self._end_radius = self.bin_radius
        else:
            self._end_radius = self.bin_radius + 1

        self.reference_sequence = reference_sequence

        self.n_features = len(self._features)

        if isinstance(target, str):
            self.target = GenomicFeatures(
                target, self._features,
                feature_thresholds=feature_thresholds)
        elif isinstance(target, Target) or isinstance(target, list):
            self.target = target
        elif target is None:
            self.target = None
        else:
            raise ValueError("target must be one of str, "
            "selene_sdk.targets.Target object, list, or None")
            
        self._save_filehandles = {}

    def get_feature_from_index(self, index):
        """
        Returns the feature corresponding to an index in the feature
        vector.

        Parameters
        ----------
        index : int
            The index of the feature to retrieve the name for.

        Returns
        -------
        str
            The name of the feature occurring at the specified index.
        """
        return self.target.index_feature_dict[index]

    def get_sequence_from_encoding(self, encoding):
        """
        Gets the string sequence from the one-hot encoding
        of the sequence.

        Parameters
        ----------
        encoding : numpy.ndarray
            An :math:`L \\times N` array (where :math:`L` is the length
            of the sequence and :math:`N` is the size of the sequence
            type's alphabet) containing the one-hot encoding of the
            sequence.

        Returns
        -------
        str
            The sequence of :math:`L` characters decoded from the input.
        """
        return self.reference_sequence.encoding_to_sequence(encoding)

    def save_dataset_to_file(self, mode, close_filehandle=False):
        """
        Save samples for each partition (i.e. train/validate/test) to
        disk.

        Parameters
        ----------
        mode : str
            Must be one of the modes specified in `save_datasets` during
            sampler initialization.
        close_filehandle : bool, optional
            Default is False. `close_filehandle=True` assumes that all
            data corresponding to the input `mode` has been saved to
            file and `save_dataset_to_file` will not be called with
            `mode` again.
        """
        if mode not in self._save_datasets:
            return
        samples = self._save_datasets[mode]
        if mode not in self._save_filehandles:
            self._save_filehandles[mode] = open(
                os.path.join(self._output_dir,
                             "{0}_data.bed".format(mode)),
                'w+')
        file_handle = self._save_filehandles[mode]
        while len(samples) > 0:
            cols = samples.pop(0)
            line = '\t'.join([str(c) for c in cols])
            file_handle.write("{0}\n".format(line))
        if close_filehandle:
            file_handle.close()

    def get_data_and_targets(self, batch_size, n_samples=None, mode=None):
        """
        This method fetches a subset of the data from the sampler,
        divided into batches. This method also allows the user to
        specify what operating mode to run the sampler in when fetching
        the data.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is None. The total number of samples to retrieve.
            If `n_samples` is None and the mode is `validate`, will
            set `n_samples` to 32000; if the mode is `test`, will set
            `n_samples` to 640000 if it is None. If the mode is `train`
            you must have specified a value for `n_samples`.
        mode : str, optional
            Default is None. The mode to run the sampler in when
            fetching the samples. See
            `selene_sdk.samplers.IntervalsSampler.modes` for more
            information. If None, will use the current mode `self.mode`.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S =` `n_samples`.

        """
        if mode is not None:
            self.set_mode(mode)
        else:
            mode = self.mode
        sequences_and_targets = []
        if n_samples is None and mode == "validate":
            n_samples = 32000
        elif n_samples is None and mode == "test":
            n_samples = 640000

        n_batches = int(n_samples / batch_size)
        for _ in range(n_batches):
            inputs, targets = self.sample(batch_size)
            sequences_and_targets.append((inputs, targets))
        targets_mat = np.vstack([t for (s, t) in sequences_and_targets])
        if mode in self._save_datasets:
            self.save_dataset_to_file(mode, close_filehandle=True)
        return sequences_and_targets, targets_mat

    def get_dataset_in_batches(self, mode, batch_size, n_samples=None):
        """
        This method returns a subset of the data for a specified run
        mode, divided into mini-batches.

        Parameters
        ----------
        mode : {'test', 'validate'}
            The mode to run the sampler in when fetching the samples.
            See `selene_sdk.samplers.IntervalsSampler.modes` for more
            information.
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is `None`. The total number of samples to retrieve.
            If `None`, it will retrieve 32000 samples if `mode` is validate
            or 640000 samples if `mode` is test or train.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            The list is length :math:`S`, where :math:`S =` `n_samples`.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`

        """
        return self.get_data_and_targets(
            batch_size, n_samples=n_samples, mode=mode)

    def get_validation_set(self, batch_size, n_samples=None):
        """
        This method returns a subset of validation data from the
        sampler, divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is `None`. The total number of validation examples
            to retrieve. If `None`, 32000 examples are retrieved.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S =` `n_samples`.

        """
        return self.get_dataset_in_batches(
            "validate", batch_size, n_samples=n_samples)

    def get_test_set(self, batch_size, n_samples=None):
        """
        This method returns a subset of testing data from the
        sampler, divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is `None`. The total number of validation examples
            to retrieve. If `None`, 640000 examples are retrieved.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S =` `n_samples`.


        Raises
        ------
        ValueError
            If no test partition of the data was specified during
            sampler initialization.
        """
        if "test" not in self.modes:
            raise ValueError("No test partition of the data was specified "
                             "during initialization. Cannot use method "
                             "`get_test_set`.")
        return self.get_dataset_in_batches("test", batch_size, n_samples)






logger = logging.getLogger(__name__)


SampleIndices = namedtuple(
    "SampleIndices", ["indices", "weights"])



class RandomPositionsSampler(OnlineSampler):
    """This sampler randomly selects a position in the genome and queries for
    a sequence centered at that position for input to the model.

    TODO: generalize to selene_sdk.sequences.Sequence?

    Parameters
    ----------
    reference_sequence : selene_sdk.sequences.Genome
        A reference sequence from which to create examples.
    target : sselene_sdk.targets.Target or list(selene_sdk.targets.Target) or str
        A `selene_sdk.targets.Target` object to provide the targets that
        we would like to predict, or a list of these objects,
        or a str to provide path to tabix-indexed,
        compressed BED file (`*.bed.gz`) of genomic coordinates mapped to
        the genomic features we want to predict. Using str as target will
        be deprecated in the future. Please consider using a GenomicFeatures
        object instead.
    features : list(str)
        List of distinct features that we aim to predict.
    seed : int, optional
        Default is 436. Sets the random seed for sampling.
    validation_holdout : list(str) or float, optional
        Default is `['chr6', 'chr7']`. Holdout can be regional or
        proportional. If regional, expects a list (e.g. `['chrX', 'chrY']`).
        Regions must match those specified in the first column of the
        tabix-indexed BED file. If proportional, specify a percentage
        between (0.0, 1.0). Typically 0.10 or 0.20.
    test_holdout : list(str) or float, optional
        Default is `['chr8', 'chr9']`. See documentation for
        `validation_holdout` for additional information.
    sequence_length : int, optional
        Default is 1000. Model is trained on sequences of `sequence_length`
        where genomic features are annotated to the center regions of
        these sequences.
    center_bin_to_predict : int, optional
        Default is 200. Query the tabix-indexed file for a region of
        length `center_bin_to_predict`.
    feature_thresholds : float [0.0, 1.0], optional
        Default is 0.5. The `feature_threshold` to pass to the
        `GenomicFeatures` object. Use str target and feature_thresholds
        is deprecated and will be removed in the future. Please consider 
        passing GenomicFeatures object directly to target instead.
    mode : {'train', 'validate', 'test'}
        Default is `'train'`. The mode to run the sampler in.
    save_datasets : list(str), optional
        Default is `['test']`. The list of modes for which we should
        save the sampled data to file.
    position_resolution : int, optional
        Default is 1. Random coordinates will be rounded to multiples 
        of position_resolution. This can be useful for example
        when target stores binned data.
    random_strand : bool, optional
        Default is True. If True, sequences are retrieved randomly
        from positive or negative strand, otherwise the positive
        strand is used by default. Note that random_strand should be
        set to False if target provides strand-specific data.
    random_shift : int, optional
        Default is 0. If True, the coordinates to retrieve sequence
        are shifted by a random integer from -random_shift to 
        random_shift independently for each sample.  
    output_dir : str or None, optional
        Default is None. The path to the directory where we should
        save sampled examples for a mode. If `save_datasets` is
        a non-empty list, `output_dir` must be specified. If
        the path in `output_dir` does not exist it will be created
        automatically.

    Attributes
    ----------
    reference_sequence : selene_sdk.sequences.Genome
        The reference sequence that examples are created from.
    target : selene_sdk.targets.Target
        The `selene_sdk.targets.Target` object holding the features that we
        would like to predict.
    validation_holdout : list(str) or float
        The samples to hold out for validating model performance. These
        can be "regional" or "proportional". If regional, this is a list
        of region names (e.g. `['chrX', 'chrY']`). These regions must
        match those specified in the first column of the tabix-indexed
        BED file. If proportional, this is the fraction of total samples
        that will be held out.
    test_holdout : list(str) or float
        The samples to hold out for testing model performance. See the
        documentation for `validation_holdout` for more details.
    sequence_length : int
        The length of the sequences to  train the model on.
    bin_radius : int
        From the center of the sequence, the radius in which to detect
        a feature annotation in order to include it as a sample's label.
    surrounding_sequence_radius : int
        The length of sequence falling outside of the feature detection
        bin (i.e. `bin_radius`) center, but still within the
        `sequence_length`.
    position_resolution : int
        Default is 1. Random coordinates will be rounded to multiples 
        of position_resolution. This can be useful for example
        when target stores binned data.
    random_strand : bool
        Default is True. If True, sequences are retrieved randomly
        from positive or negative strand, otherwise the positive
        strand is used by default. Note that random_strand should be
        set to False if target provides strand-specific data.
    random_shift : int
        Default is 0. If True, the coordinates to retrieve sequence
        are shifted by a random integer from -random_shift to 
        random_shift independently for each sample.  
    modes : list(str)
        The list of modes that the sampler can be run in.
    mode : str
        The current mode that the sampler is running in. Must be one of
        the modes listed in `modes`.

    """
    def __init__(self,
                 reference_sequence,
                 target,
                 features,
                 seed=436,
                 validation_holdout=['chr6', 'chr7'],
                 test_holdout=['chr8', 'chr9'],
                 sequence_length=1000,
                 center_bin_to_predict=200,
                 feature_thresholds=0.5,
                 mode="train",
                 save_datasets=[],
                 position_resolution=1,
                 random_shift=0,
                 random_strand=True,
                 output_dir=None):
        super(RandomPositionsSampler, self).__init__(
            reference_sequence,
            target,
            features,
            seed=seed,
            validation_holdout=validation_holdout,
            test_holdout=test_holdout,
            sequence_length=sequence_length,
            center_bin_to_predict=center_bin_to_predict,
            feature_thresholds=feature_thresholds,
            mode=mode,
            save_datasets=save_datasets,
            output_dir=output_dir)

        self._sample_from_mode = {}
        self._randcache = {}
        for mode in self.modes:
            self._sample_from_mode[mode] = None
            self._randcache[mode] = {"cache_indices": None, "sample_next": 0}

        self.sample_from_intervals = []
        self.interval_lengths = []
        self.initialized = False
        self.position_resolution = position_resolution
        self.random_shift= random_shift
        self.random_strand=random_strand

    def init(func):
        #delay initlization to allow  multiprocessing
        def dfunc(self, *args, **kwargs):
            if not self.initialized:
                if self._holdout_type == "chromosome":
                    self._partition_genome_by_chromosome()
                else:
                     self._partition_genome_by_proportion()

                for mode in self.modes:
                    self._update_randcache(mode=mode)
                self.initialized = True
            return func(self, *args, **kwargs)
        return dfunc


    def _partition_genome_by_proportion(self):
        for chrom, len_chrom in self.reference_sequence.get_chr_lens():
            self.sample_from_intervals.append(
                (chrom,
                 self.sequence_length,
                 len_chrom - self.sequence_length))
            self.interval_lengths.append(len_chrom)
        n_intervals = len(self.sample_from_intervals)

        select_indices = list(range(n_intervals))
        np.random.shuffle(select_indices)
        n_indices_validate = int(n_intervals * self.validation_holdout)
        val_indices, val_weights = get_indices_and_probabilities(
            self.interval_lengths, select_indices[:n_indices_validate])
        self._sample_from_mode["validate"] = SampleIndices(
            val_indices, val_weights)

        if self.test_holdout:
            n_indices_test = int(n_intervals * self.test_holdout)
            test_indices_end = n_indices_test + n_indices_validate
            test_indices, test_weights = get_indices_and_probabilities(
                self.interval_lengths,
                select_indices[n_indices_validate:test_indices_end])
            self._sample_from_mode["test"] = SampleIndices(
                test_indices, test_weights)

            tr_indices, tr_weights = get_indices_and_probabilities(
                self.interval_lengths, select_indices[test_indices_end:])
            self._sample_from_mode["train"] = SampleIndices(
                tr_indices, tr_weights)
        else:
            tr_indices, tr_weights = get_indices_and_probabilities(
                self.interval_lengths, select_indices[n_indices_validate:])
            self._sample_from_mode["train"] = SampleIndices(
                tr_indices, tr_weights)

    def _partition_genome_by_chromosome(self):
        for mode in self.modes:
            self._sample_from_mode[mode] = SampleIndices([], [])
        for index, (chrom, len_chrom) in enumerate(self.reference_sequence.get_chr_lens()):
            if chrom in self.validation_holdout:
                self._sample_from_mode["validate"].indices.append(
                    index)
            elif self.test_holdout and chrom in self.test_holdout:
                self._sample_from_mode["test"].indices.append(
                    index)
            else:
                self._sample_from_mode["train"].indices.append(
                    index)

            self.sample_from_intervals.append(
                (chrom,
                 self.sequence_length,
                 len_chrom - self.sequence_length))
            self.interval_lengths.append(len_chrom - 2 * self.sequence_length)

        for mode in self.modes:
            sample_indices = self._sample_from_mode[mode].indices
            indices, weights = get_indices_and_probabilities(
                self.interval_lengths, sample_indices)
            self._sample_from_mode[mode] = \
                self._sample_from_mode[mode]._replace(
                    indices=indices, weights=weights)

    def _retrieve(self, chrom, position):
        bin_start = position - self._start_radius
        bin_end = position + self._end_radius
        if self.target is not None:
            if isinstance(self.target, list):
                retrieved_targets = [t.get_feature_data(
                        chrom, bin_start, bin_end) for t in self.target]
            else:
                retrieved_targets = self.target.get_feature_data(
                    chrom, bin_start, bin_end)
            if retrieved_targets is None:
                logger.info("Target returns None. Sampling again.".format(
                                chrom, position))
                return None
        else:
            retrieved_targets = None


        window_start = bin_start - self.surrounding_sequence_radius
        window_end = bin_end + self.surrounding_sequence_radius
        if window_end - window_start < self.sequence_length:
            print(bin_start, bin_end,
                  self._start_radius, self._end_radius,
                  self.surrounding_sequence_radius)
            return None
        if self.random_strand:
            strand = self.STRAND_SIDES[random.randint(0, 1)]
        else:
            strand = '+'
            
        if self.random_shift > 0:
            r = np.random.randint(-self.random_shift, self.random_shift)
        else:
            r = 0
        retrieved_seq = \
            self.reference_sequence.get_encoding_from_coords(
                chrom, window_start+r, window_end+r, strand)

        if retrieved_seq.shape[0] == 0 or retrieved_seq.shape[0] != self.sequence_length:
            logger.info("Full sequence centered at {0} position {1} "
                        "could not be retrieved. Sampling again.".format(
                            chrom, position))
            return None
        elif np.mean(retrieved_seq==0.25) >0.30: 
            logger.info("Over 30% of the bases in the sequence centered "
                        "at {0} position {1} are ambiguous ('N'). "
                        "Sampling again.".format(chrom, position))
            return None



        if self.mode in self._save_datasets and not isinstance(retrieved_targets, list):
            feature_indices = ';'.join(
                [str(f) for f in np.nonzero(retrieved_targets)[0]])
            self._save_datasets[self.mode].append(
                [chrom,
                 window_start,
                 window_end,
                 strand,
                 feature_indices])
            if len(self._save_datasets[self.mode]) > 200000:
                self.save_dataset_to_file(self.mode)
        return (retrieved_seq, retrieved_targets)

    def _update_randcache(self, mode=None):
        if not mode:
            mode = self.mode
        self._randcache[mode]["cache_indices"] = np.random.choice(
            self._sample_from_mode[mode].indices,
            size=200000,
            replace=True,
            p=self._sample_from_mode[mode].weights)
        self._randcache[mode]["sample_next"] = 0

    @init
    def sample(self, batch_size=1, mode=None, return_coordinates=False, coordinates_only=False):
        """
        Randomly draws a mini-batch of examples and their corresponding
        labels.

        Parameters
        ----------
        batch_size : int, optional
            Default is 1. The number of examples to include in the
            mini-batch.
        mode : str, optional
            Default is None. The operating mode that the object should run in.
            If None, will use the current mode `self.mode`.
            
        Returns
        -------
        sequences, targets : tuple(numpy.ndarray, numpy.ndarray)
            A tuple containing the numeric representation of the
            sequence examples and their corresponding labels. The
            shape of `sequences` will be
            :math:`B \\times L \\times N`, where :math:`B` is
            `batch_size`, :math:`L` is the sequence length, and
            :math:`N` is the size of the sequence type's alphabet.
            The shape of `targets` will be :math:`B \\times F`,
            where :math:`F` is the number of features.

        """
        mode = mode if mode else self.mode
        if coordinates_only:
            assert return_coordinates == True
        else:
            sequences = np.zeros((batch_size, self.sequence_length, 4))
            
            if self.target is None:
                targets = None
            elif isinstance(self.target, list):
                targets = [np.zeros((batch_size, *t.shape)) for t in self.target]
            elif isinstance(self.target.shape, list):
                targets = [np.zeros((batch_size, *tshape)) for tshape in self.target.shape]
            else:
                targets = np.zeros((batch_size, *self.target.shape))
        if return_coordinates:
            coords = []

        n_samples_drawn = 0
        while n_samples_drawn < batch_size:
            sample_index = self._randcache[mode]["sample_next"]
            if sample_index == len(self._randcache[mode]["cache_indices"]):
                self._update_randcache()
                sample_index = 0

            rand_interval_index = \
                self._randcache[mode]["cache_indices"][sample_index]
            self._randcache[mode]["sample_next"] += 1

            chrom, cstart, cend = \
                self.sample_from_intervals[rand_interval_index]
            position = np.random.randint(cstart, cend)
            position -= position % self.position_resolution
            
            if not coordinates_only:
                retrieve_output = self._retrieve(chrom, position)
                if not retrieve_output:
                    continue

            if return_coordinates:
                coords.append((chrom, position))

            if not coordinates_only:
                seq, seq_targets = retrieve_output
                sequences[n_samples_drawn, :, :] = seq
                if isinstance(targets, list):
                    assert isinstance(seq_targets, (list, tuple))
                    for target, seq_target in zip(targets, seq_targets):
                        target[n_samples_drawn, :] = seq_target
                elif targets is not None:
                    targets[n_samples_drawn, :] = seq_targets
            n_samples_drawn += 1
            

        if return_coordinates:
            if coordinates_only:
                return coords
            else:
                if target is None:
                    return sequences, coords
                else:
                    return sequences, targets, coords
        else:
            if targets is None:
                return sequences,
            else:
                return sequences, targets

def get_genomes(path):
    genome = selene_sdk.sequences.Genome(
                    input_path=path,
                    blacklist_regions= 'hg38'
                )
    noblacklist_genome = selene_sdk.sequences.Genome(
                    input_path=path )
    return genome, noblacklist_genome


class GenomicSignalFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """
    def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None, 
        replacement_indices=None, replacement_scaling_factors=None):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors

            
        self.n_features = len(features)
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True, feature_indices=None):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist)  for blacklist in self.blacklists]
            self.initialized=True

        if feature_indices is None:
            feature_indices = np.arange(len(self.data))

        wigmat = np.zeros((len(feature_indices), end - start), dtype=np.float32)
        for i in feature_indices:
            try:
                wigmat[i, :] = self.data[i].values(chrom, start, end, numpy=True)
            except:
                print(chrom, start, end, self.input_paths[i], flush=True)
                raise
        
        if self.blacklists is not None:
            if self.replacement_indices is None:
                if self.blacklists_indices is not None:
                    for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[blacklist_indices, np.fmax(int(s)-start,0): int(e)-start] = 0
                else:
                    for blacklist in self.blacklists:
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[:, np.fmax(int(s)-start,0): int(e)-start] = 0
            else:
                for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(self.blacklists, self.blacklists_indices, self.replacement_indices, self.replacement_scaling_factors):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s)-start,0): int(e)-start] = wigmat[replacement_indices, np.fmax(int(s)-start,0): int(e)-start] * replacement_scaling_factor

        if nan_as_zero:
            wigmat[np.isnan(wigmat)]=0
        return wigmat
    
    
    

"""

Puffin Dataset, from:

# transcription_initiation_signal_prediction

"""

char_list = ['A', 'C', 'G', 'T', 'N']
idx2char = {i: char_list[i] for i in range(4)}
char2idx = {char_list[i]: i for i in range(4)}


def convert_from_one_hot(sequence):
    x_str = "" 
    for x in sequence:
        if x.max() == 0.25:
            x_str += 'N'
        else:
            x_str += idx2char[x.argmax()]
    return x_str


class GenomicSignalFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """
    def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None, 
        replacement_indices=None, replacement_scaling_factors=None):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.input_paths = input_paths
        self.initialized = False
        self.blacklists = blacklists
        self.blacklists_indices = blacklists_indices
        self.replacement_indices = replacement_indices
        self.replacement_scaling_factors = replacement_scaling_factors

            
        self.n_features = len(features)
        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])
        self.shape = (len(input_paths), *shape)

    def get_feature_data(self, chrom, start, end, nan_as_zero=True):
        if not self.initialized:
            self.data = [pyBigWig.open(path) for path in self.input_paths]
            if self.blacklists is not None:
                self.blacklists = [tabix.open(blacklist)  for blacklist in self.blacklists]
            self.initialized=True
            
        wigmat = np.vstack([c.values(chrom, start, end, numpy=True)
                           for c in self.data])
        
        if self.blacklists is not None:
            if self.replacement_indices is None:
                for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
                    for _, s, e in blacklist.query(chrom, start, end):
                        wigmat[blacklist_indices, np.fmax(int(s)-start,0): int(e)-start] = 0
            else:
                for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(self.blacklists, self.blacklists_indices, self.replacement_indices, self.replacement_scaling_factors):
                        for _, s, e in blacklist.query(chrom, start, end):
                            wigmat[blacklist_indices, np.fmax(int(s)-start,0): int(e)-start] = wigmat[replacement_indices, np.fmax(int(s)-start,0): int(e)-start] * replacement_scaling_factor

        if nan_as_zero:
            wigmat[np.isnan(wigmat)]=0
        return wigmat



class PuffinDataset(torch.utils.data.IterableDataset):

    '''
    Puffin Dataset
    '''

    def __init__(
        self,
        split,
        max_length,
        dataset_name="Puffin",
        d_output=1, # default regression
        track=None,
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        return_mask=False,
    ):

        self.dataset_name = dataset_name
        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.return_mask = return_mask
        self.split = split
        self.dest_path = str(dest_path)
        self.track = track

        if isinstance(dest_path, Path):
            dest_path = str(dest_path) + '/'

        # use Path object
        self.input_files = [dest_path + "agg.plus.bw.bedgraph.bw",
                dest_path + "agg.encodecage.plus.v2.bedgraph.bw",
                dest_path + "agg.encoderampage.plus.v2.bedgraph.bw",
                dest_path + "agg.plus.grocap.bedgraph.sorted.merged.bw",
                dest_path + "agg.plus.allprocap.bedgraph.sorted.merged.bw",
                dest_path + "agg.minus.allprocap.bedgraph.sorted.merged.bw",
                dest_path + "agg.minus.grocap.bedgraph.sorted.merged.bw",
                dest_path + "agg.encoderampage.minus.v2.bedgraph.bw",
                dest_path + "agg.encodecage.minus.v2.bedgraph.bw",
                dest_path + "agg.minus.bw.bedgraph.bw"]

        tfeature = GenomicSignalFeatures(self.input_files,
                ['cage_plus','encodecage_plus','encoderampage_plus', 'grocap_plus','procap_plus','procap_minus','grocap_minus'
                ,'encoderampage_minus', 'encodecage_minus',
                'cage_minus'],
                (100000,),
                [dest_path + "fantom.blacklist8.plus.bed.gz",dest_path + "fantom.blacklist8.minus.bed.gz"],
                [0,9], [1,8], [0.61357, 0.61357])
        
        self.genome = Genome(
                    input_path=dest_path + "Homo_sapiens.GRCh38.dna.primary_assembly.fa",
                    blacklist_regions= 'hg38'
                )

        self.noblacklist_genome = Genome(
                    input_path=dest_path + "Homo_sapiens.GRCh38.dna.primary_assembly.fa" )

        self.sampler = RandomPositionsSampler(
                            reference_sequence=self.genome,
                            target= tfeature,
                            features = [''],
                            test_holdout=['chr8', 'chr9'],
                            validation_holdout= ['chr10'],
                            sequence_length= 100_000,
                            center_bin_to_predict= 100_000,
                            position_resolution=1,
                            random_shift=0,
                            random_strand=False
                            )

        self.sampler.mode = "validate" if self.split == "val" else self.split

        seed = 3
        np.random.seed(seed)

        self.buffer = []
        self.buffer_size = 1024

    def get_test_data(self):
        slide_window = 50_000
        sequence_length = self.max_length
        ignore_last_len = 25_000
        remove_unk_interval = 1000


        def check_unk(position):
            start_position = position - remove_unk_interval
            end_position = position + remove_unk_interval
            sub_seq = self.sampler.reference_sequence.get_sequence_from_coords(chrom, start_position, end_position)
            if 'N' in sub_seq:
                return True
            return False


        test_position_index = []
        for index, (chrom, len_chrom) in enumerate(self.sampler.reference_sequence.get_chr_lens()):
            if chrom == 'chr8' or chrom == 'chr9':

                length = len_chrom - ignore_last_len
                for position in range(sequence_length, length - sequence_length, slide_window):
                    start = position - sequence_length // 2
                    end = position + sequence_length // 2
                    left = check_unk(start)
                    right = check_unk(end)
                    if left == True or right == True:
                        continue
                    sequence = self.sampler.reference_sequence.get_sequence_from_coords(chrom, start, end)
                    if 'N' in sequence:
                        continue    
                    test_position_index.append((chrom, position))
        print("Test Dataset", len(test_position_index))
        return test_position_index

    def update_buffer(self):
        # print("Update Buffer")
        sequence, y = self.sampler.sample(batch_size=self.buffer_size)   # (1, 100000, 4), (1, 10, 100000)
        x_str = [convert_from_one_hot(s) for s in sequence]

        seq = self.tokenizer(x_str,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else "do_not_pad",
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids
        seq_ids = torch.LongTensor(seq_ids)
        mask = torch.BoolTensor(seq['attention_mask'])

        # need to wrap in list
        target = torch.FloatTensor(y)
        target = target.permute(0, 2, 1)  # (bz, 10, 100000) -> (bz, 100000, 10)
        if self.track is not None:
            target = target[:, :, self.track].unsqueeze(-1)

        self.buffer = [(seq_ids[i], target[i], mask[i]) for i in range(self.buffer_size)]


    def generator(self):
        while True:
            self.update_buffer()
            
            for i in range(self.buffer_size):
                seq_ids, target, mask = self.buffer[i]

                if self.return_mask:
                    yield seq_ids, target, {'mask': mask}
                else:
                    yield seq_ids, target

        
    def valid_generator(self):
        test_position_index = self.get_test_data()
        # test_dataset = [self.sampler._retrieve(chrom, position) for chrom, position in test_position_index]
        # print("Finish Test Dataset Generation")

        for chrom, position in test_position_index:
            return_result = self.sampler._retrieve(chrom, position)
            if return_result is None:
                continue
            sequence, y = return_result
            x_str = convert_from_one_hot(sequence)

            seq = self.tokenizer(x_str,
                add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
                padding="max_length" if self.use_padding else "do_not_pad",
                max_length=self.max_length,
                truncation=True,
            )
            seq_ids = seq["input_ids"]
            seq_ids = torch.LongTensor(seq_ids)
            target = torch.FloatTensor(y)
            target = target.permute(1,0)  # (10, 100000) -> (100000, 10)

            if self.track is not None:
                target = target[:, self.track].unsqueeze(-1)

            if self.return_mask:
                yield seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
            else:
                yield seq_ids, target

    def test_generator(self):
        test_file = os.path.join(self.dest_path, "evaluation/allpred.str.npy")
        test_data = np.load(test_file, allow_pickle=True)
        print("Finish load test dataset")
        for x_str in test_data:
            seq = self.tokenizer(x_str,
                add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
                padding="max_length" if self.use_padding else "do_not_pad",
                max_length=self.max_length,
                truncation=True,
            )
            seq_ids = seq["input_ids"]
            seq_ids = torch.LongTensor(seq_ids)
            target = torch.zeros(self.max_length, 10)

            if self.track is not None:
                target = target[:, self.track].unsqueeze(-1)

            if self.return_mask:
                yield seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
            else:
                yield seq_ids, target


        
    def __iter__(self):
        if self.split == "train":
            return self.generator()
        elif self.split == "val":
            return self.valid_generator()
        else:
            return self.test_generator()


