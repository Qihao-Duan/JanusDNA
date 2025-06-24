"""Decoder heads.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train

log = src.utils.train.get_logger(__name__)


class Decoder(nn.Module):
    """This class doesn't do much but just signals the interface that Decoders are expected to adhere to
    TODO: is there a way to enforce the signature of the forward method?
    """

    def forward(self, x, **kwargs):
        """
        x: (batch, length, dim) input tensor
        state: additional state from the model backbone
        *args, **kwargs: additional info from the dataset

        Returns:
        y: output tensor
        *args: other arguments to pass into the loss function
        """
        return x

    def step(self, x):
        """
        x: (batch, dim)
        """
        return self.forward(x.unsqueeze(1)).squeeze(1)


class SequenceDecoder(Decoder):
    def __init__(
        self, d_model, d_output=None, l_output=None, use_lengths=False, mode="last",
            conjoin_train=False, conjoin_test=False
    ):
        super().__init__()

        # TODO: maybe only for SNP classification.
        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)

        if isinstance(self.output_transform, nn.Linear):
            nn.init.normal_(self.output_transform.weight, std=0.1)  # 正确的初始化
            if self.output_transform.bias is not None:
                nn.init.zeros_(self.output_transform.bias)  # 初始化 bias
                
        print("decoder norm init")
        
        

        if l_output is None:
            self.l_output = None
            self.squeeze = False
        elif l_output == 0:
            # Equivalent to getting an output of length 1 and then squeezing
            self.l_output = 1
            self.squeeze = True
        else:
            assert l_output > 0
            self.l_output = l_output
            self.squeeze = False

        self.use_lengths = use_lengths
        self.mode = mode

        if mode == 'ragged':
            assert not use_lengths

        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test

    def forward(self, x, state=None, lengths=None, l_output=None):
        """
        x: (n_batch, l_seq, d_model) or potentially (n_batch, l_seq, d_model, 2) if using rc_conjoin
        Returns: (n_batch, l_output, d_output)
        """
        # input = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
        # print("change input as ", input)
        # x = input
        # print("x shape before decoder", x.shape)
        # print("before decoder, x[0]: ", x[0, 0, :10])
        # print("before decoder, x[1]: ", x[1, 0, :10])
        # nn.init.zeros_(self.output_transform.bias)  
        # nn.init.zeros_(self.output_transform.weight)
        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(1)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        if self.mode == "last":
            def restrict(x_seq):
                """Use last l_output elements of sequence."""
                return x_seq[..., -l_output:, :]

        elif self.mode == "first":
            def restrict(x_seq):
                """Use first l_output elements of sequence."""
                return x_seq[..., :l_output, :]

        elif self.mode == "pool":
            def restrict(x_seq):
                """Pool sequence over a certain range"""
                L = x_seq.size(1)
                s = x_seq.sum(dim=1, keepdim=True)
                if l_output > 1:
                    c = torch.cumsum(x_seq[..., -(l_output - 1):, ...].flip(1), dim=1)
                    c = F.pad(c, (0, 0, 1, 0))
                    s = s - c  # (B, l_output, D)
                    s = s.flip(1)
                denom = torch.arange(
                    L - l_output + 1, L + 1, dtype=x_seq.dtype, device=x_seq.device
                )
                s = s / denom
                return s

        elif self.mode == "sum":
            # TODO use same restrict function as pool case
            def restrict(x_seq):
                """Cumulative sum last l_output elements of sequence."""
                return torch.cumsum(x_seq, dim=-2)[..., -l_output:, :]
        elif self.mode == 'ragged':
            assert lengths is not None, "lengths must be provided for ragged mode"

            def restrict(x_seq):
                """Ragged aggregation."""
                # remove any additional padding (beyond max length of any sequence in the batch)
                return x_seq[..., : max(lengths), :]
        else:
            raise NotImplementedError(
                "Mode must be ['last' | 'first' | 'pool' | 'sum' | 'ragged']"
            )

        # Restrict to actual length of sequence
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = restrict(x)

        if squeeze:
            assert x.size(1) == 1
            x = x.squeeze(1)

        if self.conjoin_train or (self.conjoin_test and not self.training):
            x, x_rc = x.chunk(2, dim=-1)
            x = self.output_transform(x.squeeze())
            x_rc = self.output_transform(x_rc.squeeze())
            x = (x + x_rc) / 2
        else:
            x = self.output_transform(x)

        return x

    def step(self, x, state=None):
        # Ignore all length logic
        x_fwd = self.output_transform(x.mean(dim=1))
        x_rc = self.output_transform(x.flip(dims=[1, 2]).mean(dim=1)).flip(dims=[1])
        x_out = (x_fwd + x_rc) / 2
        return x_out


# class ContactMapDecoder(nn.Module):
#     def __init__(self, d_model, out_channels=32, seq_len=1048576, num_bins=512, 
#                  output_size=448, conjoin_train=False, conjoin_test=False):
#         super().__init__()
        
#         self.conjoin_train = conjoin_train
#         self.conjoin_test = conjoin_test
#         self.seq_len = seq_len
#         self.num_bins = num_bins
#         self.output_size = output_size
        
#         # Normalize input dimensions (reduce channel count)
#         self.input_norm = nn.Linear(d_model, out_channels)
        
#         # 1D convolution (small kernel_size + pooling)
#         self.conv1d = nn.Sequential(
#             nn.Conv1d(out_channels, out_channels, kernel_size=65, stride=64, padding=32),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(),
#             nn.AvgPool1d(kernel_size=32, stride=32)  # Compress to 512 bins
#         )
        
#         # Simple linear transformation to generate 2D features
#         self.linear_2d = nn.Linear(out_channels, out_channels)
        
#         # Single-layer 2D convolution (reduce complexity)
#         self.conv2d = nn.Conv2d(out_channels + 1, out_channels, kernel_size=3, padding=1)
#         self.bn2d = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
        
#         # Residual connection
#         self.residual_conv = nn.Conv2d(out_channels + 1, out_channels, kernel_size=1)
        
#         # Final convolution
#         self.final_conv = nn.Conv1d(out_channels, 1, kernel_size=1)
#         self.tanh = nn.Tanh()
        
#         # Precompute upper triangular indices
#         self.register_buffer('triu_indices', torch.triu_indices(output_size, output_size, offset=2))
        
#         # Parameter initialization
#         self.apply(self._init_weights)
    
#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv1d, nn.Conv2d)):
#             if m.kernel_size == (1, 1):  # Final convolution and residual convolution
#                 nn.init.normal_(m.weight, mean=0, std=0.01)
#             else:  # Other convolution layers
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#             nn.init.constant_(m.weight, 1.0)
#             nn.init.constant_(m.bias, 0.0)
    
#     def uppertriu(self, x):
#         return x[:, :, self.triu_indices[0], self.triu_indices[1]]
    
#     def process(self, x):
#         # x: [B, S, D]
#         # Input validation
#         if x.size(1) != self.seq_len or x.size(2) != self.input_norm.in_features:
#             raise ValueError(f"Expected input shape [B, {self.seq_len}, {self.input_norm.in_features}], got {x.shape}")
        
#         # Normalize input
#         x = self.input_norm(x)  # [B, S, out_channels]
#         x = x.permute(0, 2, 1)  # [B, out_channels, S]
        
#         # 1D convolution and pooling
#         x = self.conv1d(x)  # [B, out_channels, 512]
        
#         # 2D feature generation (simple outer product)
#         x = x.unsqueeze(3) @ x.unsqueeze(2)  # [B, out_channels, 512, 512]
#         x = self.linear_2d(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, out_channels, 512, 512]
        
#         # Distance encoding (normalized)
#         distance_encoding = torch.log1p(torch.abs(torch.arange(self.num_bins).unsqueeze(0) - torch.arange(self.num_bins).unsqueeze(1)))
#         distance_encoding = distance_encoding / distance_encoding.max()  # Normalize to [0, 1]
#         distance_encoding = distance_encoding.unsqueeze(0).unsqueeze(0).repeat(x.shape[0], 1, 1, 1).to(x.device)
#         x_input = torch.cat([x, distance_encoding], dim=1)  # [B, out_channels+1, 512, 512]
        
#         # 2D convolution with residual connection
#         x = self.conv2d(x_input)
#         x = self.bn2d(x)
#         x = self.relu(x)
#         x = x + self.residual_conv(x_input)  # Residual connection
        
#         # Cropping
#         crop_size = (self.num_bins - self.output_size) // 2
#         x = x[:, :, crop_size:self.num_bins-crop_size, crop_size:self.num_bins-crop_size]  # [B, out_channels, 448, 448]
        
#         # Upper triangular extraction
#         x = self.uppertriu(x)  # [B, out_channels, 99681]
        
#         # Final output
#         x = self.final_conv(x)  # [B, 1, 99681]
#         assert x.size(1) == 1, "Decoder: Final output should have 1 channel!"
#         x = x.squeeze(1)
#         x = self.tanh(x) * 2  # Scale to (-2, 2)
#         return x
    
#     def forward(self, x):
#         """
#         x: (n_batch, l_seq, d_model) or (n_batch, l_seq, d_model, 2) if using rc_conjoin
#         Returns: (n_batch, 1, 99681)
#         """
#         if self.conjoin_train or (self.conjoin_test and not self.training):
#             if x.size(-1) != 2:
#                 raise ValueError(f"Expected input shape [B, {self.seq_len}, {self.input_norm.in_features}, 2] for conjoin mode, got {x.shape}")
#             x, x_rc = x.chunk(2, dim=-1)  # [B, S, D, 1] -> [B, S, D], [B, S, D]
#             x = self.process(x)
#             x_rc = self.process(x_rc)
#             x = (x + x_rc) / 2
#         else:
#             if x.dim() != 3:
#                 raise ValueError(f"Expected input shape [B, {self.seq_len}, {self.input_norm.in_features}], got {x.shape}")
#             x = self.process(x)
        
#         return x
        
class LightweightContactMapDecoder(nn.Module):
    def __init__(self, d_model, out_channels=32, seq_len=1048576, num_bins=512,
                 output_size=448, conjoin_train=False, conjoin_test=False, use_distance_encoding=True):
        super().__init__()
        
        self.seq_len = seq_len
        self.num_bins = num_bins
        self.output_size = output_size
        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test
        self.use_distance_encoding = use_distance_encoding
        
        # Reduce dimensionality
        self.input_norm = nn.Linear(d_model, out_channels)
        
        # 1D convolution and pooling to compress sequence
        self.conv1d = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=65, stride=64, padding=32),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=32, stride=32)  # To 512 bins
        )
        
        # Pairwise linear transformation
        self.pairwise_proj = nn.Linear(out_channels * 2, out_channels)

        if use_distance_encoding:
            self.distance_proj = nn.Linear(1, out_channels)
        
        # Upper triangle indices for final output
        self.register_buffer('triu_indices', torch.triu_indices(output_size, output_size, offset=2))
        
        # Final output head
        self.final_proj = nn.Linear(out_channels, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def uppertriu(self, x):
        return x[:, self.triu_indices[0], self.triu_indices[1]]

    def process(self, x):
        B, L, D = x.shape
        if x.size(1) != self.seq_len or x.size(2) != self.input_norm.in_features:
            raise ValueError(f"Expected input shape [B, {self.seq_len}, {self.input_norm.in_features}], got {x.shape}")

        # Dimension reduction
        x = self.input_norm(x)  # [B, L, C]
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.conv1d(x)  # [B, C, 512]

        # Outer product to generate 2D features
        h = x.permute(0, 2, 1)  # [B, 512, C]
        h_i = h.unsqueeze(2).expand(-1, -1, self.num_bins, -1)  # [B, 512, 512, C]
        h_j = h.unsqueeze(1).expand(-1, self.num_bins, -1, -1)  # [B, 512, 512, C]
        pair_feats = torch.cat([h_i, h_j], dim=-1)  # [B, 512, 512, 2C]

        x = self.pairwise_proj(pair_feats)  # [B, 512, 512, C]

        # Optional: Distance Encoding
        if self.use_distance_encoding:
            distance = torch.abs(torch.arange(self.num_bins)[:, None] - torch.arange(self.num_bins)[None, :])  # [512, 512]
            distance = torch.log1p(distance.float()).unsqueeze(0).to(x.device)  # [1, 512, 512]
            distance = (distance / distance.max())  # Normalize
            dist_feat = self.distance_proj(distance.unsqueeze(-1))  # [1, 512, 512, C]
            x = x + dist_feat

        x = self.final_proj(x).squeeze(-1)  # [B, 512, 512]

        # Crop to target size
        crop = (self.num_bins - self.output_size) // 2
        x = x[:, crop:self.num_bins - crop, crop:self.num_bins - crop]  # [B, 448, 448]

        # Extract upper triangle
        x = self.uppertriu(x) #.unsqueeze(1)  # [B, 99681]
        return x

    def forward(self, x):
        if x.dim() == 4 and (self.conjoin_train or (self.conjoin_test and not self.training)):
            if x.size(-1) != 2:
                raise ValueError("Expected input shape with last dim 2 for conjoin mode")
            x, x_rc = x.chunk(2, dim=-1)
            x = self.process(x)
            x_rc = self.process(x_rc)
            return (x + x_rc) / 2
        else:
            if x.dim() != 3:
                raise ValueError(f"Expected input shape [B, L, D], got {x.shape}")
            return self.process(x)


class SequenceDecoder_longrangeSNP(Decoder):
    def __init__(
        self, d_model, d_output=None, l_output=None, use_lengths=False, mode="last",
            conjoin_train=False, conjoin_test=False
    ):
        super().__init__()

        # TODO: maybe only for SNP classification.
        # self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model * 2, d_output)
        self.decoder_layers = nn.ModuleList(
            [
                nn.Linear(d_model * 2, d_model * 2),
                nn.Softplus(),
                # nn.Linear(d_model, d_model),
                # nn.Softplus(),
                nn.Linear(d_model * 2, d_output),
            ]
        )
        
        for layer in self.decoder_layers:
            if isinstance(layer, nn.Linear):
                # nn.init.normal_(layer.weight, std=0.02)  # 正确的初始化
                nn.init.xavier_uniform_(layer.weight)  # 正确的初始化
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # 初始化 bias
                
        print("decoder norm init")
        
        

        if l_output is None:
            self.l_output = None
            self.squeeze = False
        elif l_output == 0:
            # Equivalent to getting an output of length 1 and then squeezing
            self.l_output = 1
            self.squeeze = True
        else:
            assert l_output > 0
            self.l_output = l_output
            self.squeeze = False

        self.use_lengths = use_lengths
        self.mode = mode

        if mode == 'ragged':
            assert not use_lengths

        self.conjoin_train = conjoin_train
        self.conjoin_test = conjoin_test

    def forward(self, x, state=None, lengths=None, l_output=None):
        """
        x: (n_batch, l_seq, d_model) or potentially (n_batch, l_seq, d_model, 2) if using rc_conjoin
        Returns: (n_batch, l_output, d_output)
        """
        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(1)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        if self.mode == "last":
            def restrict(x_seq):
                """Use last l_output elements of sequence."""
                return x_seq[..., -l_output:, :]

        elif self.mode == "first":
            def restrict(x_seq):
                """Use first l_output elements of sequence."""
                return x_seq[..., :l_output, :]

        elif self.mode == "pool":
            def restrict(x_seq):
                """Pool sequence over a certain range"""
                L = x_seq.size(1)
                s = x_seq.sum(dim=1, keepdim=True)
                if l_output > 1:
                    c = torch.cumsum(x_seq[..., -(l_output - 1):, ...].flip(1), dim=1)
                    c = F.pad(c, (0, 0, 1, 0))
                    s = s - c  # (B, l_output, D)
                    s = s.flip(1)
                denom = torch.arange(
                    L - l_output + 1, L + 1, dtype=x_seq.dtype, device=x_seq.device
                )
                s = s / denom
                return s

        elif self.mode == "sum":
            # TODO use same restrict function as pool case
            def restrict(x_seq):
                """Cumulative sum last l_output elements of sequence."""
                return torch.cumsum(x_seq, dim=-2)[..., -l_output:, :]
        elif self.mode == 'ragged':
            assert lengths is not None, "lengths must be provided for ragged mode"

            def restrict(x_seq):
                """Ragged aggregation."""
                # remove any additional padding (beyond max length of any sequence in the batch)
                return x_seq[..., : max(lengths), :]
        else:
            raise NotImplementedError(
                "Mode must be ['last' | 'first' | 'pool' | 'sum' | 'ragged']"
            )

        # Restrict to actual length of sequence
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = restrict(x)

        if squeeze:
            assert x.size(1) == 1
            x = x.squeeze(1)

        if self.conjoin_train or (self.conjoin_test and not self.training):
            x, x_rc = x.chunk(2, dim=-1)
            for layer in self.decoder_layers:
                x = layer(x.squeeze())
                x_rc = layer(x_rc.squeeze())
            
            # x = self.output_transform(x.squeeze())
            # x_rc = self.output_transform(x_rc.squeeze())
            x = (x + x_rc) / 2
        else:
            # x = self.output_transform(x)
            for layer in self.decoder_layers:
                x = layer(x.squeeze())

        return x

    def step(self, x, state=None):
        # Ignore all length logic
        x_fwd = self.output_transform(x.mean(dim=1))
        x_rc = self.output_transform(x.flip(dims=[1, 2]).mean(dim=1)).flip(dims=[1])
        x_out = (x_fwd + x_rc) / 2
        return x_out
    
    
# For every type of encoder/decoder, specify:
# - constructor class
# - list of attributes to grab from dataset
# - list of attributes to grab from model

registry = {
    "stop": Decoder,
    "id": nn.Identity,
    "linear": nn.Linear,
    "sequence": SequenceDecoder,
    "sequence_snp": SequenceDecoder_longrangeSNP,
    "contact_map": LightweightContactMapDecoder,
}

model_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output"],
    "sequence_snp": ["d_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_state", "state_to_tensor"],
    "forecast": ["d_output"],
    "token": ["d_output"],
    "contact_map": ["d_output"],
}

dataset_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output", "l_output"],
    "sequence_snp": ["d_output", "l_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_output"],
    "forecast": ["d_output", "l_output"],
    "token": ["d_output"],
    "contact_map": ["d_output"],
}


def _instantiate(decoder, model=None, dataset=None):
    """Instantiate a single decoder"""
    if decoder is None:
        return None

    if isinstance(decoder, str):
        name = decoder
    else:
        name = decoder["_name_"]

    # Extract arguments from attribute names
    dataset_args = utils.config.extract_attrs_from_obj(
        dataset, *dataset_attrs.get(name, [])
    )
    model_args = utils.config.extract_attrs_from_obj(model, *model_attrs.get(name, []))
    print("dataset_args", dataset_args)
    print("model_args", model_args)
    # Instantiate decoder
    obj = utils.instantiate(registry, decoder, *model_args, *dataset_args)
    return obj


def instantiate(decoder, model=None, dataset=None):
    """Instantiate a full decoder config, e.g. handle list of configs
    Note that arguments are added in reverse order compared to encoder (model first, then dataset)
    """
    decoder = utils.to_list(decoder)
    return U.PassthroughSequential(
        *[_instantiate(d, model=model, dataset=dataset) for d in decoder]
    )
