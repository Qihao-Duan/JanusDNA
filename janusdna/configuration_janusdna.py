# This code is based on the original work by AI21 Labs Ltd. and HuggingFace Inc.,
# which is derived from EleutherAI's GPT-NeoX library and the GPT-NeoX/OPT implementations.
# Significant modifications have been made by AILS Lab for the purpose of developing the JanusDNA model.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" JanusDNA model configuration"""
import math

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import List, Union
# from genfoundry.crab.modeling_crab import CrabFlexAttention

logger = logging.get_logger(__name__)


class JanusDNAConfig(PretrainedConfig):
    r"""
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 65536):
            Vocabulary size of the JanusDNA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`JanusDNAModel`]
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
            Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
            integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
            logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
            sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
            significantly.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss. See [here]() for more details
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        sliding_window (`int`, *optional*):
            Sliding window attention window size. If not specified, will default to `None`.
        max_position_embeddings (`int`, *optional*, defaults to 262144):
            This value doesn't have any real effect. The maximum sequence length that this model is intended to be
            used with. It can be used with longer sequences, but performance may degrade.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            The number of experts to root per-token, can be also interpreted as the `top-p` routing
            parameter
        num_experts (`int`, *optional*, defaults to 16):
            Number of experts per Sparse MLP layer.
        expert_layer_period (`int`, *optional*, defaults to 2):
            Once in this many layers, we will have an expert layer
        expert_layer_offset (`int`, *optional*, defaults to 1):
            The first layer index that contains an expert mlp layer
        attn_layer_period (`int`, *optional*, defaults to 8):
            Once in this many layers, we will have a vanilla attention layer
        attn_layer_offset (`int`, *optional*, defaults to 4):
            The first layer index that contains a vanilla attention mlp layer
        use_mamba_kernels (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use the fast mamba kernels. These are available only if `mamba-ssm` and
            `causal-conv1d` are installed, and the mamba modules are running on a CUDA device. Raises ValueError if
            `True` and kernels are not available
        mamba_d_state (`int`, *optional*, defaults to 16):
            The dimension the mamba state space latents
        mamba_d_conv (`int`, *optional*, defaults to 4):
            The size of the mamba convolution kernel
        mamba_expand (`int`, *optional*, defaults to 2):
            Expanding factor (relative to hidden_size) used to determine the mamba intermediate size
        mamba_dt_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
            Rank of the the mamba discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
        mamba_conv_bias (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use bias in the convolution layer of the mamba mixer block.
        mamba_proj_bias (`bool`, *optional*, defaults to `False`):
            Flag indicating whether or not to use bias in the input and output projections (["in_proj", "out_proj"]) of the mamba mixer block

    """

    model_type = "JanusDNA"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=6,  # 6
        phenotypes_size=2,  # 6
        tie_word_embeddings=False,
        hidden_size=1024,
        intermediate_size=1024,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        num_logits_to_keep=1,
        output_router_logits=False,
        router_aux_loss_coef=0.2,
        pad_token_id=0,
        # bos_token_id=1,
        # eos_token_id=2,
        sliding_window=None,
        max_position_embeddings=262144,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_experts=16,
        expert_layer_period=2,
        expert_layer_offset=1,
        attn_layer_period=8,
        attn_layer_offset=4,
        use_mamba_kernels=True,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank="auto",
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        bidirectional: bool = True,
        bidirectional_strategy: Union[str, None] = "add",# todo: # should be "add", "concat", "ew_multiply" and "final layer transformer"
        bidirectional_weight_tie: bool = True,
        layer_fusion: bool = False,
        attention_mask: bool = False,
        final_attention: bool = False,
        layer_fusion_strategy: Union[str, None] = "pool",  # should be "pool" and "half", but frankly, half is not reasonable, for only fwd info without bwd info
        mid_single_direction_attention: bool = False,
        final_attention_class: str = "flex_attention",
        bidirectional_attn_tie: bool = False,
        is_causal: bool = True,
        gradient_checkpointing: bool = False,
        flex_attn_n_embd: int = 0,
        pad_vocab_size_multiple: int = 8,
        flexattn_fullattn: bool = False,
        **kwargs,
    ):
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.layer_fusion = layer_fusion
        self.attention_mask = attention_mask
        self.final_attention = final_attention
        self.layer_fusion_strategy = layer_fusion_strategy
        self.mid_single_direction_attention = mid_single_direction_attention
        self.final_attention_class = final_attention_class
        self.bidirectional_attn_tie = bidirectional_attn_tie
        self.is_causal = is_causal
        self.gradient_checkpointing = gradient_checkpointing
        self.flex_attn_n_embd = flex_attn_n_embd
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        
        self.vocab_size = vocab_size
        self.phenotypes_size = phenotypes_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.expert_layer_period = expert_layer_period
        self.expert_layer_offset = expert_layer_offset
        self.attn_layer_period = attn_layer_period
        self.attn_layer_offset = attn_layer_offset

        self.use_mamba_kernels = use_mamba_kernels
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = (
            math.ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        )
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        
        
        self.flexattn_fullattn = flexattn_fullattn

        super().__init__(
            pad_token_id=pad_token_id,
            # bos_token_id=bos_token_id,
            # eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        return [
            "attention" if i % self.attn_layer_period == self.attn_layer_offset else "mamba"
            for i in range(self.num_hidden_layers)
        ]

    @property
    def layers_num_experts(self):
        if self.expert_layer_period == 0:
            return 1 
        else:
            return [
                self.num_experts if i % self.expert_layer_period == self.expert_layer_offset else 1
                for i in range(self.num_hidden_layers)
            ]
