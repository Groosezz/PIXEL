# coding=utf-8
# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
#
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
"""
PIXEL model configuration
adapted from ViT MAE model configuration:
https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/vit_mae/configuration_vit_mae.py
"""

from transformers import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

PIXEL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Team-PIXEL/pixel-base": "https://huggingface.co/Team-PIXEL/pixel-base/resolve/main/config.json",
}


class PIXELConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PIXELModel`]. It is used to instantiate a PIXEL
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the PIXEL
    [Team-PIXEL/pixel-base](https://huggingface.co/Team-PIXEL/pixel-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`Union[Tuple[int, int], int]`, *optional*, defaults to (16, 8464)]):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        decoder_num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the decoder.
        decoder_hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the decoder.
        decoder_num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers in the decoder.
        decoder_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the decoder.
        mask_ratio (`float`, *optional*, defaults to 0.25):
            The ratio of the number of masked tokens in the input sequence.
        norm_pix_loss (`bool`, *optional*, defaults to `True`):
            Whether or not to train with normalized pixels (see Table 3 in the paper).
        
        ### below for generative decoder###
        max_position_embeddings (`int`, *optional*, defaults to 1024 to 529):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the BART model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BartModel`] or [`TFBartModel`].
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        
        # FOR Binary Pixel model
        isBinary (`bool`, *optional*, defaults to `False`):
            whether or not the model output binary pixel values
        BinaryLoss (`str` or `function`, *optional*, defaults to "mse"): 
            can select from ["mse", "focal", "bce"]
    Example:

    ```python
    >>> from pixel import PIXELModel, PIXELConfig

    >>> # Initializing a PIXEL pixel-base style configuration
    >>> configuration = PIXELConfig()

    >>> # Initializing a model from the vit-mae-base style configuration
    >>> model = PIXELModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "pixel"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "decoder_num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        image_size=(16, 8464),
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        decoder_num_attention_heads=16,
        decoder_hidden_size=512,
        decoder_num_hidden_layers=8,
        decoder_intermediate_size=2048,
        mask_ratio=0.25,
        norm_pix_loss=True,
        max_position_embeddings=529,
        #vocab_size=50265,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
        forced_eos_token_id=2,
        add_cross_attention = True,
        scale_embedding = False,
        # for binary model
        isBinary = False,
        BinaryLoss = 'mse',
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        #generative decoder
        self.max_position_embeddings =  max_position_embeddings
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.is_encoder_decoder = True
        self.decoder_start_token_id = decoder_start_token_id
        self.forced_eos_token_id = forced_eos_token_id
        self.scale_attn_by_inverse_layer_idx = False
        self.reorder_and_upcast_attn = False
        self.scale_attn_weights = True
        self.resid_pdrop = attention_probs_dropout_prob
        self.embd_pdrop = attention_probs_dropout_prob
        self.attn_pdrop = attention_probs_dropout_prob
        self.activation_function = hidden_act
        self.n_inner = decoder_intermediate_size
        self.layer_norm_epsilon = layer_norm_eps
        self.add_cross_attention = add_cross_attention
        self.vocab_size = patch_size*patch_size*num_channels
        self.n_layer = decoder_num_hidden_layers
        self.n_head = num_attention_heads
        self.n_positions = max_position_embeddings
        self.n_embd = hidden_size
        self.scale_embedding = scale_embedding
        
        #binary pixel
        self.isBinary = isBinary
        self.BinaryLoss = BinaryLoss


    
    
        
        
        
