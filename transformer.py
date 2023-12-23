from typing import Tuple, Dict
from fairseq import utils
from fairseq.models import *
from fairseq.modules import *
import torch

#https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

# https://gitlab.maastrichtuniversity.nl/dsri-examples/dsri-pytorch-workspace/-/blob/c8a88cdeb8e1a0f3a2ccd3c6119f43743cbb01e9/examples/transformer/fairseq/models/transformer.py
#https://github.com/pytorch/fairseq/issues/568
# fairseq/fairseq/models/fairseq_encoder.py

# https://github.com/pytorch/fairseq/blob/master/fairseq/modules/transformer_layer.py
class TransformerEncode(FairseqEncoder):

    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__({})
        #print('my TransformerEncode()')

        self.layer = torch.nn.ModuleList([
            TransformerEncoderLayer(Namespace({
                'encoder_embed_dim': dim,
                'encoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'encoder_normalize_before': True,
                'encoder_ffn_embed_dim': ff_dim,
            })) for i in range(num_layer)
        ])
        self.layer_norm = torch.nn.LayerNorm(dim)

    def forward(self, x):# T x B x C

        for layer in self.layer:
            x = layer(x, encoder_padding_mask = None)
        x = self.layer_norm(x)
        return x


# https://fairseq.readthedocs.io/en/latest/tutorial_simple_lstm.html
# see https://gitlab.maastrichtuniversity.nl/dsri-examples/dsri-pytorch-workspace/-/blob/c8a88cdeb8e1a0f3a2ccd3c6119f43743cbb01e9/examples/transformer/fairseq/models/transformer.py
class TransformerDecode(FairseqIncrementalDecoder):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__({})
        #print('my TransformerDecode()')

        self.layer = torch.nn.ModuleList([
            TransformerDecoderLayer(Namespace({
                'decoder_embed_dim': dim,
                'decoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'decoder_normalize_before': True,
                'decoder_ffn_embed_dim': ff_dim,
            })) for i in range(num_layer)
        ])
        self.layer_norm = torch.nn.LayerNorm(dim)


    def forward(self, x, mem, x_mask):
        #print('my TransformerDecode forward()')
        for layer in self.layer:
            x = layer(x, mem, self_attn_mask=x_mask)[0]

        x = self.layer_norm(x)
        return x  # T x B x C

    #def forward_one(self, x, mem, incremental_state):
    def forward_one(self,x,mem,incremental_state):
        x = x[-1:]
        for layer in self.layer:
            x = layer(x, mem, incremental_state=incremental_state)[0]

        x = self.layer_norm(x)
        return x