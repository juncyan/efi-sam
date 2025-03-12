import paddle
import paddle.nn as nn
from paddlenlp.transformers.mamba.modeling import MambaConfig, MambaBlock
# from .mamba import MambaBlock


class Mamba(nn.Layer):
    def __init__(self, num_features=6, embedding_dim=256):
        super().__init__()
        config = MambaConfig(embedding_dim,num_features)
        self.blocks = nn.Sequential(*[MambaBlock(config, i) for i in range(config.num_hidden_layers)])
    
    def forward(self, x):
        return self.blocks(x)