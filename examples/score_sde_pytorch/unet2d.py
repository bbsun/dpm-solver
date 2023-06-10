import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Tuple, Optional
# UNET2D for DDPM
channel_c = 128
from torch.utils.checkpoint import checkpoint
#卷积模块
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetConv2, self).__init__()
        self.conv1_X    = nn.Conv2d(1,  out_size,  (3,3), 1, 'same')
        self.conv2_X    = nn.Conv2d(out_size, out_size,  (3,3), 1, 'same')
        self.conv3_XTOL = nn.Conv2d(out_size, out_size,  (3,3), 1, 'same')
        self.batch1_X   = nn.BatchNorm2d(out_size)
        self.batch2_X   = nn.BatchNorm2d(out_size)
        self.relu1_X    = nn.ReLU()
        self.relu2_X    = nn.ReLU()
        self.conv1_L    = nn.Conv2d(in_size,  out_size,  (3,3), 1, 'same')
        self.conv2_L    = nn.Conv2d(out_size, out_size,  (3,3), 1, 'same')
        self.batch1_L   = nn.BatchNorm2d(out_size)
        self.batch2_L   = nn.BatchNorm2d(out_size)
        self.batch3_L   = nn.BatchNorm2d(out_size)
        self.relu1_L    = nn.ReLU()
        self.relu2_L    = nn.ReLU()
        self.relu3_L    = nn.ReLU()
        self.time_emb   = nn.Linear(channel_c,  channel_c)
        self.conv1_T    = nn.Conv2d(channel_c,  out_size,  (3,3), 1, 'same')
        self.conv2_T    = nn.Conv2d( out_size,  out_size,  (3,3), 1, 'same')
        self.conv3_TTOL = nn.Conv2d( out_size,  out_size,  (3,3), 1, 'same')
        self.batch1_T   = nn.BatchNorm2d(out_size)
        self.batch2_T   = nn.BatchNorm2d(out_size)
        self.relu1_T    = nn.ReLU()
        self.relu2_T    = nn.ReLU()
    def forward(self, inputs):
        L0, X0, T0 = inputs
        nb,nc,nh,nw = L0.shape
        L1    = self.relu1_L(self.batch1_L(self.conv1_L(L0)))
        L     = self.relu2_L(self.batch2_L(self.conv2_L(L1)))
        X1    = self.relu1_X(self.batch1_X(self.conv1_X(X0)))
        X     = self.relu2_X(self.batch2_X(self.conv2_X(X1)))
        T     = self.time_emb(T0)[:,:,None,None].repeat(1,1,nh,nw)
        T     = self.conv1_T(T)
        XTOL  = self.conv3_XTOL(X)
        #TTOL  = self.conv3_TTOL(T)
        TTOL  = T
        #print(f'L shape {L.shape} XTOL shape {XTOL.shape} TTOL shape {TTOL.shape}')
        L     = self.relu3_L(self.batch3_L(L+ XTOL + TTOL))
        L     = L + L1
        X     = X + X1
        return L, X0, T0
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
class unetDown(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size)
        self.down = nn.MaxPool2d((2, 2),ceil_mode=True)
    def forward(self, inputs):
        outputs = self.conv(inputs)
        L,X,T = outputs
        L = self.down(L)
        X = self.down(X)
        outputs_down = (L,X,T)
        return outputs,outputs_down
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv = unetConv2( in_size, out_size )
        self.up = nn.Upsample( scale_factor=2)
    def forward(self, inputs1, inputs2 ):
        L1,X1,T1 = inputs1
        L,X,T = inputs2
        L2    = self.up(L)
        L     = torch.cat([L1,L2],1)
        X     = self.up(X)
        return self.conv((L,X,T))
import math
class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = nn.ReLU()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb
class AttentionBlock(nn.Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.reshape(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res
class  UnetModel_DDPM(nn.Module):
    def __init__(self):
        super(UnetModel_DDPM, self).__init__()
        filters = [16*4, 32*4, 64*4, 128*4]                            # original is 16,32,64,128
        self.down1   = unetDown(1, filters[0])                 # 16
        self.down2   = unetDown(filters[0], filters[1] )       # 32
        self.down3   = unetDown(filters[1], filters[2] )       # 64
        self.center  = unetConv2(filters[2], filters[3])       #128
        self.up3     = unetUp(filters[3]+filters[2], filters[2])
        self.up2     = unetUp(filters[2]+filters[1], filters[1])
        self.up1     = unetUp(filters[1]+filters[0], filters[0])
        self.final   = nn.Conv2d(filters[0], 1, (1,1), 1)
        self.time_emb= TimeEmbedding(channel_c)
        self.att     = AttentionBlock(filters[3],8)
    def forward(self,L00,T00):
        L = L00
        T = T00
        X = torch.zeros_like(L)
        T = self.time_emb(T)
        inputs = L,X,T
        down10,down1  = self.down1(inputs)
        down20,down2  = self.down2(down1)
        down30,down3  = self.down3(down2)
        center        = self.center(down3)
        L,X,T         = center
        #L = self.att(L,T)
        center        = L,X,T
        up3           = self.up3(down30,center)
        up2           = self.up2(down20,up3)
        up1           = self.up1(down10,up2)
        L,X,T         = up1
        out = self.final(L)
        return out
