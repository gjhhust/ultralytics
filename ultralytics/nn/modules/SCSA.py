import torch
import torch.nn as nn
from einops import rearrange

class MultiScaleConv1d(nn.Module):
    """Multi-scale 1D convolutions for spatial attention."""

    def __init__(self, in_channels, kernel_sizes):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, k, padding=k // 2, groups=in_channels) for k in kernel_sizes
        ])

    def forward(self, x):
        return torch.cat([conv(x) for conv in self.convs], dim=1)


class SpatialAttention(nn.Module):
    """Spatial-attention module with multi-scale convolutions."""

    def __init__(self, channels, group_kernel_sizes=[3, 5, 7, 9]):
        super().__init__()
        self.group_chans = channels // 4
        self.multi_scale_conv = MultiScaleConv1d(self.group_chans, group_kernel_sizes)
        self.norm = nn.GroupNorm(4, channels)
        self.sa_gate = nn.Sigmoid()

    def forward(self, x_h, x_w):
        # Apply multi-scale convolutions on the mean along W and H dimensions
        x_h_attn = self.sa_gate(self.norm(self.multi_scale_conv(x_h).view_as(x_h)))
        x_w_attn = self.sa_gate(self.norm(self.multi_scale_conv(x_w).view_as(x_w)))
        return x_h_attn.unsqueeze(-1), x_w_attn.unsqueeze(-2)


class ChannelAttention(nn.Module):
    """Channel-attention module using self-attention mechanism."""

    def __init__(self, dim, head_num, window_size=-1, down_sample_mode='avg_pool', attn_drop_ratio=0.):
        super().__init__()
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.window_size = window_size
        self.down_sample_mode = down_sample_mode

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                raise NotImplementedError("Recombination mode is not implemented.")
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

        self.norm = nn.GroupNorm(1, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Sigmoid()

    def forward(self, x):
        y = self.down_func(x)
        y = self.norm(y)
        q, k, v = self.qkv(y).chunk(3, dim=1)

        b, c, h, w = q.size()
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num (h w) head_dim', head_num=self.head_num)
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num (h w) head_dim', head_num=self.head_num)
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num (h w) head_dim', head_num=self.head_num)

        attn = (q @ k.transpose(-2, -1)) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        attn = (attn @ v).mean(dim=2, keepdim=True)
        return self.ca_gate(attn)


class SCSA(nn.Module):
    """Spatial and Channel-wise Squeeze and Attention module."""

    def __init__(
        self,
        dim: int,
        head_num: int,
        window_size: int = 7,
        group_kernel_sizes: list = [3, 5, 7, 9],
        qkv_bias: bool = False,
        fuse_bn: bool = False,
        norm_cfg: dict = None,
        act_cfg: dict = None,
        down_sample_mode: str = 'avg_pool',
        attn_drop_ratio: float = 0.,
        gate_layer: str = 'sigmoid',
    ):
        super().__init__()
        self.spatial_attention = SpatialAttention(dim, group_kernel_sizes)
        self.channel_attention = ChannelAttention(
            dim, head_num, window_size, down_sample_mode, attn_drop_ratio)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = x.mean(dim=3)
        x_w = x.mean(dim=2)
        x_h_attn, x_w_attn = self.spatial_attention(x_h, x_w)
        x = x * x_h_attn * x_w_attn
        return self.channel_attention(x) * x
    
    
    
import torch
import torch.nn as nn
from einops import rearrange

class SCSA_(nn.Module):

    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: list[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(SCSA_, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = self.dim // 4

        self.local_dwc = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=self.group_chans)
        self.global_dwc_s = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=self.group_chans)
        self.global_dwc_m = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=self.group_chans)
        self.global_dwc_l = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=self.group_chans)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                # dimensionality reduction
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def space_to_chans(self, x):
        # Implement space to chans if needed
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The dim of x is (B, C, H, W)
        """
        # Spatial attention priority calculation
        b, c, h_, w_ = x.size()
        # (B, C, H)
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        # (B, C, W)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        x = x * x_h_attn * x_w_attn

        # Channel attention based on self attention
        # reduce calculations
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()

        # normalization first, then reshape -> (B, H, W, C) -> (B, C, H * W) and generate q, k and v
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # (B, head_num, head_dim, head_dim)
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        # (B, head_num, head_dim, N)
        attn = attn @ v
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * x
    
    
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

# [这里插入之前提供的SCSA, ChannelAttention, SpatialAttention, 和 MultiScaleConv1d 类定义]

def test_scsa_equivalence():
    # 定义测试参数
    batch_size = 2
    channels = 64
    height = 56
    width = 56
    head_num = 8
    window_size = 7
    group_kernel_sizes = [3, 5, 7, 9]
    qkv_bias = False
    fuse_bn = False
    norm_cfg = None
    act_cfg = None
    down_sample_mode = 'avg_pool'
    attn_drop_ratio = 0.0
    gate_layer = 'sigmoid'

    # 创建测试输入
    x = torch.randn(batch_size, channels, height, width)

    # 实例化原始SCSA模块
    original_scsa = SCSA_(
        dim=channels,
        head_num=head_num,
        window_size=window_size,
        group_kernel_sizes=group_kernel_sizes,
        qkv_bias=qkv_bias,
        fuse_bn=fuse_bn,
        down_sample_mode=down_sample_mode,
        attn_drop_ratio=attn_drop_ratio,
        gate_layer=gate_layer,
    )

    # 实例化重构后的SCSA模块
    refactored_scsa = SCSA(
        dim=channels,
        head_num=head_num,
        window_size=window_size,
        group_kernel_sizes=group_kernel_sizes,
        qkv_bias=qkv_bias,
        fuse_bn=fuse_bn,
        down_sample_mode=down_sample_mode,
        attn_drop_ratio=attn_drop_ratio,
        gate_layer=gate_layer,
    )

    # 复制原始SCSA的权重到重构后的SCSA
    refactored_scsa.load_state_dict(original_scsa.state_dict())

    # 设置为评估模式以关闭dropout等
    original_scsa.eval()
    refactored_scsa.eval()

    with torch.no_grad():  # 禁用梯度计算
        output_original = original_scsa(x)
        output_refactored = refactored_scsa(x)

        # 检查输出形状是否相同
        if output_original.shape != output_refactored.shape:
            print("Output shapes are different!")
            return False

        # 计算两者的最大绝对误差
        max_abs_diff = (output_original - output_refactored).abs().max().item()
        print(f"Maximum absolute difference: {max_abs_diff}")

        # 定义一个可接受的最大误差阈值
        tolerance = 1e-5  # 根据实际情况调整此值
        if max_abs_diff < tolerance:
            print("The outputs are consistent within the given tolerance.")
            return True
        else:
            print("The outputs differ beyond the given tolerance.")
            return False

if __name__ == "__main__":
    if test_scsa_equivalence():
        print("Test passed.")
    else:
        print("Test failed.")