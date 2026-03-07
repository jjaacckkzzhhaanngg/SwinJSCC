import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint  # 用于梯度检查点，节省显存
from timm.layers import DropPath, to_2tuple, trunc_normal_  # timm库的通用层工具
import numpy as np


class Mlp(nn.Module):
    """
    多层感知机（MLP）模块，用于Transformer块的前馈网络部分
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        初始化Mlp模块
        Args:
            in_features: 输入特征维度
            hidden_features: 隐藏层维度，默认等于in_features
            out_features: 输出特征维度，默认等于in_features
            act_layer: 激活函数层，默认GELU
            drop: Dropout概率，默认0.0
        """
        super(Mlp, self).__init__()
        # 输出维度默认等于输入维度，隐藏层维度默认等于输入维度
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 第一层线性变换：输入 -> 隐藏层
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()  # 激活函数
        # 第二层线性变换：隐藏层 -> 输出
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)  # Dropout层

    def forward(self, x):
        """前向传播"""
        x = self.fc1(x)  # 线性变换1
        x = self.act(x)  # 激活
        x = self.drop(x) # Dropout
        x = self.fc2(x)  # 线性变换2
        x = self.drop(x) # Dropout
        return x


def window_partition(x, window_size):
    """
    将特征图按照窗口大小划分成多个不重叠的窗口
    Args:
        x: 输入特征图，形状为 (B, H, W, C)，B=批量大小，H=高度，W=宽度，C=通道数
        window_size: 窗口大小（int）
    
    Returns:
        windows: 划分后的窗口，形状为 (num_windows*B, window_size, window_size, C)
                 num_windows = (H/window_size) * (W/window_size)
    """
    B, H, W, C = x.shape
    # 重塑为 [B, H//ws, ws, W//ws, ws, C]，ws=window_size
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 维度重排：[B, H//ws, W//ws, ws, ws, C] -> 展平为 [B*num_windows, ws, ws, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将划分的窗口还原回原始特征图形状（window_partition的逆操作）
    Args:
        windows: 窗口特征，形状为 (num_windows*B, window_size, window_size, C)
        window_size: 窗口大小（int）
        H: 原始特征图高度
        W: 原始特征图宽度
    
    Returns:
        x: 还原后的特征图，形状为 (B, H, W, C)
    """
    # 计算批量大小B
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # 重塑窗口为 [B, H//ws, W//ws, ws, ws, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # 维度重排：[B, H//ws, ws, W//ws, ws, C] -> 展平为 [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def cal_relative_position_index(window_size:tuple[int]):
    """
    生成窗口内token的两两相对位置索引, 参考https://blog.csdn.net/weixin_42392454/article/details/141395092
    Args:
        window_size: 窗口大小（tuple[int]）
    Returns:
        relative_position_index: 相对位置索引，形状为 (Wh * Ww, Wh * Ww)
    """
    # 生成窗口内绝对坐标：[0, Wh-1] 和 [0, Ww-1]
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Wh, Ww]
    coords_flatten = torch.flatten(coords, 1)  # [2, Wh*Ww]
    # 计算两两相对坐标：[2, Wh*Ww, Wh*Ww]
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    # 维度重排：[Wh*Ww, Wh*Ww, 2]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    # 偏移坐标到非负数区间（便于索引）
    relative_coords[:, :, 0] += window_size[0] - 1
    relative_coords[:, :, 1] += window_size[1] - 1
    # 计算一维索引（合并高宽维度）
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)  # [Wh*Ww, Wh*Ww]
    return relative_position_index

def cal_attention_mask(input_resolution:tuple[int], window_size:int, shift_size:int):
    """
    生成注意力掩码矩阵，用于防止窗口内token的自注意力计算
    Args:
        input_resolution: 输入特征图分辨率（tuple[int]）
        window_size: 窗口大小（int）
        shift_size: 窗口移位大小（int）

    Returns:
        attn_mask: 注意力掩码矩阵，形状为 (num_windows, Wh * Ww, Wh * Ww)
    """
    H, W = input_resolution
    img_mask = torch.zeros((1, H, W, 1))  # 初始化掩码图 [1, H, W, 1]
    # 划分图像区域（3x3）用于生成掩码
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    # 划分窗口并生成掩码
    mask_windows = window_partition(img_mask, window_size)  # [nW, ws, ws, 1]
    mask_windows = mask_windows.view(-1, window_size * window_size)  # [nW, N]
    # 计算两两窗口的掩码：不同区域为-100（无效），相同区域为0（有效）
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

class WindowAttention(nn.Module):
    """
    基于窗口的多头自注意力（W-MSA）模块，支持移位窗口（SW-MSA）
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        初始化函数
        Args:
            dim: 输入特征通道数
            window_size: 窗口的高和宽（tuple[int]）
            num_heads: 注意力头数
            qkv_bias: 是否为Q/K/V添加可学习偏置，默认True
            qk_scale: Q/K缩放因子，默认None（使用head_dim的平方根）
            attn_drop: 注意力权重的Dropout概率，默认0.0
            proj_drop: 输出的Dropout概率，默认0.0
        """
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww) 窗口的高和宽
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子

        # 生成相对位置偏置矩阵（离散）
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        
        # 生成相对位置索引
        relative_position_index = cal_relative_position_index(self.window_size) # [Wh*Ww, Wh*Ww]
        self.register_buffer("relative_position_index", relative_position_index) # 注册为不可学习的缓冲区

        # QKV投影层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, add_token=True, token_num=0, mask=None):
        """
        前向传播函数
        Args:
            x: 输入特征，形状为 (num_windows*B, N, C) 
               - num_windows: 总窗口数
               - B: batch size
               - N: 每个窗口的token数（含额外token），无额外token时 N=Wh*Ww
               - C: 特征通道数
            add_token: 是否添加了额外token（如cls_token），默认True
            token_num: 额外token的数量，默认0
            mask: 窗口注意力掩码，形状为 (num_windows, Wh * Ww, Wh * Ww) 或 None
                  - 0: 有效位置，-inf: 无效位置（被mask）
        Returns:
            x: 注意力计算后的输出，形状同输入 (num_windows * B, N, C)
        """
        # 获取输入张量的形状
        B_, N, C = x.shape  # B_ = num_windows * B, N = 窗口内token数（含额外token）

        # 1. QKV投影与维度变换
        # - 先通过线性层得到QKV: (B_, N, 3*C)
        # - 重塑维度: (B_, N, 3, num_heads, head_dim)，其中 head_dim = C//num_heads
        # - 维度置换: (3, B_, num_heads, N, head_dim)，方便拆分Q/K/V
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 拆分Q/K/V，形状均为 (B_, num_heads, N, head_dim)

        # 2. 计算注意力分数
        q = q * self.scale  # 对Q进行缩放，防止内积值过大
        # Q @ K^T: 计算注意力相似度，形状 (B_, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1))

        # 3. 应用相对位置偏置
        # - 根据相对位置索引取出对应的偏置值: [Wh*Ww*Wh*Ww, num_heads]
        # - 重塑为窗口内位置对的偏置矩阵: [Wh*Ww, Wh*Ww, num_heads]
        # - 维度置换: [num_heads, Wh*Ww, Wh*Ww]，适配注意力分数的维度
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        # 如果有额外token，仅对原始窗口区域（排除额外token）添加相对位置偏置
        if add_token:
            # token_num: 额外token数量，如cls_token=1，则从第1个位置开始是窗口内的特征
            attn[:, :, token_num:, token_num:] = attn[:, :, token_num:, token_num:] + relative_position_bias.unsqueeze(0)
        else:
            # 无额外token，直接对整个注意力矩阵添加偏置
            attn = attn + relative_position_bias.unsqueeze(0)

        # 4. 应用窗口掩码（处理移位窗口的边界问题）
        if mask is not None:
            if add_token:
                # 对掩码矩阵进行padding，适配额外token的维度
                # padding格式: (left, right, top, bottom)，此处为额外token位置添加0填充
                mask = F.pad(mask, (token_num, 0, token_num, 0), "constant", 0)
            
            nW = mask.shape[0]  # 窗口数量
            # 将注意力分数重塑并添加掩码: 
            # - attn重塑: (B//nW, nW, num_heads, N, N)
            # - mask扩展: (1, nW, 1, N, N)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)  # 恢复原形状
            attn = self.softmax(attn)  # 注意力权重归一化
        else:
            # 无掩码时直接归一化
            attn = self.softmax(attn)

        # 5. 注意力权重dropout
        attn = self.attn_drop(attn)

        # 6. 注意力加权求和 + 维度变换
        # - attn @ v: (B_, num_heads, N, head_dim)，注意力加权V
        # - transpose(1,2): (B_, N, num_heads, head_dim)
        # - reshape: (B_, N, C)，拼接所有注意力头的输出
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        # 7. 输出投影 + dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
    
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    核心作用：对输入特征图进行下采样（空间尺寸减半），同时将通道数扩展（通常翻倍）
    类似于CNN中的池化层，但能保留更多特征信息（通过拼接2×2窗口内的像素而非取最大/平均）
    """
    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm):
        """
        初始化函数
        Args:
            input_resolution (tuple[int]): 输入特征的空间分辨率 (H, W)
            dim (int): 输入特征的通道数 C
            out_dim (int, optional): 输出特征的通道数，默认None（此时out_dim=dim，通常实际使用时会设为2*dim）
            norm_layer (nn.Module, optional): 归一化层，默认 nn.LayerNorm
        """
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution  # 保存输入分辨率 (H, W)
        if out_dim is None:
            out_dim = dim  # 若未指定输出通道数，默认与输入相同
        self.dim = dim  # 输入通道数
        
        # 定义特征降维/变换层：将拼接后的4*C通道映射到out_dim（通常out_dim=2*C）
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        # 定义归一化层：对拼接后的4*C通道进行LayerNorm
        self.norm = norm_layer(4 * dim)
        
        # 注释掉的是用卷积实现Patch Merging的另一种方式（可选参考）
        # self.proj = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
        # self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        """
        Args:
            x: 输入特征，形状为 (B, H*W, C)
               - B: batch size
               - H*W: 序列长度（Transformer中特征图被展平为序列）
               - C: 输入通道数
        Returns:
            x: 输出特征，形状为 (B, (H/2) * (W/2), out_dim)
               - 空间尺寸减半（H→H/2, W→W/2）
               - 通道数通常翻倍（C→2 * C）
        """
        H, W = self.input_resolution  # 获取输入的空间分辨率 H, W
        B, L, C = x.shape  # 获取输入形状：B=batch, L=H*W, C=通道数
        
        # 校验输入的合法性
        assert L == H * W, "input feature has wrong size"  # 确保序列长度等于H*W
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."  # 确保H和W是偶数，否则无法2×2下采样

        # 1. 将序列形式的特征恢复为空间特征图形式：(B, H*W, C) → (B, H, W, C)
        x = x.view(B, H, W, C)

        # 2. 按2×2不重叠窗口采样，提取每个窗口内的4个像素（切片语法：start:end:step，0::2表示从0开始，步长为2）
        x0 = x[:, 0::2, 0::2, :]  # 取窗口左上角像素：(B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # 取窗口左下角像素：(B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # 取窗口右上角像素：(B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # 取窗口右下角像素：(B, H/2, W/2, C)

        # 3. 在通道维度拼接4个采样结果：4个(B, H/2, W/2, C) → (B, H/2, W/2, 4*C)
        x = torch.cat([x0, x1, x2, x3], -1)

        # 4. 恢复为Transformer的序列形式：(B, H/2, W/2, 4*C) → (B, (H/2)*(W/2), 4*C)
        x = x.view(B, H*W//4, 4 * C)

        # 5. 对拼接后的通道进行LayerNorm归一化
        x = self.norm(x)

        # 6. 通过线性层进行通道变换（通常将4*C降为2*C）：(B, (H/2)*(W/2), 4*C) → (B, (H/2)*(W/2), out_dim)
        x = self.reduction(x)

        # 注释掉的是用卷积实现的另一种方式（可选参考）
        # x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B,C,H,W)
        # x = self.proj(x).flatten(2).transpose(1, 2)  # (B, (H/2)(W/2), out_dim)
        # x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        """
        额外的字符串表示，用于打印模块信息时显示关键参数
        """
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        """
        计算该层的浮点运算量（FLOPs），用于模型复杂度分析
        Returns:
            flops: 该层的总浮点运算量
        """
        H, W = self.input_resolution
        # 1. LayerNorm的FLOPs：H*W*dim（归一化操作）
        flops = H * W * self.dim
        # 2. Linear层的FLOPs：(H/2)*(W/2) * 4*dim * 2*dim（假设out_dim=2*dim，矩阵乘法）
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchMerging4x(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        super(PatchMerging4x, self).__init__()
        H, W = input_resolution
        self.patch_merging1 = PatchMerging((H, W), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)
        self.patch_merging2 = PatchMerging((H // 2, W // 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)

    def forward(self, x, H=None, W=None):
        if H is None:
            H, W = self.input_resolution
        x = self.patch_merging1(x, H, W)
        x = self.patch_merging2(x, H//2, W//2)
        return x

class PatchReverseMerging(nn.Module):
    """ 
    Patch Reverse Merging Layer.
    核心作用：Patch Merging的逆操作，对输入特征进行上采样（空间尺寸翻倍），同时将通道数压缩
    类似于CNN中的转置卷积或上采样层，但通过PixelShuffle实现更高效的上采样
    """
    def __init__(self, input_resolution, dim, out_dim, norm_layer=nn.LayerNorm):
        """
        初始化函数
        Args:
            input_resolution (tuple[int]): 输入特征的空间分辨率 (H, W)
            dim (int): 输入特征的通道数 C
            out_dim (int): 输出特征的通道数
            norm_layer (nn.Module, optional): 归一化层，默认 nn.LayerNorm
        """
        super(PatchReverseMerging, self).__init__()
        self.input_resolution = input_resolution  # 保存输入分辨率 (H, W)
        self.dim = dim  # 输入通道数
        self.out_dim = out_dim  # 输出通道数
        
        # 定义通道扩展层：将输入通道dim映射到out_dim*4（为PixelShuffle做准备）
        # PixelShuffle(2)需要通道数是输出通道数的4倍（2^2）
        self.increment = nn.Linear(dim, out_dim * 4, bias=False)
        # 定义归一化层：对输入通道dim进行LayerNorm
        self.norm = norm_layer(dim)
        
        # 注释掉的是用转置卷积实现上采样的另一种方式（可选参考）
        # self.proj = nn.ConvTranspose2d(dim // 4, 3, 3, stride=1, padding=1)
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: 输入特征，形状为 (B, H*W, C)
               - B: batch size
               - H*W: 序列长度（Transformer中特征图被展平为序列）
               - C: 输入通道数
        Returns:
            x: 输出特征，形状为 (B, (2H) * (2W), out_dim)
               - 空间尺寸翻倍（H→2H, W→2W）
               - 通道数压缩为out_dim
        """
        H, W = self.input_resolution  # 获取输入的空间分辨率 H, W
        B, L, C = x.shape  # 获取输入形状：B=batch, L=H*W, C=通道数
        
        # 校验输入的合法性
        assert L == H * W, "input feature has wrong size"  # 确保序列长度等于H*W
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."  # 确保H和W是偶数，否则无法2×2上采样

        # 1. 对输入特征进行LayerNorm归一化
        x = self.norm(x)

        # 2. 通过线性层扩展通道数：dim → out_dim*4：(B, H*W, C) → (B, H*W, 4*out_dim)
        x = self.increment(x)

        # 3. 调整维度为CNN格式，为PixelShuffle做准备
        # 第一步：view恢复为空间特征图形式：(B, H*W, 4*out_dim) → (B, H, W, 4*out_dim)
        # 第二步：permute调整通道维度到前面：(B, H, W, 4*out_dim) → (B, 4*out_dim, H, W)
        # （PixelShuffle要求通道维度在第1维）
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        # 4. 使用PixelShuffle进行上采样
        # PixelShuffle(2)的作用：将通道数除以4（2^2），空间尺寸乘以2
        # 维度变化：(B, 4*out_dim, H, W) → (B, out_dim, 2H, 2W)
        x = nn.PixelShuffle(2)(x)

        # 注释掉的是用转置卷积实现的另一种方式（可选参考）
        # x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.norm(x)

        # 5. 恢复为Transformer的序列格式
        # 第一步：flatten(2)将空间维度展平：(B, out_dim, 2H, 2W) → (B, out_dim, 2H*2W)
        # 第二步：permute调整维度顺序：(B, out_dim, 2H*2W) → (B, 2H*2W, out_dim)
        x = x.flatten(2).permute(0, 2, 1)

        return x

    def extra_repr(self) -> str:
        """
        额外的字符串表示，用于打印模块信息时显示关键参数
        """
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        """
        计算该层的浮点运算量（FLOPs），用于模型复杂度分析
        Returns:
            flops: 该层的总浮点运算量
        """
        H, W = self.input_resolution
        # 1. PixelShuffle相关的FLOPs（估算）
        flops = H * 2 * W * 2 * self.dim // 4
        # 2. Linear层的FLOPs：H*W * dim * 4*out_dim（矩阵乘法）
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops
    
class PatchReverseMerging4x(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, use_conv=False):
        super(PatchReverseMerging4x, self).__init__()
        self.use_conv = use_conv
        self.input_resolution = input_resolution
        self.dim = dim
        H, W = input_resolution
        self.patch_reverse_merging1 = PatchReverseMerging((H, W), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)
        self.patch_reverse_merging2 = PatchReverseMerging((H * 2, W * 2), dim, norm_layer=nn.LayerNorm, use_conv=use_conv)

    def forward(self, x, H=None, W=None):
        if H is None:
            H, W = self.input_resolution
        x = self.patch_reverse_merging1(x, H, W)
        x = self.patch_reverse_merging2(x, H*2, W*2)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * 2 * W * 2 * self.dim // 4
        flops += (H * 2) * (W * 2) * self.dim // 4 * self.dim
        return flops
    
class PatchEmbed(nn.Module):
    """
    图像Patch嵌入模块
    核心作用：将原始图像分割成不重叠的patch，并将每个patch映射为一个特征向量
    是Vision Transformer（ViT/Swin）的第一步，将图像数据转换为Transformer可处理的序列格式
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        """
        初始化函数
        Args:
            img_size (int/tuple): 输入图像的尺寸，默认224（若为int则自动转为(224,224)）
            patch_size (int/tuple): 每个patch的尺寸，默认4（若为int则自动转为(4,4)）
            in_chans (int): 输入图像的通道数，默认3（RGB图像）
            embed_dim (int): 嵌入后的特征维度，默认96
            norm_layer (nn.Module, optional): 归一化层，默认None
        """
        super(PatchEmbed, self).__init__()
        # 将输入尺寸转为二元组（如224→(224,224)）
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        # 计算patch的分辨率：图像尺寸 / patch尺寸
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        
        # 保存关键参数
        self.img_size = img_size  # 输入图像尺寸 (H, W)
        self.patch_size = patch_size  # patch尺寸 (Ph, Pw)
        self.patches_resolution = patches_resolution  # patch的分辨率 (H/Ph, W/Pw)
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # 总patch数量

        self.in_chans = in_chans  # 输入图像通道数
        self.embed_dim = embed_dim  # 嵌入后的特征维度

        # 定义patch嵌入的卷积层：
        # - 卷积核大小=patch_size，步长=patch_size → 实现不重叠的patch分割
        # - 输出通道数=embed_dim → 将每个patch映射为一个embed_dim维的向量
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 定义归一化层（若指定）
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        Args:
            x: 输入图像，形状为 (B, C, H, W)
               - B: batch size
               - C: 输入通道数（如3）
               - H: 图像高度
               - W: 图像宽度
        Returns:
            x: 嵌入后的patch序列，形状为 (B, num_patches, embed_dim)
               - num_patches: 总patch数量 (H/Ph) * (W/Pw)
               - embed_dim: 嵌入后的特征维度
        """
        # B, C, H, W = x.shape
        # 注释掉的是输入尺寸校验（FIXME表示可能需要放宽约束）
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # 1. 通过卷积层进行patch嵌入
        # 维度变化：(B, C, H, W) → (B, embed_dim, H/Ph, W/Pw)
        # 2. flatten(2)：将空间维度展平
        # 维度变化：(B, embed_dim, H/Ph, W/Pw) → (B, embed_dim, num_patches)
        # 3. transpose(1,2)：调整维度顺序为Transformer的序列格式
        # 维度变化：(B, embed_dim, num_patches) → (B, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)

        # 若指定归一化层，则对嵌入后的特征进行归一化
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        """
        计算该层的浮点运算量（FLOPs），用于模型复杂度分析
        Returns:
            flops: 该层的总浮点运算量
        """
        Ho, Wo = self.patches_resolution  # patch的分辨率
        # 1. 卷积层的FLOPs：输出像素数 * 输出通道数 * 输入通道数 * 卷积核大小
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        # 2. 归一化层的FLOPs（若指定）：输出像素数 * 输出通道数
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
    
class SwinTransformerBlock(nn.Module):
    r"""
    Swin Transformer的基本模块
    核心特点：交替使用窗口多头自注意力（W-MSA）和移位窗口多头自注意力（SW-MSA）
    结构：Norm1 → W-MSA/SW-MSA → 残差连接 → Norm2 → MLP → 残差连接
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        """
        初始化函数
        Args:
            dim: 输入特征的通道数
            input_resolution: 输入特征的空间分辨率 (H, W)
            num_heads: 注意力头数
            window_size: 窗口大小，默认7
            shift_size: 窗口移位大小，默认0（0表示W-MSA，>0表示SW-MSA）
            mlp_ratio: MLP隐藏层维度与输入维度的比例，默认4.0
            qkv_bias: 是否为Q/K/V添加可学习偏置，默认True
            qk_scale: Q/K缩放因子，默认None
            act_layer: 激活函数，默认nn.GELU
            norm_layer: 归一化层，默认nn.LayerNorm
        """
        super(SwinTransformerBlock, self).__init__()
        # 保存关键参数
        self.dim = dim  # 输入通道数
        self.input_resolution = input_resolution  # 输入分辨率 (H, W)
        self.num_heads = num_heads  # 注意力头数
        self.window_size = window_size  # 窗口大小
        self.shift_size = shift_size  # 移位大小
        self.mlp_ratio = mlp_ratio  # MLP隐藏层比例
        
        # 特殊情况处理：如果输入分辨率小于等于窗口大小，则不需要窗口化
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0  # 移位大小设为0
            self.window_size = min(self.input_resolution)  # 窗口大小设为输入分辨率的最小值
        # 校验移位大小的合法性：必须在0到window_size之间
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # 1. 定义第一个归一化层（Norm1）
        self.norm1 = norm_layer(dim)
        # 2. 定义窗口注意力模块（W-MSA/SW-MSA）
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=0., proj_drop=0.)

        # 3. 定义第二个归一化层（Norm2）
        self.norm2 = norm_layer(dim)
        # 4. 定义MLP模块
        mlp_hidden_dim = int(dim * mlp_ratio)  # MLP隐藏层维度
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=0.)
        
        # 5. 生成注意力掩码（仅当shift_size>0时需要，用于SW-MSA）
        if self.shift_size > 0:
            attn_mask = cal_attention_mask(self.input_resolution, self.window_size, self.shift_size)
        else:
            attn_mask = None
        # 将掩码注册为buffer（不参与训练，但会随模型移动到device）
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        """
        Args:
            x: 输入特征，形状为 (B, H*W, C)
               - B: batch size
               - H*W: 序列长度
               - C: 输入通道数
        Returns:
            x: 输出特征，形状为 (B, H*W, C)
        """
        H, W = self.input_resolution  # 获取输入分辨率
        B, L, C = x.shape  # 获取输入形状
        # 校验输入的合法性：序列长度必须等于H*W
        assert L == H * W, "input feature has wrong size"

        # 保存残差连接的输入
        shortcut = x

        # --------------------------
        # 第一步：Norm1 + W-MSA/SW-MSA
        # --------------------------
        x = self.norm1(x)  # 归一化
        # 将序列形式恢复为空间特征图形式：(B, H*W, C) → (B, H, W, C)
        x = x.view(B, H, W, C)

        # 1. 循环移位（Cyclic Shift）：仅当shift_size>0时执行（SW-MSA）
        if self.shift_size > 0:
            # 沿H和W方向分别向左、向上移位shift_size个像素
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x  # 不移位（W-MSA）

        # 2. 划分窗口（Partition Windows）
        # window_partition将特征图划分为不重叠的窗口：(B, H, W, C) → (nW*B, ws, ws, C)
        x_windows = window_partition(shifted_x, self.window_size)
        # 将每个窗口展平为序列：(nW*B, ws, ws, C) → (nW*B, ws*ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        B_, N, C = x_windows.shape  # B_=nW*B, N=ws*ws

        # 3. 窗口注意力计算（W-MSA/SW-MSA）
        # 传入attn_mask（仅SW-MSA需要）
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # (nW*B, ws*ws, C)

        # 4. 合并窗口（Merge Windows）
        # 恢复窗口形状：(nW*B, ws*ws, C) → (nW*B, ws, ws, C)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # window_reverse将窗口合并回特征图：(nW*B, ws, ws, C) → (B, H, W, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # 5. 反向循环移位（Reverse Cyclic Shift）：仅当shift_size>0时执行
        if self.shift_size > 0:
            # 沿H和W方向分别向右、向下移位shift_size个像素，恢复原始位置
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        # 恢复为序列形式：(B, H, W, C) → (B, H*W, C)
        x = x.view(B, H * W, C)

        # 6. 第一个残差连接
        x = shortcut + x

        # --------------------------
        # 第二步：Norm2 + MLP + 残差连接
        # --------------------------
        # 保存第二个残差连接的输入（即第一个残差的输出）
        shortcut = x
        # 归一化 → MLP → 残差连接
        x = x + self.mlp(self.norm2(x))

        return x

    def flops(self):
        """
        计算该模块的浮点运算量（FLOPs）
        Returns:
            flops: 该模块的总浮点运算量
        """
        flops = 0
        H, W = self.input_resolution
        # 1. Norm1的FLOPs：H*W*dim
        flops += self.dim * H * W
        # 2. W-MSA/SW-MSA的FLOPs：nW * 单个窗口的FLOPs
        nW = H * W / self.window_size / self.window_size  # 窗口总数
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # 3. MLP的FLOPs：2*H*W*dim*(dim*mlp_ratio)（两个线性层）
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # 4. Norm2的FLOPs：H*W*dim
        flops += self.dim * H * W
        return flops
    
    def update_mask(self, device="cuda"):
        """
        更新注意力掩码并移动到指定device
        （用于动态调整device时重新生成掩码）
        Args:
            device: 目标device，默认"cuda"
        """
        if self.shift_size > 0:
            # 重新生成注意力掩码
            attn_mask = cal_attention_mask(self.input_resolution, self.window_size, self.shift_size)
            # 将掩码移动到指定device
            device = next(self.parameters()).device if device is None else device
            self.attn_mask = attn_mask.to(device)
        else:
            pass
