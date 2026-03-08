from SwinTransformerModel import *
from channel import *
import torch
from types import SimpleNamespace
from torch import nn
from random import choice

class BasicLayer_Encoder(nn.Module):
    r"""
    Swin Transformer的基本层（Stage）
    核心组成：可选的 Patch Merging 下采样层 + 多个交替使用 W-MSA/SW-MSA 的 SwinTransformerBlock
    作用：在该层内进行多尺度特征提取，同时可通过下采样降低空间分辨率、提升通道维度
    """
    
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                 downsample=None):
        """
        初始化函数
        Args:
            dim: 输入通道数（下采样前）
            out_dim: 输出通道数（下采样后，也是 Block 的输入/输出通道数）
            input_resolution: 输入分辨率 (H, W)（下采样前）
            depth: 该层包含的 SwinTransformerBlock 数量
            num_heads: 注意力头数（该层内所有 Block 共享）
            window_size: 窗口大小（该层内所有 Block 共享）
            mlp_ratio: MLP 隐藏层维度与输入维度的比例，默认 4.0
            qkv_bias: 是否为 Q/K/V 添加可学习偏置，默认 True
            qk_scale: Q/K 缩放因子，默认 None
            norm_layer: 归一化层，默认 nn.LayerNorm
            downsample: 下采样层类（如 PatchMerging），默认 None（表示该层不下采样）
        """
        super(BasicLayer_Encoder, self).__init__()
        self.dim = dim  # 输入通道数（下采样前）
        self.input_resolution = input_resolution  # 输入分辨率 (H, W)（下采样前）
        self.depth = depth  # 该层包含的 SwinTransformerBlock 数量

        # 构建多个 SwinTransformerBlock（交替使用 W-MSA 和 SW-MSA）
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=out_dim,  # Block 的输入通道数是 out_dim（因为先下采样再进 Block）
                input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2),  # Block 的输入分辨率是下采样后的（H/2, W/2）
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,  # 交替设置移位大小：偶数块用 W-MSA（shift=0），奇数块用 SW-MSA（shift=window_size//2）
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # 定义 Patch Merging 下采样层（可选）
        if downsample is not None:
            # 下采样层的输入分辨率是 input_resolution（H, W），输入通道是 dim，输出通道是 out_dim
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征，形状为 (B, H*W, C)
               - B: batch size
               - H*W: 序列长度（下采样前的空间分辨率）
               - C: 输入通道数（dim）
        Returns:
            x: 输出特征，形状为 (B, L, out_dim)
               - L: 输出序列长度
                 - 若有下采样：L = (H/2) * (W/2)
                 - 若无下采样：L = H * W
               - out_dim: 输出通道数
        """
        # 1. 先执行下采样（若有）
        if self.downsample is not None:
            # 下采样维度变化：(B, H*W, dim) → (B, (H/2)*(W/2), out_dim)
            x = self.downsample(x)
        
        # 2. 依次通过所有 SwinTransformerBlock
        for _, blk in enumerate(self.blocks):
            # Block 不改变特征维度：输入 (B, L, out_dim) → 输出 (B, L, out_dim)
            x = blk(x)
        
        return x

    def extra_repr(self) -> str:
        """
        额外的字符串表示，用于打印模块信息时显示关键参数
        Returns:
            包含 dim、input_resolution、depth 的字符串
        """
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        """
        计算该层的浮点运算量（FLOPs）
        Returns:
            flops: 该层的总浮点运算量（包含所有 Block 和下采样层）
        """
        flops = 0
        # 累加所有 SwinTransformerBlock 的 FLOPs
        for blk in self.blocks:
            flops += blk.flops()
        # 累加下采样层的 FLOPs（若有）
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def update_resolution(self, H, W):
        """
        动态更新输入分辨率（用于支持不同大小的输入图像）
        Args:
            H: 更新后的输入特征高度（下采样前的高度）
            W: 更新后的输入特征宽度（下采样前的宽度）
        """
        # 更新每个 SwinTransformerBlock 的输入分辨率和注意力掩码
        for _, blk in enumerate(self.blocks):
            # Block 的输入分辨率是下采样后的：(H/2, W/2)
            blk.input_resolution = (H, W)
            blk.update_mask()  # 重新生成 SW-MSA 的注意力掩码
        
        # 更新下采样层的输入分辨率（若有）
        if self.downsample is not None:
            # 下采样层的输入分辨率是下采样前的：(H*2, W*2)（因为当前 H/W 是下采样后的，下采样层的输入是原来的 2 倍）
            self.downsample.input_resolution = (H * 2, W * 2)

class BasicLayer_Decoder(nn.Module):
    """
    Swin Transformer 解码器的基本层（Stage）
    与编码器 BasicLayer 的核心区别：
    1. 使用 Patch Reverse Merging（上采样）代替 Patch Merging（下采样）
    2. 执行顺序不同：先通过 SwinTransformerBlock 提取特征，再进行上采样
       （编码器是先下采样，再通过 Block）
    """

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, upsample=None,):
        """
        初始化
        Args:
            dim: 输入特征的通道数（上采样前的通道数）
            out_dim: 输出特征的通道数（上采样后的通道数）
            input_resolution: 输入特征的空间分辨率 (H, W)（上采样前的分辨率）
            depth: 该层包含的 SwinTransformerBlock 的数量
            num_heads: 注意力头数（该层内所有 Block 共享）
            window_size: 窗口大小（该层内所有 Block 共享）
            mlp_ratio: MLP 隐藏层维度与输入维度的比例，默认 4.0
            qkv_bias: 是否为 Q/K/V 添加可学习偏置，默认 True
            qk_scale: Q/K 缩放因子，默认 None
            norm_layer: 归一化层，默认 nn.LayerNorm
            upsample: 上采样层类（如 PatchReverseMerging），默认 None（表示该层不上采样）
        """
        super().__init__()
        # 保存关键参数
        self.dim = dim  # 输入通道数（上采样前）
        self.input_resolution = input_resolution  # 输入分辨率 (H, W)（上采样前）
        self.depth = depth  # 该层包含的 SwinTransformerBlock 数量

        # 构建多个 SwinTransformerBlock（交替使用 W-MSA 和 SW-MSA）
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,  # Block 的输入通道数是 dim（因为先过 Block 再上采样）
                input_resolution=input_resolution,  # Block 的输入分辨率是上采样前的 (H, W)
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,  # 交替设置移位大小
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # 定义 Patch Reverse Merging 上采样层（可选）
        if upsample is not None:
            # 上采样层的输入分辨率是 input_resolution（H, W），输入通道是 dim，输出通道是 out_dim
            self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征，形状为 (B, H*W, C)
               - B: batch size
               - H*W: 序列长度（上采样前的空间分辨率）
               - C: 输入通道数（dim）
        Returns:
            x: 输出特征，形状为 (B, L, out_dim)
               - L: 输出序列长度
                 - 若有上采样：L = (2H)*(2W)
                 - 若无下采样：L = H*W
               - out_dim: 输出通道数
        """
        # 1. 先依次通过所有 SwinTransformerBlock
        for _, blk in enumerate(self.blocks):
            # Block 不改变特征维度：输入 (B, H*W, dim) → 输出 (B, H*W, dim)
            x = blk(x)

        # 2. 再执行上采样（若有）
        if self.upsample is not None:
            # 上采样维度变化：(B, H*W, dim) → (B, (2H)*(2W), out_dim)
            x = self.upsample(x)
        
        return x

    def extra_repr(self) -> str:
        """
        额外的字符串表示，用于打印模块信息时显示关键参数
        Returns:
            包含 dim、input_resolution、depth 的字符串
        """
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        """
        计算该层的浮点运算量（FLOPs）
        Returns:
            flops: 该层的总浮点运算量（包含所有 Block 和上采样层）
        """
        flops = 0
        # 累加所有 SwinTransformerBlock 的 FLOPs
        for blk in self.blocks:
            flops += blk.flops()
            print("blk.flops()", blk.flops())  # 打印每个 Block 的 FLOPs（调试用）
        # 累加上采样层的 FLOPs（若有）
        if self.upsample is not None:
            flops += self.upsample.flops()
            print("upsample.flops()", self.upsample.flops())  # 打印上采样层的 FLOPs（调试用）
        return flops

    def update_resolution(self, H, W):
        """
        动态更新输入分辨率（用于支持不同大小的输入特征）
        Args:
            H: 更新后的输入特征高度（上采样前的高度）
            W: 更新后的输入特征宽度（上采样前的宽度）
        """
        # 更新当前层的输入分辨率
        self.input_resolution = (H, W)
        # 更新每个 SwinTransformerBlock 的输入分辨率和注意力掩码
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H, W)
            blk.update_mask()  # 重新生成 SW-MSA 的注意力掩码
        # 更新上采样层的输入分辨率（若有）
        if self.upsample is not None:
            # 上采样层的输入分辨率就是当前层的输入分辨率 (H, W)
            self.upsample.input_resolution = (H, W)

class AdaptiveModulator(nn.Module):
    """
    自适应调制器（MLP）
    输入：SNR或Rate（标量）
    输出：调制系数（用于调整特征权重）
    """

    def __init__(self, M):
        """
        初始化
        Args:
            M: 调制网络的隐藏层维度
        """
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid()  # 输出范围[0,1]
        )

    def forward(self, snr):
        return self.fc(snr)
    

class ChannelModulator(nn.Module):
    """
    自适应调制器，包含多层特征变换和调制器，用于根据SNR或Rate生成调制系数，调整特征权重
    """

    def __init__(self, embed_dim, hidden_dim, layer_num):
        """
        初始化：
        - 创建自适应调制器列表（bm_list）和特征变换层列表（sm_list）
        - 第一层特征变换层将嵌入维度转换为隐藏维度
        - 后续层根据需要选择隐藏维度或嵌入维度作为输出维度
        - 使用Sigmoid函数将调制系数归一化到[0,1]
        
        Args:
            embed_dim: 输入特征的维度（如SwinJSCC最后一层的嵌入维度）
            hidden_dim: 调制网络的隐藏层维度（建议设置为embed_dim的1.5倍）
            layer_num: 调制网络的层数（建议设置为7层，经验值）
        """
        super(ChannelModulator, self).__init__()
        self.embed_dim = embed_dim
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.bm_list = nn.ModuleList() # 自适应调制器列表
        self.sm_list = nn.ModuleList() # 特征变换层列表
        self.sm_list.append(nn.Linear(embed_dim, hidden_dim)) # 第一层：嵌入维度→隐藏维度
        for i in range(layer_num):
            outdim = embed_dim if i == layer_num - 1 else hidden_dim
            self.bm_list.append(AdaptiveModulator(hidden_dim)) # 
            self.sm_list.append(nn.Linear(hidden_dim, outdim)) # 特征变换层
        self.sigmoid = nn.Sigmoid() # 最终调制系数归一化到[0,1]
    
    def forward(self, x, snr_or_rate, mode):
        """
        前向传播：
        - 根据输入的SNR或Rate生成调制系数
        - 应用调制系数调整特征权重
        - 返回调整后的特征和速率掩码（若模式为RA）

        Args:
            x: 输入特征，形状为 (B, C, H, W)
               - B: batch size
               - C: 输入通道数（dim）
               - H, W: 输入特征的空间分辨率
            snr_or_rate: 信噪比或传输速率（标量）
            mode: 模式选择，'SA'表示SNR自适应，'RA'表示速率自适应
        """
        B = x.size()[0]
        device = x.device

        snr_or_rate_cuda = torch.as_tensor(snr_or_rate, dtype=torch.float, device=device) # 将输入转换为CUDA张量
        snr_or_rate_batch = snr_or_rate_cuda.unsqueeze(0).expand(B, -1) # 扩展为batch维度：(B, 1)
        temp = x.detach() # 从计算图中分离，避免梯度回传到特征提取部分
        for i in range(self.layer_num):
            temp = self.sm_list[i](temp) # 特征变换
            bm = self.bm_list[i](snr_or_rate_batch).unsqueeze(1).expand(-1, temp.size(1), -1) # 生成调制系数并扩展到特征维度
            temp = temp * bm # 应用调制系数
        mod_val = self.sigmoid(self.sm_list[-1](temp)) # 最终调制系数归一化到[0,1]
        x = x * mod_val # 调整原始特征

        if mode == 'SA':
            return x, None # SA模式不返回掩码
        elif mode == 'RA':
            # 生成速率掩码：选择重要性最高的rate个通道
            mask = torch.sum(mod_val, dim=1) # 对每个通道的重要性求和
            _, indices = mask.sort(dim=1, descending=True) # 按重要性降序排序
            rate = int(snr_or_rate) # 将输入转换为整数速率
            c_indices = indices[:, :rate] # 选择前rate个重要的通道索引
            
            # 构建全局索引（处理batch维度）
            add = torch.arange(0, B * x.size()[2], x.size()[2], device=device).unsqueeze(1).repeat(1, rate)
            c_indices = c_indices + add.int().to(device)
            
            # 生成掩码：重要通道为1，其余为0
            mask = torch.zeros(mask.size()).reshape(-1).to(device)
            mask[c_indices.reshape(-1)] = 1
            mask = mask.reshape(B, x.size()[2])
            mask = mask.unsqueeze(1).expand(-1, temp.size(1), -1) # 扩展到特征维度
            
            x = x * mask # 应用掩码，只保留重要通道
            return x, mask # RA模式返回掩码
        else:
            raise ValueError("Invalid mode for ChannelModulator. Use 'SA' or 'RA'.")


class SwinJSCC_Encoder(nn.Module):
    """
    SwinJSCC编码器（语义通信编码器）
    核心功能：将图像编码为适合信道传输的特征，并支持根据信噪比（SNR）和传输速率（Rate）自适应调整
    支持4种模式：
    1. SwinJSCC_w/o_SAandRA：无自适应（Baseline）
    2. SwinJSCC_w/_SA：仅SNR自适应（SNR Adaptation, SA）
    3. SwinJSCC_w/_RA：仅速率自适应（Rate Adaptation, RA）
    4. SwinJSCC_w/_SAandRA：同时支持SA和RA
    """

    def __init__(self, model, img_size, patch_size, in_chans,
                 embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 bottleneck_dim=16):
        """
        初始化SwinJSCC编码器
        Args:
            model: 模型模式字符串，用于选择是否启用SA/RA
                - 'SwinJSCC_w/o_SAandRA'：无自适应（Baseline）
                - 'SwinJSCC_w/_SA'：仅SNR自适应（SNR Adaptation, SA）
                - 'SwinJSCC_w/_RA'：仅速率自适应（Rate Adaptation, RA）
                - 'SwinJSCC_w/_SAandRA'：同时支持SA和RA
            img_size: 输入图像尺寸 (H, W)
            patch_size: Patch大小（注：PatchEmbed中默认为2）
            in_chans: 输入图像通道数（如3为RGB）
            embed_dims: 每一层的嵌入维度列表（如[96, 192, 384]）
            depths: 每一层包含的SwinTransformerBlock数量列表（如[2, 2, 6]）
            num_heads: 每一层的注意力头数列表（如[3, 6, 12]）
            C: 最终传输的特征维度（若为None则不使用head_list）
            window_size: 窗口大小，默认4
            mlp_ratio: MLP隐藏层维度与输入维度的比例，默认4.0
            qkv_bias: 是否为Q/K/V添加可学习偏置，默认True
            qk_scale: Q/K缩放因子，默认None
            norm_layer: 归一化层，默认nn.LayerNorm
            patch_norm: 是否对Patch Embedding进行归一化（注：代码中实际未使用）
            bottleneck_dim: 瓶颈维度（注：代码中实际未使用）
        """
        super().__init__()
        # 保存基础参数
        self.num_layers = len(depths)  # 总层数（Stage数量）
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patches_resolution = img_size
        
        # 计算最终特征图的分辨率（经过num_layers次下采样，每次下采样空间尺寸减半）
        self.H = img_size[0] // (2 ** self.num_layers)
        self.W = img_size[1] // (2 ** self.num_layers)

        # --------------------------
        # 1. Patch Embedding（将图像分割为patch并嵌入）
        # --------------------------
        # 注：这里PatchEmbed的参数硬编码为patch_size=2, in_chans=3，忽略了传入的patch_size和in_chans
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        
        # 调制网络的隐藏层维度（最后一层嵌入维度的1.5倍）
        self.hidden_dim = int(self.embed_dims[len(embed_dims)-1] * 1.5)
        self.layer_num = layer_num = 7  # 调制网络的层数

        # --------------------------
        # 2. 构建多个BasicLayer（Swin Transformer的Stage）
        # --------------------------
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer_Encoder(
                dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else self.in_chans,  # 第一层输入通道为3（RGB），后续为上一层的输出维度
                out_dim=int(embed_dims[i_layer]),  # 当前层的输出通道数
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),  # 当前层的输入分辨率（第i层经过i次下采样）
                                  self.patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],  # 当前层包含的Block数量
                num_heads=num_heads[i_layer],  # 当前层的注意力头数
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer != 0 else None  # 第一层不下采样，后续层使用Patch Merging下采样
            )
            print("Encoder ", layer.extra_repr())
            self.layers.append(layer)

        # --------------------------
        # 3. 最终归一化层和投影层
        # --------------------------
        self.norm = norm_layer(embed_dims[-1])  # 对最后一层的输出进行归一化
        if C != None:
            self.head_list = nn.Linear(embed_dims[-1], C)  # 将特征映射到最终传输维度C

        # --------------------------
        # 4. 初始化权重
        # --------------------------
        self.apply(self._init_weights)

        # --------------------------
        # 5. 构建SNR自适应（SA）和速率自适应（RA）的调制网络
        # --------------------------
        if model == 'SwinJSCC_w/_SAandRA':
            # --- SA调制网络（用于SNR自适应）---
            self.channel_modnet = ChannelModulator(self.embed_dims[len(embed_dims) - 1], self.hidden_dim, self.layer_num)           

            # --- RA调制网络（用于速率自适应，结构同SA）---
            self.channel_modnet1 = ChannelModulator(self.embed_dims[len(embed_dims) - 1], self.hidden_dim, self.layer_num)
        else:
            # 仅SA或仅RA的情况（共享一套调制网络）
            self.channel_modnet = ChannelModulator(self.embed_dims[len(embed_dims) - 1], self.hidden_dim, self.layer_num)

    def forward(self, x, snr, rate, model):
        """
        前向传播
        Args:
            x: 输入图像，形状为 (B, C, H, W)
               - B: batch size
               - C: 输入通道数（如3）
               - H: 图像高度
               - W: 图像宽度
            snr: 信噪比（标量，如10.0）
            rate: 传输速率（标量，如32）
            model: 模型模式字符串
        Returns:
            x: 编码后的特征，形状为 (B, L, C_trans)
               - L: 序列长度（最终特征图的空间尺寸）
               - C_trans: 传输维度
            mask: 速率掩码（仅RA模式返回，形状同x，用于指示哪些通道被保留）
        """
        # B, C, H, W = x.size()
        # device = x.get_device()  # 获取当前设备（cuda/cpu）

        # --------------------------
        # 第一步：Patch Embedding
        # --------------------------
        # 维度变化：(B, 3, H, W) → (B, H/2*W/2, embed_dims[0])
        x = self.patch_embed(x)

        # --------------------------
        # 第二步：通过多个BasicLayer提取多尺度特征
        # --------------------------
        for i_layer, layer in enumerate(self.layers):
            x = layer(x)  # 每个Layer可能包含下采样，空间尺寸减半，通道数翻倍
        x = self.norm(x)  # 最终归一化：(B, L, embed_dims[-1])，L=(H/(2^num_layers))*(W/(2^num_layers))

        # --------------------------
        # 第三步：根据模型模式进行自适应处理
        # --------------------------
        if model == 'SwinJSCC_w/o_SAandRA':
            # 模式1：无自适应，直接投影到传输维度
            x = self.head_list(x)
            return x, None

        elif model == 'SwinJSCC_w/_SA':
            # 模式2：仅SNR自适应（SA）
            # 核心逻辑：根据SNR生成调制系数，调整特征权重（低SNR时抑制不重要特征）
            x = self.channel_modnet(x, snr, mode='SA')[0]  # 获取SNR调制后的特征（不返回掩码）
            x = self.head_list(x)  # 投影到传输维度
            return x, None

        elif model == 'SwinJSCC_w/_RA':
            # 模式3：仅速率自适应（RA）
            # 核心逻辑：根据Rate生成重要性分数，选择最重要的rate个通道进行传输
            return self.channel_modnet(x, rate, mode='RA')  # 获取速率调制后的特征和掩码

        elif model == 'SwinJSCC_w/_SAandRA':
            # 模式4：同时SA和RA（先SNR调整，再速率选择）
            # --- 1. SNR自适应（使用sm_list1和bm_list1）---
            x = self.channel_modnet1(x, snr, mode='SA')[0]  # 获取SNR调制后的特征（不返回掩码）
            # --- 2. 速率自适应（使用sm_list和bm_list）---
            return self.channel_modnet(x, rate, mode='RA')  # 获取速率调制后的特征和掩码
        else:
            raise ValueError(f'Invalid model type: {model}')

    def _init_weights(self, m):
        """
        初始化模型权重
        - Linear层：权重截断正态分布，偏置置0
        - LayerNorm层：权重置1，偏置置0
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        指定不需要权重衰减的参数（如绝对位置编码）
        注：代码中未使用绝对位置编码，此处保留接口
        """
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """
        指定不需要权重衰减的参数关键词（如相对位置偏置表）
        """
        return {'relative_position_bias_table'}

    def flops(self):
        """
        计算模型的总浮点运算量（FLOPs）
        Returns:
            flops: 总FLOPs
        """
        flops = 0
        flops += self.patch_embed.flops()  # Patch Embedding的FLOPs
        for i, layer in enumerate(self.layers):
            flops += layer.flops()  # 所有BasicLayer的FLOPs
        # 注：这里的计算可能不完整，缺少head_list和调制网络的FLOPs
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        return flops

    def update_resolution(self, H, W):
        """
        动态更新输入分辨率（用于支持不同大小的输入图像）
        Args:
            H: 新的输入图像高度
            W: 新的输入图像宽度
        """
        self.input_resolution = (H, W)
        # 更新每个BasicLayer的分辨率（第i层的分辨率是H/(2^(i+1))）
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H // (2 ** (i_layer + 1)),
                                    W // (2 ** (i_layer + 1)))

class SwinJSCC_Decoder(nn.Module):
    """
    SwinJSCC解码器（语义通信解码器）
    核心功能：将接收到的特征解码为原始图像，并支持根据信噪比（SNR）自适应调整
    与编码器对称的设计：
    1. 通道维度递减（如 [384, 192, 96]）
    2. 空间分辨率递增（通过 PatchReverseMerging 上采样）
    3. 支持4种模式（RA模式在编码器已完成掩码选择，解码器仅需SA）
    """
    def __init__(self, model, img_size, embed_dims, depths, num_heads, C, out_chans=3, 
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 bottleneck_dim=16):
        """
        初始化SwinJSCC解码器
        Args:
            model: 模型模式字符串，用于选择是否启用SA
            img_size: 输出图像尺寸 (H, W)
            embed_dims: 每一层的嵌入维度列表（解码器顺序与编码器相反，如 [384, 192, 96]）
            depths: 每一层包含的SwinTransformerBlock数量列表（与编码器一致）
            num_heads: 每一层的注意力头数列表（解码器顺序与编码器相反）
            C: 接收到的传输特征维度（与编码器的C一致）
            out_chans: 输出通道数，默认3（RGB图像）
            window_size: 窗口大小，默认4
            mlp_ratio: MLP隐藏层维度与输入维度的比例，默认4.0
            qkv_bias: 是否为Q/K/V添加可学习偏置，默认True
            qk_scale: Q/K缩放因子，默认None
            norm_layer: 归一化层，默认nn.LayerNorm
            ape: 是否使用绝对位置编码，默认False，废弃接口
            patch_norm: 是否对Patch Embedding进行归一化（注：解码器未使用Patch Embedding，保留接口）
            bottleneck_dim: 瓶颈维度（注：代码中实际未使用）
        """
        super().__init__()

        # 保存基础参数
        self.num_layers = len(depths)  # 总层数（Stage数量）
        self.ape = ape
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.H = img_size[0]  # 最终输出图像高度
        self.W = img_size[1]  # 最终输出图像宽度
        self.out_chans = out_chans
        
        # 计算初始特征图的分辨率（编码器的最终分辨率，解码器的初始分辨率）
        self.patches_resolution = (
            img_size[0] // (2 ** self.num_layers),
            img_size[1] // (2 ** self.num_layers)
        )
        
        # 绝对位置编码（可选，与编码器对应）
        """
        num_patches = self.H // 4 * self.W // 4
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        """

        # --------------------------
        # 1. 构建多个BasicLayer_Decoder（Swin Transformer解码器的Stage）
        # --------------------------
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 确定当前层的输入/输出通道
            dim = int(embed_dims[i_layer])
            out_dim = int(embed_dims[i_layer + 1]) if (i_layer < self.num_layers - 1) else self.out_chans  # 最后一层输出3通道（RGB）
            
            # 确定当前层的输入分辨率（初始分辨率 * 2^i_layer）
            input_resolution = (
                self.patches_resolution[0] * (2 ** i_layer),
                self.patches_resolution[1] * (2 ** i_layer)
            )
            
            layer = BasicLayer_Decoder(
                dim=dim,
                out_dim=out_dim,
                input_resolution=input_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                upsample=PatchReverseMerging
            )
            self.layers.append(layer)
            print("Decoder ", layer.extra_repr())

        # --------------------------
        # 2. 初始投影层（将传输维度C映射回解码器初始嵌入维度）
        # --------------------------
        if C is not None:
            self.head_list = nn.Linear(C, embed_dims[0])
        else:
            self.head_list = None

        # --------------------------
        # 3. 初始化权重
        # --------------------------
        self.apply(self._init_weights)

        # --------------------------
        # 4. 构建SNR自适应（SA）调制网络（使用封装好的ChannelModulator）
        # --------------------------
        self.hidden_dim = int(self.embed_dims[0] * 1.5)  # 调制网络隐藏层维度（初始嵌入维度的1.5倍）
        self.layer_num = layer_num = 7  # 调制网络层数（经验值）
        
        if model != "SwinJSCC_w/_RA":
            # 仅非RA模式构建SA调制网络（RA模式的掩码在编码器已应用）
            self.channel_modnet = ChannelModulator(
                embed_dim=self.embed_dims[0],
                hidden_dim=self.hidden_dim,
                layer_num=self.layer_num
            )
        else:
            self.channel_modnet = None

    def forward(self, x, snr, model):
        """
        前向传播
        Args:
            x: 接收到的传输特征，形状为 (B, L, C)
               - B: batch size
               - L: 序列长度（初始特征图的空间尺寸）
               - C: 传输维度
            snr: 信噪比（标量，如10.0）
            model: 模型模式字符串
        Returns:
            x: 重建的图像，形状为 (B, 3, H, W)
               - 3: RGB通道数
               - H, W: 输出图像尺寸
        """
        # --------------------------
        # 模式1：无自适应（Baseline）
        # --------------------------
        if model == 'SwinJSCC_w/o_SAandRA':
            # 1. 初始投影：传输维度C → 解码器初始嵌入维度
            x = self.head_list(x)

        # --------------------------
        # 模式2：仅SNR自适应（SA）
        # --------------------------
        elif model == 'SwinJSCC_w/_SA':
            # 1. 初始投影：传输维度C → 解码器初始嵌入维度
            x = self.head_list(x)
            
            # 2. 使用封装好的ChannelModulator进行SA调制
            # 注意：RA模式在编码器已完成，这里仅做SA
            x, _ = self.channel_modnet(x, snr, mode='SA')

        # --------------------------
        # 模式3：仅速率自适应（RA），不需要任何前置操作
        # --------------------------

        # --------------------------
        # 模式4：同时SA和RA
        # --------------------------
        elif model == 'SwinJSCC_w/_SAandRA':
            x, _ = self.channel_modnet(x, snr, mode='SA')
            
        # 通过所有解码器层（上采样+特征提取）
        for i_layer, layer in enumerate(self.layers):
            x = layer(x)
            
        B, L, N = x.shape
        x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
        x = torch.clamp(x, min=0.0, max=1.0)
        return x

    def _init_weights(self, m):
        """
        初始化模型权重（与编码器一致）
        - Linear层：权重截断正态分布，偏置置0
        - LayerNorm层：权重置1，偏置置0
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        指定不需要权重衰减的参数（与编码器一致）
        """
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """
        指定不需要权重衰减的参数关键词（与编码器一致）
        """
        return {'relative_position_bias_table'}

    def flops(self):
        """
        计算模型的总浮点运算量（FLOPs）
        Returns:
            flops: 总FLOPs
        """
        flops = 0
        # 累加所有BasicLayer的FLOPs
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        # 注：这里的计算可能不完整，缺少head_list和调制网络的FLOPs
        return flops

    def update_resolution(self, H, W):
        """
        动态更新输出分辨率（用于支持不同大小的输出图像）
        Args:
            H: 新的输出图像高度
            W: 新的输出图像宽度
        """
        self.input_resolution = (H, W)
        self.H = H * (2 ** len(self.layers))  # 修复原代码的运算符优先级问题（2**len先算）
        self.W = W * (2 ** len(self.layers))
        # 更新每个BasicLayer的分辨率（第i层的分辨率是H/(2^(num_layers - i))）
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(
                H * (2 ** i_layer),
                W * (2 ** i_layer)
            )

class SwinJSCC(nn.Module):
    def __init__(self, config: SimpleNamespace):
        """
        初始化SwinJSCC模型
        Args:
            config: 配置参数
                - model: 模型模式（'SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_SAandRA'）
                - img_size: 输入图像尺寸
                - patch_size: 初始图像块尺寸
                - in_chans: 输入通道数
                - embed_dims: 嵌入维度列表
                - depths: 每个阶段的层数列表
                - num_heads: 每个阶段的注意力头数列表
                - C: 传输维度（偶数）
                - multiple_snrs: 多信噪比列表（True/False）
                - pass_channel: 是否传递信道信息（True/False）
                - channel_numbers: 传输维度列表，用于速率控制
                - device: 设备（'cuda'/'cpu'）
                - window_size: 窗口大小，默认为4
                - mlp_ratio: MLP比例，默认为4.0
                - qkv_bias: QKV偏置，默认为True
                - qk_scale: QK缩放，默认为None
                - norm_layer: 归一化层，默认为nn.LayerNorm
                - patch_norm: 图像块归一化，默认为True（未使用）
                - bottleneck_dim: 瓶颈维度，默认为16（不使用）
                - channel_type: 信道类型 ('none'/'awgn'/'rayleigh' 或 0/1/2)
        """
        super().__init__()
        self.config = config
        self.model_mode = config.model
        self.pass_channel = config.pass_channel
        self.downsample = len(config.depths)
        self.C = config.C
        # 强制编码器输出维度为偶数（核心约束）
        if self.C is not None:
            assert self.C % 2 == 0, "特征维度C必须为偶数，才能拆分为实部+虚部"
        self.channel_numbers = config.channel_numbers
        self.multiple_snrs = config.multiple_snrs
        self.img_size = config.img_size
        self.device = config.device if hasattr(config, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 编码器/解码器/信道初始化逻辑不变（仅需保证编码器输出维度为C）
        self.encoder = SwinJSCC_Encoder(
            model=self.model_mode,
            img_size=config.img_size,
            patch_size=getattr(config, 'patch_size', 2),
            in_chans=getattr(config, 'in_chans', 3),
            embed_dims=config.embed_dims,
            depths=config.depths,
            num_heads=config.num_heads,
            C=self.C,  # 输出维度为偶数C
            window_size=getattr(config, 'window_size', 4),
            mlp_ratio=getattr(config, 'mlp_ratio', 4.0),
            qkv_bias=getattr(config, 'qkv_bias', True),
            qk_scale=getattr(config, 'qk_scale', None),
            norm_layer=getattr(config, 'norm_layer', nn.LayerNorm),
            patch_norm=getattr(config, 'patch_norm', True),
            bottleneck_dim=getattr(config, 'bottleneck_dim', 16)
        )
        self.decoder = SwinJSCC_Decoder(
            model=self.model_mode,
            img_size=config.img_size,
            embed_dims=config.embed_dims[::-1],
            depths=config.depths,
            num_heads=config.num_heads[::-1],
            C=self.C,  # 解码器输入维度为C（合并实/虚部后）
            out_chans = getattr(config, 'in_chans', 3), # 输出通道与输入通道相等。
            window_size=getattr(config, 'window_size', 4),
            mlp_ratio=getattr(config, 'mlp_ratio', 4.0),
            qkv_bias=getattr(config, 'qkv_bias', True),
            qk_scale=getattr(config, 'qk_scale', None),
            norm_layer=getattr(config, 'norm_layer', nn.LayerNorm),
            patch_norm=getattr(config, 'patch_norm', True),
            bottleneck_dim=getattr(config, 'bottleneck_dim', 16)
        )
        self.channel = Channel1(config)

    def forward(self, x, snr=None, rate=None):
        snr = snr if snr is not None else choice(self.multiple_snrs)
        rate = rate if rate is not None else choice(self.channel_numbers)
        
        # ========== 1. 编码器前向 ==========
        enc_output, mask = self.encoder(x, snr, rate, self.model_mode)

        # ========== 2. 信道前向 ==========
        if self.model_mode == 'SwinJSCC_w/o_SAandRA' or self.model_mode == 'SwinJSCC_w/_SA':
            CBR = enc_output.numel() / 2 / x.numel()
            if self.pass_channel:
                channel_output = self.channel.forward(enc_output, snr)
            else:
                channel_output = enc_output
        elif self.model_mode == 'SwinJSCC_w/_RA' or self.model_mode == 'SwinJSCC_w/_SAandRA':
            CBR = rate / (2 * 3 * 4 ** (self.downsample))
            avg_pwr = torch.sum(enc_output ** 2) / mask.sum()
            if self.pass_channel:
                channel_output = self.channel.forward(enc_output, snr, True, avg_pwr)
            else:
                channel_output = enc_output
            channel_output *= mask
        #========== 3. 解码器前向 ==========
        recon_image = self.decoder.forward(channel_output, snr, self.model_mode)
        return recon_image, CBR
        

    # 其余方法（flops/update_resolution/extra_repr）保持不变
    def flops(self):
        encoder_flops = self.encoder.flops()
        decoder_flops = self.decoder.flops()
        return encoder_flops + decoder_flops

    def update_resolution(self, H, W):
        self.encoder.update_resolution(H, W)
        self.decoder.update_resolution(
            H // (2 ** self.encoder.num_layers),
            W // (2 ** self.encoder.num_layers)
        )
        self.config.img_size = (H, W)

    def extra_repr(self):
        return (
            f"model_mode={self.model_mode}, "
            f"img_size={self.img_size}, "
            f"trans_dim={self.C}, "
            f"channel_type={self.config.channel_type}"
        )
    
