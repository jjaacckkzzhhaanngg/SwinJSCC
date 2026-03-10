from types import SimpleNamespace
import torch.nn as nn
import random
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
import torch.nn.functional as F
import torchvision.models as models

# 假设你的模型文件在当前目录
from SwinJSCCModel import SwinJSCC
from distortion import Distortion

import matplotlib.pyplot as plt

# 解决matplotlib中文显示/服务器无GUI问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False
import matplotlib
# 服务器环境自动切换非交互式后端
if os.environ.get('DISPLAY', '') == '' and matplotlib.get_backend() != 'Agg':
    matplotlib.use('Agg')

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        # 加载预训练的 VGG16 的前几层特征提取块
        blocks.append(models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:4].eval())
        blocks.append(models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[4:9].eval())
        blocks.append(models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[9:16].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False # 冻结 VGG 参数，不参与训练
        self.blocks = nn.ModuleList(blocks)
        self.transform = F.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        # 将输入规范化到 ImageNet 分布
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            # 在特征图层级使用 L1 误差，强化纹理相似度
            loss += F.l1_loss(x, y)
        return loss

def get_config():
    """
    SwinJSCC 模型的完整配置参数
    针对 Tiny-ImageNet 64×64 数据集优化
    """
    config = SimpleNamespace(
        # ==================== 模型架构参数 ====================
        model='SwinJSCC_w/_SAandRA',  # 模型模式：'SwinJSCC_w/o_SAandRA', 'SwinJSCC_w/_SA', 'SwinJSCC_w/_RA', 'SwinJSCC_w/_SAandRA'
        
        # 输入图像参数
        img_size=(256, 256),           # 输入图像尺寸 (H, W)
        patch_size=2,                # Patch 大小（必须能整除 img_size）
        in_chans=3,                  # 输入通道数（RGB=3）
        
        # Swin Transformer 层级配置
        # 64×64 → patch_embed → 32×32 → stage1(下采样) → 16×16 → stage2(下采样) → 8×8
        embed_dims=[96, 192, 256, 320],   # 每个 stage 的嵌入维度（逐层增加）
        depths=[2, 2, 6, 2],            # 每个 stage 的 Transformer Block 数量
        num_heads=[4, 6, 8, 10],        # 每个 stage 的注意力头数
        
        # 传输参数
        C=192,                       # 传输维度，对应CBR为0.125（必须是偶数，用于实部/虚部分离）
        
        # 窗口和 MLP 参数
        window_size=8,               # 窗口大小（必须能整除各层特征图尺寸：32, 16, 8）
        mlp_ratio=4.0,               # MLP 隐藏层维度 = mlp_ratio × embed_dim
        qkv_bias=True,               # Q/K/V 是否使用偏置
        qk_scale=None,               # Q/K 缩放因子（None 则自动计算）
        
        # 归一化和正则化
        norm_layer=nn.LayerNorm,     # 归一化层类型
        patch_norm=True,             # Patch Embedding 后是否归一化
        bottleneck_dim=16,           # 瓶颈维度（当前代码未使用）
        
        # ==================== 信道参数 ====================
        channel_type='awgn',         # 信道类型：'none', 'awgn', 'rayleigh'
        pass_channel=True,           # 是否通过信道（False 则跳过信道模拟）
        
        # SNR 自适应参数
        multiple_snrs=[1, 4, 7, 10, 13],  # 训练时的 SNR 列表（dB）
        
        # 速率自适应参数
        channel_numbers=[32, 64, 96, 128, 192],    # 可选的传输维度（用于速率控制）
        
        # ==================== 训练参数 ====================
        # 数据集
        trainset='mini-imagenet',     # 数据集名称
        train_data_dir='./data/mini-imagenet/train',  # 训练集路径
        val_data_dir='./data/mini-imagenet/val',      # 验证集路径
        
        # 损失函数
        distortion_metric='MS-SSIM',      # 失真度量：'MSE', 'MS-SSIM'
        ms_ssim_levels=5,                 # MS-SSIM 层数（针对 64×64 使用 3 层）
        ms_ssim_window_size=11, # 7,            # 减小窗口尺寸到 11(针对 64×64 使用 7)
        ms_ssim_weights=None,  # [0.4, 0.3, 0.3],  # MS-SSIM 权重（针对 64×64 使用 [0.4, 0.3, 0.3]）
        
        # 优化器
        optimizer='AdamW',                # 优化器类型
        learning_rate=1e-4,               # 初始学习率
        weight_decay=0.01,                # 权重衰减
        betas=(0.9, 0.999),               # Adam 的 beta 参数
        
        # 学习率调度
        scheduler='CosineAnnealingLR',    # 学习率调度器：'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'
        lr_step_size=30,                  # StepLR 的步长
        lr_gamma=0.1,                     # StepLR 的衰减率
        T_max=100,                        # CosineAnnealingLR 的周期
        
        # 训练超参数
        batch_size=16,                    # 批次大小
        num_epochs=80,                   # 训练轮数
        num_workers=4,                    # DataLoader 的工作进程数
        alpha=0.84,                        # 损失函数超参数（用于平衡 L2-loss 和 MS-SSIM）
        
        # 梯度和数值稳定性
        grad_clip=1.0,                    # 梯度裁剪阈值（None 则不裁剪）
        amp=False,                         # 是否使用混合精度训练
        
        # ==================== 日志和保存 ====================
        # 设备
        device='cuda',                    # 训练设备：'cuda', 'cpu'
        
        # 日志
        logger=None,                      # 日志记录器（可选）
        log_interval=10,                  # 每隔多少 batch 打印一次日志
        
        # 模型保存
        model_name = "SwinJSCC_SAandRA_B3_", # 保存的模型名称
        save_dir='./checkpoints',         # 模型保存目录
        save_interval=10,                 # 每隔多少 epoch 保存一次
        resume=None,                      # 恢复训练的检查点路径（None 则从头训练）
        
        # 验证
        val_interval=5,                   # 每隔多少 epoch 验证一次
        
        # ==================== 数据增强 ====================
        # 训练时的数据增强
        train_transforms=[
            'RandomHorizontalFlip',       # 随机水平翻转
            'RandomCrop',                 # 随机裁剪（如果需要）
            'ColorJitter',                # 颜色抖动
        ],
        
        # 验证时的数据增强（通常只做归一化）
        val_transforms=[
            'CenterCrop',                 # 中心裁剪
        ],
        
        # 归一化参数（ImageNet 标准）
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        
        # ==================== 其他参数 ====================
        seed=43,                          # 随机种子
        deterministic=False,              # 是否使用确定性算法（影响性能）
    )
    
    # 参数验证
    _validate_config(config)
    
    return config


def _validate_config(config):
    """验证配置参数的合法性"""
    # 检查 img_size 能否被 patch_size 整除
    assert config.img_size[0] % config.patch_size == 0, \
        f"img_size[0] ({config.img_size[0]}) 必须能被 patch_size ({config.patch_size}) 整除"
    assert config.img_size[1] % config.patch_size == 0, \
        f"img_size[1] ({config.img_size[1]}) 必须能被 patch_size ({config.patch_size}) 整除"
    
    # 检查 C 是否为偶数
    assert config.C % 2 == 0, f"传输维度 C ({config.C}) 必须是偶数"
    
    # 检查 embed_dims, depths, num_heads 长度一致
    assert len(config.embed_dims) == len(config.depths) == len(config.num_heads), \
        "embed_dims, depths, num_heads 的长度必须一致"
    
    # 检查 window_size 能否整除各层特征图尺寸
    patch_resolution = config.img_size[0] // config.patch_size
    for i in range(len(config.depths)):
        resolution = patch_resolution // (2 ** i)
        assert resolution % config.window_size == 0, \
            f"第 {i} 层的特征图尺寸 ({resolution}) 必须能被 window_size ({config.window_size}) 整除"
    
    # 检查 MS-SSIM 层数
    if config.distortion_metric == 'MS-SSIM':
        max_levels = 0
        size = min(config.img_size)
        while size >= 11:  # MS-SSIM 最小需要 11×11
            max_levels += 1
            size //= 2
        assert config.ms_ssim_levels <= max_levels, \
            f"对于 {config.img_size} 的图像，MS-SSIM 最多支持 {max_levels} 层，但配置了 {config.ms_ssim_levels} 层"
    
    print("✓ 配置参数验证通过")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path

# 假设你的模型文件在当前目录
from SwinJSCCModel import SwinJSCC
from distortion import Distortion


# ==================== 数据集定义 ====================
class TinyImageNetDataset(Dataset):
    """Tiny-ImageNet 数据集加载器"""
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Args:
            root_dir: 数据集根目录
            transform: 数据增强
            is_train: 是否为训练集
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_train = is_train
        self.images = []
        
        if is_train:
            # 训练集结构：train/n01443537/images/*.JPEG
            for class_dir in self.root_dir.iterdir():
                if class_dir.is_dir():
                    img_dir = class_dir / 'images'
                    if img_dir.exists():
                        self.images.extend(list(img_dir.glob('*.JPEG')))
        else:
            # 验证集结构：val/images/*.JPEG
            img_dir = self.root_dir / 'images'
            if img_dir.exists():
                self.images.extend(list(img_dir.glob('*.JPEG')))
        
        print(f"{'训练集' if is_train else '验证集'} 加载完成：{len(self.images)} 张图像")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
class MiniImageNetDataset(Dataset):
    """Mini-ImageNet 数据集加载器"""
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Args:
            root_dir: 数据集根目录
            transform: 数据增强
            is_train: 是否为训练集
        """
        self.root_dir = Path(root_dir)
        
        self.transform_train = transform if transform is not None else transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),# 颜色抖动
            # 转为张量
            transforms.ToTensor(),
        ])
        self.transform_val = transform if transform is not None else transforms.Compose([
            transforms.Resize((256, 256)), 
            # 转为张量
            transforms.ToTensor(),
        ])

        self.is_train = is_train
        self.images = []

        # 支持的图像格式（全覆盖常见格式）
        img_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG', '.bmp', '.BMP')
        
        # 直接遍历根目录下的所有图像（无需额外拼接train/val）
        for ext in img_extensions:
            self.images.extend(list(self.root_dir.glob(f'*{ext}')))
        
        # 空值检查（关键！提前报错，避免后续num_samples=0）
        if len(self.images) == 0:
            raise ValueError(
                f"在 {self.root_dir} 下未找到任何图像！\n"
                f"请检查：1. 路径是否正确 2. 图像格式是否在 {img_extensions} 中"
            )
        
        print(f"{'训练集' if is_train else '验证集'} 加载完成：{len(self.images)} 张图像")
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.is_train:
            image = self.transform_train(image)
        else:
            image = self.transform_val(image)

        return image
    

def get_transforms(config, is_train=True):
    """获取数据增强"""
    if is_train:
        transform_list = [
            transforms.Resize(config.img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
        ]
    else:
        transform_list = [
            transforms.Resize(config.img_size),
            transforms.ToTensor(),
        ]
    
    # 可选：归一化（如果使用预训练权重）
    # transform_list.append(transforms.Normalize(
    #     mean=config.normalize_mean,
    #     std=config.normalize_std
    # ))
    
    return transforms.Compose(transform_list)


# ==================== 训练器 ====================
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        
        # 初始化模型
        self.model = SwinJSCC(config).to(self.device)
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        
        # 初始化损失函数
        self.criterion = Distortion(config).to(self.device)
        self.perceptual_criterion = VGGPerceptualLoss().to(self.device)
        
        # 初始化优化器
        self.optimizer = self._get_optimizer()
        
        # 初始化学习率调度器
        self.scheduler = self._get_scheduler()
        
        # 混合精度训练
        self.scaler = torch.amp.GradScaler('cuda') if config.amp else None 
        
        # 训练状态
        self.start_epoch = 0
        self.best_loss = float('inf')
        
        # 恢复训练
        if config.resume:
            self._load_checkpoint(config.resume)
    
    def _set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _get_optimizer(self):
        """获取优化器"""
        if self.config.optimizer == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=self.config.betas
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer}")
    
    def _get_scheduler(self):
        """获取学习率调度器"""
        if self.config.scheduler == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.T_max,
                eta_min=1e-6
            )
        elif self.config.scheduler == 'StepLR':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
        elif self.config.scheduler == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return None
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        print(f"从 {checkpoint_path} 恢复训练...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"从 epoch {self.start_epoch} 继续训练，最佳损失: {self.best_loss:.6f}")
    
    def _save_checkpoint(self, epoch, loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'best_loss': self.best_loss,
            'config': self.config,
        }
        # 保存在self.config.save_dir中以self.config.model_name命名的文件夹下
        save_dir = os.path.join(self.config.save_dir, self.config.model_name) # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)    
        
        # 保存最新检查点
        latest_path = os.path.join(save_dir, self.config.model_name + 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 定期保存
        if (epoch + 1) % self.config.save_interval == 0:
            epoch_path = os.path.join(save_dir, self.config.model_name + f'epoch_{epoch+1}.pth')
            torch.save(checkpoint, epoch_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(save_dir, self.config.model_name + 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ 保存最佳模型 (loss: {loss:.6f})")
    
    def train_epoch(self, train_loader, epoch):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        total_distortion = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        for batch_idx, images in enumerate(pbar):
            images = images.to(self.device)
            
            # 随机选择 SNR 和 rate（课程学习策略）
            # 早期训练使用高 SNR/高 rate，后期逐渐降低
            progress = epoch / self.config.num_epochs
            if progress < 0.3:
                snr = random.choice(self.config.multiple_snrs[-1:])  # 高 SNR
                rate = random.choice(self.config.channel_numbers[-1:])  # 高 rate
            elif progress < 0.6:
                snr = random.choice(self.config.multiple_snrs[-1:])  # 中 SNR
                rate = random.choice(self.config.channel_numbers)  # 全部 rate
            else:
                snr = random.choice(self.config.multiple_snrs)  # 全部 SNR
                rate = random.choice(self.config.channel_numbers)  # 全部 rate
            
            # 前向传播
            if self.config.amp:
                with torch.amp.autocast('cuda'):
                    recon_images, CBR = self.model(images, snr=snr, rate=rate)
                    images_fp32 = images.float()
                    recon_images_fp32 = recon_images.float()
                    # 1. 结构失真损失 (MS-SSIM)
                    loss = self.criterion(recon_images_fp32, images_fp32)
                    # 2. 像素失真损失 (使用 L1 替代 MSE，L1 不容易模糊)
                    loss_l1 = F.l1_loss(recon_images_fp32, images_fp32)
                    # loss_l2 = F.mse_loss(recon_images_fp32, images_fp32)  # 可选：同时计算 L2 以供分析，但不参与最终损失
                    # 3. 感知/纹理损失 (VGG Loss)
                    loss_perceptual = self.perceptual_criterion(recon_images_fp32, images_fp32)
            else:
                recon_images, CBR = self.model(images, snr=snr, rate=rate)
                loss = self.criterion(recon_images, images)
                # loss_l2 = F.mse_loss(recon_images, images)
                loss_l1 = F.l1_loss(recon_images, images)
                loss_perceptual = self.perceptual_criterion(recon_images, images)
            loss = 0.3 * loss + 0.6 * loss_l1 + 0.1 * loss_perceptual
            
            # 反向传播
            self.optimizer.zero_grad()
            if self.config.amp:
                self.scaler.scale(loss).backward()
                if self.config.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_distortion += loss.item()
            
            # 更新进度条
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.6f}',
                    'SNR': f'{snr}dB',
                    'rate': f'{rate}',
                    'CBR': f'{CBR:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        results = {snr: [] for snr in self.config.multiple_snrs}
        
        pbar = tqdm(val_loader, desc='验证中')
        for images in pbar:
            images = images.to(self.device)
            
            # 对每个 SNR 进行评估
            for snr in self.config.multiple_snrs:
                rate = self.config.channel_numbers[-1]  # 使用最高 rate
                recon_images, _ = self.model(images, snr=snr, rate=rate)
                loss = self.criterion(recon_images, images)
                results[snr].append(loss.item())
            
            total_loss += sum(results[snr][-1] for snr in self.config.multiple_snrs) / len(self.config.multiple_snrs)
        
        avg_loss = total_loss / len(val_loader)
        
        # 打印各 SNR 的结果
        print("验证结果:")
        for snr in self.config.multiple_snrs:
            snr_loss = np.mean(results[snr])
            print(f"  SNR {snr}dB: {snr_loss:.6f}")
        print(f"  平均: {avg_loss:.6f}\n")
        
        return avg_loss
    
    @torch.no_grad()
    def visualize_results(self, val_loader, num_samples=5, snr=10, rate=None):
        """
        可视化原图与重建图，并计算每张图的MS-SSIM值
        Args:
            val_loader: 验证集数据加载器
            num_samples: 要展示的样本数量
            snr: 评估使用的SNR值 (dB)
            rate: 传输速率（None则使用配置中最高的rate）
        """
        if rate is None:
            rate = self.config.channel_numbers[-1]
        
        self.model.eval()
        device = self.device
        
        # 获取一个batch的验证数据
        for images in val_loader:
            images = images.to(device)
            # 生成重建图像（使用指定SNR和rate）
            recon_images, _ = self.model(images, snr=snr, rate=rate)
            break  # 只取第一个batch
        
        # 限制展示的样本数量（防止超出batch大小）
        num_samples = min(num_samples, len(images))
        images = images[:num_samples]
        recon_images = recon_images[:num_samples]
        
        # 创建画布：2行（原图/重建图）×num_samples列
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*4, 8))
        fig.suptitle(f'Original vs Reconstructed (SNR={snr}dB, Rate={rate})', fontsize=16)
        
        # 遍历每个样本，绘制并计算MS-SSIM
        ms_ssim_list = []
        for i in range(num_samples):
            # ========== 处理图像张量 → 可显示的PIL格式 ==========
            # 原图：(C,H,W) → (H,W,C)，[0,1] → [0,255]
            orig_img = images[i].cpu().permute(1, 2, 0).numpy()
            orig_img = np.clip(orig_img, 0, 1)  # 防止数值溢出
            orig_img = (orig_img * 255).astype(np.uint8)
            
            # 重建图：同上
            recon_img = recon_images[i].cpu().permute(1, 2, 0).numpy()
            recon_img = np.clip(recon_img, 0, 1)
            recon_img = (recon_img * 255).astype(np.uint8)
            
            # ========== 计算单样本MS-SSIM值 ==========
            # 保持batch维度 (1,C,H,W)，复用训练时的Distortion类
            single_orig = images[i:i+1]
            single_recon = recon_images[i:i+1]
            distortion_loss = self.criterion(single_recon, single_orig)
            ms_ssim_val = 1 - distortion_loss.item()  # Distortion返回1-MS-SSIM，需转换
            ms_ssim_list.append(ms_ssim_val)
            
            # ========== 绘制图像 ==========
            # 绘制原图
            ax_orig = axes[0, i] if num_samples > 1 else axes[0]
            ax_orig.imshow(orig_img)
            ax_orig.set_title(f'Original\nSample {i+1}', fontsize=12)
            ax_orig.axis('off')
            
            # 绘制重建图（标注MS-SSIM）
            ax_recon = axes[1, i] if num_samples > 1 else axes[1]
            ax_recon.imshow(recon_img)
            ax_recon.set_title(f'Reconstructed\nMS-SSIM: {ms_ssim_val:.4f}', fontsize=12)
            ax_recon.axis('off')
        
        # ========== 保存/显示图像 ==========
        plt.tight_layout()
        save_path = os.path.join(self.config.save_dir, self.config.model_name + f'visual_SNR{snr}_Rate{rate}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ 可视化结果已保存至: {save_path}")
        plt.show()
        
        # ========== 打印统计信息 ==========
        avg_ms_ssim = np.mean(ms_ssim_list)
        avg_distortion = 1 - avg_ms_ssim
        print(f"\n📊 统计结果 (SNR={snr}dB, Rate={rate}):")
        print(f"   单个样本MS-SSIM: {[f'{v:.4f}' for v in ms_ssim_list]}")
        print(f"   平均MS-SSIM值: {avg_ms_ssim:.4f}")
        print(f"   平均Distortion Loss: {avg_distortion:.4f}")
    
    def train(self, train_loader, val_loader):
        """完整训练流程"""
        print(f"\n开始训练 (共 {self.config.num_epochs} 个 epoch)...")
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 验证
            if (epoch + 1) % self.config.val_interval == 0:
                val_loss = self.validate(val_loader)
                
                # 保存最佳模型
                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
                
                self._save_checkpoint(epoch, val_loss, is_best)
                
                # 更新学习率（ReduceLROnPlateau）
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
            else:
                self._save_checkpoint(epoch, train_loss)
            
            # 更新学习率（其他调度器）
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1} 完成，用时: {epoch_time:.2f}s, 训练损失: {train_loss:.6f}\n")
        
        print("训练完成！")


# ==================== 主函数 ====================
def main():
    # 获取配置
    config = get_config()
    
    # 创建数据集
    train_dataset = MiniImageNetDataset(
        root_dir=config.train_data_dir,
        transform=get_transforms(config, is_train=True),
        is_train=True
    )
    
    val_dataset = MiniImageNetDataset(
        root_dir=config.val_data_dir,
        transform=get_transforms(config, is_train=False),
        is_train=False
    )

    # ===== 快速验证：随机采样一定比例的数据 =====
    sample_ratio = 0.4  # 修改这里控制比例，1.0 表示使用全部数据
    if sample_ratio < 1.0:
        train_size = int(len(train_dataset) * sample_ratio)
        val_size = int(len(val_dataset) * sample_ratio)
        train_indices = random.sample(range(len(train_dataset)), train_size)
        val_indices = random.sample(range(len(val_dataset)), val_size)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        print(f"快速验证模式：训练集 {train_size} 张，验证集 {val_size} 张")
    # ==========================================
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # 创建训练器
    trainer = Trainer(config)
    
    # 开始训练
    trainer.train(train_loader, val_loader)

    # ========== 训练完成后：可视化结果 ==========
    print("\n📈 开始可视化训练结果...")
    # 可视化10dB SNR（常用值）的结果
    trainer.visualize_results(
        val_loader=val_loader,
        num_samples=5,  # 展示5个样本
        snr=10,         # 选择10dB SNR评估
        rate=None       # 使用最高rate（192）
    )
    
    # 可选：遍历所有SNR值可视化
    # for snr in config.multiple_snrs:
    #     trainer.visualize_results(val_loader, num_samples=3, snr=snr)


if __name__ == '__main__':
    main()
