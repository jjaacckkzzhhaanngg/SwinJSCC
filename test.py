import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 请根据您的实际主文件类名调整导入
# 这里假设主模型类名为 SwinJSCC (包含 .encoder 和 .decoder)
from SwinJSCCModel import SwinJSCC 
from trainmodel import get_config, MiniImageNetDataset, get_transforms
from distortion import Distortion

def evaluate_and_visualize_superposition():
    # 1. 初始化配置与设备
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 测试参数设置
    test_snr = 1      # 建议设置为 1dB
    test_rate = 192   # 最大传输速率

    # 2. 初始化两个模型并加载权重
    print("正在初始化并加载模型权重...")
    model1 = SwinJSCC(config).to(device)
    model2 = SwinJSCC(config).to(device)

    # 请确保 checkpoint 文件夹在当前运行目录下
    checkpoint1 = torch.load('checkpoints/SwinJSCC_SAandRA_B1_/SwinJSCC_SAandRA_B1_best.pth', map_location=device, weights_only=False)
    checkpoint2 = torch.load('checkpoints/SwinJSCC_SAandRA_B2_/SwinJSCC_SAandRA_B2_best.pth', map_location=device, weights_only=False)

    # 提取纯粹的模型权重字典
    # 使用 .get() 方法可以兼容字典格式。请根据您 trainmodel.py 中实际保存的键名（如 'model_state_dict' 或 'state_dict'）进行调整
    # 如果 checkpoint 本身就是单纯的权重字典，则回退直接使用 checkpoint
    state_dict1 = checkpoint1.get('model_state_dict', checkpoint1.get('state_dict', checkpoint1))
    state_dict2 = checkpoint2.get('model_state_dict', checkpoint2.get('state_dict', checkpoint2))

    # 加载权重
    model1.load_state_dict(state_dict1)
    model2.load_state_dict(state_dict2)

    # 切换到评估模式（关键！这会固定 Dropout 和 LayerNorm/BatchNorm 等）
    model1.eval()
    model2.eval()

    # 3. 准备 Tiny-ImageNet 验证数据集
    # 根据 trainmodel.py 中的 config.val_data_dir 加载数据
    transform = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.ToTensor() # 转换为 [0, 1] 的张量
    ])
    # val_dataset = datasets.ImageFolder(config.val_data_dir, transform=transform)
    val_dataset = MiniImageNetDataset(
        root_dir=config.val_data_dir,
        transform=transform,
        is_train=False
    )
    # batch_size=2 刚好可以将数据集两两分组
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

    # 初始化 MS-SSIM 计算工具 (使用您 distortion.py 中的类)
    distortion_criterion = Distortion(config).to(device)

    results = []

    # 4. 在整个数据集上进行推断
    print("开始在整个数据集上进行推断测试，这可能需要一些时间...")
    with torch.no_grad():
        for batch_idx, images in enumerate(val_loader):
            # 如果最后一组只有一张图片，则跳过
            if len(images) < 2:
                continue
                
            img1 = images[0:1].to(device) # Shape: (1, C, H, W)
            img2 = images[1:2].to(device) # Shape: (1, C, H, W)

            # === 编码器特征提取 ===
            # SwinJSCC_w/_SAandRA 模式的 encoder 返回 (features, mask)
            latent1, _ = model1.encoder(img1, test_snr, test_rate, model1.model_mode)
            latent2, _ = model2.encoder(img2, test_snr, test_rate, model2.model_mode)

            # === 信号叠加（不经过信道，直接物理叠加） ===
            superposed_latent = latent1 + latent2

            # === 解码器重建 ===
            # 解码器可能仅需要 snr 作为自适应条件
            recon1 = model1.decoder(superposed_latent, test_snr, model1.model_mode)
            recon2 = model2.decoder(superposed_latent, test_snr, model2.model_mode)

            # === 计算 MS-SSIM ===
            # Distortion 返回 1 - MS-SSIM
            loss1 = distortion_criterion(recon1, img1)
            loss2 = distortion_criterion(recon2, img2)
            
            msssim1 = 1 - loss1.item()
            msssim2 = 1 - loss2.item()
            min_msssim = min(msssim1, msssim2)

            # 收集结果用于后续排序 (将张量转至 CPU 以节省显存)
            results.append({
                'img1': img1.cpu().squeeze(0),
                'img2': img2.cpu().squeeze(0),
                'recon1': recon1.cpu().squeeze(0),
                'recon2': recon2.cpu().squeeze(0),
                'msssim1': msssim1,
                'msssim2': msssim2,
                'min_msssim': min_msssim,
                'batch_idx': batch_idx
            })

            if (batch_idx + 1) % 500 == 0:
                print(f"已处理 {batch_idx + 1} 组图片...")

    # 5. 根据最小 MS-SSIM 进行排序
    print("推断完成！正在筛选最好和最差的5组...")
    results.sort(key=lambda x: x['min_msssim'])
    
    worst_5 = results[:5]       # MS-SSIM 最小的5组
    best_5 = results[-5:]       # MS-SSIM 最大的5组 (默认升序，最后5个最大)
    best_5.reverse()            # 将最好的倒序，使最最好的在第一个

    # 6. 可视化函数
    def plot_results(selected_results, title_prefix):
        num_samples = len(selected_results)
        # 画布：4行（原图1, 重建1, 原图2, 重建2），num_samples 列
        fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 3, 12))
        fig.suptitle(f'{title_prefix} 5 Pairs (Min MS-SSIM sorted)', fontsize=16)

        for i, res in enumerate(selected_results):
            # 将张量 (C,H,W) 转换为 numpy (H,W,C) 用于绘图
            def to_img(tensor):
                img = tensor.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                return (img * 255).astype(np.uint8)

            img1_np = to_img(res['img1'])
            recon1_np = to_img(res['recon1'])
            img2_np = to_img(res['img2'])
            recon2_np = to_img(res['recon2'])

            # 第一行：原图1
            axes[0, i].imshow(img1_np)
            axes[0, i].set_title(f"Pair {res['batch_idx']} Original 1")
            axes[0, i].axis('off')

            # 第二行：重建图1
            axes[1, i].imshow(recon1_np)
            axes[1, i].set_title(f"Recon 1 MS-SSIM: {res['msssim1']:.4f}")
            axes[1, i].axis('off')

            # 第三行：原图2
            axes[2, i].imshow(img2_np)
            axes[2, i].set_title(f"Original 2")
            axes[2, i].axis('off')

            # 第四行：重建图2
            axes[3, i].imshow(recon2_np)
            axes[3, i].set_title(f"Recon 2 MS-SSIM: {res['msssim2']:.4f}")
            axes[3, i].axis('off')

            # 底部添加全局标注信息
            axes[3, i].text(0.5, -0.2, f"Min MS-SSIM: {res['min_msssim']:.4f}", 
                            ha='center', va='top', transform=axes[3, i].transAxes, 
                            fontsize=11, fontweight='bold', color='red')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 建立保存文件夹并保存图片
        os.makedirs(config.save_dir, exist_ok=True)
        save_path = os.path.join(config.save_dir, f'superposition_{title_prefix.lower()}_5_pairs.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ {title_prefix} 结果已保存至: {save_path}")
        plt.show()

    # 展示并保存结果
    plot_results(best_5, "Top")
    plot_results(worst_5, "Bottom")

if __name__ == '__main__':
    evaluate_and_visualize_superposition()

