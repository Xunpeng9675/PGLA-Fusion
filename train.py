# -*- coding: utf-8 -*-

import os
import time
import datetime
import torch
import torch.nn as nn
import numpy as np
import random
import yaml                                    # --- 新增 ---
from torch.utils.data import DataLoader
from model.loss import Fusionloss, cc, infoNCE_loss, PerceptualLoss
import kornia
from model.kernel_loss import kernelLoss
from utils.evaluator import average_similarity
from model.PGLA_Fusion import (
    DualStreamContextEncoder,
    DualStreamContextDecoder,
    SemanticFeatureEncoder,
    ProgressiveTextureEncoder,
)
from utils.dataset import H5Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_config(config_path="config.yaml"):     # --- 新增 ---
    """加载yaml配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


# ==================== 加载配置 ====================
cfg = load_config()

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['hardware']['gpu_id'])
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# 设置随机种子
set_seed(cfg['train']['seed'])

# ==================== 从配置读取参数 ====================
batch_size = cfg['train']['batch_size']
num_epochs = cfg['train']['num_epochs']
windows_size = cfg['train']['windows_size']
lr = cfg['train']['lr']
weight_decay = cfg['train']['weight_decay']
clip_grad_norm_value = cfg['train']['clip_grad_norm_value']

# 损失权重
coeff_ssim = cfg['loss_weights']['ssim']
coeff_mse = cfg['loss_weights']['mse']
coeff_tv = cfg['loss_weights']['tv']
coeff_decomp = cfg['loss_weights']['decomp']
coeff_nice = cfg['loss_weights']['nice']
coeff_cc_basic = cfg['loss_weights']['cc_basic']
coeff_gauss = cfg['loss_weights']['gauss']
coeff_laplace = cfg['loss_weights']['laplace']

# 数据路径
train_h5_path = cfg['data']['train_h5_path']
vgg_weight_path = cfg['model']['vgg_weight_path']

# ==================== 初始化损失函数 ====================
gaussianLoss = kernelLoss("gaussian")
laplaceLoss = kernelLoss("laplace")
MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
criteria_fusion = Fusionloss()
Loss_ssim = kornia.losses.SSIMLoss(11, reduction="mean")
perceptual_loss = PerceptualLoss(weight_path=vgg_weight_path)

# ==================== 设备与模型 ====================
device = cfg['hardware']['device'] if torch.cuda.is_available() else "cpu"

Encoder = nn.DataParallel(DualStreamContextEncoder()).to(device)
Decoder = nn.DataParallel(DualStreamContextDecoder()).to(device)
BaseFuseLayer = nn.DataParallel(
    SemanticFeatureEncoder(dim=cfg['model']['dim'], num_heads=cfg['model']['num_heads'])
).to(device)
DetailFuseLayer = nn.DataParallel(
    ProgressiveTextureEncoder(num_layers=cfg['model']['detail_num_layers'])
).to(device)

# ==================== 优化器与调度器 ====================
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer1 = optim.Adam(Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = optim.Adam(Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = optim.Adam(BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = optim.Adam(DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = CosineAnnealingLR(
    optimizer1,
    T_max=cfg['train']['scheduler']['T_max'],
    eta_min=cfg['train']['scheduler']['eta_min']
)
scheduler2 = CosineAnnealingLR(
    optimizer2,
    T_max=cfg['train']['scheduler']['T_max'],
    eta_min=cfg['train']['scheduler']['eta_min']
)
scheduler3 = CosineAnnealingLR(
    optimizer3,
    T_max=cfg['train']['scheduler']['T_max'],
    eta_min=cfg['train']['scheduler']['eta_min']
)
scheduler4 = CosineAnnealingLR(
    optimizer4,
    T_max=cfg['train']['scheduler']['T_max'],
    eta_min=cfg['train']['scheduler']['eta_min']
)

# ==================== 数据加载 ====================
trainloader = DataLoader(
    H5Dataset(train_h5_path),
    batch_size=batch_size,
    shuffle=True,
    num_workers=cfg['hardware']['num_workers'],
)

loader = {'train': trainloader}

# ==================== 训练 ====================
result_name = f"{cfg['logging']['result_name_prefix']}_batch{batch_size}_epoch{num_epochs}_WIN{windows_size}_cuda{cfg['hardware']['gpu_id']}"
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

torch.backends.cudnn.benchmark = True
prev_time = time.time()
Encoder.train()
Decoder.train()
BaseFuseLayer.train()
DetailFuseLayer.train()

for epoch in range(num_epochs):
    ''' train '''
    for i, (img_VI, img_IR) in enumerate(loader['train']):

        # Phase I
        img_VI, img_IR = img_VI.cuda(), img_IR.cuda()

        Encoder.zero_grad()
        Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        feature_V_B, feature_V_D, _ = Encoder(img_VI)
        feature_I_B, feature_I_D, _ = Encoder(img_IR)
        data_VI_hat, _ = Decoder(img_VI, feature_V_B, feature_V_D)
        data_IR_hat, _ = Decoder(img_IR, feature_I_B, feature_I_D)

        cc_loss_B = cc(feature_V_B, feature_I_B)
        cc_loss_D = cc(feature_V_D, feature_I_D)

        ssim_loss = coeff_ssim * (
            Loss_ssim(img_IR, data_IR_hat) + Loss_ssim(img_VI, data_VI_hat)
        )

        mse_loss = coeff_mse * (
            MSELoss(img_VI, data_VI_hat) + MSELoss(img_IR, data_IR_hat)
        )
        perceptual_loss_value = cfg['loss_weights'].get('perceptual', 1.0) * (
            perceptual_loss(data_VI_hat, img_VI) + perceptual_loss(data_IR_hat, img_IR)
        )
        tv_loss = coeff_tv * (
            L1Loss(
                kornia.filters.SpatialGradient()(img_VI),
                kornia.filters.SpatialGradient()(data_VI_hat),
            )
            + L1Loss(
                kornia.filters.SpatialGradient()(img_IR),
                kornia.filters.SpatialGradient()(data_IR_hat),
            )
        )

        cc_loss = coeff_decomp * (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

        laplace_loss = coeff_laplace * laplaceLoss(feature_V_B, feature_I_B)
        gauss_loss = coeff_gauss * gaussianLoss(feature_V_B, feature_I_B)
        ince_loss = coeff_nice * infoNCE_loss(feature_V_B, feature_I_B)
        basic_cc_loss = coeff_cc_basic * cc_loss_B

        mmd_loss = laplace_loss + gauss_loss + basic_cc_loss + ince_loss

        loss1 = ssim_loss + mse_loss + cc_loss + tv_loss + mmd_loss + perceptual_loss_value

        similarity_cos = average_similarity(feature_V_B, feature_I_B, "cosine")
        similarity_pearson = average_similarity(feature_V_B, feature_I_B, "pearson")
        distance_euclidean = average_similarity(feature_V_B, feature_I_B, "euclidean")

        loss1.backward()

        # Phase II
        nn.utils.clip_grad_norm_(
            Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        optimizer1.step()
        optimizer2.step()

        feature_V_B, feature_V_D, feature_V = Encoder(img_VI)
        feature_I_B, feature_I_D, feature_I = Encoder(img_IR)
        feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
        feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)
        data_Fuse, feature_F = Decoder(img_VI, feature_F_B, feature_F_D)

        cc_loss_B = cc(feature_V_B, feature_I_B)
        cc_loss_D = cc(feature_V_D, feature_I_D)
        cc_loss = coeff_decomp * ((cc_loss_D) ** 2 / (1.01 + cc_loss_B))
        fusionloss, _, _ = criteria_fusion(img_VI, img_IR, data_Fuse)

        loss2 = fusionloss + cc_loss

        loss2.backward()
        nn.utils.clip_grad_norm_(
            Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        nn.utils.clip_grad_norm_(
            DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2
        )
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()

        # 日志输出
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        print(
            f"[E:{epoch}/{num_epochs}][B:{i}/{len(loader['train'])}][L1:{loss1.item():.2f},mse:{mse_loss.item():.2f},cc:{cc_loss.item():.2f},tv:{tv_loss.item():.2f},mmd:{mmd_loss.item():.2f},lap:{laplace_loss.item():.2f},gauss:{gauss_loss.item():.2f},ince:{ince_loss.item():.2f},ccb:{basic_cc_loss.item():.2f}, perceptual:{perceptual_loss_value.item():.2f}][L2:{loss2.item():.2f},f:{fusionloss.item():.2f},cc:{cc_loss.item():.2f}][{similarity_cos:.2f},{similarity_pearson:.2f},{distance_euclidean:.2f}]"
        )

    # 保存checkpoint
    os.makedirs(cfg['logging']['save_dir'], exist_ok=True)  # --- 新增：确保目录存在 ---
    save_path = os.path.join(cfg['logging']['save_dir'], f"1.0learning rate halved{result_name}_{epoch}.pth")
    checkpoint = {
        'Encoder': Encoder.state_dict(),
        'Decoder': Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    torch.save(checkpoint, save_path)

    # 更新学习率调度器（你原来的代码没有调用 scheduler.step()，这里加上）
    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    scheduler4.step()