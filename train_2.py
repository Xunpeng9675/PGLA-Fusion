# -*- coding: utf-8 -*-

import os
import time
import datetime
import torch
import torch.nn as nn
import numpy as np
import random
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
import torch.nn.functional as F
import math

# ========== Environment & Seed ==========
# reduce fragmentation (可选，建议在 shell 中也设置)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU
    torch.backends.cudnn.benchmark = True  # Keep performance optimization

# Set seed for reproducibility (without sacrificing performance)
set_seed(42)  # Use any desired seed value

# ========== Debug / Numeric-safety helpers ==========
def is_finite_tensor(t):
    if isinstance(t, torch.Tensor):
        return torch.isfinite(t).all().item()
    else:
        # scalar check
        return not (math.isnan(float(t)) or math.isinf(float(t)))

def save_problematic_batch(epoch, batch_idx, info_dict):
    """
    Save suspicious tensors to disk for later debugging.
    All tensors will be detached and moved to CPU.
    """
    os.makedirs("debug_nan", exist_ok=True)
    fname = f"debug_nan/problem_epoch{epoch}_batch{batch_idx}_{int(time.time())}.pth"
    to_save = {}
    for k, v in info_dict.items():
        try:
            if isinstance(v, torch.Tensor):
                to_save[k] = v.detach().cpu()
            else:
                to_save[k] = v
        except Exception as e:
            to_save[k] = f"<save_failed:{e}>"
    torch.save(to_save, fname)
    print(f"[DEBUG] Saved problematic batch info to {fname}")

# ========= Loss / Metrics / Model setup ==========
gaussianLoss = kernelLoss("gaussian")
laplaceLoss = kernelLoss("laplace")
MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
criteria_fusion = Fusionloss()
Loss_ssim = kornia.losses.SSIMLoss(11, reduction="mean")
# 实例化感知损失
perceptual_loss = PerceptualLoss(weight_path="model/vgg16-397923af.pth")

# ---------- hyperparams ----------
batch_size = 8
num_epochs = 5
windows_size = 11
lr = 1e-4
weight_decay = 0
# clip grad norm: 0.01 太小，容易出异常，设为 1.0 更合理
clip_grad_norm_value = 1.0
optim_step = 20
optim_gamma = 0.5

coeff_ssim = 5.0
coeff_mse = 1.0
coeff_tv = 5.0
coeff_decomp = 2.0
coeff_nice = 0.1
coeff_cc_basic = 2.0
coeff_gauss = 1.0
coeff_laplace = 1.0

result_name = f"mkmmd_batch{batch_size}_epoch{num_epochs}_WIN{windows_size}_cuda1"

device = "cuda" if torch.cuda.is_available() else "cpu"
Encoder = nn.DataParallel(DualStreamContextEncoder()).to(device)
Decoder = nn.DataParallel(DualStreamContextDecoder()).to(device)
BaseFuseLayer = nn.DataParallel(SemanticFeatureEncoder(dim=64, num_heads=8)).to(
    device
)
DetailFuseLayer = nn.DataParallel(ProgressiveTextureEncoder(num_layers=1)).to(device)

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# 定义优化器
optimizer1 = optim.Adam(Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = optim.Adam(Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = optim.Adam(BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = optim.Adam(DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

# 使用 CosineAnnealingLR 学习率调度器
scheduler1 = CosineAnnealingLR(optimizer1, T_max=num_epochs, eta_min=1e-6)
scheduler2 = CosineAnnealingLR(optimizer2, T_max=num_epochs, eta_min=1e-6)
scheduler3 = CosineAnnealingLR(optimizer3, T_max=num_epochs, eta_min=1e-6)
scheduler4 = CosineAnnealingLR(optimizer4, T_max=num_epochs, eta_min=1e-6)

# ========== DataLoader ==========
trainloader = DataLoader(
    H5Dataset(r"data/dataSet4Training_imgsize_128_stride_200.h5"),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

torch.backends.cudnn.benchmark = True
prev_time = time.time()
Encoder.train()
Decoder.train()
BaseFuseLayer.train()
DetailFuseLayer.train()

# Optional: enable anomaly detection for autograd (slower). Uncomment to use:
# torch.autograd.set_detect_anomaly(True)

for epoch in range(num_epochs):
    ''' train '''
    for i, (img_VI, img_IR) in enumerate(loader['train']):

        # Move to device
        img_VI = img_VI.to(device)
        img_IR = img_IR.to(device)

        # Zero grads
        Encoder.zero_grad()
        Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        # ========== Phase I ==========
        try:
            # forward
            feature_V_B, feature_V_D, _ = Encoder(img_VI)
            feature_I_B, feature_I_D, _ = Encoder(img_IR)
            data_VI_hat, _ = Decoder(img_VI, feature_V_B, feature_V_D)
            data_IR_hat, _ = Decoder(img_IR, feature_I_B, feature_I_D)

            # compute cc losses
            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)

            # ssim + mse (clamp images to valid range before comparing)
            ssim_loss = coeff_ssim * (
                Loss_ssim(img_IR.clamp(0.0,1.0), data_IR_hat.clamp(0.0,1.0))
                + Loss_ssim(img_VI.clamp(0.0,1.0), data_VI_hat.clamp(0.0,1.0))
            )

            mse_loss = coeff_mse * (
                MSELoss(img_VI, data_VI_hat) + MSELoss(img_IR, data_IR_hat)
            )

            # perceptual loss: clamp inputs to [0,1] (depends on your PerceptualLoss expectation)
            perceptual_loss_value = perceptual_loss(data_VI_hat.clamp(0.0,1.0), img_VI.clamp(0.0,1.0)) \
                                     + perceptual_loss(data_IR_hat.clamp(0.0,1.0), img_IR.clamp(0.0,1.0))

            # tv loss
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

            # cc decomposition loss - ensure denominator not too small
            denom = (1.01 + cc_loss_B)
            denom = denom.clamp(min=1e-6)
            cc_loss = coeff_decomp * (cc_loss_D ** 2) / denom

            # laplace / gauss / infoNCE (normalize features before infoNCE)
            laplace_loss = coeff_laplace * laplaceLoss(feature_V_B, feature_I_B)
            gauss_loss = coeff_gauss * gaussianLoss(feature_V_B, feature_I_B)

            # normalize features for infoNCE
            try:
                feat_V_B_norm = F.normalize(feature_V_B, dim=1)
                feat_I_B_norm = F.normalize(feature_I_B, dim=1)
            except Exception:
                feat_V_B_norm = feature_V_B
                feat_I_B_norm = feature_I_B

            ince_loss = coeff_nice * infoNCE_loss(feat_V_B_norm, feat_I_B_norm)
            basic_cc_loss = coeff_cc_basic * cc_loss_B

            mmd_loss = laplace_loss + gauss_loss + basic_cc_loss + ince_loss

            loss1 = ssim_loss + mse_loss + cc_loss + tv_loss + mmd_loss + perceptual_loss_value

            # ======= Safety checks for Phase I losses =======
            # check all component losses
            comp_losses = {
                "ssim_loss": ssim_loss,
                "mse_loss": mse_loss,
                "perceptual_loss_value": perceptual_loss_value,
                "tv_loss": tv_loss,
                "cc_loss": cc_loss,
                "laplace_loss": laplace_loss,
                "gauss_loss": gauss_loss,
                "ince_loss": ince_loss,
                "basic_cc_loss": basic_cc_loss,
                "loss1": loss1
            }
            bad_flag = False
            for name, val in comp_losses.items():
                if not is_finite_tensor(val):
                    print(f"[NaN/Inf in Phase I] epoch {epoch} batch {i} -- {name} is not finite. val={val}")
                    bad_flag = True

            if bad_flag:
                # save debug info and skip this batch
                info_dict = {
                    "img_VI": img_VI.cpu(),
                    "img_IR": img_IR.cpu(),
                    "data_VI_hat": data_VI_hat.detach().cpu(),
                    "data_IR_hat": data_IR_hat.detach().cpu(),
                    "feature_V_B": feature_V_B.detach().cpu(),
                    "feature_I_B": feature_I_B.detach().cpu(),
                    "feature_V_D": feature_V_D.detach().cpu(),
                    "feature_I_D": feature_I_D.detach().cpu(),
                    "comp_losses": {k: (float(v) if not isinstance(v, torch.Tensor) else float(v.detach().cpu().item())) for k,v in comp_losses.items()}
                }
                save_problematic_batch(epoch, i, info_dict)
                # zero grads and continue to next batch
                Encoder.zero_grad(); Decoder.zero_grad(); BaseFuseLayer.zero_grad(); DetailFuseLayer.zero_grad()
                optimizer1.zero_grad(); optimizer2.zero_grad(); optimizer3.zero_grad(); optimizer4.zero_grad()
                continue

            # backprop for phase I
            loss1.backward()

            # check gradients finite before optimizer.step()
            grads_finite = True
            for p in Encoder.parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        grads_finite = False
                        break
            if not grads_finite:
                print(f"[NaN gradient in Phase I] epoch {epoch} batch {i} - skipping optimizer step and saving debug")
                info_dict = {
                    "img_VI": img_VI.cpu(),
                    "img_IR": img_IR.cpu(),
                }
                save_problematic_batch(epoch, i, info_dict)
                Encoder.zero_grad(); Decoder.zero_grad(); BaseFuseLayer.zero_grad(); DetailFuseLayer.zero_grad()
                optimizer1.zero_grad(); optimizer2.zero_grad(); optimizer3.zero_grad(); optimizer4.zero_grad()
                continue

            # gradient clipping & step
            nn.utils.clip_grad_norm_(Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()

        except Exception as e:
            print(f"[Exception in Phase I] epoch {epoch} batch {i}: {e}")
            # try save batch and continue
            try:
                save_problematic_batch(epoch, i, {"img_VI": img_VI.cpu(), "img_IR": img_IR.cpu(), "exception": str(e)})
            except Exception as e2:
                print("Failed to save problematic batch:", e2)
            continue

        # ========== Phase II ==========
        try:
            feature_V_B, feature_V_D, feature_V = Encoder(img_VI)
            feature_I_B, feature_I_D, feature_I = Encoder(img_IR)
            # fuse features
            feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)
            data_Fuse, feature_F = Decoder(img_VI, feature_F_B, feature_F_D)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            denom = (1.01 + cc_loss_B).clamp(min=1e-6)
            cc_loss = coeff_decomp * ((cc_loss_D ** 2) / denom)

            fusionloss, _, _ = criteria_fusion(img_VI, img_IR, data_Fuse)

            loss2 = fusionloss + cc_loss

            # safety checks for Phase II
            comp_losses2 = {"fusionloss": fusionloss, "cc_loss": cc_loss, "loss2": loss2}
            bad2 = False
            for name, val in comp_losses2.items():
                if not is_finite_tensor(val):
                    print(f"[NaN/Inf in Phase II] epoch {epoch} batch {i} -- {name} is not finite. val={val}")
                    bad2 = True

            if bad2:
                info_dict = {
                    "img_VI": img_VI.cpu(),
                    "img_IR": img_IR.cpu(),
                    "data_Fuse": data_Fuse.detach().cpu(),
                    "feature_F_B": feature_F_B.detach().cpu() if 'feature_F_B' in locals() else None,
                    "feature_F_D": feature_F_D.detach().cpu() if 'feature_F_D' in locals() else None,
                }
                save_problematic_batch(epoch, i, info_dict)
                Encoder.zero_grad(); Decoder.zero_grad(); BaseFuseLayer.zero_grad(); DetailFuseLayer.zero_grad()
                optimizer1.zero_grad(); optimizer2.zero_grad(); optimizer3.zero_grad(); optimizer4.zero_grad()
                continue

            # backprop for phase II
            loss2.backward()

            # gradient check
            grads_finite2 = True
            for p in list(Encoder.parameters()) + list(Decoder.parameters()) + list(BaseFuseLayer.parameters()) + list(DetailFuseLayer.parameters()):
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        grads_finite2 = False
                        break
            if not grads_finite2:
                print(f"[NaN gradient in Phase II] epoch {epoch} batch {i} - skipping optimizer step and saving debug")
                save_problematic_batch(epoch, i, {"img_VI": img_VI.cpu(), "img_IR": img_IR.cpu(), "note": "grad NaN in phase II"})
                Encoder.zero_grad(); Decoder.zero_grad(); BaseFuseLayer.zero_grad(); DetailFuseLayer.zero_grad()
                optimizer1.zero_grad(); optimizer2.zero_grad(); optimizer3.zero_grad(); optimizer4.zero_grad()
                continue

            # gradient clipping and stepping
            nn.utils.clip_grad_norm_(Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

            # logging metrics & similarity
            similarity_cos = average_similarity(feature_V_B, feature_I_B, "cosine")
            similarity_pearson = average_similarity(feature_V_B, feature_I_B, "pearson")
            distance_euclidean = average_similarity(feature_V_B, feature_I_B, "euclidean")

            print(
                f"[E:{epoch}/{num_epochs}][B:{i}/{len(loader['train'])}][L1:{loss1.item():.2f},mse:{mse_loss.item():.2f},cc:{cc_loss.item():.2f},tv:{tv_loss.item():.2f},mmd:{mmd_loss.item():.2f},lap:{laplace_loss.item():.2f},gauss:{gauss_loss.item():.2f},ince:{ince_loss.item():.2f},ccb:{basic_cc_loss.item():.2f}, perceptual:{perceptual_loss_value.item():.2f}][L2:{loss2.item():.2f},f:{fusionloss.item():.2f},cc:{cc_loss.item():.2f}][{similarity_cos:.2f},{similarity_pearson:.2f},{distance_euclidean:.2f}]"
            )

        except Exception as e:
            print(f"[Exception in Phase II] epoch {epoch} batch {i}: {e}")
            try:
                save_problematic_batch(epoch, i, {"img_VI": img_VI.cpu(), "img_IR": img_IR.cpu(), "exception": str(e)})
            except Exception as e2:
                print("Failed to save problematic batch:", e2)
            # Zero grads to avoid propagation
            Encoder.zero_grad(); Decoder.zero_grad(); BaseFuseLayer.zero_grad(); DetailFuseLayer.zero_grad()
            optimizer1.zero_grad(); optimizer2.zero_grad(); optimizer3.zero_grad(); optimizer4.zero_grad()
            continue

        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

    # End of epoch: save checkpoint
    save_path = os.path.join(f"checkPoints/clip/1.0稳定版{result_name}_{epoch}.pth")
    checkpoint = {
        'Encoder': Encoder.state_dict(),
        'Decoder': Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)

    # Scheduler step (optional, keep existing behavior)
    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    scheduler4.step()

    # Prevent learning rate from going too low (preserve original logic)
    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6
