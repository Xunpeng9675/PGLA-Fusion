from model.PGLA_Fusion import (
    DualStreamContextEncoder,
    DualStreamContextDecoder,
    SemanticFeatureEncoder,
    ProgressiveTextureEncoder,
)
import os
import numpy as np
from utils.evaluator import Evaluator
import torch
import torch.nn as nn
from utils.imageUtils import img_save, image_read_cv2
import warnings
import logging
import cv2  # 导入 OpenCV
import yaml
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

def load_config(config_path="config.yaml"):     # --- 新增 ---
    """加载yaml配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

cfg = load_config()

os.environ["CUDA_VISIBLE_DEVICES" ] = str(cfg['hardware']['gpu_id'])
ckpt_path = str(cfg['test']['ckpt_path'])

print(f"{ckpt_path}\n")
for dataset_name in ["TNO", "RoadScene", "MSRS_test"]:  #
    print(f"The test result of {dataset_name}:")
    test_folder = os.path.join("test_img", dataset_name)
    test_out_folder = os.path.join("test_result", dataset_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Encoder = nn.DataParallel(DualStreamContextEncoder()).to(device)
    Decoder = nn.DataParallel(DualStreamContextDecoder()).to(device)
    BaseFuseLayer = nn.DataParallel(SemanticFeatureEncoder(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(ProgressiveTextureEncoder(num_layers=1)).to(device)

    Encoder.load_state_dict(torch.load(ckpt_path)["Encoder"])
    Decoder.load_state_dict(torch.load(ckpt_path)["Decoder"])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)["BaseFuseLayer"])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)["DetailFuseLayer"])
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder, "ir")):

            data_IR = (
                image_read_cv2(os.path.join(test_folder, "ir", img_name), mode="GRAY")[
                    np.newaxis, np.newaxis, ...
                ]
                / 255.0
            )
            data_VIS = (
                image_read_cv2(os.path.join(test_folder, "vi", img_name), mode="GRAY")[
                    np.newaxis, np.newaxis, ...
                ]
                / 255.0
            )

            data_IR, data_VIS = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

            feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
            data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)
            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (
                torch.max(data_Fuse) - torch.min(data_Fuse)
            )
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
            img_save(fi, img_name.split(sep=".")[0], test_out_folder)

    eval_folder = test_out_folder
    ori_img_folder = test_folder

    metric_result = np.zeros((7))
    for img_name in os.listdir(os.path.join(ori_img_folder, "ir")):
        ir = image_read_cv2(os.path.join(ori_img_folder, "ir", img_name), "GRAY")
        vi = image_read_cv2(os.path.join(ori_img_folder, "vi", img_name), "GRAY")
        fi = image_read_cv2(
            os.path.join(eval_folder, img_name.split(".")[0] + ".png"), "GRAY"
        )
        metric_result += np.array(
            [
                Evaluator.EN(fi),
                Evaluator.SD(fi),
                Evaluator.SF(fi),
                Evaluator.SCD(fi, ir, vi),
                Evaluator.VIFF(fi, ir, vi),
                Evaluator.Qabf(fi, ir, vi),
                Evaluator.SSIM(fi, ir, vi),
            ]
        )#参数量 (Params)、FLOPs、内存占用

    metric_result /= len(os.listdir(eval_folder))
    print("\t\t EN\t\t SD\t\t SF\t\t SCD\t\t VIF\t\t Qabf\t\t SSIM")
    print(
        "\t\t "
        + str(np.round(metric_result[0], 4))
        + "\t\t"
        + str(np.round(metric_result[1], 4))
        + "\t\t"
        + str(np.round(metric_result[2], 4))
        + "\t\t"
        + str(np.round(metric_result[3], 4))
        + "\t\t"
        + str(np.round(metric_result[4], 4))
        + "\t\t"
        + str(np.round(metric_result[5], 4))
        + "\t\t"
        + str(np.round(metric_result[6], 4))
        + "\t\t"
    )
    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder, "ir")):
            # 读取红外图像（保持不变）
            data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[
                          np.newaxis, np.newaxis, ...] / 255.0

            # 读取可见光图像并转换为 YCrCb 色彩空间
            data_VIS_BGR = cv2.imread(os.path.join(test_folder, "vi", img_name))
            data_VIS_YCrCb = cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb)
            data_VIS_Y, data_VIS_Cr, data_VIS_Cb = cv2.split(data_VIS_YCrCb)
            data_VIS_Y = data_VIS_Y[np.newaxis, np.newaxis, ...] / 255.0

            # 将数据转换为 Tensor
            data_IR, data_VIS_Y = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS_Y)
            data_VIS_Y, data_IR = data_VIS_Y.cuda(), data_IR.cuda()

            # 进行融合
            feature_V_B, feature_V_D, feature_V = Encoder(data_VIS_Y)
            feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
            data_Fuse, _ = Decoder(data_VIS_Y, feature_F_B, feature_F_D)

            # 处理并保存图像
            fi = np.squeeze((data_Fuse * 255.0).cpu().numpy())
            fi = fi.astype(np.uint8)

            # 将融合后的 Y 通道与原始的 Cr 和 Cb 通道合并
            ycrcb_fi = np.dstack((fi, data_VIS_Cr, data_VIS_Cb))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)

            # 保存彩色融合图像
            img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)
print("测试结束")