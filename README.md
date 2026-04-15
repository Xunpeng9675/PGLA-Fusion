PGLA-Fusion：渐进式图引导与局部注意力红外与可见光图像融合
本仓库是论文 PGLA-Fusion 的官方实现。该方法采用双流编解码器架构，通过语义特征编码器与渐进式纹理编码器，分别对基础语义特征和细节纹理特征进行渐进式融合，并引入多种损失函数引导网络学习。

📌 主要特点
双流上下文编码器/解码器：分别提取红外和可见光图像的多尺度特征并完成重建。

语义特征编码器：基于交叉注意力机制，融合基础语义特征。

渐进式纹理编码器：逐步融合细节纹理特征。

混合损失函数：包括 SSIM 损失、MSE 损失、TV 损失、MMD 损失（高斯核 + 拉普拉斯核）、InfoNCE 损失、相关系数损失、感知损失等。

🛠️ 环境配置
bash
# 创建虚拟环境（Python 3.8.10）
conda create -n PGLA python=3.8.10
conda activate PGLA

# 安装依赖
pip install -r requirements.txt

# 安装 PyTorch（示例：CUDA 11.1）
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
📂 数据准备
下载 MSRS 数据集
从 MSRS 官方仓库 下载，并放置于 ./dataSet4Training/。

预处理
运行预处理脚本生成 H5 格式的训练数据：

bash
python prepare_data.py
处理后文件位于 ./data/MSRS_train_imgsize_128_stride_200.h5。

注：预处理脚本 prepare_data.py 需自行准备（原项目包含），若无则需根据数据集格式编写。

🏊 训练
配置文件
所有训练参数均在 config.yaml 中定义，包括：

硬件设置（GPU ID、数据加载线程数）

训练超参数（batch size、epochs、学习率、权重衰减、梯度裁剪）

损失权重（SSIM、MSE、TV、decomp、NCE、CC、高斯核损失、拉普拉斯核损失）

数据路径（训练 H5 文件路径、VGG 感知损失权重路径）

模型参数（语义特征编码器维度、注意力头数、纹理编码器层数）

日志保存目录

开始训练
bash
python train.py
训练过程中会：

交替优化编码器、解码器及两个融合模块。

每个 epoch 结束后保存检查点至 ./checkPoints/。

控制台输出各项损失及特征相似度指标。

训练完成后，模型权重保存在 ./checkPoints/ 下，命名格式如：
1.0learning rate halvedPGLA_batch4_epoch100_WIN200_cuda0_epoch_99.pth

🏄 测试
测试前准备
将测试数据集（如 TNO、RoadScene、MSRS_test）按以下结构存放：

text
test_img/
├── TNO/
│   ├── ir/
│   └── vi/
├── RoadScene/
│   ├── ir/
│   └── vi/
└── MSRS_test/
    ├── ir/
    └── vi/
修改 config.yaml 中 test.ckpt_path 为训练好的模型路径。

运行测试
bash
python test.py
测试过程将：

对每个测试集逐对图像进行融合。

计算并输出 EN、SD、SF、SCD、VIF、Qabf、SSIM 等客观评价指标。

保存融合结果（默认彩色融合结果保存在 test_result/数据集名称/ 下，格式为 PNG）。

📁 项目结构
text
PGLA-Fusion/
├── .idea/                     # IDE 配置
├── checkPoints/               # 训练保存的模型权重
├── model/                     # 模型定义
│   ├── PGLA_Fusion.py         # 编码器、解码器、融合模块
│   ├── loss.py                # 融合损失、相关系数、InfoNCE 等
│   ├── kernel_loss.py         # 高斯核/拉普拉斯核 MMD 损失
│   └── __pycache__/
├── utils/                     # 工具函数
│   ├── dataset.py             # H5 数据集加载
│   ├── evaluator.py           # 评价指标计算
│   ├── imageUtils.py          # 图像读写与保存
│   └── ...
├── config.yaml                # 总配置文件
├── train.py                   # 训练脚本
├── test.py                    # 测试脚本
├── requirements.txt           # 依赖列表
├── README.md                  # 本文档
└── .gitignore
⚙️ 关键参数说明（config.yaml）
参数	说明	示例值
hardware.gpu_id	使用的 GPU 编号	0
train.batch_size	批大小	4
train.num_epochs	训练轮数	100
train.lr	初始学习率	1e-4
loss_weights.ssim	SSIM 损失权重	0.5
loss_weights.perceptual	感知损失权重	0.1
model.dim	语义特征编码器内部维度	64
model.num_heads	注意力头数	8
data.train_h5_path	训练数据 H5 文件路径	./data/MSRS_train_imgsize_128_stride_200.h5
