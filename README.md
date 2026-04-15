# PGLA-Fusion：渐进式图引导与局部注意力红外与可见光图像融合
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

⚙️ 环境配置
执行以下命令安装指定版本的 PyTorch 及相关依赖（适配 CUDA 11.1）：
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
其他依赖请参考项目根目录下的 requirements.txt 文件，执行 pip install -r requirements.txt 完成安装。

1. 下载数据集
从 MSRS 官方仓库 下载 MSRS 训练数据集，下载后放置于项目目录下的 ./dataSet4Training/ 文件夹中。

运行预处理脚本，将原始数据集转换为 H5 格式的训练数据，方便后续高效加载：
python prepare_data.py
预处理完成后，生成的 H5 文件将位于 ./data/MSRS_train_imgsize_128_stride_200.h5。
注意：预处理脚本 prepare_data.py 需自行准备（原项目包含该脚本），若未获取到，需根据数据集格式自行编写。
🏊 训练流程
1. 配置文件设置
所有训练相关参数均在根目录的 config.yaml 文件中定义，关键配置项如下（详细参数见下文说明）：
- 硬件设置：GPU ID、数据加载线程数
- 训练超参数：批大小、训练轮数、学习率、权重衰减、梯度裁剪
- 损失权重：SSIM、MSE、TV、decomp、NCE、CC、高斯核损失、拉普拉斯核损失
- 数据路径：训练 H5 文件路径、VGG 感知损失权重路径
- 模型参数：语义特征编码器维度、注意力头数、纹理编码器层数
- 日志保存目录
2. 开始训练
配置完成后，执行以下命令启动训练：
python train.py
3. 训练过程说明
- 训练过程中，将交替优化编码器、解码器及两个融合模块，确保模型性能稳定提升。
- 每个 epoch 结束后，会自动保存模型检查点至 ./checkPoints/ 目录。
- 控制台会实时输出各项损失值及特征相似度指标，便于监控训练效果。
- 训练完成后，最终模型权重保存在 ./checkPoints/ 下，命名格式示例：1.0learning rate halvedPGLA_batch4_epoch100_WIN200_cuda0_epoch_99.pth
- 
🏄 测试流程

1. 测试前准备
（1）数据集结构
将测试数据集（如 TNO、RoadScene、MSRS_test）按以下目录结构存放至项目中：
test_img/
├── TNO/
│   ├── ir/    # 红外图像目录
│   └── vi/    # 可见光图像目录
├── RoadScene/
│   ├── ir/
│   └── vi/
└── MSRS_test/
    ├── ir/
    └── vi/
（2）模型路径配置
修改 config.yaml 文件中的 test.ckpt_path 字段，指定训练好的模型权重路径（即 ./checkPoints/ 下的 .pth 文件）。
2. 运行测试
执行以下命令启动测试：
python test.py
3. 测试结果说明
- 测试过程中，将对每个测试集的红外-可见光图像逐对进行融合。
- 自动计算并输出 EN（熵）、SD（标准差）、SF（空间频率）、SCD（空间相关性）、VIF（视觉信息保真度）、Qabf（融合质量评价）、SSIM（结构相似性）等客观评价指标。
- 融合结果默认以 PNG 格式保存至 test_result/数据集名称/ 目录下（默认保存彩色融合结果）。

📁 项目结构
PGLA-Fusion/
├── .idea/          # IDE 配置文件（可忽略）
├── checkPoints/    # 训练保存的模型权重
├── model/          # 模型核心代码
│   ├── PGLA_Fusion.py  # 编码器、解码器、融合模块定义
│   ├── loss.py         # 融合损失、相关系数、InfoNCE 等损失函数
│   ├── kernel_loss.py  # 高斯核/拉普拉斯核 MMD 损失
│   └── __pycache__     # 编译缓存（可忽略）
├── utils/          # 工具函数目录
│   ├── dataset.py      # H5 数据集加载工具
│   ├── evaluator.py    # 客观评价指标计算工具
│   ├── imageUtils.py   # 图像读写、保存工具
│   └── ...             # 其他辅助工具
├── config.yaml     # 总配置文件（训练/测试参数）
├── train.py        # 训练主脚本
├── test.py         # 测试主脚本
├── requirements.txt # 项目依赖列表
├── README.md       # 项目说明文档（本文档）
└── .gitignore      # Git 忽略文件

⚙️ 关键参数说明（config.yaml）

参数

说明

示例值

hardware.gpu_id
使用的 GPU 编号（单卡填0，多卡填对应编号）
0
train.batch_size
训练批大小（根据 GPU 显存调整）
4
train.num_epochs
训练总轮数
100
train.lr
初始学习率
1e-4
loss_weights.ssim
SSIM 损失的权重
0.5
loss_weights.perceptual
VGG 感知损失的权重
0.1
model.dim
语义特征编码器内部维度
64
model.num_heads
注意力机制的头数
8
data.train_h5_path
训练用 H5 数据文件的路径
./data/MSRS_train_imgsize_128_stride_200.h5

📝 注意事项
- 确保 CUDA 版本与 PyTorch 版本匹配（本文档使用 CUDA 11.1，对应 PyTorch 1.8.1+cu111）。
- 预处理脚本 prepare_data.py 需与数据集格式匹配，若数据集结构变更，需修改脚本。
- 训练时若出现显存不足，可减小 train.batch_size 或调整图像输入尺寸。
- 测试时需确保 test.ckpt_path 路径正确，否则会导致模型加载失败。
