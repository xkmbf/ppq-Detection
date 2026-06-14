# 乒乓球目标检测 (Table Tennis Ball Detection)

基于 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) 框架的乒乓球目标检测项目。使用轻量级 **PicoDet** 模型对比赛视频帧中的乒乓球进行实时检测。

## 项目结构

```
PaddleDetection/
├── configs/                    # 模型配置文件
│   ├── picodet/                # PicoDet 模型配置
│   │   ├── _base_/             # 基础配置（骨架、优化器、数据读取器）
│   │   │   ├── picodet_v2.yml          # 模型架构定义
│   │   │   ├── optimizer_300e.yml      # 优化器与学习率调度
│   │   │   └── picodet_640_reader.yml  # 数据预处理与增强
│   │   └── ppq.yml             # ★ 乒乓球检测训练配置
│   ├── datasets/               # 数据集定义配置
│   ├── runtime.yml             # 运行时通用配置
│   └── ...                     # 其他模型配置
├── dataset/                    # 数据集目录
│   ├── train/                  # 训练集（10386 张图片）
│   │   ├── JPEGImages/         # 训练图片
│   │   ├── Annotations/        # VOC XML 标注
│   │   ├── labels.txt          # 类别标签：pingpang（乒乓球）
│   │   └── train_list.txt      # 训练图片-标注映射列表
│   └── val/                    # 验证集（1588 张图片）
│       ├── JPEGImages/         # 验证图片
│       ├── Annotations/        # VOC XML 标注
│       ├── labels.txt          # 类别标签
│       └── val_list.txt        # 验证图片-标注映射列表
├── ppdet/                      # PaddleDetection 核心库
│   ├── core/                   # 核心模块（工作空间、配置管理）
│   │   ├── workspace.py        # 全局注册与组件管理
│   │   └── config/             # 配置解析
│   ├── data/                   # 数据模块
│   │   ├── reader.py           # 数据读取器
│   │   ├── source/             # 数据集类（VOC、COCO等）
│   │   │   └── voc.py          # VOC 数据集读取
│   │   └── transform/          # 数据增强与预处理
│   ├── engine/                 # 训练引擎
│   │   ├── trainer.py          # 标准训练器
│   │   ├── callbacks.py        # 训练回调
│   │   └── export_utils.py     # 模型导出
│   ├── modeling/               # 模型组件
│   │   ├── backbones/          # 骨干网络（LCNet 等）
│   │   │   └── lcnet.py        # ★ LCNet 轻量级骨干网络
│   │   ├── necks/              # 特征融合颈部
│   │   │   └── lc_pan.py       # ★ LCPAN 轻量级特征金字塔
│   │   ├── heads/              # 检测头
│   │   │   └── pico_head.py    # ★ PicoHead / PicoHeadV2
│   │   ├── losses/             # 损失函数
│   │   │   └── varifocal_loss.py # Varifocal Loss
│   │   ├── architectures/      # 完整模型架构
│   │   └── assigners/          # 标签分配策略
│   ├── optimizer/              # 优化器与 EMA
│   │   ├── optimizer.py        # 优化器构建
│   │   └── ema.py              # 指数移动平均
│   ├── metrics/                # 评估指标
│   │   ├── metrics.py          # 指标计算入口
│   │   └── map_utils.py        # mAP 计算
│   └── utils/                  # 工具函数
│       ├── cli.py              # 命令行参数解析
│       ├── logger.py           # 日志
│       ├── checkpoint.py       # 模型保存与加载
│       └── visualizer.py       # 结果可视化
├── tools/                      # 训练与推理脚本
│   ├── train.py                # ★ 训练脚本
│   ├── eval.py                 # ★ 评估脚本
│   ├── infer.py                # 推理脚本
│   └── export_model.py         # 模型导出脚本
├── work/                       # 输出目录（日志、模型权重）
├── ppq.yml                     # 顶层入口配置（指向 configs 中的子配置）
├── voc_ppq.yml                 # 数据集定义配置
├── setup.py                    # 包安装脚本
└── requirements.txt            # Python 依赖
```

## 关键模块说明

### 模型架构 — PicoDet

PicoDet 是 PaddleDetection 推出的轻量级目标检测模型，特别适合移动端和边缘设备部署。本项目的模型结构为：

| 组件 | 模块 | 说明 |
|------|------|------|
| **Backbone** | `LCNet` | 轻量级 CNN 骨干网络，使用深度可分离卷积，scale=2.0，输出 [C2, C3, C4, C5] 四个特征层 |
| **Neck** | `LCPAN` | 轻量级路径聚合网络（Path Aggregation Network），通道数 160，融合多尺度特征 |
| **Head** | `PicoHeadV2` | 改进的轻量检测头，共享分类与回归分支，带 SE 注意力机制，基于 Anchor-free GFL 范式 |

**核心损失函数**：
- **VarifocalLoss** — 分类损失，对 IoU 加权的正样本进行聚焦
- **GIoULoss** — 边界框回归损失
- **DistributionFocalLoss (DFL)** — 分布式焦点损失，提升框回归精度

**标签分配**：
- 初始阶段使用 **ATSSAssigner**（静态分配）
- 后期切换为 **TaskAlignedAssigner**（任务对齐动态分配，topk=13）

### 数据集

| 项目 | 说明 |
|------|------|
| 格式 | Pascal VOC (XML 标注) |
| 类别 | `pingpang`（乒乓球） |
| 训练集 | 10,386 张图片 |
| 验证集 | 1,588 张图片 |
| 评估指标 | VOC mAP (11-point) |

## 环境依赖

- Python 3.10
- PaddlePaddle >= 2.3.0
- 其他依赖见 `requirements.txt`

```bash
python setup.py install
pip install -r requirements.txt

// 安装cuda 13.0版本
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
```

## 快速开始

### 训练

使用 PicoDet 配置训练乒乓球检测模型：

```bash
python tools/train.py \
    -c configs/picodet/ppq.yml \
    --use_vdl=true \
    --vdl_log_dir=work/vdl_dir \
    --eval \
    -o save_dir=./work/model
```

**参数说明**：
| 参数 | 说明 |
|------|------|
| `-c` | 指定配置文件路径 |
| `--use_vdl` | 启用 VisualDL 训练可视化 |
| `--vdl_log_dir` | VisualDL 日志输出目录 |
| `--eval` | 训练过程中自动评估 |
| `-o` | 覆盖配置项，这里指定模型保存路径 |

训练启动后，可通过 VisualDL 查看训练曲线：

```bash
visualdl --logdir work/vdl_dir
```

### 验证

在验证集上评估训练好的模型：

```bash
python tools/eval.py \
    -c configs/picodet/ppq.yml \
    -o weights=work/model/best_model/model.pdparams
```

### 推理

使用训练好的模型对图片或视频进行推理：

```bash
python tools/infer.py \
    -c configs/picodet/ppq.yml \
    --infer_img=/path/to/image.jpg \
    -o weights=work/model/best_model/model.pdparams \
    --output_dir=/path/to/image_infer.jpg
```

### 模型导出

将训练好的模型导出为推理部署格式：

```bash
python tools/export_model.py \
    -c configs/picodet/ppq.yml \
    -o weights=work/model/best_model/model.pdparams \
    --output_dir=work/export_model
```
