# 安装指南

本文档提供了基于AI的化学反应过渡态结构预测系统的详细安装指南。

## 📋 系统要求

### 最低要求
- **操作系统**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8 或更高版本
- **内存**: 8GB RAM (推荐16GB+)
- **存储**: 10GB 可用空间

### 推荐配置
- **GPU**: NVIDIA GPU (支持CUDA 11.8+)
- **显存**: 4GB+ (推荐8GB+)
- **CPU**: 多核处理器
- **内存**: 16GB+ RAM

## 🚀 快速安装

### 方法1: 自动安装脚本 (推荐)

#### Windows
```bash
# 下载项目
git clone https://github.com/example/ts-prediction.git
cd ts-prediction

# 运行安装脚本
install.bat
```

#### Linux/macOS
```bash
# 下载项目
git clone https://github.com/example/ts-prediction.git
cd ts-prediction

# 给脚本执行权限
chmod +x install.sh

# 运行安装脚本
./install.sh
```

### 方法2: 手动安装

#### 1. 创建虚拟环境
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

#### 2. 安装PyTorch
```bash
# GPU版本 (推荐)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 3. 安装PyTorch Geometric
```bash
# GPU版本
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# CPU版本
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

#### 4. 安装其他依赖
```bash
pip install -r requirements.txt
```

#### 5. 安装项目
```bash
pip install -e .
```

## 🐳 Docker安装

### 使用Docker Compose (推荐)
```bash
# 构建并启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 进入容器
docker-compose exec ts-prediction bash
```

### 使用Dockerfile
```bash
# 构建镜像
docker build -t ts-prediction .

# 运行容器
docker run --gpus all -it -v $(pwd)/data:/app/data ts-prediction
```

## 🔧 Conda安装

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate ts-prediction

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## ✅ 验证安装

运行以下命令验证安装是否成功：

```bash
# 检查核心依赖
python -c "
import torch
import torch_geometric
import numpy as np
import matplotlib.pyplot as plt
print('✅ 核心依赖导入成功')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'当前GPU: {torch.cuda.get_device_name(0)}')
print(f'PyTorch Geometric版本: {torch_geometric.__version__}')
"

# 运行GPU检查脚本
python check_gpu.py

# 测试数据加载
python -c "
import pickle
import os
if os.path.exists('data/processed/train_data.pkl'):
    with open('data/processed/train_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f'✅ 训练数据加载成功: {len(data)} 个样本')
else:
    print('⚠️ 训练数据不存在，请先运行 python download_data.py')
"
```

## 🔧 常见问题解决

### 1. CUDA版本不匹配
```bash
# 检查CUDA版本
nvidia-smi

# 安装对应版本的PyTorch
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. PyTorch Geometric安装失败
```bash
# 方法1: 使用conda
conda install pyg -c pyg

# 方法2: 从源码安装
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```

### 3. 内存不足
```bash
# 减少批次大小
python train_advanced_model.py --batch_size 2

# 使用CPU训练
python train_advanced_model.py --device cpu
```

### 4. 权限问题 (Linux/macOS)
```bash
# 给脚本执行权限
chmod +x install.sh
chmod +x src/*.py

# 使用sudo安装系统依赖
sudo apt-get update
sudo apt-get install python3-dev build-essential
```

### 5. 网络问题
```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 或使用阿里云镜像
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## 📦 可选组件安装

### 开发工具
```bash
pip install -e ".[dev]"
```

### 分析工具
```bash
pip install -e ".[analysis]"
```

### 完整安装
```bash
pip install -e ".[all]"
```

## 🔄 更新安装

```bash
# 更新代码
git pull origin main

# 更新依赖
pip install -r requirements.txt --upgrade

# 重新安装项目
pip install -e . --force-reinstall
```

## 🗑️ 卸载

```bash
# 删除虚拟环境
rm -rf venv

# 或删除conda环境
conda env remove -n ts-prediction

# 删除Docker容器和镜像
docker-compose down
docker rmi ts-prediction
```

## 📞 获取帮助

如果遇到安装问题，请：

1. 查看 [常见问题解决](#常见问题解决) 部分
2. 检查 [GitHub Issues](https://github.com/example/ts-prediction/issues)
3. 提交新的Issue并提供详细的错误信息
4. 联系开发团队: research@example.com

## 📚 下一步

安装完成后，请查看：
- [快速开始指南](QUICKSTART.md)
- [完整使用指南](COMPLETE_GUIDE.md)
- [API文档](docs/api.md)