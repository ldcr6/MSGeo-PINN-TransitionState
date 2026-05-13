# 依赖环境配置文件总览

本项目提供了多种依赖环境配置方式，适应不同的安装需求和使用场景。

## 📋 配置文件列表

### 🐍 Python包管理
| 文件 | 用途 | 适用场景 |
|------|------|----------|
| `requirements.txt` | pip依赖列表 | 标准Python环境 |
| `setup.py` | 传统安装脚本 | 兼容性安装 |
| `pyproject.toml` | 现代项目配置 | 新版Python项目 |

### 🔧 环境管理
| 文件 | 用途 | 适用场景 |
|------|------|----------|
| `environment.yml` | Conda环境配置 | Anaconda/Miniconda用户 |
| `Dockerfile` | Docker镜像构建 | 容器化部署 |
| `docker-compose.yml` | Docker服务编排 | 完整服务栈 |

### 🚀 安装脚本
| 文件 | 用途 | 适用场景 |
|------|------|----------|
| `install.sh` | Linux/macOS自动安装 | Unix系统 |
| `install.bat` | Windows自动安装 | Windows系统 |
| `check_environment.py` | 环境检查脚本 | 验证安装 |

## 🎯 核心依赖包

### 深度学习框架
```
torch>=2.0.0                 # PyTorch深度学习框架
torchvision>=0.15.0          # 计算机视觉工具
torchaudio>=2.0.0            # 音频处理工具
```

### 图神经网络
```
torch-geometric>=2.3.0       # 图神经网络库
torch-scatter>=2.1.0         # 图数据散射操作
torch-sparse>=0.6.0          # 稀疏张量操作
torch-cluster>=1.6.0         # 图聚类算法
```

### 科学计算
```
numpy>=1.21.0                # 数值计算基础
scipy>=1.9.0                 # 科学计算工具
pandas>=1.5.0                # 数据分析工具
scikit-learn>=1.1.0          # 机器学习工具
```

### 化学计算
```
ase>=3.22.0                  # 原子模拟环境
rdkit-pypi>=2022.9.1         # 化学信息学工具
openbabel-wheel>=3.1.1       # 分子格式转换
```

### 数据处理
```
h5py>=3.7.0                  # HDF5数据格式
pickle5>=0.0.11              # 序列化工具
joblib>=1.2.0                # 并行计算工具
```

### 可视化
```
matplotlib>=3.5.0            # 基础绘图
seaborn>=0.11.0              # 统计可视化
plotly>=5.10.0               # 交互式图表
```

### 工具库
```
tqdm>=4.64.0                 # 进度条显示
tensorboard>=2.10.0          # 训练监控
psutil>=5.9.0                # 系统监控
```

## 🔧 可选依赖包

### 开发工具
```
pytest>=7.0.0               # 单元测试
black>=22.0.0                # 代码格式化
flake8>=5.0.0                # 代码检查
jupyter>=1.0.0               # 交互式开发
```

### 分析工具
```
MDAnalysis>=2.3.0            # 分子动力学分析
nglview>=3.0.0               # 3D分子可视化
py3Dmol>=1.8.0               # 分子3D显示
```

### 优化工具
```
optuna>=3.0.0                # 超参数优化
rmsd>=1.5.0                  # RMSD计算
```

## 🚀 快速安装指南

### 方法1: 自动安装 (推荐)
```bash
# Windows
install.bat

# Linux/macOS
chmod +x install.sh
./install.sh
```

### 方法2: pip安装
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
pip install -e .
```

### 方法3: conda安装
```bash
# 创建环境
conda env create -f environment.yml
conda activate ts-prediction
```

### 方法4: Docker安装
```bash
# 使用Docker Compose
docker-compose up -d

# 或直接构建
docker build -t ts-prediction .
docker run --gpus all -it ts-prediction
```

## ✅ 环境验证

安装完成后运行环境检查：
```bash
python check_environment.py
```

预期输出：
```
🎉 环境检查完全通过！系统已准备就绪。

下一步:
1. 下载数据: python download_data.py
2. 训练模型: python train_advanced_model.py
3. 评估模型: python realistic_evaluation.py
```

