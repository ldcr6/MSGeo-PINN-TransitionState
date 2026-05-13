# 🧪 基于AI的化学反应过渡态结构预测 - 完整指南

## 📋 项目概述

本项目是为"全球校园人工智能算法精英大赛·算法挑战赛·基于AI的化学反应过渡态结构预测"开发的完整解决方案。

### 🎯 核心功能
- **数据处理**: 支持 Transition1x HDF5 格式和传统 XYZ 格式
- **模型架构**: 基础GCN模型和高级能量感知模型
- **训练优化**: 超参数优化、早停、学习率调度
- **集成预测**: 多模型集成提高预测精度
- **全面评估**: 详细的性能分析和可视化

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone <your-repo-url>
cd transition-state-prediction

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备
```bash
# 自动下载 Transition1x 数据集
python download_data.py

# 或手动下载（如果自动下载失败）
# 参考 manual_download.md
```

### 3. 数据分析
```bash
# 分析数据集特征
python analyze_data.py
```

### 4. 模型训练
```bash
# 基础训练
python run_training.py

# 或使用高级模型
python run_training.py --config optimized_config.yaml
```

### 5. 模型预测
```bash
# 单模型预测
python run_prediction.py --model models/best_model.pth --test_data data/test

# 集成预测
python ensemble_predict.py --models models/model1.pth models/model2.pth --test_data data/test
```

## 📁 项目结构详解

```
├── src/                          # 核心源代码
│   ├── data_processing.py        # 数据预处理（支持多种格式）
│   ├── model.py                  # 基础GCN模型
│   ├── advanced_model.py         # 高级能量感知模型
│   ├── train.py                  # 训练逻辑
│   ├── predict.py                # 预测逻辑
│   └── utils.py                  # 工具函数
├── data/                         # 数据目录
│   ├── transition1x.h5          # Transition1x数据集
│   ├── train/                    # 传统格式训练数据
│   └── test/                     # 传统格式测试数据
├── models/                       # 保存的模型
├── results/                      # 预测结果
├── scripts/                      # 辅助脚本
│   ├── download_data.py          # 数据下载
│   ├── analyze_data.py           # 数据分析
│   ├── evaluate_model.py         # 模型评估
│   ├── ensemble_predict.py       # 集成预测
│   └── hyperparameter_tuning.py  # 超参数优化
├── config.yaml                   # 基础配置
├── optimized_config.yaml         # 优化后配置
└── requirements.txt              # 依赖包
```

## 🔧 高级功能

### 超参数优化
```bash
# 随机搜索
python hyperparameter_tuning.py --method random --trials 50

# 贝叶斯优化
python hyperparameter_tuning.py --method bayesian --trials 30

# 网格搜索
python hyperparameter_tuning.py --method grid
```

### 模型评估
```bash
# 全面评估
python evaluate_model.py --model models/best_model.pth --data data/transition1x.h5

# 限制样本数量
python evaluate_model.py --model models/best_model.pth --data data/test --max_samples 500
```

### 集成预测
```bash
# 多模型集成
python ensemble_predict.py \
    --models models/model1.pth models/model2.pth models/model3.pth \
    --test_data data/test \
    --output results/ensemble_predictions \
    --analyze_agreement
```

## 📊 性能指标

### 主要评估指标
- **RMSD (Root Mean Square Deviation)**: 几何误差
- **Success Rate**: RMSD ≤ 0.5 Å 的反应比例
- **Inference Time**: 单个反应的预测时间

### 预期性能
- **平均 RMSD**: < 0.3 Å
- **成功率**: > 70%
- **推理速度**: < 1 秒/反应

## 🎨 模型架构

### 基础模型 (TransitionStatePredictor)
- **编码器**: 4层 GCN + 批归一化
- **特征融合**: 多头注意力机制
- **预测器**: 3层全连接网络
- **损失函数**: 坐标损失 + 平滑性约束

### 高级模型 (EnergyAwarePredictor)
- **多尺度GNN**: GCN + GAT + TransformerConv
- **能量感知**: 集成反应能量信息
- **路径建模**: 双向LSTM建模反应路径
- **不确定性估计**: 预测置信度
- **高级损失**: 能量一致性 + 不确定性正则化

## 🔬 数据格式支持

### Transition1x HDF5 格式
```python
# 自动检测和加载
dataloader = transition1x.Dataloader('data/transition1x.h5', datasplit='train')
for reaction in dataloader:
    reactant = reaction['reactant']
    ts = reaction['transition_state']
    product = reaction['product']
```

### 传统 XYZ 格式
```
data/train/reaction_001/
├── reactant.xyz    # 反应物结构
├── product.xyz     # 产物结构
└── ts.xyz          # 过渡态结构
```

## ⚙️ 配置选项

### 模型配置
```yaml
model:
  hidden_dim: 128        # 隐藏层维度
  num_layers: 4          # GCN层数
  dropout: 0.1           # Dropout率
  learning_rate: 0.001   # 学习率
  batch_size: 32         # 批次大小
  use_energy_features: true  # 是否使用能量特征
```

### 训练配置
```yaml
training:
  device: "cuda"         # 设备选择
  epochs: 100            # 训练轮数
  early_stopping_patience: 10  # 早停耐心值
  save_path: "models/best_model.pth"
```

### 损失函数配置
```yaml
loss:
  coord_weight: 1.0      # 坐标损失权重
  smooth_weight: 0.1     # 平滑性损失权重
  energy_weight: 0.05    # 能量损失权重
  uncertainty_weight: 0.01  # 不确定性权重
```

## 🐛 故障排除

### 常见问题

#### 1. CUDA内存不足
```yaml
# 解决方案：减小批次大小和模型维度
model:
  batch_size: 16
  hidden_dim: 64
```

#### 2. 数据加载失败
```bash
# 检查数据路径和格式
python src/data_processing.py

# 重新下载数据
python download_data.py
```

#### 3. 模型训练不收敛
```bash
# 尝试超参数优化
python hyperparameter_tuning.py --method random --trials 20

# 或调整学习率
# config.yaml: learning_rate: 0.0001
```

#### 4. 预测精度不高
- 增加训练数据量
- 使用集成预测
- 调整损失函数权重
- 尝试高级模型架构

### 性能优化建议

#### 训练加速
- 使用GPU训练
- 减少数据加载时间
- 使用混合精度训练
- 并行数据处理

#### 推理优化
- 模型量化
- 批量预测
- ONNX导出
- TensorRT加速

## 📈 实验建议

### 基础实验流程
1. **数据探索**: 运行 `analyze_data.py` 了解数据特征
2. **基线训练**: 使用默认配置训练基础模型
3. **超参数优化**: 寻找最佳参数组合
4. **高级模型**: 尝试能量感知模型
5. **集成预测**: 组合多个模型提高性能
6. **全面评估**: 分析模型在不同条件下的表现

### 消融实验
- 不同GNN架构的对比
- 损失函数组件的影响
- 能量特征的作用
- 注意力机制的效果

### 泛化能力测试
- 不同反应类型的性能
- 不同分子大小的表现
- 跨数据集的迁移能力

## 🏆 竞赛提交

### 提交文件清单
```
submission/
├── src/                    # 完整源代码
├── models/                 # 训练好的模型
├── results/                # 预测结果
├── requirements.txt        # 依赖包
├── config.yaml            # 配置文件
├── README.md              # 项目说明
├── 技术方案报告.pdf        # 技术报告
└── run_prediction.py      # 预测脚本
```

### 结果格式
- **文件格式**: XYZ
- **命名规则**: `{reaction_id}_ts_pred.xyz`
- **坐标精度**: 保留6位小数
- **原子顺序**: 与输入保持一致

### 评分标准
- **RMSD (40%)**: 几何误差越小越好
- **成功率 (30%)**: RMSD ≤ 0.5 Å 的比例
- **推理时间 (10%)**: 预测效率
- **代码质量 (10%)**: 可读性和可复现性
- **报告质量 (10%)**: 技术创新和分析深度

## 🤝 贡献指南

### 代码规范
- 使用 Python 3.8+
- 遵循 PEP 8 代码风格
- 添加详细的文档字符串
- 包含单元测试

### 提交流程
1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 创建 Pull Request

## 📞 支持与联系

如果遇到问题或需要帮助：
1. 查看 `QUICKSTART.md` 快速开始指南
2. 参考 `manual_download.md` 数据下载指南
3. 检查 GitHub Issues
4. 联系项目维护者

---

🎉 **祝您在竞赛中取得优异成绩！** 🎉