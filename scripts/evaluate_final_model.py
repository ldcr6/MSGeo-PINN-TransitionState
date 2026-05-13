#!/usr/bin/env python3
"""
最终模型评估脚本
按照评分标准计算RMSE和预测成功率
"""

import os
import sys
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append('src')
from advanced_ts_model import AdvancedTransitionStatePredictor
from utils import calculate_rmsd, batch_rmsd_calculation, calculate_success_rate


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
        # 加载模型
        self.model = self.load_model(model_path)
        
        # 评估结果
        self.evaluation_results = {}
        
    def load_model(self, model_path: str):
        """加载训练好的模型"""
        print(f"📂 加载模型: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 重建模型配置
        config = {
            'hidden_dim': 256,
            'num_layers': 4,
            'dropout': 0.1,
            'max_atoms': 50
        }
        
        # 创建模型
        model = AdvancedTransitionStatePredictor(config).to(self.device)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ 模型加载成功 (训练轮数: {checkpoint.get('epoch', 'Unknown')})")
        return model
    
    def load_test_data(self, data_path: str) -> List[Dict]:
        """加载测试数据"""
        print(f"📊 加载测试数据: {data_path}")
        
        with open(data_path, 'rb') as f:
            test_data = pickle.load(f)
        
        print(f"✅ 测试数据加载成功: {len(test_data)} 个样本")
        return test_data
    
    def predict_single_sample(self, sample: Dict) -> np.ndarray:
        """预测单个样本的过渡态坐标"""
        try:
            # 提取数据
            reactant_graph = sample['reactant_graph']
            product_graph = sample['product_graph']
            
            # 转换为张量
            r_x = torch.tensor(reactant_graph.x, dtype=torch.float32, device=self.device)
            r_edge_index = torch.tensor(reactant_graph.edge_index, dtype=torch.long, device=self.device)
            r_pos = torch.tensor(reactant_graph.pos, dtype=torch.float32, device=self.device)
            
            p_x = torch.tensor(product_graph.x, dtype=torch.float32, device=self.device)
            p_edge_index = torch.tensor(product_graph.edge_index, dtype=torch.long, device=self.device)
            p_pos = torch.tensor(product_graph.pos, dtype=torch.float32, device=self.device)
            
            # 创建图数据
            from torch_geometric.data import Data
            r_data = Data(x=r_x, edge_index=r_edge_index, pos=r_pos)
            p_data = Data(x=p_x, edge_index=p_edge_index, pos=p_pos)
            
            # 模型预测
            with torch.no_grad():
                predicted_coords, _ = self.model(r_data, p_data)
            
            return predicted_coords.cpu().numpy()
            
        except Exception as e:
            print(f"❌ 预测错误: {e}")
            return None
    
    def calculate_rmse(self, predicted_coords: List[np.ndarray], 
                      true_coords: List[np.ndarray]) -> float:
        """计算平均RMSE"""
        rmsd_values = []
        
        for pred, true in zip(predicted_coords, true_coords):
            if pred is not None and true is not None:
                try:
                    rmsd = calculate_rmsd(pred, true)
                    if not np.isnan(rmsd) and not np.isinf(rmsd):
                        rmsd_values.append(rmsd)
                except:
                    continue
        
        if not rmsd_values:
            return float('inf')
        
        return np.mean(rmsd_values)
    
    def calculate_success_rate_custom(self, predicted_coords: List[np.ndarray], 
                                    true_coords: List[np.ndarray], 
                                    threshold: float = 0.5) -> float:
        """计算预测成功率"""
        success_count = 0
        total_count = 0
        
        for pred, true in zip(predicted_coords, true_coords):
            if pred is not None and true is not None:
                try:
                    rmsd = calculate_rmsd(pred, true)
                    if not np.isnan(rmsd) and not np.isinf(rmsd):
                        total_count += 1
                        if rmsd <= threshold:
                            success_count += 1
                except:
                    continue
        
        if total_count == 0:
            return 0.0
        
        return (success_count / total_count) * 100
    
    def calculate_score(self, rmse: float, success_rate: float) -> Dict[str, float]:
        """根据评分标准计算分数"""
        # RMSE评分 (40分)
        if rmse >= 0.5:
            rmse_score = 0
        elif rmse <= 0.2:
            rmse_score = 40
        else:
            # 0.2 < RMSE < 0.5: 40 - ((RMSE - 0.2) / 0.3) * 40
            rmse_score = 40 - ((rmse - 0.2) / 0.3) * 40
        
        # 预测成功率评分 (30分)
        success_score = (success_rate / 100) * 30
        
        # 总分
        total_score = rmse_score + success_score
        
        return {
            'rmse_score': rmse_score,
            'success_score': success_score,
            'total_score': total_score
        }
    
    def evaluate_model(self, test_data_path: str, max_samples: int = None) -> Dict:
        """完整模型评估"""
        print("="*80)
        print("🚀 开始模型评估")
        print("="*80)
        
        # 加载测试数据
        test_data = self.load_test_data(test_data_path)
        
        if max_samples:
            test_data = test_data[:max_samples]
            print(f"📊 限制评估样本数: {max_samples}")
        
        # 预测所有样本
        predicted_coords = []
        true_coords = []
        
        print("🔮 开始预测...")
        for i, sample in enumerate(tqdm(test_data, desc="预测进度")):
            try:
                # 预测
                pred_coords = self.predict_single_sample(sample)
                
                # 真实坐标
                true_pos = sample['ts_pos']
                
                if pred_coords is not None:
                    predicted_coords.append(pred_coords)
                    true_coords.append(true_pos)
                
                # 每100个样本显示一次进度
                if (i + 1) % 100 == 0:
                    current_rmse = self.calculate_rmse(predicted_coords, true_coords)
                    current_success = self.calculate_success_rate_custom(predicted_coords, true_coords)
                    print(f"📈 当前进度 ({i+1}/{len(test_data)}): RMSE={current_rmse:.4f}, 成功率={current_success:.2f}%")
                    
            except Exception as e:
                print(f"❌ 样本 {i} 处理错误: {e}")
                continue
        
        print(f"✅ 预测完成: {len(predicted_coords)} 个有效样本")
        
        # 计算评估指标
        print("\n📊 计算评估指标...")
        
        # 1. 平均RMSE
        avg_rmse = self.calculate_rmse(predicted_coords, true_coords)
        
        # 2. 预测成功率 (RMSE ≤ 0.5)
        success_rate = self.calculate_success_rate_custom(predicted_coords, true_coords, threshold=0.5)
        
        # 3. 不同阈值的成功率
        success_rates = {}
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for threshold in thresholds:
            success_rates[threshold] = self.calculate_success_rate_custom(
                predicted_coords, true_coords, threshold
            )
        
        # 4. RMSE分布统计
        rmsd_values = []
        for pred, true in zip(predicted_coords, true_coords):
            try:
                rmsd = calculate_rmsd(pred, true)
                if not np.isnan(rmsd) and not np.isinf(rmsd):
                    rmsd_values.append(rmsd)
            except:
                continue
        
        rmse_stats = {
            'mean': np.mean(rmsd_values),
            'std': np.std(rmsd_values),
            'min': np.min(rmsd_values),
            'max': np.max(rmsd_values),
            'median': np.median(rmsd_values),
            'q25': np.percentile(rmsd_values, 25),
            'q75': np.percentile(rmsd_values, 75)
        }
        
        # 5. 计算评分
        scores = self.calculate_score(avg_rmse, success_rate)
        
        # 整理结果
        results = {
            'total_samples': len(test_data),
            'valid_predictions': len(predicted_coords),
            'avg_rmse': avg_rmse,
            'success_rate': success_rate,
            'success_rates_by_threshold': success_rates,
            'rmse_statistics': rmse_stats,
            'scores': scores,
            'rmsd_values': rmsd_values
        }
        
        self.evaluation_results = results
        return results
    
    def print_results(self, results: Dict):
        """打印评估结果"""
        print("\n" + "="*80)
        print("📊 模型评估结果")
        print("="*80)
        
        print(f"📈 总样本数: {results['total_samples']}")
        print(f"✅ 有效预测: {results['valid_predictions']}")
        print(f"📉 预测成功率: {results['valid_predictions']/results['total_samples']*100:.2f}%")
        
        print(f"\n🎯 关键指标:")
        print(f"  平均RMSE: {results['avg_rmse']:.4f} Å")
        print(f"  预测成功率 (≤0.5Å): {results['success_rate']:.2f}%")
        
        print(f"\n📊 RMSE统计:")
        stats = results['rmse_statistics']
        print(f"  均值: {stats['mean']:.4f} ± {stats['std']:.4f} Å")
        print(f"  中位数: {stats['median']:.4f} Å")
        print(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}] Å")
        print(f"  四分位数: [{stats['q25']:.4f}, {stats['q75']:.4f}] Å")
        
        print(f"\n🎯 不同阈值的成功率:")
        for threshold, rate in results['success_rates_by_threshold'].items():
            print(f"  ≤{threshold:.1f}Å: {rate:.2f}%")
        
        print(f"\n🏆 评分结果:")
        scores = results['scores']
        print(f"  RMSE评分 (40分): {scores['rmse_score']:.2f}")
        print(f"  成功率评分 (30分): {scores['success_score']:.2f}")
        print(f"  总分 (70分): {scores['total_score']:.2f}")
        
        # 评级
        total_score = scores['total_score']
        if total_score >= 60:
            grade = "优秀 🏆"
        elif total_score >= 50:
            grade = "良好 👍"
        elif total_score >= 40:
            grade = "中等 📈"
        elif total_score >= 30:
            grade = "及格 ✅"
        else:
            grade = "需改进 📉"
        
        print(f"  评级: {grade}")
    
    def plot_results(self, results: Dict, save_path: str = "evaluation_results.png"):
        """绘制评估结果图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. RMSE分布直方图
        rmsd_values = results['rmsd_values']
        axes[0, 0].hist(rmsd_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(results['avg_rmse'], color='red', linestyle='--', 
                          label=f'Mean: {results["avg_rmse"]:.4f}')
        axes[0, 0].axvline(results['rmse_statistics']['median'], color='green', linestyle='--', 
                          label=f'Median: {results["rmse_statistics"]["median"]:.4f}')
        axes[0, 0].set_xlabel('RMSE (Å)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('RMSE Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 成功率随阈值变化
        thresholds = list(results['success_rates_by_threshold'].keys())
        success_rates = list(results['success_rates_by_threshold'].values())
        axes[0, 1].plot(thresholds, success_rates, 'o-', color='green', linewidth=2, markersize=6)
        axes[0, 1].axhline(50, color='red', linestyle='--', alpha=0.7, label='50% Line')
        axes[0, 1].axvline(0.5, color='orange', linestyle='--', alpha=0.7, label='0.5Å Threshold')
        axes[0, 1].set_xlabel('RMSE Threshold (Å)')
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_title('Success Rate vs Threshold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 评分可视化
        scores = results['scores']
        score_names = ['RMSE Score\n(40分)', 'Success Score\n(30分)', 'Total Score\n(70分)']
        score_values = [scores['rmse_score'], scores['success_score'], scores['total_score']]
        max_scores = [40, 30, 70]
        
        x_pos = np.arange(len(score_names))
        bars = axes[1, 0].bar(x_pos, score_values, color=['lightcoral', 'lightgreen', 'gold'], 
                             alpha=0.8, edgecolor='black')
        
        # 添加满分线
        for i, max_score in enumerate(max_scores):
            axes[1, 0].axhline(max_score, xmin=i/len(score_names), xmax=(i+1)/len(score_names), 
                              color='red', linestyle='--', alpha=0.5)
        
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Scoring Results')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(score_names)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, score_values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 累积分布函数
        sorted_rmsd = np.sort(rmsd_values)
        cumulative = np.arange(1, len(sorted_rmsd) + 1) / len(sorted_rmsd) * 100
        axes[1, 1].plot(sorted_rmsd, cumulative, color='purple', linewidth=2)
        axes[1, 1].axvline(0.5, color='red', linestyle='--', alpha=0.7, 
                          label=f'0.5Å: {results["success_rate"]:.1f}%')
        axes[1, 1].axvline(results['avg_rmse'], color='orange', linestyle='--', alpha=0.7, 
                          label=f'Mean: {results["avg_rmse"]:.3f}Å')
        axes[1, 1].set_xlabel('RMSE (Å)')
        axes[1, 1].set_ylabel('Cumulative Percentage (%)')
        axes[1, 1].set_title('Cumulative Distribution of RMSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 评估图表已保存: {save_path}")
    
    def save_results(self, results: Dict, save_path: str = "evaluation_results.json"):
        """保存评估结果"""
        # 转换numpy数组为列表以便JSON序列化
        results_json = results.copy()
        results_json['rmsd_values'] = [float(x) for x in results['rmsd_values']]
        
        # 转换嵌套字典中的numpy类型
        for key, value in results_json['rmse_statistics'].items():
            results_json['rmse_statistics'][key] = float(value)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        
        print(f"💾 评估结果已保存: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估训练好的模型')
    parser.add_argument('--model_path', type=str, default='models/best_advanced_ts_model.pth',
                       help='模型文件路径')
    parser.add_argument('--test_data', type=str, default='data/processed/test_data.pkl',
                       help='测试数据路径')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='最大评估样本数 (None表示全部)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建评估器
    evaluator = ModelEvaluator(args.model_path)
    
    # 执行评估
    results = evaluator.evaluate_model(args.test_data, args.max_samples)
    
    # 打印结果
    evaluator.print_results(results)
    
    # 绘制图表
    plot_path = output_dir / "evaluation_plots.png"
    evaluator.plot_results(results, str(plot_path))
    
    # 保存结果
    json_path = output_dir / "evaluation_results.json"
    evaluator.save_results(results, str(json_path))
    
    print(f"\n🎉 评估完成! 结果保存在: {output_dir}")


if __name__ == "__main__":
    main()