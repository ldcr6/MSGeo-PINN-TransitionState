#!/usr/bin/env python3
"""
模型评估脚本
全面评估过渡态预测模型的性能
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import yaml

sys.path.append('src')
from data_processing import MolecularDataProcessor
from model import TransitionStatePredictor
from advanced_model import EnergyAwarePredictor
from utils import calculate_rmsd, calculate_success_rate


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载模型
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # 数据处理器
        self.data_processor = MolecularDataProcessor(config_path)
        
        # 评估结果
        self.results = {
            'rmsd_values': [],
            'success_rates': [],
            'inference_times': [],
            'atom_counts': [],
            'activation_energies': [],
            'reaction_types': [],
            'element_compositions': []
        }
    
    def load_model(self, model_path: str):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint.get('config', {}).get('model', self.config['model'])
        
        # 检查是否为高级模型
        if 'use_energy_features' in model_config:
            model = EnergyAwarePredictor(model_config).to(self.device)
        else:
            model = TransitionStatePredictor(model_config).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded: {type(model).__name__}")
        
        return model
    
    def evaluate_single_reaction(self, reaction_data: Dict) -> Dict:
        """评估单个反应"""
        import time
        
        try:
            start_time = time.time()
            
            # 获取数据
            reactant_graph = reaction_data['reactant_graph'].to(self.device)
            product_graph = reaction_data['product_graph'].to(self.device)
            true_coords = reaction_data['ts_pos'].numpy()
            
            # 预测
            with torch.no_grad():
                if isinstance(self.model, EnergyAwarePredictor):
                    # 高级模型可能需要能量信息
                    energies = None
                    if 'activation_energy' in reaction_data:
                        energies = torch.tensor([
                            reaction_data.get('reactant_energy', 0),
                            reaction_data.get('ts_energy', 0),
                            reaction_data.get('product_energy', 0)
                        ]).unsqueeze(0).to(self.device)
                    
                    pred_coords, uncertainty = self.model(reactant_graph, product_graph, energies)
                    pred_coords = pred_coords.cpu().numpy()
                else:
                    pred_coords = self.model(reactant_graph, product_graph).cpu().numpy()
            
            end_time = time.time()
            
            # 计算RMSD
            rmsd = calculate_rmsd(pred_coords, true_coords)
            
            # 收集结果
            result = {
                'rmsd': rmsd,
                'inference_time': end_time - start_time,
                'success': rmsd <= 0.5,
                'atom_count': len(true_coords),
                'activation_energy': reaction_data.get('activation_energy', 0),
                'reaction_type': reaction_data.get('rxn_name', 'unknown'),
                'formula': reaction_data.get('formula', ''),
                'predicted_coords': pred_coords,
                'true_coords': true_coords
            }
            
            return result
            
        except Exception as e:
            print(f"Error evaluating reaction: {e}")
            return None
    
    def evaluate_dataset(self, data_path: str, max_samples: int = None) -> Dict:
        """评估整个数据集"""
        print("Loading evaluation dataset...")
        
        # 加载数据
        if data_path.endswith('.h5'):
            eval_data = self.data_processor.load_transition1x_dataset(data_path, 'val')
        else:
            eval_data = self.data_processor.load_dataset(data_path, is_training=False)
        
        if not eval_data:
            print("No evaluation data found!")
            return {}
        
        if max_samples:
            eval_data = eval_data[:max_samples]
        
        print(f"Evaluating {len(eval_data)} reactions...")
        
        # 评估每个反应
        results = []
        for i, reaction_data in enumerate(tqdm(eval_data, desc="Evaluating")):
            result = self.evaluate_single_reaction(reaction_data)
            if result:
                results.append(result)
        
        # 汇总统计
        return self.compute_statistics(results)
    
    def compute_statistics(self, results: List[Dict]) -> Dict:
        """计算统计信息"""
        if not results:
            return {}
        
        # 提取数据
        rmsd_values = [r['rmsd'] for r in results]
        inference_times = [r['inference_time'] for r in results]
        success_flags = [r['success'] for r in results]
        atom_counts = [r['atom_count'] for r in results]
        activation_energies = [r['activation_energy'] for r in results]
        
        # 计算统计量
        stats = {
            'total_reactions': len(results),
            'average_rmsd': np.mean(rmsd_values),
            'median_rmsd': np.median(rmsd_values),
            'rmsd_std': np.std(rmsd_values),
            'success_rate': np.mean(success_flags) * 100,
            'average_inference_time': np.mean(inference_times),
            'rmsd_percentiles': {
                '25th': np.percentile(rmsd_values, 25),
                '75th': np.percentile(rmsd_values, 75),
                '90th': np.percentile(rmsd_values, 90),
                '95th': np.percentile(rmsd_values, 95)
            }
        }
        
        # 按分子大小分析
        stats['by_molecule_size'] = self.analyze_by_molecule_size(results)
        
        # 按激活能分析
        stats['by_activation_energy'] = self.analyze_by_activation_energy(results)
        
        # 按反应类型分析
        stats['by_reaction_type'] = self.analyze_by_reaction_type(results)
        
        return stats
    
    def analyze_by_molecule_size(self, results: List[Dict]) -> Dict:
        """按分子大小分析性能"""
        size_bins = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 100)]
        analysis = {}
        
        for min_atoms, max_atoms in size_bins:
            bin_name = f"{min_atoms}-{max_atoms} atoms"
            bin_results = [r for r in results 
                          if min_atoms <= r['atom_count'] < max_atoms]
            
            if bin_results:
                rmsd_values = [r['rmsd'] for r in bin_results]
                success_flags = [r['success'] for r in bin_results]
                
                analysis[bin_name] = {
                    'count': len(bin_results),
                    'average_rmsd': np.mean(rmsd_values),
                    'success_rate': np.mean(success_flags) * 100
                }
        
        return analysis
    
    def analyze_by_activation_energy(self, results: List[Dict]) -> Dict:
        """按激活能分析性能"""
        # 过滤有效的激活能数据
        valid_results = [r for r in results if r['activation_energy'] != 0]
        
        if not valid_results:
            return {}
        
        activation_energies = [r['activation_energy'] for r in valid_results]
        energy_bins = np.percentile(activation_energies, [0, 33, 67, 100])
        
        analysis = {}
        bin_names = ['Low Energy', 'Medium Energy', 'High Energy']
        
        for i, bin_name in enumerate(bin_names):
            min_energy = energy_bins[i]
            max_energy = energy_bins[i + 1]
            
            bin_results = [r for r in valid_results 
                          if min_energy <= r['activation_energy'] <= max_energy]
            
            if bin_results:
                rmsd_values = [r['rmsd'] for r in bin_results]
                success_flags = [r['success'] for r in bin_results]
                
                analysis[bin_name] = {
                    'count': len(bin_results),
                    'energy_range': f"{min_energy:.3f} - {max_energy:.3f} eV",
                    'average_rmsd': np.mean(rmsd_values),
                    'success_rate': np.mean(success_flags) * 100
                }
        
        return analysis
    
    def analyze_by_reaction_type(self, results: List[Dict]) -> Dict:
        """按反应类型分析性能"""
        from collections import defaultdict
        
        type_results = defaultdict(list)
        for result in results:
            rxn_type = result['reaction_type']
            type_results[rxn_type].append(result)
        
        analysis = {}
        for rxn_type, type_data in type_results.items():
            if len(type_data) >= 5:  # 至少5个样本
                rmsd_values = [r['rmsd'] for r in type_data]
                success_flags = [r['success'] for r in type_data]
                
                analysis[rxn_type] = {
                    'count': len(type_data),
                    'average_rmsd': np.mean(rmsd_values),
                    'success_rate': np.mean(success_flags) * 100
                }
        
        return analysis
    
    def create_evaluation_report(self, stats: Dict, output_path: str = "evaluation_report.html"):
        """创建评估报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Transition State Prediction Model Evaluation</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧪 Transition State Prediction Model Evaluation</h1>
                <p>Comprehensive performance analysis of the trained model</p>
            </div>
            
            <div class="section">
                <h2>📊 Overall Performance</h2>
                <div class="metric">Total Reactions Evaluated: {stats['total_reactions']:,}</div>
                <div class="metric">Average RMSD: {stats['average_rmsd']:.4f} Å</div>
                <div class="metric">Median RMSD: {stats['median_rmsd']:.4f} Å</div>
                <div class="metric">Success Rate (RMSD ≤ 0.5 Å): {stats['success_rate']:.1f}%</div>
                <div class="metric">Average Inference Time: {stats['average_inference_time']:.3f} seconds</div>
            </div>
            
            <div class="section">
                <h2>📈 RMSD Distribution</h2>
                <table>
                    <tr><th>Percentile</th><th>RMSD (Å)</th></tr>
                    <tr><td>25th</td><td>{stats['rmsd_percentiles']['25th']:.4f}</td></tr>
                    <tr><td>75th</td><td>{stats['rmsd_percentiles']['75th']:.4f}</td></tr>
                    <tr><td>90th</td><td>{stats['rmsd_percentiles']['90th']:.4f}</td></tr>
                    <tr><td>95th</td><td>{stats['rmsd_percentiles']['95th']:.4f}</td></tr>
                </table>
            </div>
        """
        
        # 按分子大小的分析
        if 'by_molecule_size' in stats:
            html_content += """
            <div class="section">
                <h2>🔬 Performance by Molecule Size</h2>
                <table>
                    <tr><th>Size Range</th><th>Count</th><th>Avg RMSD (Å)</th><th>Success Rate (%)</th></tr>
            """
            for size_range, data in stats['by_molecule_size'].items():
                html_content += f"""
                    <tr>
                        <td>{size_range}</td>
                        <td>{data['count']}</td>
                        <td>{data['average_rmsd']:.4f}</td>
                        <td>{data['success_rate']:.1f}</td>
                    </tr>
                """
            html_content += "</table></div>"
        
        # 按激活能的分析
        if 'by_activation_energy' in stats and stats['by_activation_energy']:
            html_content += """
            <div class="section">
                <h2>⚡ Performance by Activation Energy</h2>
                <table>
                    <tr><th>Energy Range</th><th>Count</th><th>Avg RMSD (Å)</th><th>Success Rate (%)</th></tr>
            """
            for energy_range, data in stats['by_activation_energy'].items():
                html_content += f"""
                    <tr>
                        <td>{energy_range}</td>
                        <td>{data['count']}</td>
                        <td>{data['average_rmsd']:.4f}</td>
                        <td>{data['success_rate']:.1f}</td>
                    </tr>
                """
            html_content += "</table></div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 Evaluation report saved to: {output_path}")
    
    def create_visualizations(self, stats: Dict, results: List[Dict]):
        """创建可视化图表"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 提取数据
        rmsd_values = [r['rmsd'] for r in results]
        inference_times = [r['inference_time'] for r in results]
        atom_counts = [r['atom_count'] for r in results]
        success_flags = [r['success'] for r in results]
        
        # 1. RMSD分布
        axes[0, 0].hist(rmsd_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Success Threshold')
        axes[0, 0].set_xlabel('RMSD (Å)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('RMSD Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 推理时间分布
        axes[0, 1].hist(inference_times, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_xlabel('Inference Time (s)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Inference Time Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. RMSD vs 分子大小
        axes[0, 2].scatter(atom_counts, rmsd_values, alpha=0.5, s=10)
        axes[0, 2].set_xlabel('Number of Atoms')
        axes[0, 2].set_ylabel('RMSD (Å)')
        axes[0, 2].set_title('RMSD vs Molecule Size')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 成功率 vs 分子大小
        size_bins = np.arange(0, max(atom_counts) + 5, 5)
        success_rates_by_size = []
        bin_centers = []
        
        for i in range(len(size_bins) - 1):
            mask = (np.array(atom_counts) >= size_bins[i]) & (np.array(atom_counts) < size_bins[i + 1])
            if np.sum(mask) > 0:
                success_rate = np.mean(np.array(success_flags)[mask]) * 100
                success_rates_by_size.append(success_rate)
                bin_centers.append((size_bins[i] + size_bins[i + 1]) / 2)
        
        axes[1, 0].bar(bin_centers, success_rates_by_size, width=4, alpha=0.7, color='lightgreen')
        axes[1, 0].set_xlabel('Number of Atoms')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].set_title('Success Rate by Molecule Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 累积分布函数
        sorted_rmsd = np.sort(rmsd_values)
        cumulative_prob = np.arange(1, len(sorted_rmsd) + 1) / len(sorted_rmsd)
        
        axes[1, 1].plot(sorted_rmsd, cumulative_prob, linewidth=2)
        axes[1, 1].axvline(0.5, color='red', linestyle='--', label='Success Threshold')
        axes[1, 1].set_xlabel('RMSD (Å)')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('RMSD Cumulative Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 性能摘要
        axes[1, 2].axis('off')
        summary_text = f"""
        Model Performance Summary
        
        Total Reactions: {len(results):,}
        Average RMSD: {np.mean(rmsd_values):.4f} Å
        Success Rate: {np.mean(success_flags) * 100:.1f}%
        Avg Inference: {np.mean(inference_times):.3f}s
        
        Best 10% RMSD: {np.percentile(rmsd_values, 10):.4f} Å
        Worst 10% RMSD: {np.percentile(rmsd_values, 90):.4f} Å
        """
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Evaluation visualizations saved as 'model_evaluation.png'")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate transition state prediction model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to evaluation data')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples to evaluate')
    parser.add_argument('--output', type=str, default='evaluation_report.html', help='Output report path')
    
    args = parser.parse_args()
    
    print("🧪 Starting Model Evaluation")
    print("=" * 50)
    
    # 创建评估器
    evaluator = ModelEvaluator(args.model)
    
    # 评估模型
    stats = evaluator.evaluate_dataset(args.data, args.max_samples)
    
    if stats:
        # 打印结果
        print(f"\n📊 Evaluation Results:")
        print(f"Total reactions: {stats['total_reactions']:,}")
        print(f"Average RMSD: {stats['average_rmsd']:.4f} Å")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Average inference time: {stats['average_inference_time']:.3f}s")
        
        # 生成报告
        evaluator.create_evaluation_report(stats, args.output)
        
        print(f"\n✅ Evaluation completed!")
    else:
        print("❌ Evaluation failed - no results generated")


if __name__ == "__main__":
    main()