#!/usr/bin/env python3
"""
集成预测脚本
使用多个模型进行集成预测以提高准确性
"""

import os
import sys
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse

sys.path.append('src')
from data_processing import MolecularDataProcessor
from model import TransitionStatePredictor
from advanced_model import EnergyAwarePredictor
from utils import create_xyz_file


class EnsemblePredictor:
    """集成预测器"""
    
    def __init__(self, model_paths: List[str], config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载多个模型
        self.models = []
        self.model_weights = []
        
        for model_path in model_paths:
            model, weight = self.load_model(model_path)
            if model is not None:
                self.models.append(model)
                self.model_weights.append(weight)
        
        if not self.models:
            raise ValueError("No valid models loaded!")
        
        print(f"Loaded {len(self.models)} models for ensemble prediction")
        
        # 归一化权重
        total_weight = sum(self.model_weights)
        self.model_weights = [w / total_weight for w in self.model_weights]
        
        # 数据处理器
        self.data_processor = MolecularDataProcessor(config_path)
    
    def load_model(self, model_path: str) -> Tuple[torch.nn.Module, float]:
        """加载单个模型"""
        try:
            if not os.path.exists(model_path):
                print(f"Warning: Model not found: {model_path}")
                return None, 0.0
            
            checkpoint = torch.load(model_path, map_location=self.device)
            model_config = checkpoint.get('config', {}).get('model', self.config['model'])
            
            # 根据配置选择模型类型
            if 'use_energy_features' in model_config:
                model = EnergyAwarePredictor(model_config).to(self.device)
            else:
                model = TransitionStatePredictor(model_config).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # 使用验证损失作为权重（损失越小权重越大）
            val_loss = checkpoint.get('best_val_loss', 1.0)
            weight = 1.0 / (val_loss + 1e-8)  # 避免除零
            
            print(f"Loaded model: {model_path} (weight: {weight:.4f})")
            return model, weight
            
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return None, 0.0
    
    def predict_single_reaction(self, reactant_graph, product_graph, energies=None) -> np.ndarray:
        """对单个反应进行集成预测"""
        predictions = []
        uncertainties = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                try:
                    if isinstance(model, EnergyAwarePredictor):
                        pred_coords, uncertainty = model(reactant_graph, product_graph, energies)
                        predictions.append(pred_coords.cpu().numpy())
                        uncertainties.append(uncertainty.cpu().numpy())
                    else:
                        pred_coords = model(reactant_graph, product_graph)
                        predictions.append(pred_coords.cpu().numpy())
                        uncertainties.append(np.ones_like(pred_coords.cpu().numpy()))
                        
                except Exception as e:
                    print(f"Error in model {i}: {e}")
                    continue
        
        if not predictions:
            raise RuntimeError("All models failed to predict")
        
        # 加权平均预测
        weighted_pred = np.zeros_like(predictions[0])
        total_weight = 0
        
        for pred, weight, uncertainty in zip(predictions, self.model_weights, uncertainties):
            # 使用不确定性进行加权（不确定性越小权重越大）
            uncertainty_weight = 1.0 / (np.mean(uncertainty) + 1e-8)
            combined_weight = weight * uncertainty_weight
            
            weighted_pred += pred * combined_weight
            total_weight += combined_weight
        
        weighted_pred /= total_weight
        
        return weighted_pred
    
    def predict_batch(self, test_data_path: str, output_path: str) -> Dict:
        """批量集成预测"""
        print("Loading test data...")
        
        # 加载测试数据
        if test_data_path.endswith('.h5'):
            test_data = self.data_processor.load_transition1x_dataset(test_data_path, 'test')
        else:
            test_data = self.data_processor.load_dataset(test_data_path, is_training=False)
        
        if not test_data:
            print("No test data found!")
            return {}
        
        print(f"Found {len(test_data)} test reactions")
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 预测统计
        successful_predictions = 0
        total_time = 0
        
        # 逐个预测
        for i, reaction_data in enumerate(tqdm(test_data, desc="Ensemble Predicting")):
            try:
                import time
                start_time = time.time()
                
                # 获取输入数据
                reactant_graph = reaction_data['reactant_graph'].to(self.device)
                product_graph = reaction_data['product_graph'].to(self.device)
                
                # 准备能量数据（如果可用）
                energies = None
                if 'activation_energy' in reaction_data:
                    energies = torch.tensor([
                        reaction_data.get('reactant_energy', 0),
                        reaction_data.get('ts_energy', 0),
                        reaction_data.get('product_energy', 0)
                    ]).unsqueeze(0).to(self.device)
                
                # 集成预测
                predicted_coords = self.predict_single_reaction(
                    reactant_graph, product_graph, energies
                )
                
                # 获取原子类型
                atomic_numbers = []
                for atom_feature in reactant_graph.x:
                    atom_idx = torch.argmax(atom_feature).item()
                    if atom_idx < len(list(self.data_processor.atom_types.values())):
                        atomic_numbers.append(list(self.data_processor.atom_types.values())[atom_idx])
                    else:
                        atomic_numbers.append(6)  # 默认为碳
                
                # 保存预测结果
                reaction_id = reaction_data.get('reaction_id', f'reaction_{i:04d}')
                output_file = os.path.join(output_path, f"{reaction_id}_ts_ensemble.xyz")
                create_xyz_file(atomic_numbers, predicted_coords, output_file)
                
                end_time = time.time()
                total_time += (end_time - start_time)
                successful_predictions += 1
                
            except Exception as e:
                print(f"Error predicting reaction {i}: {e}")
                continue
        
        # 统计结果
        avg_time = total_time / successful_predictions if successful_predictions > 0 else 0
        success_rate = successful_predictions / len(test_data) * 100
        
        results = {
            'total_reactions': len(test_data),
            'successful_predictions': successful_predictions,
            'success_rate': success_rate,
            'average_inference_time': avg_time,
            'ensemble_size': len(self.models)
        }
        
        print(f"\nEnsemble Prediction Results:")
        print(f"Models used: {len(self.models)}")
        print(f"Successful predictions: {successful_predictions}/{len(test_data)} ({success_rate:.1f}%)")
        print(f"Average inference time: {avg_time:.3f} seconds per reaction")
        print(f"Results saved to: {output_path}")
        
        return results
    
    def analyze_model_agreement(self, test_data_path: str, num_samples: int = 100) -> Dict:
        """分析模型间的一致性"""
        print(f"Analyzing model agreement on {num_samples} samples...")
        
        # 加载少量测试数据
        if test_data_path.endswith('.h5'):
            test_data = self.data_processor.load_transition1x_dataset(test_data_path, 'test')
        else:
            test_data = self.data_processor.load_dataset(test_data_path, is_training=False)
        
        test_data = test_data[:num_samples]
        
        agreement_stats = {
            'pairwise_rmsd': [],
            'prediction_variance': [],
            'model_correlations': []
        }
        
        for reaction_data in tqdm(test_data, desc="Analyzing Agreement"):
            try:
                reactant_graph = reaction_data['reactant_graph'].to(self.device)
                product_graph = reaction_data['product_graph'].to(self.device)
                
                # 获取所有模型的预测
                predictions = []
                with torch.no_grad():
                    for model in self.models:
                        if isinstance(model, EnergyAwarePredictor):
                            pred_coords, _ = model(reactant_graph, product_graph)
                        else:
                            pred_coords = model(reactant_graph, product_graph)
                        predictions.append(pred_coords.cpu().numpy())
                
                # 计算模型间的RMSD
                from utils import calculate_rmsd
                pairwise_rmsds = []
                for i in range(len(predictions)):
                    for j in range(i + 1, len(predictions)):
                        rmsd = calculate_rmsd(predictions[i], predictions[j])
                        pairwise_rmsds.append(rmsd)
                
                agreement_stats['pairwise_rmsd'].extend(pairwise_rmsds)
                
                # 计算预测方差
                pred_array = np.array(predictions)  # [n_models, n_atoms, 3]
                variance = np.var(pred_array, axis=0)  # [n_atoms, 3]
                agreement_stats['prediction_variance'].append(np.mean(variance))
                
            except Exception as e:
                print(f"Error analyzing reaction: {e}")
                continue
        
        # 汇总统计
        summary = {
            'average_pairwise_rmsd': np.mean(agreement_stats['pairwise_rmsd']),
            'rmsd_std': np.std(agreement_stats['pairwise_rmsd']),
            'average_prediction_variance': np.mean(agreement_stats['prediction_variance']),
            'high_disagreement_threshold': np.percentile(agreement_stats['pairwise_rmsd'], 90)
        }
        
        print(f"\nModel Agreement Analysis:")
        print(f"Average pairwise RMSD: {summary['average_pairwise_rmsd']:.4f} Å")
        print(f"RMSD standard deviation: {summary['rmsd_std']:.4f} Å")
        print(f"Average prediction variance: {summary['average_prediction_variance']:.6f}")
        print(f"High disagreement threshold (90th percentile): {summary['high_disagreement_threshold']:.4f} Å")
        
        return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Ensemble prediction for transition states')
    parser.add_argument('--models', nargs='+', required=True, 
                       help='Paths to trained models')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--output', type=str, default='results/ensemble_predictions',
                       help='Output directory for predictions')
    parser.add_argument('--analyze_agreement', action='store_true',
                       help='Analyze agreement between models')
    
    args = parser.parse_args()
    
    print("🔮 Starting Ensemble Prediction")
    print("=" * 50)
    print(f"Models: {args.models}")
    print(f"Test data: {args.test_data}")
    print(f"Output: {args.output}")
    print("=" * 50)
    
    # 检查模型文件
    valid_models = [m for m in args.models if os.path.exists(m)]
    if not valid_models:
        print("❌ No valid model files found!")
        return
    
    if len(valid_models) < len(args.models):
        print(f"⚠️  Only {len(valid_models)}/{len(args.models)} models found")
    
    try:
        # 创建集成预测器
        ensemble = EnsemblePredictor(valid_models)
        
        # 执行预测
        results = ensemble.predict_batch(args.test_data, args.output)
        
        # 分析模型一致性（如果需要）
        if args.analyze_agreement:
            agreement = ensemble.analyze_model_agreement(args.test_data)
        
        print("\n✅ Ensemble prediction completed!")
        
    except Exception as e:
        print(f"❌ Error during ensemble prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()