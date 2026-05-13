"""
过渡态结构预测脚本
"""

import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
import time
from ase import Atoms
from ase.io import write

from data_processing import MolecularDataProcessor
from model import TransitionStatePredictor


class TSPredictor:
    """过渡态预测器"""
    
    def __init__(self, model_path: str, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载模型
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # 初始化数据处理器
        self.data_processor = MolecularDataProcessor(config_path)
    
    def load_model(self, model_path: str):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint.get('config', {}).get('model', self.config['model'])
        
        model = TransitionStatePredictor(model_config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {model_path}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
        
        return model
    
    def predict_single_reaction(self, reactant_graph, product_graph):
        """预测单个反应的过渡态"""
        with torch.no_grad():
            # 将图数据移到设备上
            reactant_graph = reactant_graph.to(self.device)
            product_graph = product_graph.to(self.device)
            
            # 预测
            predicted_coords = self.model(reactant_graph, product_graph)
            
            return predicted_coords.cpu().numpy()
    
    def predict_batch(self, test_data_path: str, output_path: str):
        """批量预测测试集"""
        print("Loading test data...")
        test_data = self.data_processor.load_dataset(test_data_path, is_training=False)
        
        if len(test_data) == 0:
            print("No test data found!")
            return
        
        print(f"Found {len(test_data)} test reactions")
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 预测统计
        total_time = 0
        successful_predictions = 0
        
        # 逐个预测
        for i, reaction_data in enumerate(tqdm(test_data, desc="Predicting")):
            try:
                start_time = time.time()
                
                # 获取反应物和产物图
                reactant_graph = reaction_data['reactant_graph']
                product_graph = reaction_data['product_graph']
                
                # 预测过渡态坐标
                predicted_coords = self.predict_single_reaction(reactant_graph, product_graph)
                
                # 获取原子类型（从反应物获取）
                atomic_numbers = []
                for atom_feature in reactant_graph.x:
                    # 从one-hot编码恢复原子类型
                    atom_idx = torch.argmax(atom_feature).item()
                    atomic_numbers.append(list(self.data_processor.atom_types.values())[atom_idx])
                
                # 创建ASE Atoms对象
                atoms = Atoms(numbers=atomic_numbers, positions=predicted_coords)
                
                # 保存为XYZ文件
                reaction_id = reaction_data.get('reaction_id', f'reaction_{i:04d}')
                output_file = os.path.join(output_path, f"{reaction_id}_ts_pred.xyz")
                write(output_file, atoms)
                
                end_time = time.time()
                total_time += (end_time - start_time)
                successful_predictions += 1
                
            except Exception as e:
                print(f"Error predicting reaction {i}: {e}")
                continue
        
        # 输出统计信息
        avg_time = total_time / successful_predictions if successful_predictions > 0 else 0
        success_rate = successful_predictions / len(test_data) * 100
        
        print(f"\nPrediction completed!")
        print(f"Successful predictions: {successful_predictions}/{len(test_data)} ({success_rate:.1f}%)")
        print(f"Average inference time: {avg_time:.3f} seconds per reaction")
        print(f"Results saved to: {output_path}")
        
        return {
            'total_reactions': len(test_data),
            'successful_predictions': successful_predictions,
            'success_rate': success_rate,
            'average_inference_time': avg_time
        }
    
    def evaluate_predictions(self, predictions_path: str, ground_truth_path: str):
        """评估预测结果（如果有真实标签）"""
        try:
            from rmsd import rmsd, kabsch_rmsd
        except ImportError:
            print("rmsd package not installed. Install with: pip install rmsd")
            return
        
        print("Evaluating predictions...")
        
        pred_files = [f for f in os.listdir(predictions_path) if f.endswith('.xyz')]
        
        rmsd_values = []
        success_count = 0
        
        for pred_file in tqdm(pred_files, desc="Evaluating"):
            try:
                # 读取预测结果
                pred_path = os.path.join(predictions_path, pred_file)
                pred_atoms = self.data_processor.read_xyz_file(pred_path)
                
                # 查找对应的真实标签
                reaction_id = pred_file.replace('_ts_pred.xyz', '')
                true_path = os.path.join(ground_truth_path, reaction_id, 'ts.xyz')
                
                if os.path.exists(true_path):
                    true_atoms = self.data_processor.read_xyz_file(true_path)
                    
                    if pred_atoms[1] is not None and true_atoms[1] is not None:
                        # 计算RMSD
                        rmsd_val = kabsch_rmsd(pred_atoms[1], true_atoms[1])
                        rmsd_values.append(rmsd_val)
                        
                        if rmsd_val <= 0.5:  # 成功阈值
                            success_count += 1
                
            except Exception as e:
                print(f"Error evaluating {pred_file}: {e}")
                continue
        
        if rmsd_values:
            avg_rmsd = np.mean(rmsd_values)
            success_rate = success_count / len(rmsd_values) * 100
            
            print(f"\nEvaluation Results:")
            print(f"Average RMSD: {avg_rmsd:.4f} Å")
            print(f"Success Rate (RMSD ≤ 0.5 Å): {success_rate:.1f}%")
            print(f"Total evaluated: {len(rmsd_values)} reactions")
            
            return {
                'average_rmsd': avg_rmsd,
                'success_rate': success_rate,
                'total_evaluated': len(rmsd_values)
            }
        else:
            print("No valid predictions to evaluate")
            return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict transition states')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--test_data', type=str, default='data/test',
                       help='Path to test data')
    parser.add_argument('--output', type=str, default='results/predictions',
                       help='Output directory for predictions')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate predictions if ground truth available')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Please train the model first using: python src/train.py")
        return
    
    # 创建预测器
    predictor = TSPredictor(args.model)
    
    # 批量预测
    results = predictor.predict_batch(args.test_data, args.output)
    
    # 评估（如果需要）
    if args.evaluate:
        eval_results = predictor.evaluate_predictions(args.output, args.test_data)


if __name__ == "__main__":
    main()