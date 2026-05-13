"""
数据预处理模块
处理Transition1x数据集，提取特征用于模型训练
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import torch
from torch_geometric.data import Data, DataLoader
from ase import Atoms
from ase.io import read
import yaml
import h5py
from pathlib import Path


class MolecularDataProcessor:
    """分子数据处理器 - 针对Transition1x数据集优化"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Transition1x数据集中的原子类型（基于论文统计）
        self.atom_types = {
            'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Si': 14, 'P': 15, 'S': 16, 
            'Cl': 17, 'Br': 35, 'I': 53
        }
        
        # 数据集统计信息（基于Transition1x特征）
        self.max_atoms = 50  # 大多数反应的原子数限制
        self.bond_cutoff = 3.0  # 键连接的距离阈值
        
    def read_xyz_file(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        读取XYZ文件
        返回: (原子类型数组, 坐标数组)
        """
        try:
            atoms = read(filepath)
            atomic_numbers = atoms.get_atomic_numbers()
            positions = atoms.get_positions()
            return atomic_numbers, positions
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None, None
    
    def calculate_distance_matrix(self, positions: np.ndarray) -> np.ndarray:
        """计算原子间距离矩阵"""
        n_atoms = len(positions)
        dist_matrix = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
                
        return dist_matrix
    
    def create_molecular_graph(self, atomic_numbers: np.ndarray, 
                             positions: np.ndarray, 
                             cutoff: float = 3.0) -> Data:
        """
        创建分子图表示
        """
        n_atoms = len(atomic_numbers)
        
        # 节点特征：原子类型的one-hot编码
        node_features = []
        for atom_num in atomic_numbers:
            feature = [0] * len(self.atom_types)
            if atom_num in self.atom_types.values():
                idx = list(self.atom_types.values()).index(atom_num)
                feature[idx] = 1
            node_features.append(feature)
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # 边连接：基于距离阈值
        edge_indices = []
        edge_attrs = []
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < cutoff:
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])  # 无向图
                    edge_attrs.extend([dist, dist])
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        
        # 位置信息
        pos = torch.tensor(positions, dtype=torch.float)
        
        return Data(x=node_features, edge_index=edge_index, 
                   edge_attr=edge_attr, pos=pos)
    
    def process_reaction_data(self, reaction_folder: str) -> Dict:
        """
        处理单个反应的数据
        """
        reactant_path = os.path.join(reaction_folder, "reactant.xyz")
        product_path = os.path.join(reaction_folder, "product.xyz")
        ts_path = os.path.join(reaction_folder, "ts.xyz")
        
        # 读取结构文件
        r_atoms, r_pos = self.read_xyz_file(reactant_path)
        p_atoms, p_pos = self.read_xyz_file(product_path)
        
        if r_atoms is None or p_atoms is None:
            return None
        
        # 创建图表示
        reactant_graph = self.create_molecular_graph(r_atoms, r_pos)
        product_graph = self.create_molecular_graph(p_atoms, p_pos)
        
        result = {
            'reactant_graph': reactant_graph,
            'product_graph': product_graph,
            'reactant_pos': torch.tensor(r_pos, dtype=torch.float),
            'product_pos': torch.tensor(p_pos, dtype=torch.float)
        }
        
        # 如果是训练数据，包含过渡态
        if os.path.exists(ts_path):
            ts_atoms, ts_pos = self.read_xyz_file(ts_path)
            if ts_atoms is not None:
                result['ts_pos'] = torch.tensor(ts_pos, dtype=torch.float)
                result['ts_graph'] = self.create_molecular_graph(ts_atoms, ts_pos)
        
        return result
    
    def load_transition1x_dataset(self, h5_path: str, datasplit: str = 'train') -> List[Dict]:
        """
        加载Transition1x HDF5数据集
        Args:
            h5_path: HDF5文件路径
            datasplit: 'train', 'val', 'test'
        """
        try:
            import transition1x
        except ImportError:
            print("Error: transition1x package not installed!")
            print("Please run: python download_data.py")
            return []
        
        dataset = []
        print(f"Loading Transition1x dataset from {h5_path} (split: {datasplit})")
        
        try:
            # 使用only_final=True获取反应物、过渡态、产物
            dataloader = transition1x.Dataloader(h5_path, datasplit=datasplit, only_final=True)
            
            for i, reaction in enumerate(dataloader):
                try:
                    reaction_data = self.process_transition1x_reaction(reaction, i)
                    if reaction_data is not None:
                        dataset.append(reaction_data)
                        
                    if (i + 1) % 100 == 0:
                        print(f"Processed {i + 1} reactions...")
                        
                except Exception as e:
                    print(f"Error processing reaction {i}: {e}")
                    continue
            
            print(f"Successfully loaded {len(dataset)} reactions from {datasplit} split")
            return dataset
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
    
    def process_transition1x_reaction(self, reaction: Dict, reaction_id: int) -> Optional[Dict]:
        """
        处理单个Transition1x反应数据
        """
        try:
            # 提取反应物、过渡态、产物
            reactant = reaction['reactant']
            transition_state = reaction['transition_state']
            product = reaction['product']
            
            # 提取坐标和原子信息
            r_atoms = np.array(reactant['atomic_numbers'])
            r_pos = np.array(reactant['positions'])
            
            ts_atoms = np.array(transition_state['atomic_numbers'])
            ts_pos = np.array(transition_state['positions'])
            
            p_atoms = np.array(product['atomic_numbers'])
            p_pos = np.array(product['positions'])
            
            # 验证原子数一致性
            if not (len(r_atoms) == len(ts_atoms) == len(p_atoms)):
                print(f"Warning: Inconsistent atom counts in reaction {reaction_id}")
                return None
            
            # 创建图表示
            reactant_graph = self.create_molecular_graph(r_atoms, r_pos)
            product_graph = self.create_molecular_graph(p_atoms, p_pos)
            ts_graph = self.create_molecular_graph(ts_atoms, ts_pos)
            
            result = {
                'reaction_id': f"t1x_{reaction_id:06d}",
                'rxn_name': reaction.get('rxn', f'reaction_{reaction_id}'),
                'formula': reactant.get('formula', ''),
                'reactant_graph': reactant_graph,
                'product_graph': product_graph,
                'ts_graph': ts_graph,
                'reactant_pos': torch.tensor(r_pos, dtype=torch.float),
                'product_pos': torch.tensor(p_pos, dtype=torch.float),
                'ts_pos': torch.tensor(ts_pos, dtype=torch.float),
                'reactant_energy': reactant.get('wB97x_6-31G(d).energy', 0.0),
                'ts_energy': transition_state.get('wB97x_6-31G(d).energy', 0.0),
                'product_energy': product.get('wB97x_6-31G(d).energy', 0.0),
                'activation_energy': (transition_state.get('wB97x_6-31G(d).energy', 0.0) - 
                                    reactant.get('wB97x_6-31G(d).energy', 0.0))
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing Transition1x reaction {reaction_id}: {e}")
            return None
    
    def load_dataset(self, data_path: str, is_training: bool = True) -> List[Dict]:
        """
        加载数据集 - 支持多种格式
        """
        # 检查是否为HDF5文件
        if data_path.endswith('.h5') or data_path.endswith('.hdf5'):
            datasplit = 'train' if is_training else 'test'
            return self.load_transition1x_dataset(data_path, datasplit)
        
        # 检查是否包含HDF5文件的目录
        h5_files = list(Path(data_path).glob("*.h5"))
        if h5_files:
            print(f"Found HDF5 file: {h5_files[0]}")
            datasplit = 'train' if is_training else 'test'
            return self.load_transition1x_dataset(str(h5_files[0]), datasplit)
        
        # 传统XYZ文件夹格式
        return self.load_xyz_dataset(data_path, is_training)
    
    def load_xyz_dataset(self, data_path: str, is_training: bool = True) -> List[Dict]:
        """
        加载传统XYZ格式数据集
        """
        dataset = []
        
        if not os.path.exists(data_path):
            print(f"Data path {data_path} does not exist!")
            return dataset
        
        reaction_folders = [f for f in os.listdir(data_path) 
                          if os.path.isdir(os.path.join(data_path, f))]
        
        print(f"Processing {len(reaction_folders)} reactions...")
        
        for folder in reaction_folders:
            folder_path = os.path.join(data_path, folder)
            reaction_data = self.process_reaction_data(folder_path)
            
            if reaction_data is not None:
                reaction_data['reaction_id'] = folder
                dataset.append(reaction_data)
        
        print(f"Successfully processed {len(dataset)} reactions")
        return dataset


def main():
    """测试数据处理功能"""
    processor = MolecularDataProcessor()
    
    # 创建示例数据目录结构
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)
    
    print("Data processor initialized successfully!")
    
    # 检查是否有Transition1x数据
    h5_files = list(Path("data").glob("*.h5"))
    if h5_files:
        print(f"Found Transition1x HDF5 file: {h5_files[0]}")
        
        # 测试加载少量数据
        try:
            import transition1x
            dataloader = transition1x.Dataloader(str(h5_files[0]), datasplit='train', only_final=True)
            sample = next(iter(dataloader))
            print(f"Sample reaction: {sample.get('rxn', 'unknown')}")
            print(f"Reactant formula: {sample['reactant']['formula']}")
            print(f"Number of atoms: {len(sample['reactant']['atomic_numbers'])}")
            
        except ImportError:
            print("transition1x package not installed. Run: python download_data.py")
        except Exception as e:
            print(f"Error testing data: {e}")
    else:
        print("No HDF5 files found. For Transition1x dataset, run: python download_data.py")
        print("For XYZ format, place files in:")
        print("data/train/reaction_001/")
        print("  ├── reactant.xyz")
        print("  ├── product.xyz")
        print("  └── ts.xyz")


if __name__ == "__main__":
    main()