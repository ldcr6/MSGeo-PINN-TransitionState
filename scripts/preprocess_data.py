#!/usr/bin/env python3
"""
预处理数据集 - 转换为Pickle格式以加速训练
"""
import os
import pickle
import torch
from pathlib import Path
from tqdm import tqdm
from ase.io import read
import numpy as np

def create_molecular_graph_cpu(xyz_file):
    """Create molecular graph (CPU version for preprocessing)"""
    atoms = read(xyz_file)
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    
    n_atoms = len(atomic_numbers)
    atom_type_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 14: 5, 15: 6, 16: 7, 17: 8, 35: 9, 53: 10}
    
    node_features = []
    for atom_num in atomic_numbers:
        feat = [0.0] * 11
        if atom_num in atom_type_map:
            feat[atom_type_map[atom_num]] = 1.0
        node_features.append(feat)
    
    x = torch.tensor(node_features, dtype=torch.float32)
    pos = torch.tensor(positions, dtype=torch.float32)
    
    # Create edges
    edge_indices = []
    cutoff = 3.5
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < cutoff:
                edge_indices.append([i, j])
                edge_indices.append([j, i])
    
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.tensor([[i, i] for i in range(n_atoms)], dtype=torch.long).t()
    
    return {
        'x': x,
        'edge_index': edge_index,
        'pos': pos,
        'atomic_numbers': atomic_numbers
    }

def preprocess_dataset(data_dir, output_file):
    """Preprocess dataset and save to pickle"""
    data_dir = Path(data_dir)
    samples = []
    
    rxn_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"Preprocessing {len(rxn_dirs)} reactions...")
    
    for rxn_dir in tqdm(rxn_dirs, desc="Processing"):
        r_file = rxn_dir / "r.xyz"
        p_file = rxn_dir / "p.xyz"
        ts_file = rxn_dir / "ts.xyz"
        
        if r_file.exists() and p_file.exists() and ts_file.exists():
            try:
                r_graph = create_molecular_graph_cpu(r_file)
                p_graph = create_molecular_graph_cpu(p_file)
                ts_atoms = read(ts_file)
                ts_coords = torch.tensor(ts_atoms.get_positions(), dtype=torch.float32)
                
                samples.append({
                    'r_graph': r_graph,
                    'p_graph': p_graph,
                    'ts_coords': ts_coords,
                    'rxn_id': rxn_dir.name
                })
            except Exception as e:
                print(f"Error processing {rxn_dir.name}: {e}")
    
    # Save
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(samples, f)
    
    print(f"\nPreprocessed {len(samples)} samples")
    print(f"Saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1e6:.1f} MB")
    
    return len(samples)

if __name__ == "__main__":
    data_dir = "data/tar_extracted/train_data"
    output_file = "data/preprocessed_train_data.pkl"
    
    print("="*80)
    print("Data Preprocessing")
    print("="*80)
    
    num_samples = preprocess_dataset(data_dir, output_file)
    
    print("\nPreprocessing complete!")
    print(f"Use this file for faster training: {output_file}")
    print("="*80)

