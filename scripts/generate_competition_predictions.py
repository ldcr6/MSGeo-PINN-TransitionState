#!/usr/bin/env python3
"""
Competition Prediction Script
Generate predictions for all 500 test reactions in the required format
"""

import os
import sys
import torch
import numpy as np
from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm
from ase.io import read

# Suppress OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add src to path
sys.path.append('src')
from advanced_ts_model import AdvancedTransitionStatePredictor


def load_model(model_path, device='cuda'):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        config = checkpoint['config']['model']
    else:
        raise ValueError("No config found in checkpoint")
    
    model = AdvancedTransitionStatePredictor(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def create_molecular_graph(xyz_file, device='cuda'):
    """Create molecular graph from XYZ file"""
    atoms = read(xyz_file)
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()
    
    n_atoms = len(atomic_numbers)
    
    # One-hot encoding for atom types
    atom_type_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 14: 5, 15: 6, 16: 7, 17: 8, 35: 9, 53: 10}
    
    node_features = []
    for atom_num in atomic_numbers:
        feat = [0.0] * 11
        if atom_num in atom_type_map:
            feat[atom_type_map[atom_num]] = 1.0
        node_features.append(feat)
    
    x = torch.tensor(node_features, dtype=torch.float32)
    pos = torch.tensor(positions, dtype=torch.float32)
    
    # Create edges (cutoff = 3.0 Angstrom)
    edge_indices = []
    cutoff = 3.0
    
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
    
    data = Data(x=x, edge_index=edge_index, pos=pos)
    return data.to(device), atomic_numbers


def save_prediction(coords, atomic_numbers, output_path):
    """Save predicted coordinates to XYZ file as TS_pred.xyz"""
    atom_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 
                   14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
    
    with open(output_path, 'w') as f:
        f.write(f"{len(coords)}\n")
        f.write("Predicted Transition State\n")
        for atom_num, (x, y, z) in zip(atomic_numbers, coords):
            symbol = atom_symbols.get(atom_num, 'X')
            f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")


def main():
    """Main prediction function"""
    print("="*80)
    print("Competition Prediction - Generate TS_pred.xyz for all test reactions")
    print("="*80)
    
    # Configuration
    model_path = 'models/best_advanced_ts_model.pth'
    test_data_dir = '初赛数据/test_data_1'
    output_dir = 'competition_submission'
    
    # Check paths
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return
    
    if not os.path.exists(test_data_dir):
        print(f"[ERROR] Test data not found: {test_data_dir}")
        return
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = load_model(model_path, device)
    print(f"Model loaded successfully")
    
    # Get all reaction directories
    reaction_dirs = sorted([d for d in os.listdir(test_data_dir) 
                           if os.path.isdir(os.path.join(test_data_dir, d))])
    
    print(f"\nFound {len(reaction_dirs)} test reactions")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate predictions
    print(f"\nGenerating predictions...")
    success_count = 0
    failed_reactions = []
    
    for rxn_dir in tqdm(reaction_dirs, desc="Processing"):
        rxn_path = os.path.join(test_data_dir, rxn_dir)
        
        # Input files (RS.xyz = reactant, PS.xyz = product)
        r_file = os.path.join(rxn_path, "RS.xyz")
        p_file = os.path.join(rxn_path, "PS.xyz")
        
        if not os.path.exists(r_file) or not os.path.exists(p_file):
            print(f"\n[SKIP] Missing files in {rxn_dir}")
            failed_reactions.append(rxn_dir)
            continue
        
        try:
            # Create graphs
            r_graph, r_atoms = create_molecular_graph(r_file, device)
            p_graph, p_atoms = create_molecular_graph(p_file, device)
            
            # Predict
            with torch.no_grad():
                pred_coords, _ = model(r_graph, p_graph)
            
            # Convert to numpy
            pred_coords_np = pred_coords.cpu().numpy()
            
            # Create output directory for this reaction
            rxn_output_dir = os.path.join(output_dir, rxn_dir)
            os.makedirs(rxn_output_dir, exist_ok=True)
            
            # Save as TS_pred.xyz (required format)
            output_file = os.path.join(rxn_output_dir, "TS_pred.xyz")
            save_prediction(pred_coords_np, r_atoms, output_file)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n[ERROR] Failed for {rxn_dir}: {e}")
            failed_reactions.append(rxn_dir)
            continue
    
    # Summary
    print(f"\n{'='*80}")
    print("Prediction Summary")
    print(f"{'='*80}")
    print(f"Total reactions: {len(reaction_dirs)}")
    print(f"Successful predictions: {success_count}")
    print(f"Failed predictions: {len(failed_reactions)}")
    
    if failed_reactions:
        print(f"\nFailed reactions:")
        for rxn in failed_reactions[:10]:  # Show first 10
            print(f"  - {rxn}")
        if len(failed_reactions) > 10:
            print(f"  ... and {len(failed_reactions) - 10} more")
    
    print(f"\nPredictions saved to: {output_dir}/")
    print(f"Each reaction has: {output_dir}/<rxn_name>/TS_pred.xyz")
    print(f"\n{'='*80}")
    print("Ready for competition submission!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
