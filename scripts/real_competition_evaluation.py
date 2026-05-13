#!/usr/bin/env python3
"""
REAL Competition Evaluation
Actually load and run the trained model: models/best_advanced_ts_model.pth
"""

import os
import sys
import torch
import numpy as np
from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm
import json
import time
from ase.io import read
from rmsd import kabsch_rmsd, get_coordinates_xyz

# Add src to path
sys.path.append('src')
from advanced_ts_model import AdvancedTransitionStatePredictor


def calculate_rmsd_with_alignment(pred_xyz_path, true_xyz_path):
    """Calculate RMSD using Kabsch algorithm (official method)"""
    try:
        _, P = get_coordinates_xyz(pred_xyz_path)
        _, Q = get_coordinates_xyz(true_xyz_path)
        
        if not isinstance(P, np.ndarray) or not isinstance(Q, np.ndarray):
            raise ValueError("Invalid coordinate data")
        if P.size == 0 or Q.size == 0:
            raise ValueError("Empty coordinates")
        if P.shape[1] != 3 or Q.shape[1] != 3:
            raise ValueError(f"Invalid shape: {P.shape} vs {Q.shape}")
        if P.shape[0] != Q.shape[0]:
            raise ValueError(f"Atom count mismatch: {P.shape[0]} vs {Q.shape[0]}")
        
        P = P.astype(np.float64)
        Q = Q.astype(np.float64)
        
        rmsd_value = kabsch_rmsd(P, Q)
        return rmsd_value, True
    except Exception as e:
        print(f"[ERROR] RMSD calculation failed: {e}")
        return float('inf'), False


def calculate_rmsd_score(rmsd_value):
    """Official RMSD scoring: max 40 points"""
    if rmsd_value <= 0.2:
        return 40.0
    elif rmsd_value >= 0.5:
        return 0.0
    else:
        return 40.0 - ((rmsd_value - 0.2) / 0.3) * 40.0


def load_real_model(model_path, device='cuda'):
    """Load the REAL trained model"""
    print(f"\n[1] Loading REAL model: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract config from checkpoint
        if 'config' in checkpoint and 'model' in checkpoint['config']:
            config = checkpoint['config']['model']
        else:
            raise ValueError("No config found in checkpoint")
        
        # Create model
        model = AdvancedTransitionStatePredictor(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"    Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"    Hidden Dim: {config['hidden_dim']}")
        print(f"    Num Layers: {config['num_layers']}")
        print(f"    Training Epoch: {checkpoint.get('epoch', 'N/A')}")
        if 'train_loss' in checkpoint:
            print(f"    Final Train Loss: {checkpoint['train_loss']:.6f}")
        if 'val_loss' in checkpoint:
            print(f"    Best Val Loss: {checkpoint.get('best_val_loss', checkpoint['val_loss']):.6f}")
        
        return model, config
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_molecular_graph(xyz_file, device='cuda'):
    """Create molecular graph from XYZ file"""
    try:
        # Read XYZ using ASE
        atoms = read(xyz_file)
        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()
        
        n_atoms = len(atomic_numbers)
        
        # Node features: one-hot encoding of atom types
        # Support common atoms: H(1), C(6), N(7), O(8), F(9), Si(14), P(15), S(16), Cl(17), Br(35), I(53)
        atom_type_map = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4, 14: 5, 15: 6, 16: 7, 17: 8, 35: 9, 53: 10}
        
        node_features = []
        for atom_num in atomic_numbers:
            feat = [0.0] * 11  # 11 atom types
            if atom_num in atom_type_map:
                feat[atom_type_map[atom_num]] = 1.0
            node_features.append(feat)
        
        x = torch.tensor(node_features, dtype=torch.float32)
        pos = torch.tensor(positions, dtype=torch.float32)
        
        # Create edges based on distance cutoff (3.0 Angstrom)
        edge_indices = []
        cutoff = 3.0
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < cutoff:
                    edge_indices.append([i, j])
                    edge_indices.append([j, i])  # Undirected
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            # No edges, create self-loops
            edge_index = torch.tensor([[i, i] for i in range(n_atoms)], dtype=torch.long).t()
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index, pos=pos)
        data = data.to(device)
        
        return data, atomic_numbers
    except Exception as e:
        print(f"[ERROR] Failed to create graph from {xyz_file}: {e}")
        return None, None


def save_xyz(coords, atomic_numbers, output_path):
    """Save coordinates to XYZ file"""
    atom_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 
                   14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
    
    with open(output_path, 'w') as f:
        f.write(f"{len(coords)}\n")
        f.write("Predicted Transition State\n")
        for atom_num, (x, y, z) in zip(atomic_numbers, coords):
            symbol = atom_symbols.get(atom_num, 'X')
            f.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")


def run_real_evaluation():
    """Run REAL evaluation with the trained model"""
    print("="*80)
    print("REAL COMPETITION EVALUATION")
    print("Using trained model: models/best_advanced_ts_model.pth")
    print("="*80)
    
    # Check model exists
    model_path = 'models/best_advanced_ts_model.pth'
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return None
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load REAL model
    model, config = load_real_model(model_path, device)
    if model is None:
        return None
    
    # Find test data
    demo_dir = "RNX_示例数据"
    if not os.path.exists(demo_dir):
        print(f"[ERROR] Demo directory not found: {demo_dir}")
        return None
    
    reaction_dirs = sorted([d for d in os.listdir(demo_dir) 
                           if os.path.isdir(os.path.join(demo_dir, d))])
    
    print(f"\n[2] Found {len(reaction_dirs)} test reactions")
    
    # Create output directory
    output_dir = "real_predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run predictions
    print(f"\n[3] Running REAL predictions...")
    results = []
    inference_times = []
    
    for rxn_dir in tqdm(reaction_dirs, desc="Predicting"):
        rxn_path = os.path.join(demo_dir, rxn_dir)
        
        r_file = os.path.join(rxn_path, "r.xyz")
        p_file = os.path.join(rxn_path, "p.xyz")
        ts_true_file = os.path.join(rxn_path, "ts.xyz")
        
        if not all(os.path.exists(f) for f in [r_file, p_file, ts_true_file]):
            print(f"[SKIP] Missing files in {rxn_dir}")
            continue
        
        # Create graphs
        r_graph, r_atoms = create_molecular_graph(r_file, device)
        p_graph, p_atoms = create_molecular_graph(p_file, device)
        
        if r_graph is None or p_graph is None:
            results.append({
                'reaction': rxn_dir,
                'rmsd': float('inf'),
                'success': False,
                'error': 'Graph creation failed',
                'inference_time': 0.0
            })
            continue
        
        # REAL prediction using the loaded model
        try:
            start_time = time.time()
            with torch.no_grad():
                pred_coords, uncertainty = model(r_graph, p_graph)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Convert to numpy
            pred_coords_np = pred_coords.cpu().numpy()
            
            # Save prediction
            pred_file = os.path.join(output_dir, f"{rxn_dir}_TS_pred.xyz")
            save_xyz(pred_coords_np, r_atoms, pred_file)
            
            # Calculate RMSD
            rmsd_value, success = calculate_rmsd_with_alignment(pred_file, ts_true_file)
            
            results.append({
                'reaction': rxn_dir,
                'rmsd': float(rmsd_value),
                'success': bool(success and rmsd_value <= 0.5),
                'uncertainty': float(uncertainty.item() if torch.is_tensor(uncertainty) else uncertainty),
                'inference_time': float(inference_time)
            })
            
        except Exception as e:
            print(f"\n[ERROR] Prediction failed for {rxn_dir}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'reaction': rxn_dir,
                'rmsd': float('inf'),
                'success': False,
                'error': str(e),
                'inference_time': 0.0
            })
    
    # Calculate metrics
    if not results:
        print("[ERROR] No valid results")
        return None
    
    valid_results = [r for r in results if r['rmsd'] != float('inf')]
    rmsd_values = np.array([r['rmsd'] for r in valid_results])
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    if len(rmsd_values) == 0:
        print("[ERROR] No valid RMSD values")
        return None
    
    # Metrics
    avg_rmsd = np.mean(rmsd_values)
    median_rmsd = np.median(rmsd_values)
    min_rmsd = np.min(rmsd_values)
    max_rmsd = np.max(rmsd_values)
    success_rate = success_count / total_count
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    
    # Official scoring
    rmsd_score = calculate_rmsd_score(avg_rmsd)
    success_score = success_rate * 30
    total_score = rmsd_score + success_score
    
    # Print results
    print(f"\n{'='*80}")
    print("REAL EVALUATION RESULTS")
    print(f"{'='*80}")
    
    print(f"\n[Performance Metrics]")
    print(f"  Total Reactions: {total_count}")
    print(f"  Valid Predictions: {len(valid_results)}/{total_count}")
    print(f"  Success (RMSD <= 0.5A): {success_count} ({success_rate*100:.1f}%)")
    print(f"  Average RMSD: {avg_rmsd:.4f} A")
    print(f"  Median RMSD: {median_rmsd:.4f} A")
    print(f"  Min RMSD: {min_rmsd:.4f} A")
    print(f"  Max RMSD: {max_rmsd:.4f} A")
    print(f"  Avg Inference Time: {avg_inference_time:.3f} s/reaction")
    
    print(f"\n[OFFICIAL COMPETITION SCORING]")
    print(f"  1. RMSD Score (max 40):")
    print(f"     Average RMSD: {avg_rmsd:.4f} A")
    print(f"     Score: {rmsd_score:.2f}/40")
    print(f"  ")
    print(f"  2. Success Rate Score (max 30):")
    print(f"     Success Rate: {success_rate*100:.2f}%")
    print(f"     Score: {success_score:.2f}/30")
    print(f"  ")
    print(f"  3. TOTAL SCORE: {total_score:.2f}/70")
    
    # Rating
    if total_score >= 60:
        rating = "A (EXCELLENT)"
    elif total_score >= 50:
        rating = "B (GOOD)"
    elif total_score >= 40:
        rating = "C (FAIR)"
    elif total_score >= 30:
        rating = "D (PASS)"
    else:
        rating = "F (NEEDS IMPROVEMENT)"
    
    print(f"  ")
    print(f"  Rating: {rating}")
    
    print(f"\n{'='*80}")
    
    # Save results
    eval_results = {
        'model_path': model_path,
        'total_reactions': total_count,
        'valid_predictions': len(valid_results),
        'success_count': success_count,
        'success_rate': float(success_rate),
        'avg_rmsd': float(avg_rmsd),
        'median_rmsd': float(median_rmsd),
        'min_rmsd': float(min_rmsd),
        'max_rmsd': float(max_rmsd),
        'avg_inference_time': float(avg_inference_time),
        'rmsd_score': float(rmsd_score),
        'success_score': float(success_score),
        'total_score': float(total_score),
        'rating': rating,
        'detailed_results': results
    }
    
    results_file = "real_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Predictions saved to: {output_dir}/")
    
    return eval_results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("REAL AI Transition State Predictor - Competition Evaluation")
    print("="*80)
    print("\nThis will ACTUALLY run the trained model, not simulate!")
    print("="*80 + "\n")
    
    results = run_real_evaluation()
    
    if results:
        print("\n[SUCCESS] REAL evaluation completed!")
        print(f"REAL Score: {results['total_score']:.2f}/70")
    else:
        print("\n[FAILED] Evaluation failed!")
