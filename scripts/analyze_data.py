#!/usr/bin/env python3
"""
Transition1x 数据集分析脚本
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd


def analyze_transition1x_dataset(h5_path: str):
    """分析 Transition1x 数据集"""
    try:
        import transition1x
    except ImportError:
        print("Error: transition1x package not installed!")
        print("Please run: python download_data.py")
        return
    
    print("=" * 60)
    print("Transition1x Dataset Analysis")
    print("=" * 60)
    
    # 统计信息
    stats = {
        'total_reactions': 0,
        'atom_counts': [],
        'formulas': [],
        'activation_energies': [],
        'reaction_energies': [],
        'element_counts': Counter(),
        'reaction_types': Counter()
    }
    
    # 分析训练集
    print("Analyzing training data...")
    dataloader = transition1x.Dataloader(h5_path, datasplit='train', only_final=True)
    
    for i, reaction in enumerate(dataloader):
        try:
            # 基本信息
            stats['total_reactions'] += 1
            
            # 原子数统计
            n_atoms = len(reaction['reactant']['atomic_numbers'])
            stats['atom_counts'].append(n_atoms)
            
            # 分子式
            formula = reaction['reactant']['formula']
            stats['formulas'].append(formula)
            
            # 能量信息
            r_energy = reaction['reactant'].get('wB97x_6-31G(d).energy', 0)
            ts_energy = reaction['transition_state'].get('wB97x_6-31G(d).energy', 0)
            p_energy = reaction['product'].get('wB97x_6-31G(d).energy', 0)
            
            activation_energy = ts_energy - r_energy
            reaction_energy = p_energy - r_energy
            
            stats['activation_energies'].append(activation_energy)
            stats['reaction_energies'].append(reaction_energy)
            
            # 元素统计
            atomic_numbers = reaction['reactant']['atomic_numbers']
            for atom_num in atomic_numbers:
                stats['element_counts'][atom_num] += 1
            
            # 反应类型
            rxn_name = reaction.get('rxn', 'unknown')
            stats['reaction_types'][rxn_name] += 1
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} reactions...")
                
        except Exception as e:
            print(f"Error processing reaction {i}: {e}")
            continue
    
    # 打印统计结果
    print_statistics(stats)
    
    # 生成可视化
    create_visualizations(stats)


def print_statistics(stats):
    """打印统计信息"""
    print(f"\n📊 Dataset Statistics")
    print(f"Total reactions: {stats['total_reactions']:,}")
    
    # 原子数统计
    atom_counts = np.array(stats['atom_counts'])
    print(f"\n🔬 Molecular Size:")
    print(f"  Average atoms per molecule: {np.mean(atom_counts):.1f}")
    print(f"  Min atoms: {np.min(atom_counts)}")
    print(f"  Max atoms: {np.max(atom_counts)}")
    print(f"  Median atoms: {np.median(atom_counts):.1f}")
    
    # 能量统计
    activation_energies = np.array(stats['activation_energies'])
    reaction_energies = np.array(stats['reaction_energies'])
    
    print(f"\n⚡ Energy Statistics (eV):")
    print(f"  Activation Energy:")
    print(f"    Mean: {np.mean(activation_energies):.3f}")
    print(f"    Std:  {np.std(activation_energies):.3f}")
    print(f"    Range: [{np.min(activation_energies):.3f}, {np.max(activation_energies):.3f}]")
    
    print(f"  Reaction Energy:")
    print(f"    Mean: {np.mean(reaction_energies):.3f}")
    print(f"    Std:  {np.std(reaction_energies):.3f}")
    print(f"    Range: [{np.min(reaction_energies):.3f}, {np.max(reaction_energies):.3f}]")
    
    # 元素统计
    print(f"\n🧪 Element Distribution:")
    # 原子序数到元素符号的映射
    atomic_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 
                     15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
    
    for atom_num, count in stats['element_counts'].most_common(10):
        symbol = atomic_symbols.get(atom_num, f'Z{atom_num}')
        percentage = count / sum(stats['element_counts'].values()) * 100
        print(f"  {symbol:>2}: {count:>8,} ({percentage:5.1f}%)")
    
    # 反应类型统计
    print(f"\n🔄 Top Reaction Types:")
    for rxn_type, count in stats['reaction_types'].most_common(10):
        percentage = count / stats['total_reactions'] * 100
        print(f"  {rxn_type[:30]:30}: {count:>6} ({percentage:5.1f}%)")


def create_visualizations(stats):
    """创建可视化图表"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Transition1x Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. 原子数分布
    axes[0, 0].hist(stats['atom_counts'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Number of Atoms')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Molecular Size Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 活化能分布
    axes[0, 1].hist(stats['activation_energies'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Activation Energy (eV)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Activation Energy Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 反应能分布
    axes[0, 2].hist(stats['reaction_energies'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 2].set_xlabel('Reaction Energy (eV)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Reaction Energy Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 元素分布
    atomic_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si', 
                     15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
    
    top_elements = stats['element_counts'].most_common(10)
    elements = [atomic_symbols.get(atom_num, f'Z{atom_num}') for atom_num, _ in top_elements]
    counts = [count for _, count in top_elements]
    
    axes[1, 0].bar(elements, counts, color='gold', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Element')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Element Distribution (Top 10)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. 活化能 vs 原子数
    axes[1, 1].scatter(stats['atom_counts'], stats['activation_energies'], 
                      alpha=0.5, s=10, color='purple')
    axes[1, 1].set_xlabel('Number of Atoms')
    axes[1, 1].set_ylabel('Activation Energy (eV)')
    axes[1, 1].set_title('Activation Energy vs Molecular Size')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 反应类型分布（前10）
    top_reactions = stats['reaction_types'].most_common(10)
    rxn_names = [name[:15] + '...' if len(name) > 15 else name for name, _ in top_reactions]
    rxn_counts = [count for _, count in top_reactions]
    
    axes[1, 2].barh(rxn_names, rxn_counts, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 2].set_xlabel('Count')
    axes[1, 2].set_title('Top Reaction Types')
    
    plt.tight_layout()
    plt.savefig('transition1x_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📈 Visualization saved as 'transition1x_analysis.png'")


def main():
    """主函数"""
    # 查找HDF5文件
    h5_files = list(Path("data").glob("*.h5"))
    
    if not h5_files:
        print("No HDF5 files found in ./data directory")
        print("Please run: python download_data.py")
        return
    
    h5_path = str(h5_files[0])
    print(f"Analyzing dataset: {h5_path}")
    
    analyze_transition1x_dataset(h5_path)


if __name__ == "__main__":
    main()