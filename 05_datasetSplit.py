'''
Author:  Kanbe 
Date: 2025-03-15 13:12:11
LastEditors:  Kanbe 
LastEditTime: 2025-03-20 01:45:12
FilePath: /Models/05_datasetSplit.py
Description: 
'''

import argparse
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict

def load_dataset(input_path: str) -> np.ndarray:
    df = pd.read_csv(input_path)    
    data = df.to_records(index=False)    
    return data

def split_dataset(data: np.ndarray,ratios: Tuple[float, float, float],seed: int = 114514) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Validate ratios
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1"
   # assert all(r > 0 for r in ratios), "All ratios must be positive"
    
    np.random.seed(seed)
    shuffled_data = np.random.permutation(data)
    
    n = len(shuffled_data)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    
    return (
        shuffled_data[:train_end],
        shuffled_data[train_end:val_end],
        shuffled_data[val_end:]
    )

def save_splits(splits: Tuple[np.ndarray, np.ndarray, np.ndarray],output_dir: str,base_name: str) -> Dict[str, str]:
    """Save splits to files with timestamp and return paths"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    paths = {}
    for name, data in zip(['train', 'validation', 'test'], splits):
        filename = f"{base_name}{name}_{timestamp}.pkl"
        path = os.path.join(output_dir, filename)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        paths[name] = path
    
    return paths

def generate_report(splits: Tuple[np.ndarray, np.ndarray, np.ndarray],paths: Dict[str, str]) -> str:
    """Generate detailed split report"""
    report = [
        "Dataset Split Report",
        "====================",
        f"Generated at: {datetime.now().isoformat()}",
        ""
    ]
    
    # Add split statistics
    total = sum(len(s) for s in splits)
    for name, data in zip(['Train', 'Validation', 'Test'], splits):
        report.extend([
            f"{name} Set:",
            f"  Samples: {len(data):,} ({len(data)/total:.1%})",
            f"  Path: {paths[name.lower()]}",
            ""
        ])
    
    return '\n'.join(report)

def main():
    parser = argparse.ArgumentParser(description="Dataset Splitting Tool with Seed Control",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True,help='Path to input dataset pickle file')
    parser.add_argument('-o', '--output-dir',default='data',help='Output directory for split datasets')
    parser.add_argument('-r', '--ratios',type=float,nargs=3,default=[0.8, 0.1, 0.1],help='Train/Val/Test ratios (must sum to 1)')
    parser.add_argument('-s', '--seed',type=int,default=114514,help='Random seed for reproducibility')
    parser.add_argument('-b', '--base-name',default='',help='Base name for output files')
    args = parser.parse_args()
    
    data = load_dataset(args.input)
    splits = split_dataset(data, tuple(args.ratios), args.seed)
    paths = save_splits(splits, args.output_dir, args.base_name)
    print(generate_report(splits, paths))

if __name__ == '__main__':
    main()
