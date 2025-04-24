'''
Author:  Kanbe 
Date: 2025-04-14 01:51:08
LastEditors:  Kanbe 
LastEditTime: 2025-04-15 01:30:26
FilePath: /RetESP/preprocess.py
Description: 
'''

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
import pickle
from pathlib import Path

SEQ_MAP = {aa: i+1 for i, aa in enumerate('GAVLIMPSCNQTFYWDERKH')}
MOL_MAP = {sym: i+1 for i, sym in enumerate('0,C,H,O,N,P,S,B,F,Cl,Br,I,Se,s,o,n,c,Na,Mg,Fe,Zn,Cu,Ca,Ba,Hg,Al,Mn,Si,Li'.split(','))}

def process_sequence(seq: str) -> list:
    if any(c in seq for c in ['X','B','U','O']):
        return None
    if not (5 <= len(seq) <= 2000):
        return None
    return [SEQ_MAP.get(aa, 0) for aa in seq]

def process_molecule(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(str(smiles))
    if not mol:
        return None
        
    # 芳香原子标记
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for atom in mol.GetAromaticAtoms():
        atoms[atom.GetIdx()] = atoms[atom.GetIdx()].lower()
    try:
        atom_ids = [MOL_MAP[a] for a in atoms]
    except KeyError as e:
        print(f"无效原子类型: {str(e)}")
        return None
    
    # 生成邻接矩阵和边特征
    adj = rdmolops.GetAdjacencyMatrix(mol)
    edge_attr = np.zeros_like(adj, dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        edge_attr[i,j] = edge_attr[j,i] = bond_type
    
    return {
        'atom_ids': atom_ids,
        'adj_matrix': adj,
        'edge_attr': edge_attr
    }

def save_processed_data(raw_path: str, save_path: str):
    raw_data = np.load(raw_path, allow_pickle=True)
    
    processed = []
    for item in raw_data:
        label = item[0]
        seq = item[1]
        smiles = item[2]
        
        seq_feat = process_sequence(seq)
        mol_feat = process_molecule(smiles)
        if seq_feat and mol_feat:
            processed.append({
                'label': label,
                'sequence': seq_feat,
                'atom_ids': mol_feat['atom_ids'],
                'adj': mol_feat['adj_matrix'],
                'edge_attr': mol_feat['edge_attr']
            })
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            'data': processed,
            'max_seq_len': max(len(x['sequence']) for x in processed),
            'max_atoms': max(len(x['atom_ids']) for x in processed)
        }, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='原始数据路径')
    parser.add_argument('--output', type=str, required=True, help='预处理保存路径')
    args = parser.parse_args()
    
    save_processed_data(args.input, args.output)
    print(f"预处理完成！保存至 {args.output}")
