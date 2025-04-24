'''
Author:  Kanbe 
Date: 2025-04-14 01:51:08
LastEditors:  Kanbe 
LastEditTime: 2025-04-19 23:35:35
FilePath: /RetESP/MolFormer+RetNet/preprocess.py
Description: 
'''

import numpy as np
import pickle

SEQ_MAP = {aa: i+1 for i, aa in enumerate('GAVLIMPSCNQTFYWDERKH')}
MOL_MAP = {sym: i+1 for i, sym in enumerate('0,C,H,O,N,P,S,B,F,Cl,Br,I,Se,s,o,n,c,Na,Mg,Fe,Zn,Cu,Ca,Ba,Hg,Al,Mn,Si,Li'.split(','))}

def process_sequence(seq: str) -> list:
    if any(c in seq for c in ['X','B','U','O']):
        return None
    if not (5 <= len(seq) <= 2000):
        return None
    return [SEQ_MAP.get(aa, 0) for aa in seq]

def save_processed_data(raw_path: str, save_path: str):
    raw_data = np.load(raw_path, allow_pickle=True)
    
    processed = []
    for item in raw_data:
        label = item[0]
        seq = item[1]
        smiles = item[2]
        
        seq_feat = process_sequence(seq)
        
        if seq_feat and smiles:
            processed.append({
                'label': label,
                'sequence': seq_feat,
                'smiles' : smiles
            })
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            'data': processed,
            'max_seq_len': max(len(x['sequence']) for x in processed),
        }, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='原始数据路径')
    parser.add_argument('--output', type=str, required=True, help='预处理保存路径')
    args = parser.parse_args()
    
    save_processed_data(args.input, args.output)
    print(f"预处理完成！保存至 {args.output}")
