"""Convert an exported strand array to final hairutil formats in the case root."""
from argparse import ArgumentParser
from pathlib import Path
import pickle
import subprocess

import numpy as np


def export_final_strands(case_dir, model_name='stage3', iteration=10000):
    case_dir = Path(case_dir).resolve()
    strands_path = (case_dir / 'work' / 'curves_reconstruction' / model_name
                    / 'strands' / f'{iteration}_strands.pkl')
    with strands_path.open('rb') as source:
        strands = pickle.load(source)
    if not isinstance(strands, np.ndarray) or strands.ndim != 3 or strands.shape[-1] != 3:
        raise ValueError(f'Expected an N x L x 3 NumPy array in {strands_path}')

    # hairutil reads NPY, not pickle. Keep this intermediate beside the pickle.
    array_path = strands_path.with_suffix('.npy')
    np.save(array_path, strands, allow_pickle=False)
    subprocess.run([
        'hairutil', 'convert', '-i', str(array_path), '-o', 'data,ply',
        '-d', str(case_dir), '--overwrite', '--no-autofix',
    ], check=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--case_dir', type=Path, required=True)
    parser.add_argument('--model_name', default='stage3')
    parser.add_argument('--iter', type=int, default=10000)
    args = parser.parse_args()
    export_final_strands(args.case_dir, args.model_name, args.iter)
