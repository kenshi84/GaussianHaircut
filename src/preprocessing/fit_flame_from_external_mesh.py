"""
Replacement for the FLAME multiview LBFGS fit (NeuralHaircut/.../fit.py).

Skips the optimization entirely. Instead:
  1. Take the FLAME mean-shape template (zero pose, zero shape, zero expression).
  2. Procrustes-align it to an externally-fitted ICTFaceKit head mesh, using
     the inner 51 dlib face landmarks (static, pose-independent).
  3. Apply the same final 0.98 about-center scaling that the original fitter does.
  4. Save as flame_fitting/<exp>/stage_3/mesh_final.obj.

Coordinate-frame assumption: the external head mesh must live in the same world
frame as the cameras that downstream stages use ($EXP_PATH_1/cameras/30000_matrices.pkl
and the trained gaussians). In practice, that means feeding the same images +
COLMAP run into both pipelines.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh


GH_ROOT = Path(__file__).resolve().parents[2]
SMPLX_NPZ = GH_ROOT / 'ext/NeuralHaircut/PIXIE/data/SMPLX_NEUTRAL_2020.npz'
FLAME_IDS_NPY = GH_ROOT / 'ext/NeuralHaircut/PIXIE/data/SMPL-X__FLAME_vertex_ids.npy'

# dlib-68: 0-16 face contour, 17-67 inner. FLAME exposes the inner 51 as
# pose-independent static landmarks; the contour 17 are pose-dependent.
DLIB_INNER = slice(17, 68)


def umeyama(src, tgt):
    """Similarity transform (s, R, t) such that  s * R @ src.T + t ≈ tgt.T.

    src, tgt: (N, 3) point sets in correspondence.
    """
    n = src.shape[0]
    src_mu = src.mean(0)
    tgt_mu = tgt.mean(0)
    src_c = src - src_mu
    tgt_c = tgt - tgt_mu
    cov = src_c.T @ tgt_c / n
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    var_src = (src_c ** 2).sum() / n
    s = float((S * np.array([1.0, 1.0, d])).sum() / var_src)
    t = tgt_mu - s * R @ src_mu
    return s, R, t


def flame_static_landmarks(verts, faces, lmk_faces_idx, lmk_bary_coords):
    face_verts = verts[faces[lmk_faces_idx]]                 # (L, 3, 3)
    return (face_verts * lmk_bary_coords[:, :, None]).sum(1) # (L, 3)


def write_obj(path, verts, faces):
    with open(path, 'w') as fh:
        for v in verts:
            fh.write(f'v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n')
        for face in faces:
            fh.write(f'f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n')


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description=__doc__)
    ap.add_argument('--data_path', required=True, type=Path,
                    help='GaussianHaircut $DATA_PATH')
    ap.add_argument('--exp_name', required=True,
                    help='GaussianHaircut $EXP_NAME_1 (e.g. stage1)')
    ap.add_argument('--external_head_mesh', required=True, type=Path,
                    help='Path to head_fit.obj produced by your fitter '
                         '(ICTFaceKit topology, 14062 verts)')
    ap.add_argument('--ictfacekit_npz', required=True, type=Path,
                    help='Path to ictfacekit_14062.npz (must contain '
                         'idx_to_landmark_verts, 68 indices)')
    args = ap.parse_args()

    # --- FLAME / SMPL-X assets ---
    smplx = np.load(SMPLX_NPZ, allow_pickle=True)
    smplx_v = np.asarray(smplx['v_template'], dtype=np.float64)
    smplx_f = np.asarray(smplx['f'], dtype=np.int64)
    lmk_faces_idx = np.asarray(smplx['lmk_faces_idx'], dtype=np.int64)
    lmk_bary = np.asarray(smplx['lmk_bary_coords'], dtype=np.float64)
    flame_ids = np.load(FLAME_IDS_NPY).astype(np.int64)

    if lmk_faces_idx.shape[0] != 51:
        raise RuntimeError(f'expected 51 static FLAME landmarks, got {lmk_faces_idx.shape[0]}')

    flame_inner = flame_static_landmarks(smplx_v, smplx_f, lmk_faces_idx, lmk_bary)

    # --- External head mesh + ICTFaceKit landmark indices ---
    ict = np.load(args.ictfacekit_npz, allow_pickle=True)
    ict_lmk_idx = np.asarray(ict['idx_to_landmark_verts'], dtype=np.int64)
    if ict_lmk_idx.shape[0] != 68:
        raise RuntimeError(f'expected 68 ICTFaceKit landmarks, got {ict_lmk_idx.shape[0]}')

    head_mesh = trimesh.load(args.external_head_mesh, process=False)
    head_v = np.asarray(head_mesh.vertices, dtype=np.float64)
    if head_v.shape[0] != 14062:
        print(f'warning: external head has {head_v.shape[0]} verts (expected 14062)')

    ict_inner = head_v[ict_lmk_idx[DLIB_INNER]]

    # --- Procrustes align FLAME inner landmarks to ICT inner landmarks ---
    s, R, t = umeyama(flame_inner, ict_inner)
    rmsd = float(np.sqrt(np.mean(np.sum((s * (flame_inner @ R.T) + t - ict_inner) ** 2, axis=1))))
    print(f'sim(3): scale={s:.6f}  det(R)={np.linalg.det(R):+.6f}  '
          f'landmark RMSD={rmsd:.4f} (in target units)')

    # --- Build aligned FLAME-cut head ---
    flame_v = s * (smplx_v[flame_ids] @ R.T) + t  # (5023, 3)

    flame_set = set(flame_ids.tolist())
    keep = np.array([(f[0] in flame_set) and (f[1] in flame_set) and (f[2] in flame_set)
                     for f in smplx_f])
    flame_f_smplx = smplx_f[keep]
    remap = np.full(smplx_v.shape[0], -1, dtype=np.int64)
    remap[flame_ids] = np.arange(flame_ids.shape[0])
    flame_f = remap[flame_f_smplx]
    assert (flame_f >= 0).all()

    # Match runner.py:178-179 — shrink 2% about the mesh center.
    center = flame_v.mean(0, keepdims=True)
    flame_v = (flame_v - center) * 0.98 + center

    out_dir = args.data_path / 'flame_fitting' / args.exp_name / 'stage_3'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_obj = out_dir / 'mesh_final.obj'
    write_obj(out_obj, flame_v, flame_f)
    print(f'wrote {out_obj}: {flame_v.shape[0]} verts, {flame_f.shape[0]} faces')


if __name__ == '__main__':
    main()
