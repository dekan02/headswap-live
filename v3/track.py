#!/usr/bin/env python
"""
Phase 3D-1 (revised): per-frame motion tracking for the v3 geometry-first pipeline.

Runs in the GAGAvatar venv (third_party/GAGAvatar/venv). For each frame of the
245-frame regression clip (offline_demo/frames/test), produces and saves to disk:
  - SMIRK's 50-D expression, 3-D jaw (axis-angle), 3-D global head rotation
    (axis-angle) -- via the SAME per-frame path _headswap_smirk_live_bench.py's
    track_smirk_frame() already uses successfully offline (crop_face -> GPU warp
    -> SmirkEncoder), not a reimplementation of SMIRK inference. No OptimEngine
    involved (that's a separate, unrelated 300-step per-item fitting tool).
  - MediaPipe FaceLandmarker output, saved as the FULL 478-point set in pixel
    space (x, y) + mediapipe's own normalized z, exactly as run_mediapipe()
    returns it (third_party/smirk/utils/mediapipe_utils.py).

Landmark-subset finding (checked before wiring anything, per the task brief):
FLAME_with_eye.pt's 'lmk_embeddings_mediapipe' entry is NOT a full-468/478
correspondence -- it's a 105-point SUBSET, keyed by
  landmark_indices: (105,) int64, values in [0, 466] -- indices into the
                     468-point mediapipe FACE mesh (i.e. the non-iris prefix
                     of the 478-point array run_mediapipe() returns; none of
                     the 105 land on the iris points 468-477).
  lmk_face_idx:      (105,) int64 -- FLAME face index per landmark
  lmk_b_coords:      (105, 3) float -- barycentric coords into that face
So FLAME's i-th predicted landmark (via lmk_face_idx[i]/lmk_b_coords[i])
corresponds to driver landmark mediapipe_landmarks[landmark_indices[i]], NOT
to mediapipe_landmarks[i]. This script saves the full 478-point array (cheap,
~11KB/frame) rather than pre-selecting the 105-subset, so Task 3 (fit3d.py)
can index with landmark_indices itself -- keeps the correspondence logic in
one place (fit3d.py, where it's actually consumed) instead of silently baking
an assumption about it into the tracker.

Output: v3/debug_out/3d1/track_results.npz with per-frame arrays:
  expression   (245, 50)   float32  SMIRK expression_params, un-padded
  jaw          (245, 3)    float32  SMIRK jaw_params, axis-angle-ish (relu+clamp, see smirk_encoder.py)
  global_rot   (245, 3)    float32  SMIRK pose_params, axis-angle (see _headswap_smirk_adapter.py)
  eyelid       (245, 2)    float32  SMIRK eyelid_params (closure amount, NOT used by the adapter)
  landmarks_mp (245, 478, 3) float32  mediapipe landmarks: (x_px, y_px, z_norm)
  valid        (245,)      bool     False for frames where mediapipe/SMIRK found no face

Usage (from HeadSwap repo root):
    third_party/GAGAvatar/venv/Scripts/python.exe v3/track.py
    third_party/GAGAvatar/venv/Scripts/python.exe v3/track.py --clip test --limit 10
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_HEADSWAP_ROOT = Path(__file__).resolve().parent.parent
_GAGA_ROOT = _HEADSWAP_ROOT / "third_party" / "GAGAvatar"  # junction -> ../GAGAvatar

# mediapipe_utils.py resolves 'assets/face_landmarker.task' relative to cwd at
# import time (documented in _headswap_smirk_bench.py / _headswap_smirk_live_bench.py) --
# chdir into third_party/smirk, import what's needed, then chdir back before
# touching anything HeadSwap-root-relative. Same dance those scripts use.
_SMIRK_ROOT = _GAGA_ROOT / "third_party" / "smirk"
os.chdir(str(_SMIRK_ROOT))
sys.path.insert(0, ".")
import cv2  # noqa: E402
from skimage.transform import estimate_transform  # noqa: E402
from src.smirk_encoder import SmirkEncoder  # noqa: E402
from utils.mediapipe_utils import run_mediapipe  # noqa: E402
os.chdir(str(_HEADSWAP_ROOT))

import torch  # noqa: E402
import torch.nn.functional as TF  # noqa: E402

SMIRK_CHECKPOINT = _SMIRK_ROOT / "pretrained_models" / "SMIRK_em1.pt"
FRAMES_DIR_TMPL = _HEADSWAP_ROOT / "offline_demo" / "frames" / "{clip}"
OUT_DIR = _HEADSWAP_ROOT / "v3" / "debug_out" / "3d1"


def crop_face(frame_bgr, landmarks_xy, scale=1.4, image_size=224):
    """Verbatim from _headswap_smirk_live_bench.py -- crop-transform estimate only."""
    left, right = np.min(landmarks_xy[:, 0]), np.max(landmarks_xy[:, 0])
    top, bottom = np.min(landmarks_xy[:, 1]), np.max(landmarks_xy[:, 1])
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0]-size/2, center[1]+size/2],
                         [center[0]+size/2, center[1]-size/2]])
    dst_pts = np.array([[0, 0], [0, image_size-1], [image_size-1, 0]])
    return estimate_transform('similarity', src_pts, dst_pts)


def crop_warp_gpu(frame_bgr, tform, image_size, device):
    """Verbatim from _headswap_smirk_live_bench.py -- GPU crop+resample, BGR->RGB, [0,1]."""
    h, w = frame_bgr.shape[:2]
    img = torch.from_numpy(frame_bgr).to(device).float()[..., [2, 1, 0]]
    img = img.permute(2, 0, 1)[None] / 255.0

    inv_matrix = torch.tensor(tform.inverse.params, dtype=torch.float32, device=device)
    ys, xs = torch.meshgrid(
        torch.arange(image_size, device=device, dtype=torch.float32),
        torch.arange(image_size, device=device, dtype=torch.float32),
        indexing='ij',
    )
    out_coords = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
    in_coords = out_coords @ inv_matrix.T
    grid_x = 2.0 * in_coords[..., 0] / (w - 1) - 1.0
    grid_y = 2.0 * in_coords[..., 1] / (h - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)[None]
    return TF.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


def track_frame(frame_bgr, smirk_encoder, device, image_size=224):
    """Returns (smirk_out_dict, mediapipe_landmarks_478x3) or (None, None) if no face found."""
    kpt = run_mediapipe(frame_bgr)
    if kpt is None:
        return None, None
    tform = crop_face(frame_bgr, kpt[..., :2], scale=1.4, image_size=image_size)
    img_t = crop_warp_gpu(frame_bgr, tform, image_size, device)
    with torch.no_grad():
        smirk_out = smirk_encoder(img_t)
    return smirk_out, kpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', default='test')
    ap.add_argument('--limit', type=int, default=None, help='process only the first N frames (debug)')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = args.device
    frames_dir = Path(str(FRAMES_DIR_TMPL).format(clip=args.clip))
    frame_paths = sorted(frames_dir.glob('*.png'))
    if args.limit:
        frame_paths = frame_paths[:args.limit]
    if not frame_paths:
        print(f'FAIL: no frames found in {frames_dir}')
        return 1

    print(f'Loading SMIRK encoder from {SMIRK_CHECKPOINT} ...')
    smirk_encoder = SmirkEncoder().to(device)
    checkpoint = torch.load(str(SMIRK_CHECKPOINT), map_location=device, weights_only=False)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k}
    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()

    n = len(frame_paths)
    expression = np.zeros((n, 50), dtype=np.float32)
    jaw = np.zeros((n, 3), dtype=np.float32)
    global_rot = np.zeros((n, 3), dtype=np.float32)
    eyelid = np.zeros((n, 2), dtype=np.float32)
    landmarks_mp = np.zeros((n, 478, 3), dtype=np.float32)
    valid = np.zeros((n,), dtype=bool)

    print(f'Tracking {n} frames from {frames_dir} ...')
    t_start = time.perf_counter()
    n_skipped = 0
    for idx, path in enumerate(frame_paths):
        frame_bgr = cv2.imread(str(path))
        smirk_out, kpt = track_frame(frame_bgr, smirk_encoder, device)
        if smirk_out is None:
            print(f'  frame {idx}: no face detected, skipping (kept as invalid/zeros)')
            n_skipped += 1
            continue
        expression[idx] = smirk_out['expression_params'][0].detach().cpu().numpy()
        jaw[idx] = smirk_out['jaw_params'][0].detach().cpu().numpy()
        global_rot[idx] = smirk_out['pose_params'][0].detach().cpu().numpy()
        eyelid[idx] = smirk_out['eyelid_params'][0].detach().cpu().numpy()
        landmarks_mp[idx] = kpt.astype(np.float32)
        valid[idx] = True
        if idx % 40 == 0:
            print(f'  frame {idx}/{n}')

    elapsed = time.perf_counter() - t_start
    print(f'Tracked {n - n_skipped}/{n} frames ({n_skipped} skipped) in {elapsed:.1f}s '
          f'({n / elapsed:.1f} FPS)')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'track_results.npz'
    np.savez(
        out_path,
        expression=expression, jaw=jaw, global_rot=global_rot, eyelid=eyelid,
        landmarks_mp=landmarks_mp, valid=valid,
        frame_names=np.array([p.name for p in frame_paths]),
    )
    print(f'Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)')

    if n_skipped:
        print(f'WARNING: {n_skipped} frame(s) had no detected face -- check '
              f'`valid` mask before using their rows downstream.')
    else:
        print('OK: every frame produced a valid SMIRK + mediapipe result.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
