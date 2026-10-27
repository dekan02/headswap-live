#!/usr/bin/env python
"""
Phase 3D-2: v3/fit3d.py -- closed-form {scale s, translation t_xy} similarity
solve per frame, aligning FLAME's own predicted (mediapipe-corresponding)
landmarks onto the driver's real MediaPipe landmarks. Rotation comes straight
from SMIRK (baked into FLAME's pose_params, not fit here) -- no OptimEngine, no
6-DOF optimizer. Runs in the GAGAvatar venv (needs FLAMEModel + pytorch3d-free
FLAME forward -- actually no pytorch3d needed here at all, just torch).

*** ROUND 2 (2026-09-04), Phase 3D-2 continuation: superseded the round-1 design
below (rotation baked into FLAME's pose_params). Round 1's actual real-render
test (v3/render_silhouette_check.py + v3/_diagnose_rotation_bake.py) found that
baking SMIRK's global rotation into FLAME's mesh does NOT visibly rotate
GAGAvatar's rendered 3DGS output (0x/1x/2x/3x rotation scale sweep on frame 36
looked nearly identical) -- matching the design every OTHER script in this repo
(GAGAvatar_track's own engine_optim.py CoreEngine, which optimizes R/T directly
as a pytorch3d camera transform, never touching FLAME's pose) already uses.
Root cause, confirmed via v3/_step2_sign_check.py: GAGAvatar's Gaussian
generator produces per-Gaussian covariance/orientation in a canonical,
pose-independent frame; moving the mesh's vertex XYZ via a baked rotation moves
Gaussian CENTERS but not their covariance, while the renderer's own view-matrix
math (core.libs.utils_renderer.build_camera_matrices) correctly reprojects BOTH
position and covariance when rotation lives in the camera instead. Fix: FLAME
now always renders at global rotation ZERO (canonical orientation); SMIRK's
already-validated rotation (confirmed correct via landmark reprojection in
round 1, not re-derived) is composed onto the camera's R submatrix instead. See
Step 1/2/3 below (replacing round 1's Step 1/5) and NOTES_3D0_findings.md's
round-2 section for the full derivation, including the empirical (not assumed)
mesh-rotation-convention pin (FLAME's global rotation pivots about the ROOT
JOINT location J0, row-vector convention P0 @ Rg.T, residual 0.0027 FLAME
units at 38 degrees rotation) and camera composition R_cam = Rg.T @ source_R
(forced by matrix associativity once the mesh convention is known; ALSO
independently confirmed to turn the same direction as the real driver frame 36,
not mirrored -- see step2_sign_check.png).

Per frame:
  1. FLAME forward with beta_avatar LOCKED to the tracked avatar identity shape,
     THIS frame's SMIRK expression(zero-padded 50->100) + jaw(3,aa), but global
     rotation pose_params[:, :3] forced to ZERO (round 2 -- see note above).
     SMIRK's global_rot(3,aa) is instead converted to a rotation matrix via
     FLAME's own batch_rodrigues() and applied to the CAMERA (Step 5 below),
     not handed to FLAME's forward().
  2. FLAME's predicted landmark positions via the barycentric interpolation
     `lmk_embeddings_mediapipe` calls for (105-point subset, see v3/track.py's
     docstring for the landmark_indices correspondence) -- reusing
     `vertices2landmarks` from core.libs.flame_model.lbs (the same utility
     FLAMEModel.forward() itself is built on for its own 70-point set), not a
     hand-rolled barycentric lookup.
  3. Weak-perspective project: drop z, and flip FLAME's Y axis (FLAME is Y-up;
     mediapipe pixel landmarks are Y-down/image convention) -- this flip is
     REQUIRED for the closed-form solve below to work, since a pure
     scale+translation similarity cannot correct a Y-axis mirror on its own
     (that would need a reflection, which isotropic positive scale can't
     produce) -- confirmed correct via the frame-36 (41 deg turn) overlay,
     which is asymmetric enough that a wrong Y (or X) sign would visibly fail.
  4. Closed-form isotropic similarity (Procrustes/Umeyama minus the rotation
     term, derived directly, not approximated):
        minimize_{s>0, t in R^2}  sum_i || s*P_i + t - Q_i ||^2
        => s = sum_i dot(P_i - mean(P), Q_i - mean(Q)) / sum_i ||P_i - mean(P)||^2
           t = mean(Q) - s * mean(P)
  5. Camera transform for render_gaussian's t_transform ([R|T] 3x4, Task 0 Q2):
     R = the avatar identity's own fixed source_R (NOT R_live @ source_R --
     rotation is already baked into the FLAME mesh via step 1, so the camera
     orientation calibration stays fixed for every frame, same role source_R
     already plays for a canonical/neutral pose in every other script here).
     T is derived by converting the driver-pixel-space (s, t_xy) into FLAME's
     own physical unit system (dividing out s), so it stays valid regardless of
     the driver frame's pixel resolution (1280x720) vs the render's own 512x512
     canvas -- see the derivation in NOTES_3D0_findings.md's Task 3 section for
     why this produces a render whose on-screen motion is PROPORTIONAL to the
     driver's real on-screen motion (verified: multiplying t_xy or dividing s by
     a constant scales T_x/T_y/T_z consistently, and the mandatory T_z gate
     (v3/camera_convention_check.py) already confirmed size ~ 1/T_z holds):
        T_z[t] = T_z0 * s_ref / s[t]      (s_ref = s at the neutral reference frame)
        T_x[t] = t_xy[t, 0] / s[t]
        T_y[t] = t_xy[t, 1] / s[t]
     KNOWN, FLAGGED SCOPE LIMIT: this makes the render's on-screen motion track
     the driver's motion *proportionally*, but does not itself calibrate the
     ABSOLUTE zoom/crop so the 512-canvas render lines up pixel-for-pixel with
     the 1280x720 driver frame -- that absolute placement is the "stabilize"/
     later blend stage's job (same role run_composite_v2's neck-anchored
     placement plays for the 2D stack today), not fit3d's. Not guessed away
     silently -- called out here and in NOTES_3D0_findings.md.

*** The real exit criterion is visual: per-frame landmark overlays, not just a
mean-error number. *** See plot_fit3d_overlays() below.

Usage (GAGAvatar venv, from HeadSwap repo root):
    third_party/GAGAvatar/venv/Scripts/python.exe v3/fit3d.py
"""
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
from scipy.optimize import least_squares

_HEADSWAP_ROOT = Path(__file__).resolve().parent.parent
_GAGA_ROOT = _HEADSWAP_ROOT / "third_party" / "GAGAvatar"
sys.path.insert(0, str(_GAGA_ROOT))

from core.libs.flame_model import FLAMEModel  # noqa: E402
from core.libs.flame_model.lbs import vertices2landmarks, batch_rodrigues  # noqa: E402

TRACKED_PT = _GAGA_ROOT / "render_results" / "tracked" / "tracked.pt"
FLAME_CKPT = _GAGA_ROOT / "assets" / "FLAME_with_eye.pt"
TRACK_NPZ = _HEADSWAP_ROOT / "v3" / "debug_out" / "3d1" / "track_results.npz"
FRAMES_DIR = _HEADSWAP_ROOT / "offline_demo" / "frames" / "test"
OUT_DIR = _HEADSWAP_ROOT / "v3" / "debug_out" / "3d2"
OVERLAY_DIR = OUT_DIR / "overlays"

STRIP_FRAMES = [0, 36, 82, 117]
REFERENCE_FRAME = 0  # "neutral" per the baseline strip / Task 0-1 notes


def load_flame_and_identity(device):
    flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=5.0, no_lmks=True).to(device)

    tracked = torch.load(str(TRACKED_PT), map_location='cpu', weights_only=False)
    rec = tracked['headswap_target.jpg']
    beta_avatar = torch.as_tensor(rec['shapecode']).float().to(device)  # (300,)
    source_transform = torch.as_tensor(rec['transform_matrix']).float().to(device)  # (3,4)

    flame_ck = torch.load(str(FLAME_CKPT), map_location='cpu', weights_only=False)
    mp_lmk = flame_ck['lmk_embeddings_mediapipe']
    landmark_indices = mp_lmk['landmark_indices'].long()          # (105,) -> index into 478-pt mediapipe array
    lmk_face_idx = mp_lmk['lmk_face_idx'].long().to(device)       # (105,)
    lmk_b_coords = mp_lmk['lmk_b_coords'].float().to(device)      # (105,3)

    return flame_model, beta_avatar, source_transform, landmark_indices, lmk_face_idx, lmk_b_coords


def run_flame_batch(flame_model, faces_tensor, beta_avatar, lmk_face_idx, lmk_b_coords,
                     expression, jaw, global_rot, device, zero_global_rot=False):
    """expression:(N,50) jaw:(N,3) global_rot:(N,3) numpy arrays -> (N,105,3) FLAME landmarks (scaled).

    zero_global_rot=True feeds pose_params[:, :3]=0 into FLAME's own forward (root-
    joint/global rotation), keeping jaw's 3 params and expression as-is -- ROUND 2
    of Phase 3D-2: confirmed clean by reading FLAME's forward()+lbs() directly
    (posedirs' pose_feature slices OUT joint 0, i.e. global rotation contributes
    NO pose-corrective blendshape term; the neck joint is separately fixed at zero
    always). global_rot is still returned by run_flame_batch's caller (main())
    for use as the CAMERA rotation instead -- see v3/_step2_sign_check.py and
    NOTES_3D0_findings.md's round-2 Step 1/2 sections for why."""
    n = expression.shape[0]
    shape_params = beta_avatar[None].expand(n, -1)
    expr100 = torch.zeros(n, 100, device=device)
    expr100[:, :50] = torch.from_numpy(expression).to(device)
    rot_for_flame = torch.zeros(n, 3, device=device) if zero_global_rot else torch.from_numpy(global_rot).to(device)
    pose_params = torch.cat([
        rot_for_flame,  # global rotation, FLAME-native axis-angle (zeroed this round -- see docstring)
        torch.from_numpy(jaw).to(device),
    ], dim=-1).float()  # (N,6)
    eye_pose = torch.zeros(n, 6, device=device)

    vertices = flame_model(shape_params=shape_params, expression_params=expr100,
                            pose_params=pose_params, eye_pose_params=eye_pose)  # (N,5023,3), already *scale
    landmarks3d = vertices2landmarks(
        vertices, faces_tensor,
        lmk_face_idx.repeat(n, 1),
        lmk_b_coords.repeat(n, 1, 1),
    )
    return landmarks3d


def project_landmarks_to_render_canvas(landmarks_world, R, T, focal=12.0, cam_size=512):
    """Same formula as v3/render_silhouette_check.py's function of the same name
    (re-derived independently from core.libs.utils_renderer.build_camera_matrices
    both times, not copy-pasted blind -- kept in sync, see that module's
    docstring for the derivation). landmarks_world: (L,3) numpy, R:(3,3), T:(3,).
    Returns (L,2) pixel coords in the cam_size x cam_size render canvas."""
    base = landmarks_world @ R + T[None, :]
    x_view, y_view, z_view = -base[:, 0], -base[:, 1], base[:, 2]
    ndc_x = x_view * focal / z_view
    ndc_y = y_view * focal / z_view
    px = (ndc_x + 1.0) * 0.5 * cam_size
    py = (ndc_y + 1.0) * 0.5 * cam_size
    return np.stack([px, py], axis=-1)


def solve_T_least_squares(P0_frame, R_cam, target_canvas, T_init, focal=12.0, cam_size=512):
    """3-unknown Gauss-Newton (via scipy least_squares) solve for camera
    translation T=(Tx,Ty,Tz) minimizing reprojection error of P0_frame (this
    frame's FLAME landmarks, ZERO global rotation baked in) through the actual
    perspective formula against target_canvas (driver landmarks already mapped
    into render-canvas pixel space by the fixed calibration). R_cam is this
    frame's camera rotation (Rg.T @ source_R, fixed, not solved here)."""
    def residual(T):
        proj = project_landmarks_to_render_canvas(P0_frame, R_cam, T, focal, cam_size)
        return (proj - target_canvas).ravel()
    result = least_squares(residual, T_init, method='lm', max_nfev=200)
    return result.x, float(np.mean(result.fun ** 2) ** 0.5)


def similarity_solve(P, Q):
    """P, Q: (N, L, 2) torch tensors. Returns s:(N,), t:(N,2) minimizing sum||s*P+t-Q||^2."""
    centroid_P = P.mean(dim=1)  # (N,2)
    centroid_Q = Q.mean(dim=1)
    Pc = P - centroid_P[:, None, :]
    Qc = Q - centroid_Q[:, None, :]
    num = (Pc * Qc).sum(dim=(1, 2))
    den = (Pc * Pc).sum(dim=(1, 2)).clamp_min(1e-8)
    s = num / den
    t = centroid_Q - s[:, None] * centroid_P
    return s, t


def plot_fit3d_overlays(P, Q, fitted, valid, err_px, out_dir, frame_indices=None, max_frames=None):
    """Green = driver's real mediapipe landmarks (Q). Red = FLAME's fitted landmarks
    (s*P+t_xy). Drawn on top of the driver frame -- the actual exit criterion."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = P.shape[0]
    idxs = frame_indices if frame_indices is not None else range(n)
    if max_frames:
        idxs = list(idxs)[:max_frames]
    for i in idxs:
        img = cv2.imread(str(FRAMES_DIR / f'{i:05d}.png'))
        if img is None:
            continue
        for (x, y) in Q[i]:
            cv2.circle(img, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)  # green, BGR
        for (x, y) in fitted[i]:
            cv2.circle(img, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)  # red, BGR
        cv2.putText(img, f'frame {i:04d}  valid={bool(valid[i])}  mean_err={err_px[i]:.1f}px',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f'{i:04d}.png'), img)


def make_hard_frame_strip(P, Q, fitted, valid, err_px, out_path):
    panels = []
    for f in STRIP_FRAMES:
        img = cv2.imread(str(FRAMES_DIR / f'{f:05d}.png'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for (x, y) in Q[f]:
            cv2.circle(img, (int(round(x)), int(round(y))), 4, (0, 255, 0), -1)
        for (x, y) in fitted[f]:
            cv2.circle(img, (int(round(x)), int(round(y))), 3, (255, 0, 0), -1)
        cv2.putText(img, f'f{f:04d} err={err_px[f]:.1f}px', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        panels.append(img)
    h = min(p.shape[0] for p in panels)
    panels = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
    strip = np.concatenate(panels, axis=1)
    header = np.full((36, strip.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, 'green = real mediapipe landmarks   red = FLAME fitted (s*P+t_xy)',
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    full = np.concatenate([header, strip], axis=0)
    cv2.imwrite(str(out_path), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))


def main():
    device = 'cuda'
    d = np.load(TRACK_NPZ)
    expression, jaw, global_rot = d['expression'], d['jaw'], d['global_rot']
    landmarks_mp, valid = d['landmarks_mp'], d['valid']
    n = expression.shape[0]
    print(f'Loaded track_results.npz: {n} frames, {valid.sum()} valid')

    flame_model, beta_avatar, source_transform, landmark_indices, lmk_face_idx, lmk_b_coords = \
        load_flame_and_identity(device)
    faces_tensor = flame_model.get_faces()

    print('Running FLAME forward for all frames (beta_avatar locked, full SMIRK pose) '
          '-- LANDMARK-OVERLAY DIAGNOSTIC ONLY this round, not the camera-transform basis...')
    landmarks3d_full = run_flame_batch(flame_model, faces_tensor, beta_avatar, lmk_face_idx, lmk_b_coords,
                                        expression, jaw, global_rot, device, zero_global_rot=False)  # (N,105,3)

    P = landmarks3d_full[..., :2].clone()
    P[..., 1] *= -1.0  # FLAME Y-up -> image Y-down
    P = P.cpu()

    Q = torch.from_numpy(landmarks_mp[:, landmark_indices.numpy(), :2]).float()  # (N,105,2), driver pixel space

    print('Solving closed-form {s, t_xy} weak-perspective similarity per frame (diagnostic only) ...')
    s, t_xy = similarity_solve(P, Q)
    fitted = s[:, None, None] * P + t_xy[:, None, :]
    err = (fitted - Q).norm(dim=-1)  # (N,105)
    err_mean = err.mean(dim=1)  # (N,)

    print(f'\nPer-frame landmark error (px, weak-perspective diagnostic): mean={err_mean.mean():.2f}  '
          f'median={err_mean.median():.2f}  max={err_mean.max():.2f} (frame {int(err_mean.argmax())})')

    # ---- Step 1/2/3 (round 2, 2026-09-04): camera transform, superseding round 1's
    # Step 5 entirely -- see the module docstring's round-2 note for why. ----
    print('\nRunning FLAME forward at ZERO global rotation (Step 1: this is what actually '
          'gets rendered/goes into the camera-transform solve) ...')
    landmarks3d_zero = run_flame_batch(flame_model, faces_tensor, beta_avatar, lmk_face_idx, lmk_b_coords,
                                        expression, jaw, global_rot, device, zero_global_rot=True)  # (N,105,3)
    P0_np = landmarks3d_zero.cpu().numpy()  # (N,105,3), FLAME units, zero global rotation baked in

    # Step 2: camera rotation R_cam[f] = Rg[f].T @ source_R, empirically pinned in
    # v3/_step2_sign_check.py (mesh-side convention residual 0.0027 FLAME units at
    # 38 degrees; ALSO independently confirmed by eye to turn the same direction as
    # the real driver frame 36, not mirrored -- see step2_sign_check.png). Camera
    # composition is forced by matrix associativity once the mesh convention is
    # known (documented in NOTES_3D0_findings.md), not a second free guess.
    Rg_all = batch_rodrigues(torch.from_numpy(global_rot).to(device))  # (N,3,3), FLAME's own function
    R_dev = source_transform[:3, :3]  # (3,3), still on `device`
    R_cam_all = torch.matmul(Rg_all.transpose(1, 2), R_dev[None].expand(n, -1, -1))  # (N,3,3) = Rg.T @ source_R
    R_cam_np = R_cam_all.cpu().numpy()

    # Step 3: solve T[f]=(Tx,Ty,Tz) via real-perspective least-squares (NOT the
    # round-1 centering-only formula, which is now retired). First bootstrap a
    # FIXED render-canvas<->driver-frame calibration (s_cal, t_cal) from the
    # reference frame only (rotation ~0 there, so an initial T guess of source_T
    # -- the identity's own calibrated neutral camera placement -- is physically
    # reasonable), then solve every frame's T against driver landmarks mapped
    # into render-canvas space through that ONE fixed calibration (a compositing-
    # stage property, not a per-frame unknown -- re-verified post-render in
    # v3/render_silhouette_check.py's own center-point check, per the brief).
    Q_np = Q.numpy()  # (N,105,2) driver-pixel-space landmarks
    source_T_np = source_transform[:3, 3].cpu().numpy()

    T_ref_guess = source_T_np.copy()
    render_lmk_ref_guess = project_landmarks_to_render_canvas(P0_np[REFERENCE_FRAME], R_cam_np[REFERENCE_FRAME], T_ref_guess)
    s_cal_t, t_cal_t = similarity_solve(torch.from_numpy(render_lmk_ref_guess)[None], torch.from_numpy(Q_np[REFERENCE_FRAME])[None])
    s_cal, t_cal = float(s_cal_t[0]), t_cal_t[0].numpy()
    print(f'\nBootstrapped fixed render-canvas<->driver-frame calibration from frame {REFERENCE_FRAME}: '
          f's_cal={s_cal:.4f}  t_cal=({t_cal[0]:.1f},{t_cal[1]:.1f})')

    T_all = np.zeros((n, 3), dtype=np.float64)
    ls_residual_px = np.zeros(n, dtype=np.float64)
    T_prev = T_ref_guess.astype(np.float64)
    print('Solving per-frame camera translation via least_squares (real perspective formula) ...')
    for f in range(n):
        target_canvas = (Q_np[f] - t_cal) / s_cal  # driver landmarks mapped into render-canvas space
        T_f, resid_f = solve_T_least_squares(P0_np[f], R_cam_np[f], target_canvas, T_prev)
        T_all[f] = T_f
        ls_residual_px[f] = resid_f * s_cal  # convert render-canvas-px residual back to driver-px scale
        T_prev = T_f  # warm-start next frame from this one (SMIRK poses vary smoothly)
    print(f'Per-frame T-solve residual (driver px, after mapping back through s_cal): '
          f'mean={ls_residual_px.mean():.2f}  median={np.median(ls_residual_px):.2f}  max={ls_residual_px.max():.2f}')

    transform = np.zeros((n, 3, 4), dtype=np.float32)
    transform[:, :, :3] = R_cam_np
    transform[:, :, 3] = T_all

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / 'fit3d_results.npz',
        s=s.numpy(), t_xy=t_xy.numpy(), transform=transform,
        err_per_landmark=err.numpy(), err_mean=err_mean.numpy(),
        P=P.numpy(), Q=Q.numpy(), fitted=fitted.numpy(), valid=valid,
        landmark_indices=landmark_indices.numpy(),
        s_cal=s_cal, t_cal=t_cal, ls_residual_px=ls_residual_px,
        P0=P0_np, R_cam=R_cam_np, T=T_all,
    )
    print(f"Saved {OUT_DIR / 'fit3d_results.npz'}")

    print('\nRendering landmark overlays for all 245 frames...')
    plot_fit3d_overlays(P.numpy(), Q.numpy(), fitted.numpy(), valid, err_mean.numpy(), OVERLAY_DIR)
    print(f'Saved {n} overlay images to {OVERLAY_DIR}')

    strip_path = OUT_DIR / 'fit3d_hard_frame_strip.png'
    make_hard_frame_strip(P.numpy(), Q.numpy(), fitted.numpy(), valid, err_mean.numpy(), strip_path)
    print(f'Saved {strip_path}')

    print('\nHard-frame quick numbers (weak-perspective err is a diagnostic only; '
          'T-solve residual is what actually feeds the render this round):')
    for f in STRIP_FRAMES:
        print(f'  frame {f:4d}: weak-persp landmark err = {err_mean[f]:.2f}px, '
              f'T-solve residual = {ls_residual_px[f]:.2f}px, '
              f'T=({T_all[f,0]:.3f},{T_all[f,1]:.3f},{T_all[f,2]:.3f})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
