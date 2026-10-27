"""Frozen horizontal-gaze calibration, production wiring.

Origin: scratch investigation under v3/debug_out/3d4/gaze_calib/ --
    1. eye_pose_test/_run.py  -- confirmed FLAME's eye_pose_params (6-dim,
       currently hardcoded to zeros in render_gaga.py) is NOT inert: produces
       a real, correctly-localized pixel change post-render.
    2. _audit_iris_residual.py (Part A) -- used mediapipe iris landmarks
       (landmarks_mp[:,468:478]) to quantify how much raw iris motion is a
       head-yaw confound vs a real residual gaze signal (~27-46% residual
       after removing linear yaw dependence, horizontal axis).
    3. _axis_sign_sweep.py / _yaw_sign_zoom.py / _dim4_bigzoom.py (Part B) --
       empirically (render sweeps, not FLAME-convention guessing) established
       which of the 6 eye_pose_params dims drives which eye/axis/sign:
         dim1 = R eye horizontal gaze, dim4 = L eye horizontal gaze.
         R eye: medial(nose-ward) gaze -> NEGATIVE dim1.
         L eye: medial(nose-ward) gaze -> POSITIVE dim4.
       (The two eyes need OPPOSITE-sign gains -- confirmed, not assumed.)
    4. _build_calib.py -- fit separate gains (max-abs scaling against the
       highest-residual calibration frames, clipped to [-0.5,0.5]) mapping
       yaw-deconfounded residual -> dim1 / dim4.
    5. _review_and_motion.py -- visually validated: 1x static review (11
       frames: near-frontal fit, large-turn f80/f84, held-out f40/f45/f49)
       plus a 28-frame motion clip (f58-85, highest-residual-activity window
       that also spans the largest yaw swing) -- correct gaze direction,
       held-out frames stayed near-neutral under a big turn with no real
       gaze signal, no popping/artifacting in motion. User-approved to
       promote into production (see project memory).

Scope, deliberately: HORIZONTAL gaze only (dims 1, 4). Vertical gaze (dims
0, 3) is explicitly left UNRESOLVED -- not a pitch hack, not zeroed-with-
intent-to-fill-later, just untouched by this module. Roll (dims 2, 5) is
not driven by anything here either; both are the caller's responsibility to
leave alone (render_gaga.py leaves them at 0 by construction of the 6-vector
this module returns).

Blink is a separate, not-yet-solved problem (SMIRK eyelid_params dropped by
fit3d, never reaches FLAME) -- this module does NOT implement blink. It only
guards against the blink CONTAMINATING the horizontal-gaze signal: on frames
where a per-eye eyelid-closure heuristic fires, that eye's predicted dim is
zeroed rather than trusting a noisy iris-landmark position on a closed lid.
"""
import numpy as np

# Empirically-verified per-eye iris/corner landmark indices (mediapipe
# FaceLandmarker full 478-point set, landmarks_mp in track_results.npz).
# NOTE: an early scratch draft had the R eye's outer/inner corners swapped
# (outer=362, inner=263). Corrected here: outer=263 (lateral/temporal),
# inner=362 (medial/nasal) -- verified via direct pixel-coordinate check at
# f92 (landmark 362 x=743.0 < landmark 263 x=785.7; the R eye's medial side
# is the smaller-x side, since the face midline sits to the R eye's left).
EYES = {
    "L": dict(outer=33, inner=133, iris=list(range(468, 473))),
    "R": dict(outer=263, inner=362, iris=list(range(473, 478))),
}

# Frozen gains from the scratch calibration fit (v3/debug_out/3d4/gaze_calib/
# calib_gains.json): simple max-abs scaling against the highest-residual
# calibration frames (capped at |dim|<=0.3 during fitting; predict-time
# values below are additionally clipped to [-0.5,0.5]). Kept as a "small
# robust linear fit", deliberately not a nonlinear model, per the original
# scoping instruction. Provisional but visually validated -- do not re-tune
# without a visible production-regression reason (see project memory).
GAIN_R = 3.4135502750785136  # residual_R -> dim1 magnitude
GAIN_L = 2.5739787245906225  # residual_L -> dim4 magnitude

BLINK_EYELID_THRESH = 0.32  # matches the scratch audit's blink-exclusion cutoff


def _compute_h_raw(landmarks_mp, side):
    """Per-eye horizontal iris offset: positive = toward the eye's medial
    (nasal) corner, negative = toward the lateral (temporal) corner. Same
    convention for both eyes (post outer/inner fix above)."""
    idx = EYES[side]
    outer, inner = landmarks_mp[:, idx["outer"], :2], landmarks_mp[:, idx["inner"], :2]
    iris_c = landmarks_mp[:, idx["iris"], :2].mean(axis=1)
    eaxis = inner - outer
    espan = np.linalg.norm(eaxis, axis=1, keepdims=True)
    eaxis_u = eaxis / np.clip(espan, 1e-6, None)
    center_h = (outer + inner) / 2.0
    return np.einsum("ij,ij->i", iris_c - center_h, eaxis_u) / np.clip(espan[:, 0], 1e-6, None)


def _yaw_deconfound(h_raw, yaw_deg, ok_mask):
    """Linear-regress h_raw on head yaw (over ok_mask only) and return the
    full-length residual (raw - predicted), i.e. the part of iris motion NOT
    explained by head pose. This regression is learned from the whole
    sequence, so it can still retain some gaze/head-motion correlation --
    it's a deconfounding, not a perfect separation (see project memory)."""
    A = np.vstack([yaw_deg[ok_mask], np.ones(int(ok_mask.sum()))]).T
    coef, *_ = np.linalg.lstsq(A, h_raw[ok_mask], rcond=None)
    resid = h_raw - (coef[0] * yaw_deg + coef[1])
    return resid, coef


def compute_gaze_eye_pose_dims(landmarks_mp, global_rot, valid, eyelid):
    """Compute FLAME eye_pose_params dims 1 (R eye horizontal) and 4 (L eye
    horizontal) for every frame in a track_results.npz-shaped set of arrays.

    Args (all from track_results.npz, full (n,...) arrays, not per-frame):
        landmarks_mp: (n,478,3) mediapipe landmarks (x_px,y_px,z_norm).
        global_rot:   (n,3) axis-angle head rotation (index 1 = yaw, deg via
                      np.degrees -- same convention as stabilize.py).
        valid:        (n,) bool, tracking-valid mask.
        eyelid:       (n,2) closure heuristic, column 0=L, 1=R (same
                      convention as the eye/mouth diagnostic phase).

    Returns:
        dim1, dim4: (n,) float32 arrays, clipped to [-0.5,0.5], zeroed on
        blink frames (per-eye) and on invalid-tracking frames.
    """
    yaw_deg = np.degrees(global_rot[:, 1].astype(np.float64))
    n = landmarks_mp.shape[0]
    dim1 = np.zeros(n, dtype=np.float32)
    dim4 = np.zeros(n, dtype=np.float32)
    for side in ("L", "R"):
        h_raw = _compute_h_raw(landmarks_mp, side)
        blink = eyelid[:, 0 if side == "L" else 1] > BLINK_EYELID_THRESH
        ok = valid & ~blink
        resid, _coef = _yaw_deconfound(h_raw, yaw_deg, ok)
        if side == "R":
            d = np.clip(-GAIN_R * resid, -0.5, 0.5).astype(np.float32)
        else:
            d = np.clip(GAIN_L * resid, -0.5, 0.5).astype(np.float32)
        d[blink] = 0.0
        d[~valid] = 0.0
        if side == "R":
            dim1 = d
        else:
            dim4 = d
    return dim1, dim4
