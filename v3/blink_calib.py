"""Frozen blink (eyelid-closure) calibration, production wiring.

Origin: scratch investigation under v3/debug_out/3d4/blink_calib/ --
    1. _eyepose_dim0_3_probe.py -- per user instruction, checked whether
       FLAME's eye_pose_params dims 0 (R eye) / 3 (L eye) -- previously
       suspected "vertical gaze / pitch" -- are actually a pitch rotation
       (iris moves, eyelid aperture unchanged) or an EYELID CLOSURE actuator
       (upper lid drops, occludes iris). Render sweep at f92 (clean frontal,
       eyes-open reference) with a horizontal crosshair confirmed closure,
       not pitch: the eyelid itself changes shape. Both eyes share the SAME
       sign convention (negative=more open, positive=more closed) -- unlike
       the gaze dims 1/4, which need opposite-sign gains per eye.
    2. _eyepose_dim0_ceiling.py -- extended the sweep (0.4 to 1.3) and found
       the response is NON-MONOTONIC: closure increases smoothly up to
       roughly val=0.5-0.6 (near-fully-closed), then the eye visibly REOPENS
       past that. Usable range is therefore [0, ~0.55], not unbounded.
    3. Raw SMIRK eyelid_params statistics (track_results.npz 'eyelid') showed
       a large non-zero resting baseline even at known-open frames (~0.19-0.31),
       with genuine blink peaks reaching only ~0.27-0.32 baseline-subtracted --
       nowhere near SMIRK's own clamp ceiling of 1.0. A raw pass-through would
       leave the avatar permanently half-lidded, so a baseline-subtraction +
       deadband + per-eye-scale scheme (this module) is required, conceptually
       parallel to gaze's yaw-deconfounding.
    4. _build_blink_calib.py -- built and froze the mapping below. Full-
       sequence scan found the strongest genuine closures are NOT confined to
       the originally-assumed f141-146 window: a second real blink/squint
       runs f19-33 during a large head turn (R-eye closure peaks at f27,
       0.322, continuing through f33 -- wider than first assumed). This was
       documented, not suppressed -- see the deadband/scale rationale below.
    5. Visual validation (1x static review of open refs f92/173/207 + both
       blink events, plus two motion clips covering f15-37 and f137-150):
       open frames stay open, both blinks close smoothly and reopen without
       overshoot or ceiling-reopening artifacts, no snapping frame-to-frame.
       User-approved to promote into production (see project memory).

Scope, deliberately: EYELID CLOSURE only (dims 0, 3). Horizontal gaze (dims
1, 4, see gaze_calib.py) is untouched by this module -- the two calibrations
are independent and share eye_pose_params only by writing to disjoint dims.
Roll (dims 2, 5) is not driven by anything here either.
"""
import numpy as np

# Curated set of frames already established (gaze-calibration review) as
# clean, near-frontal, genuinely open-eyed -- used as the "known-open"
# reference set for the per-eye resting baseline (robust median, NOT a
# percentile over the whole clip, since the whole clip contains real blinks).
OPEN_REF_FRAMES = [92, 173, 179, 184, 189, 203, 207, 40, 45, 49]

DEADBAND = 0.04  # soft subtract, above the measured non-blink noise floor (std ~0.03-0.04)
TARGET_NORM = 0.475  # conservative target for the strongest observed closure (band: 0.45-0.50)

# Frozen from the scratch calibration fit (v3/debug_out/3d4/blink_calib/
# blink_calib_meta.json). scale = strongest observed baseline-subtracted
# closure / TARGET_NORM -- deliberately NOT derived from SMIRK's absolute
# max (per explicit instruction: that risks over-driving on a partial blink
# or an unstable baseline). Provisional but visually validated -- do not
# re-tune without a visible production-regression reason (see project memory).
BASELINE_L = 0.2641654312610626
BASELINE_R = 0.25238656997680664
SCALE_L = 0.5630510449409485   # peak closure 0.2674 (f144) / TARGET_NORM
SCALE_R = 0.6788694262504578   # peak closure 0.3225 (f27)  / TARGET_NORM


def compute_blink_eye_pose_dims(eyelid, valid):
    """Compute FLAME eye_pose_params dims 0 (R eye closure) and 3 (L eye
    closure) for every frame in a track_results.npz-shaped eyelid array.

    Args:
        eyelid: (n,2) SMIRK eyelid_params, column 0=L, 1=R (track_results.npz).
        valid:  (n,) bool, tracking-valid mask.

    Returns:
        dim0, dim3: (n,) float32 arrays. dim0 = 0.5*normalized_R (R eye),
        dim3 = 0.5*normalized_L (L eye), where normalized = clip((closure -
        DEADBAND)/scale, 0, 1) and closure = max(0, eyelid - baseline). Both
        capped at 0.5, inside the empirically-confirmed monotonic region
        (ceiling ~0.55 before the actuator reopens). Zeroed on invalid-
        tracking frames.
    """
    n = eyelid.shape[0]
    closure_L = np.clip(eyelid[:, 0] - BASELINE_L, 0, None)
    closure_R = np.clip(eyelid[:, 1] - BASELINE_R, 0, None)
    closure_db_L = np.clip(closure_L - DEADBAND, 0, None)
    closure_db_R = np.clip(closure_R - DEADBAND, 0, None)
    normalized_L = np.clip(closure_db_L / SCALE_L, 0, 1)
    normalized_R = np.clip(closure_db_R / SCALE_R, 0, 1)
    dim3 = (0.5 * normalized_L).astype(np.float32)  # L eye
    dim0 = (0.5 * normalized_R).astype(np.float32)  # R eye
    dim3[~valid] = 0.0
    dim0[~valid] = 0.0
    return dim0, dim3
