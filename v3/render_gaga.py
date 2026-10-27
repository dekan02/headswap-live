#!/usr/bin/env python
"""
Phase 3D-2 true close-out, Step 3: v3/render_gaga.py -- the PLAN_3D.md module
layout's "thin wrapper over ../GAGAvatar render, now taking {s,R,t_xy}"
(Stage 5 of the architecture in docs/PLAN_3D.md section 3). Batch-renders all
245 frames using:
  - fit3d.py's round-2 camera-extrinsic rotation + least-squares-solved T
    (v3/debug_out/3d2/fit3d_results.npz) -- unchanged this pass. Step 1 of
    this same close-out pass investigated whether the render-canvas<->
    driver-frame calibration (s_cal, t_cal) could be tightened and found it
    cannot be (it's already doing everything the 105-point face-landmark
    correspondence table can constrain -- see NOTES_3D0_findings.md's "Phase
    3D-2 true close-out" section) -- so "the corrected calibration" going
    into this batch render IS fit3d.py's existing s_cal/t_cal, confirmed
    correct for what it measures, not a new number.
  - stabilize.py's conclusion: every one of the 5 motion channels (rotation,
    T_z, T_xy, jaw, expression) FAILED the lag/smear gate this round, so all
    channels ship RAW/unfiltered here (v3/stabilize.py's apply_stabilization()
    called for completeness/future-proofing, but with every CONFIG entry
    still `enabled=False` -- this call is a pass-through no-op this round).

Per frame, saves TWO things (Stage 5 render output, not Stage 7 composite --
blend_v3.py, which does occlusion/seam/colour, is a separate future pass and
is NOT run here):
  1. `render_gaga_frames/{f:05d}_avatar.png`  -- RGBA, render-canvas-native
     avatar output at 512 x CANVAS_H (CANVAS_H default 768, the below-jaw
     coverage canvas; `--canvas-h 512` for the legacy square). The actual
     Stage-5 artifact blend_v3.py will consume later; un-warped, alpha as the
     4th channel.
  2. `render_gaga_frames/{f:05d}_naive_composite.png` -- driver-frame-sized
     naive alpha-over composite (no occlusion reasoning, no seam, no colour
     transfer -- purely a VISUAL SPOT-CHECK artifact for this pass's contact-
     sheet gate, same naive-composite formula render_silhouette_check.py
     already used for the 4 original hard frames). Saved for all 245 frames
     so the contact sheet can be re-picked without re-rendering, but only
     CONTACT_SHEET_FRAMES get assembled into the single gate image below.

Visual gate (per the brief): 12-frame contact sheet -- the 4 named hard
frames [0, 36, 82, 117] plus 8 more spanning the full yaw range (both turn
directions, near-zero-yaw tail) -- checked for the recurrence of round-1's
two known failure modes (solid-black render, frozen/non-rotating output),
and specifically annotated per-frame with whether that frame falls in the
70-95 "crown-pop" window this pass's bounded scale-check just cleared.

*** UPDATE (pre-blend_v3.py verification pass): the floater-Gaussian comb
fix from v3/_line_artifact_diagnosis.py (an alpha-profile-based row mask,
see `apply_floater_mask()` below) is now applied to EVERY frame's avatar
alpha before it's saved/composited, not just frame 0. Per-frame silhouette
area before/after the mask is logged to
`render_gaga_frames_meta.npz` (`area_before`, `area_after`, `area_dropped`,
`comb_cutoff_row`) specifically so a follow-up outlier check can confirm the
mask removes a roughly-constant small area (the comb) on every frame and
never eats a disproportionate chunk of real head/hair silhouette on any
particular pose -- see NOTES_3D0_findings.md's floater-fix-generalization
section for that analysis. The canvas-edge box (bottom-edge hard cutoff,
also found in that same diagnosis) is DELIBERATELY NOT touched here -- it's
explicitly scoped into blend_v3.py's seam design instead, not patched here.

*** UPDATE (Phase 3D-4, general silhouette-edge despeckle): the same
floater-Gaussian mechanism as the crown comb, but scattered around the
whole silhouette perimeter instead of one row-band, was found during the
final visual gate's motion check (a faint dotted/streaky texture near the
jaw/ear-to-background edge, diagnosed in NOTES_3D0_findings.md). Fixed at
this stage (not in blend_v3.py) since that's where it originates --
`despeckle_silhouette()` keeps only the single largest connected component
of each frame's (post-comb-mask) nonzero alpha and zeros everything else,
applied to every frame right after `apply_floater_mask()`.

Run in the GAGAvatar venv (renders all 245 frames -- takes several minutes):
    third_party/GAGAvatar/venv/Scripts/python.exe v3/render_gaga.py
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torchvision
import lightning

_HEADSWAP_ROOT = Path(__file__).resolve().parent.parent
_GAGA_ROOT = _HEADSWAP_ROOT / "third_party" / "GAGAvatar"
sys.path.insert(0, str(_GAGA_ROOT))
sys.path.insert(0, str(_HEADSWAP_ROOT / "v3"))

import inference as offline_inference  # noqa: E402
from core.models import build_model  # noqa: E402
from core.libs.utils import ConfigDict  # noqa: E402
from core.libs.flame_model import FLAMEModel  # noqa: E402
from core.data.loader_track import build_points_planes  # noqa: E402
from core.libs.utils_renderer import render_gaussian, NUM_CHANNELS  # noqa: E402

from stabilize import (apply_stabilization, TRACK_NPZ,  # noqa: E402
                        smooth_full_transform_zero_phase,
                        transform_frame_deltas, lock_transform_motion_gated,
                        lock_transform_motion_gated_soft_release,
                        lock_transform_motion_gated_slew_release,
                        lock_transform_motion_gated_smoothstep_release,
                        lock_transform_motion_gated_damped, DAMPED_LOCK_DAMP_FACTOR,
                        DEADBAND_LOCK_CONFIG, SOFT_RELEASE_RAMP_FRAMES,
                        SLEW_MAX_TRANS_PER_FRAME, SLEW_MAX_ROT_DEG_PER_FRAME,
                        SLEW_MAX_RAMP_FRAMES, SMOOTHSTEP_RELEASE_RAMP_FRAMES)
from jaw_expand import compute_jaw_selection, displace_jaw_vertices, JAW_DISPLACE_MM_DEFAULT  # noqa: E402
from jaw_inflate import inflate_jaw_gaussian_scales, JAW_INFLATE_FACTOR_DEFAULT  # noqa: E402
from jaw_opacity import inflate_jaw_gaussian_opacity, JAW_OPACITY_FACTOR_DEFAULT  # noqa: E402
from gaze_calib import compute_gaze_eye_pose_dims  # noqa: E402
from blink_calib import compute_blink_eye_pose_dims  # noqa: E402

IDENTITY_IMAGE = str(_HEADSWAP_ROOT / "avatar" / "reference_photos" / "headswap_target.jpg")
GAGA_CKPT = str(_GAGA_ROOT / "assets" / "GAGAvatar.pt")
FIT3D_NPZ = _HEADSWAP_ROOT / "v3" / "debug_out" / "3d2" / "fit3d_results.npz"
FRAMES_DIR = _HEADSWAP_ROOT / "offline_demo" / "frames" / "test"
OUT_DIR = _HEADSWAP_ROOT / "v3" / "debug_out" / "3d3"
FRAMES_OUT_DIR = OUT_DIR / "render_gaga_frames"

# ---------------------------------------------------------------------------
# Below-jaw coverage fix (2026-09-10, paired with blend_v3.py's neck_taper gate).
# GAGA_RENDER_SIDE_DIAGNOSIS.md established that the ~6.5-8k px "below-jaw
# shortfall" seen in every gaga_overlay is NOT geometry -- it is this file's
# old 512-square render canvas ending at ~larynx height and being warped into
# the 720-row driver frame, i.e. a straight horizontal crop of the avatar's own
# (billboard) neck. The billboards already contain neck down to the collar.
#
# Fix: render a TALLER canvas (CANVAS_W stays 512, CANVAS_H raised) with the
# vertical focal length scaled by 512/CANVAS_H so the head is *pixel-identical*
# to the 512 render in the centre band and the extra rows are pure added
# context (proof: the rasteriser maps camera Y to pixel row as
# (Y/Z)*focal_y*0.5*H + 0.5*H; holding focal_y*H constant makes the first term
# invariant, and the second differs by exactly CANVAS_PAD = (CANVAS_H-512)//2).
# So crown/hair silhouette + overshoot MUST be unchanged; only the previously
# clipped neck rows appear. The compositor warp M compensates the CANVAS_PAD
# downshift with t_cal[1] -> t_cal[1] - s_cal*CANVAS_PAD (x is untouched); see
# blend_v3.load_M(canvas_h).
#
# StyleUNet is hard-wired 512x512-square, so the tall RGB is assembled from
# two 512-crop upsampler passes (top-aligned + bottom-aligned) feathered in
# their row overlap -- see _upsample_tall().
#
# DEFAULT_CANVAS_H == 768 is the production canvas. `--canvas-h 512` is the
# byte-identical legacy path (pad 0, single upsampler pass, base M).
CANVAS_W = 512
DEFAULT_CANVAS_H = 768
# 12-frame diagnostic set used throughout the coverage investigation.
DIAG_FRAMES = [5, 15, 17, 30, 36, 45, 60, 75, 78, 80, 91, 105]

HARD_FRAMES = [0, 36, 82, 117]
# 8 more spanning the full yaw range: strong negative yaw (-21/-33), the
# transition/near-zero region on both sides of the turns, strong positive
# yaw distinct from frame 82, and a late near-zero-yaw tail frame for
# contrast against the turn window. Picked directly from the measured yaw
# curve (v3/debug_out/3d3/full_clip_results.npz), not guessed blind.
SPREAD_FRAMES = [5, 20, 45, 60, 70, 90, 125, 200]
CONTACT_SHEET_FRAMES = HARD_FRAMES + SPREAD_FRAMES
CROWN_POP_WINDOW = (70, 95)

# floater-Gaussian comb fix (v3/_line_artifact_diagnosis.py) -- SAME threshold,
# applied per-frame here (that diagnosis only checked frame 0).
COMB_ALPHA_THRESHOLD = 40
COMB_MIN_RUN = 15          # rows the profile must STAY above threshold before
                            # being trusted as "real head start", not a comb blip
SOLID_ALPHA_THRESHOLD = 128  # for area-before/after silhouette-pixel counts

# Phase 3D-4 close-out, Fix 1: the cutoff used to be a hard `alpha[:cutoff]=0`
# zero-out. Diagnosed directly (NOTES_3D0_findings.md's "second follow-up"
# section) that the RAW pre-mask alpha ramps up smoothly through the cutoff
# row -- there's no genuine hard edge in the render there, the hard edge was
# an artifact OF the masking step. TAPER_ROWS softens exactly that transition
# with a cosine ease-in centered on cutoff_row, while leaving the comb-
# detection trigger (cutoff_row itself, and everything COMB_MIN_RUN rows or
# more above it) completely unchanged -- still fully zeroed, so the comb is
# still removed exactly as before; only the few rows right at the boundary
# change from a step to a smooth ramp.
TAPER_ROWS = 5


def apply_floater_mask(avatar_alpha):
    """Zeros render-canvas rows above the first row where the row-mean alpha
    profile stays above COMB_ALPHA_THRESHOLD for COMB_MIN_RUN consecutive rows
    -- same logic validated on frame 0 in v3/_line_artifact_diagnosis.py, run
    here per-frame since the comb's image-space position is pose-dependent
    (it's fixed in the avatar's own reconstruction, not the driver frame).
    The transition at cutoff_row is a cosine taper over +/-TAPER_ROWS (Phase
    3D-4 close-out Fix 1), not a hard zero -- see TAPER_ROWS's comment above.
    Returns (masked_alpha, cutoff_row, area_before, area_after)."""
    profile = avatar_alpha.mean(axis=1).astype(np.float64)
    above = profile > COMB_ALPHA_THRESHOLD
    cutoff_row = 0
    for r in range(len(above)):
        if above[r:r + COMB_MIN_RUN].all():
            cutoff_row = r
            break
    else:
        cutoff_row = 0  # profile never sustains above threshold -- don't mask anything
    area_before = int((avatar_alpha > SOLID_ALPHA_THRESHOLD).sum())

    masked = avatar_alpha.astype(np.float64).copy()
    if cutoff_row > 0:
        lo = max(0, cutoff_row - TAPER_ROWS)
        hi = min(avatar_alpha.shape[0], cutoff_row + TAPER_ROWS)
        masked[:lo, :] = 0.0
        band_h = hi - lo
        if band_h > 0:
            # cosine ease 0->1 across the band, centered on cutoff_row --
            # rows well below lo stay fully zeroed (comb still removed),
            # rows at/after hi are untouched (same as the old hard cutoff).
            t = (np.arange(lo, hi) + 0.5 - lo) / band_h
            w = 0.5 - 0.5 * np.cos(np.pi * t)
            masked[lo:hi, :] *= w[:, None]
    masked = np.clip(np.round(masked), 0, 255).astype(np.uint8)
    area_after = int((masked > SOLID_ALPHA_THRESHOLD).sum())
    return masked, cutoff_row, area_before, area_after


# Phase 3D-4: general silhouette-edge despeckle. Diagnosed (NOTES_3D0_
# findings.md's "diagnosis of the dotted/stippled jaw-to-background seam"
# section) as the same floater-Gaussian mechanism as the crown comb above,
# but scattered around the whole silhouette perimeter (mostly near hair)
# instead of one contiguous row-band -- so apply_floater_mask's row-profile
# comb detector can't catch it; a different mask is needed.
#
# First approach tried and REJECTED: keep only the single largest connected
# component of (alpha>0). Connected-component analysis at the native
# 512-canvas resolution showed exactly one large component (>97% of all
# nonzero-alpha pixels) plus many small, clearly-spurious ones at most
# frames/poses (visually confirmed disconnected floater specks, e.g.
# diag_component2_f54.png) -- but frame 30 falsified the "just keep the
# largest CC" rule directly: its streak-shaped noise cluster turned out to
# be topologically connected to the real silhouette through a chain of
# faint (max alpha 18) bridging pixels, so it shared the SAME (single
# largest) component as real head/hair and survived un-despeckled --
# confirmed by checking the streak pixels' own component label, which
# equalled the kept main label. A pure size/connectivity rule on the raw
# nonzero mask can't distinguish "real soft hair edge, incidentally always
# connected" from "noise cluster, incidentally bridged by a few near-zero
# pixels" -- both look identical to an alpha>0 connectivity graph.
#
# Actual approach: hysteresis, analogous to Canny edge linking. `T_HIGH`
# (reusing COMB_ALPHA_THRESHOLD=40, the same "is this really solid content"
# bar apply_floater_mask already trusts) defines a STRONG mask; only the
# strong mask's largest connected component is kept as the real-content
# "core" (a stray streak or fleck practically never reaches 40/255 as a
# sizeable connected blob of its own -- verified: across 12 test frames
# spanning the clip's pose range, whatever this scheme ends up removing
# never exceeds alpha 44/255, i.e. it only ever touches noise-level alpha,
# confirmed to never touch anything resembling real solid content).
# `DILATE_RADIUS_PX` then restores the natural soft antialiased falloff
# immediately around that core (a real edge's alpha ramps from ~40 down to
# 0 over roughly 10-15px, confirmed directly on frame 100's own falloff
# profile) without reaching far enough to pull back in a genuinely distant,
# separately-floating noise cluster -- confirmed on frame 30's specific
# counter-example (diag_locate_streak_f30_v2.png): clean at this setting,
# where the largest-CC-only approach left it untouched.
T_HIGH_DESPECKLE = COMB_ALPHA_THRESHOLD  # 40 -- same "solid content" bar as the comb detector
DILATE_RADIUS_PX = 15


def despeckle_silhouette(avatar_alpha, t_high=T_HIGH_DESPECKLE, dilate_radius=DILATE_RADIUS_PX):
    """Keeps only the largest connected component of (alpha>=t_high) as the
    real-content core, dilated by dilate_radius px to restore its natural
    soft edge falloff; zeros everything else. Returns (despeckled_alpha,
    area_removed)."""
    strong = (avatar_alpha >= t_high).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(strong, connectivity=8)
    if n <= 1:
        return avatar_alpha.copy(), 0  # no solid content at all -- nothing to anchor on, leave untouched
    sizes = stats[1:, cv2.CC_STAT_AREA]
    main_label = 1 + int(np.argmax(sizes))
    core = (labels == main_label).astype(np.uint8)
    k = 2 * dilate_radius + 1
    keep = cv2.dilate(core, np.ones((k, k), np.uint8)) > 0
    despeckled = np.where(keep, avatar_alpha, 0).astype(np.uint8)
    area_removed = int((avatar_alpha[~keep] > 0).sum())
    return despeckled, area_removed


def _upsample_tall(model, gen_black, canvas_h, freeze_noise=False):
    """StyleUNet is hard-wired to a 512x512 square input, so a CANVAS_H>512
    feature map [1,32,CANVAS_H,512] can't be upsampled in one pass. Run it
    twice on 512-row crops -- one top-aligned (rows [0:512], covers the head),
    one bottom-aligned (rows [CANVAS_H-512:CANVAS_H], covers the recovered
    neck) -- and linearly feather the two RGB results across their row overlap
    (A-weight 1->0 top->bottom of the overlap). For CANVAS_H<=1024 the two
    crops always overlap, so every output row has a real upsampler value; the
    face sits entirely inside crop A and the new neck strip entirely inside
    crop B, with the blend confined to the mid-overlap band. Returns
    [1,3,canvas_h,512] in [0,1]."""
    assert 512 < canvas_h <= 1024, canvas_h
    top_b = canvas_h - 512                       # first row of the bottom crop
    up_a = model.upsampler(gen_black[:, :, 0:512, :], randomize_noise=not freeze_noise)[:, :3].clamp(0, 1)
    up_b = model.upsampler(gen_black[:, :, top_b:canvas_h, :], randomize_noise=not freeze_noise)[:, :3].clamp(0, 1)
    out = gen_black.new_zeros(1, 3, canvas_h, 512)
    out[:, :, 0:top_b, :] = up_a[:, :, 0:top_b, :]                # A-only (head)
    out[:, :, 512:canvas_h, :] = up_b[:, :, 512 - top_b:512, :]   # B-only (deep neck)
    ov = 512 - top_b                                             # overlap height
    w = torch.linspace(1.0, 0.0, ov, device=gen_black.device).view(1, 1, ov, 1)
    out[:, :, top_b:512, :] = w * up_a[:, :, top_b:512, :] + (1.0 - w) * up_b[:, :, 0:ov, :]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--canvas-h', type=int, default=DEFAULT_CANVAS_H,
                    help=f'render-canvas height (width stays 512). default {DEFAULT_CANVAS_H} '
                         '(below-jaw coverage). 512 = byte-identical legacy path. A tall '
                         'value must be even offset from 512, in (512, 1024].')
    ap.add_argument('--frames', type=str, default=None,
                    help='comma-separated frame indices to render (default: all 245). '
                         '"diag" = the 12-frame coverage-investigation set. A subset run '
                         'writes only beside its frames, never the production meta/contact sheet.')
    ap.add_argument('--out-dir', type=str, default=None,
                    help='override the frames output directory.')
    ap.add_argument('--motion-adaptive-smooth', dest='motion_adaptive_smooth',
                    action='store_true', default=True,
                    help='Default ON (validated + re-verified against the collar-seam fix, '
                         'see project_motion_adaptive_smoothing.md). Apply stabilize.py\'s '
                         'yaw-rate-driven MotionAdaptiveOneEuroFilter to fit3d T (translation) '
                         '+ FLAME jaw + expression -- heavy smoothing near |dyaw/dt|~0 (at-rest '
                         'shimmer), near pass-through during turns. Rotation (R_cam / '
                         'global_rot) is never filtered by this flag. See FLICKSRC_* diagnosis '
                         'this session. Use --no-motion-adaptive-smooth to reproduce the old '
                         'raw-channel behavior.')
    ap.add_argument('--no-motion-adaptive-smooth', dest='motion_adaptive_smooth',
                    action='store_false',
                    help='Disable motion-adaptive smoothing, reproducing the pre-ship raw '
                         'T/jaw/expression path.')
    ap.add_argument('--transform-smooth', dest='transform_smooth',
                    action='store_true', default=False,
                    help='STAGED, default OFF (V46, short-window PASS not yet full-clip '
                         'regressed -- see project_v46_transform_smoothing_positive.md). Applies '
                         'a 3-frame zero-phase FIR (no group delay) to the FULL per-frame camera '
                         'transform fed to render_gaussian -- rotation (R_cam, via '
                         'continuity-fixed quaternion) AND translation T (on top of whatever T '
                         '--motion-adaptive-smooth already produced) -- targeting the '
                         'camera-transform-reprojection jitter root-caused in the Round-3 '
                         'despeckle/R_cam-T causal ablation. Distinct from stabilize.py\'s '
                         'CONFIG[\'rotation\'] (an already-rejected causal OneEuroFilter that '
                         'lagged a 38deg turn by 5.9deg) -- this kernel has zero measured lag. '
                         'CLOSED as real-but-imperceptible on full-clip blind review -- see '
                         'project_v46_closed_imperceptible.md. Do not default this on.')
    ap.add_argument('--transform-deadband-lock', dest='transform_deadband_lock',
                    action='store_true', default=False,
                    help='STAGED, default OFF (Round-3 follow-up to the closed V46 FIR '
                         'candidate). Hysteresis-gated hard lock on the full per-frame camera '
                         'transform (R_cam+T): while frame-to-frame rotation/translation '
                         'velocity stays below DEADBAND_LOCK_CONFIG\'s lo-thresholds, the '
                         'transform is held constant (sample-and-hold) instead of merely damped; '
                         'raw pass-through resumes the instant velocity crosses the hi-threshold '
                         '(no crossfade). Distinct mechanism from --transform-smooth (which '
                         'always-on light-filters every frame) -- see the causal ablation in '
                         'this session\'s Round-3 diagnostic, which froze R_cam/T outright and '
                         'measured 100%% elimination of the deep-neck ROI jitter (vs '
                         '--transform-smooth\'s ~30%%). Mutually independent of '
                         '--transform-smooth; if both are passed, smoothing is applied first, '
                         'then the lock on top. Measured NEGATIVE this round: hard-cut RELEASE '
                         'introduces 5-8px single-frame snap jumps at the deep-neck ROI (vs '
                         '0.5-1.3px baseline motion there), net WORSE than doing nothing '
                         '(peak +16.6%%, mean +7.9%% vs baseline). Superseded by '
                         '--transform-deadband-lock-soft; kept only as the diagnostic control '
                         'that isolated the snap mechanism.')
    ap.add_argument('--transform-deadband-lock-soft', dest='transform_deadband_lock_soft',
                    action='store_true', default=False,
                    help='STAGED, default OFF. Same hysteresis-gated lock as '
                         '--transform-deadband-lock, except RELEASE blends from the locked '
                         f'anchor pose to the live tracked pose over a {SOFT_RELEASE_RAMP_FRAMES}-frame '
                         'cubic ease-out ramp (quaternion slerp for rotation, linear for T) '
                         'instead of an instant cut -- targets the snap artifact '
                         '--transform-deadband-lock measured. ENGAGE is still instant (0.00px by '
                         'construction). Mutually exclusive with --transform-deadband-lock in '
                         'effect (both gate the same transform channel); if both are passed, '
                         'this soft-release version runs last and wins. Measured on f115-185: '
                         'best ROI-jitter reduction so far (peak -37.1%%, mean -35.9%% vs '
                         'baseline), no more 5-8px snaps -- but the fixed 8-frame ramp still '
                         'allows a single-frame velocity spike up to 2.66px (at f175, the '
                         'largest-accumulated-drift release), 3-5x the local baseline motion '
                         'there. Superseded by --transform-deadband-lock-slew (slew-rate '
                         'limiter); kept as the diagnostic control that isolated the spike.')
    ap.add_argument('--transform-deadband-lock-slew', dest='transform_deadband_lock_slew',
                    action='store_true', default=False,
                    help='STAGED, default OFF. Same hysteresis-gated lock as '
                         '--transform-deadband-lock, except RELEASE is a true slew-rate '
                         f'limiter (stabilize.SLEW_MAX_TRANS_PER_FRAME={SLEW_MAX_TRANS_PER_FRAME}, '
                         f'SLEW_MAX_ROT_DEG_PER_FRAME={SLEW_MAX_ROT_DEG_PER_FRAME} deg/frame, '
                         f'capped at {SLEW_MAX_RAMP_FRAMES} frames) instead of a fixed-duration '
                         'ramp -- release duration scales with however much drift accumulated '
                         'during the lock (small drift: 1-2 frames; large drift: longer, up to '
                         'the cap), targeting the --transform-deadband-lock-soft velocity spike. '
                         'ENGAGE is still instant. If more than one --transform-deadband-lock* '
                         'flag is passed, this slew version runs last and wins. Measured NEGATIVE '
                         'on f115-185: constant-velocity chasing a continuously-moving live '
                         'target compounds during the highest-motion sub-window (2 consecutive '
                         'near-maximal steps, 3.88px+4.17px at f117/f118) and the f175 release '
                         'never fully caught up inside the window -- worse max delta (4.17px) '
                         'AND worse ROI reduction (peak 23.5%%, mean 30.8%%) than '
                         '--transform-deadband-lock-soft. Superseded by '
                         '--transform-deadband-lock-smoothstep.')
    ap.add_argument('--transform-deadband-lock-smoothstep', dest='transform_deadband_lock_smoothstep',
                    action='store_true', default=False,
                    help='STAGED, default OFF. Same fixed-duration RELEASE ramp structure as '
                         '--transform-deadband-lock-soft (anchor frozen at release trigger, '
                         'quaternion slerp + linear T interp over --transform-deadband-ramp-frames '
                         'frames), except the blend weight is smoothstep (3t^2-2t^3) instead of '
                         'cubic ease-out (1-(1-t)^3). Smoothstep has zero slope at BOTH ramp '
                         'endpoints, targeting the peak-at-t=0 velocity spike that produced the '
                         '2.66px f175 jump under cubic ease-out (its interior peak velocity is '
                         'higher, at t=0.5, by construction -- the tradeoff under test). ENGAGE '
                         'stays instant. If more than one --transform-deadband-lock* flag is '
                         'passed, this smoothstep version runs last and wins.')
    ap.add_argument('--transform-deadband-ramp-frames', dest='transform_deadband_ramp_frames',
                    type=int, default=SMOOTHSTEP_RELEASE_RAMP_FRAMES,
                    help=f'RELEASE ramp duration in frames for '
                         f'--transform-deadband-lock-smoothstep (default {SMOOTHSTEP_RELEASE_RAMP_FRAMES}).')
    ap.add_argument('--transform-damped-lock', dest='transform_damped_lock',
                    action=argparse.BooleanOptionalAction, default=True,
                    help='SHIPPED default ON. Hysteresis-gated EMA damping of the camera transform '
                         '(R_cam via quaternion slerp + T) during still windows, raw pass-through '
                         f'while moving; damping factor {DAMPED_LOCK_DAMP_FACTOR} ({DAMPED_LOCK_DAMP_FACTOR*100:.0f}%%). Suppresses '
                         'neck-boundary shimmer while keeping organic micro-sway (playback-validated; '
                         'hard-lock/smoothstep variants read robotic). Skipped if any explicit '
                         '--transform-deadband-lock* diagnostic flag is passed. '
                         '--no-transform-damped-lock restores the prior behavior.')
    ap.add_argument('--jaw-mesh-expand', dest='jaw_mesh_expand',
                    action='store_true', default=False,
                    help='Local FLAME jaw mesh expansion (task3, STAGED/default OFF pending '
                         'sign-off): push the jaw/chin/ear-hinge vertices (derived from the '
                         'FLAME model\'s own ibug68 landmark embedding, not a guessed index '
                         'range -- see jaw_expand.py) outward along their per-frame surface '
                         'normal by --jaw-expand-mm, before rasterization. Targets the '
                         'a_av==0 rasterization hole at the ear-to-jaw hinge under high yaw.')
    ap.add_argument('--jaw-expand-mm', type=float, default=JAW_DISPLACE_MM_DEFAULT,
                    help=f'outward displacement in mm at full falloff weight, only applied '
                         f'when --jaw-mesh-expand is set. default {JAW_DISPLACE_MM_DEFAULT}mm '
                         '(requested range 5-12mm).')
    ap.add_argument('--jaw-gaussian-inflate', dest='jaw_gaussian_inflate',
                    action='store_true', default=False,
                    help='Local FLAME jaw/ear-hinge peripheral Gaussian scale inflation '
                         '(task4 Pathway 3, STAGED/default OFF pending sign-off): multiply the '
                         'mesh Gaussians\' rasterizer "scales" for the same jaw/chin/ear-hinge '
                         'vertex selection used by --jaw-mesh-expand (see jaw_inflate.py), '
                         'instead of moving their xyz. One-shot, applied when _gs_params is '
                         'first cached, not per-frame.')
    ap.add_argument('--jaw-inflate-factor', type=float, default=JAW_INFLATE_FACTOR_DEFAULT,
                    help=f'Gaussian scale multiplier at full selection weight, only applied '
                         f'when --jaw-gaussian-inflate is set. default {JAW_INFLATE_FACTOR_DEFAULT}.')
    ap.add_argument('--jaw-opacity-inflate', dest='jaw_opacity_inflate',
                    action='store_true', default=False,
                    help='Local FLAME jaw/ear-hinge peripheral Gaussian OPACITY inflation '
                         '(task5 Pathway 3, STAGED/default OFF pending sign-off): multiply the '
                         'mesh Gaussians\' rasterizer "opacities" (clamped to [0,1]) for the '
                         'same jaw/chin/ear-hinge vertex selection used by --jaw-mesh-expand / '
                         '--jaw-gaussian-inflate (see jaw_opacity.py). One-shot, applied when '
                         '_gs_params is first cached, not per-frame.')
    ap.add_argument('--jaw-opacity-factor', type=float, default=JAW_OPACITY_FACTOR_DEFAULT,
                    help=f'Opacity multiplier at full selection weight (clamped to 1.0), only '
                         f'applied when --jaw-opacity-inflate is set. default '
                         f'{JAW_OPACITY_FACTOR_DEFAULT}.')
    ap.add_argument('--fit3d-npz', type=str, default=None,
                    help='Override the fit3d_results.npz path (default: the production '
                         'v3/debug_out/3d2/fit3d_results.npz). For validation-only tracking '
                         're-fit experiments (task6, silhouette-loss T re-solve) that stage a '
                         'variant npz alongside the production one without touching it.')
    ap.add_argument('--gaze-calib', dest='gaze_calib', action='store_true', default=True,
                    help='Default ON (SHIPPED, tag headswap-offline-demo-v6 -- validated via '
                         'production-regression render, see project memory). Drive FLAME '
                         'eye_pose_params dims 1 (R eye) and 4 (L eye), PER FRAME, from the '
                         'frozen horizontal-gaze calibration in gaze_calib.py: mediapipe iris '
                         'landmarks -> per-eye horizontal offset -> linear head-yaw '
                         'deconfounding -> empirically-signed per-eye gain (GAIN_R=3.4136, '
                         'GAIN_L=2.5740, opposite sign per eye) -> clip [-0.5,0.5] -> zeroed on '
                         'blink frames (eyelid>0.32) and invalid-tracking frames. This DOES '
                         'track real per-frame driver eye movement (not a static/fixed gaze) --'
                         ' it is a coarse, horizontal-only proxy (linear yaw deconfound, no '
                         'depth/vergence model), not a full 3D gaze tracker, and vertical gaze '
                         'is explicitly unresolved: dims 0/3 turned out to be an eyelid-closure '
                         'actuator (see --blink-calib), not pitch, and dims 2/5 have no visible '
                         'effect, so there is no clean remaining actuator for vertical gaze on '
                         'this checkpoint. See v3/debug_out/3d4/gaze_calib/ for the scratch '
                         'derivation + visual validation. Use --no-gaze-calib to reproduce the '
                         'pre-ship static-eyes behavior (eye_pose dims 1/4 always 0).')
    ap.add_argument('--no-gaze-calib', dest='gaze_calib', action='store_false',
                    help='Disable horizontal-gaze calibration, reproducing the pre-ship '
                         'static-eyes behavior (eye_pose dims 1/4 always 0).')
    ap.add_argument('--blink-calib', dest='blink_calib', action='store_true', default=True,
                    help='Default ON (SHIPPED, tag headswap-offline-demo-v6 -- validated via '
                         'production-regression render, see project memory). Drive FLAME '
                         'eye_pose_params dims 0 (R eye) and 3 (L eye), PER FRAME, from the '
                         'frozen blink/eyelid-closure calibration in blink_calib.py: SMIRK '
                         'eyelid_params -> subtract per-eye robust-median open-reference '
                         'baseline (BASELINE_L=0.2642, BASELINE_R=0.2524) -> soft deadband '
                         '(0.04) -> divide by per-eye empirically-fit scale (SCALE_L=0.5631, '
                         'SCALE_R=0.6789) -> clip [0,1] -> x0.5 -> zeroed on invalid-tracking '
                         'frames. Clipped to the empirically-confirmed monotonic actuator range '
                         '(dim0/dim3 response is non-monotonic past ~0.5-0.6: the eyelid '
                         're-opens if pushed further). See v3/debug_out/3d4/blink_calib/ for the '
                         'scratch derivation + visual validation. Independent of --gaze-calib '
                         '(disjoint dims 0/3 vs 1/4); both are ON by default and may be '
                         'combined freely. Use --no-blink-calib to reproduce the pre-ship '
                         'no-blink behavior (eye_pose dims 0/3 always 0).')
    ap.add_argument('--no-blink-calib', dest='blink_calib', action='store_false',
                    help='Disable blink calibration, reproducing the pre-ship no-blink '
                         'behavior (eye_pose dims 0/3 always 0).')
    ap.add_argument('--freeze-noise', dest='freeze_noise', action='store_true', default=False,
                    help='STAGED, default off, diagnostic for the render-boiling thread (see '
                         'project_render_boiling_closed_accepted_limit.md). model.upsampler '
                         '(StyleUNet -> StyleGAN2GeneratorCSFT) injects fresh independent '
                         "Gaussian noise at every StyleConv layer per call when its "
                         'randomize_noise arg is left at its default True -- neither call site '
                         'in this file has ever passed it, so every frame (and, on the tall '
                         'canvas, both the up_a/up_b crops within the same frame) gets an '
                         'uncorrelated noise draw, independent of the frozen appearance and of '
                         't_points/t_transform. This flag passes randomize_noise=False instead, '
                         "using the model's registered fixed noise buffers so the same spatial "
                         'noise pattern is reused every call -- diagnostic only, not yet '
                         'benchmarked.')
    args = ap.parse_args()

    canvas_h = int(args.canvas_h)
    tall = canvas_h != 512
    if tall:
        if not (512 < canvas_h <= 1024) or (canvas_h - 512) % 2 != 0:
            ap.error('--canvas-h must be 512, or an even value in (512, 1024]')
    canvas_pad = (canvas_h - 512) // 2
    # focal_y scaled so the vertical pixel scale is identical to the 512 render
    # (see the CANVAS_H comment block up top). focal_x / width are untouched.
    FOCAL = 12.0  # == model.cam_params focal_x/focal_y, see GAGAvatar models.py
    focal_y_tall = FOCAL * 512.0 / canvas_h

    if args.frames is None:
        frame_iter = None
    elif args.frames == 'diag':
        frame_iter = list(DIAG_FRAMES)
    else:
        frame_iter = [int(x) for x in args.frames.split(',') if x.strip() != '']

    frames_out_dir = Path(args.out_dir) if args.out_dir is not None else FRAMES_OUT_DIR

    device = 'cuda'
    lightning.fabric.seed_everything(42)
    frames_out_dir.mkdir(parents=True, exist_ok=True)
    print(f'canvas = {CANVAS_W}x{canvas_h}  pad={canvas_pad}  '
          f'focal_y={"12.0" if not tall else f"{focal_y_tall:.4f}"}  -> {frames_out_dir}')

    print('Loading GAGAvatar model...')
    fabric = lightning.Fabric(accelerator=device, strategy='auto', devices=[0])
    fabric.launch()
    full_checkpoint = fabric.load(GAGA_CKPT)
    meta_cfg = ConfigDict(init_dict=full_checkpoint['meta_cfg'])
    model = build_model(model_cfg=meta_cfg.MODEL)
    model.load_state_dict(full_checkpoint['model'])
    model = fabric.setup(model)
    model.eval()

    print('Loading avatar identity (cached tracking)...')
    from core.libs.GAGAvatar_track.engines import CoreEngine as TrackEngine
    track_engine = TrackEngine(focal_length=12.0, device=device)
    feature_data = offline_inference.get_tracked_results(IDENTITY_IMAGE, track_engine)
    del track_engine
    torch.cuda.empty_cache()

    f_image = torchvision.transforms.functional.resize(feature_data['image'].to(device), (518, 518), antialias=True)[None]
    f_transform = feature_data['transform_matrix'].float().to(device)
    f_planes_cpu = build_points_planes(meta_cfg.DATASET.POINT_PLANE_SIZE, f_transform.cpu())
    f_planes = {k: v[None].to(device) for k, v in f_planes_cpu.items()}
    f_shape = feature_data['shapecode'].float().to(device)[None]

    flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=5.0, no_lmks=True).to(device)

    jaw_idx_t = jaw_weight_t = faces_t = None
    if args.jaw_mesh_expand or args.jaw_gaussian_inflate or args.jaw_opacity_inflate:
        v_template_np = flame_model.v_template.detach().cpu().numpy()
        faces_np = flame_model.get_faces().detach().cpu().numpy()
        lmk_faces_idx_np = flame_model.full_lmk_faces_idx[0].detach().cpu().numpy()
        lmk_bary_np = flame_model.full_lmk_bary_coords[0].detach().cpu().numpy()
        jaw_idx_np, jaw_weight_np = compute_jaw_selection(v_template_np, faces_np,
                                                           lmk_faces_idx_np, lmk_bary_np)
        jaw_idx_t = torch.from_numpy(jaw_idx_np).to(device)
        jaw_weight_t = torch.from_numpy(jaw_weight_np).to(device)
        faces_t = torch.from_numpy(faces_np).to(device)
        jaw_expand_delta_m = float(args.jaw_expand_mm) / 1000.0
        if args.jaw_mesh_expand:
            print(f'jaw-mesh-expand ON: {len(jaw_idx_np)} vertices selected, '
                  f'delta={args.jaw_expand_mm}mm at full weight.')
        if args.jaw_gaussian_inflate:
            print(f'jaw-gaussian-inflate ON: {len(jaw_idx_np)} vertices selected, '
                  f'scale factor={args.jaw_inflate_factor}x at full weight.')
        if args.jaw_opacity_inflate:
            print(f'jaw-opacity-inflate ON: {len(jaw_idx_np)} vertices selected, '
                  f'opacity factor={args.jaw_opacity_factor}x at full weight (clamped to 1.0).')

    fit3d_npz_path = Path(args.fit3d_npz) if args.fit3d_npz else FIT3D_NPZ
    if args.fit3d_npz:
        print(f'fit3d-npz override: loading transforms from {fit3d_npz_path} (not production).')
    fit3d = np.load(fit3d_npz_path)
    transforms_all = fit3d['transform']  # (245,3,4) R_cam=Rg.T@source_R, T=LS-solved
    s_cal, t_cal = float(fit3d['s_cal']), fit3d['t_cal']
    n = transforms_all.shape[0]
    print(f'{n} frames. Calibration s_cal={s_cal:.4f} t_cal=({t_cal[0]:.1f},{t_cal[1]:.1f}) '
          f'(unchanged from round 2 -- see Step 1 of this pass for why)')

    # stabilize.py's old constant-cutoff CONFIG: every channel ships raw (see
    # module docstring). --motion-adaptive-smooth (STAGED, default off) is a
    # separate, opt-in yaw-rate-driven filter on T/jaw/expression only -- see
    # stabilize.py's MotionAdaptiveOneEuroFilter / FLICKSRC_* diagnosis.
    stabilized = apply_stabilization(motion_adaptive=args.motion_adaptive_smooth)
    jaw_all, expression_all = stabilized['jaw'], stabilized['expression']
    if args.motion_adaptive_smooth:
        # Override ONLY the translation column of transforms_all with the
        # filtered T -- transform = hstack([R_cam, T]), so R_cam (cols 0:3) is
        # left byte-untouched. Never filter rotation (see FLICKSRC_ diagnosis).
        transforms_all = transforms_all.copy()
        transforms_all[:, :, 3] = stabilized['T'].astype(transforms_all.dtype)
        print('stabilize.py: MOTION-ADAPTIVE smoothing ON for T/jaw/expression '
              '(rotation/R_cam untouched).')
    else:
        print('stabilize.py channels applied (all pass-through/raw this round, per its own CONFIG).')

    if args.transform_smooth:
        transforms_all = smooth_full_transform_zero_phase(transforms_all, transforms_all[:, :, 3])
        print('transform-smooth ON (V46, STAGED -- CLOSED imperceptible, see '
              'project_v46_closed_imperceptible.md): zero-phase 3-tap smoothing applied to the '
              'FULL camera transform (R_cam via quaternion + T), on top of whatever T the '
              'motion-adaptive path above already produced.')

    if args.transform_deadband_lock:
        rot_delta_deg, trans_delta = transform_frame_deltas(transforms_all)
        transforms_all = lock_transform_motion_gated(
            transforms_all, rot_delta_deg, trans_delta, **DEADBAND_LOCK_CONFIG)
        n_locked = int(((rot_delta_deg < DEADBAND_LOCK_CONFIG['rot_thresh_lo']) &
                        (trans_delta < DEADBAND_LOCK_CONFIG['trans_thresh_lo'])).sum())
        print(f'transform-deadband-lock ON (STAGED): hysteresis-gated hard lock applied to the '
              f'FULL camera transform. {n_locked}/{n} frames fall under the still-frame '
              f'thresholds this round (candidate lock frames, not a guarantee every one stayed '
              f'locked -- hysteresis exit uses the separate hi-threshold).')

    if args.transform_deadband_lock_soft:
        rot_delta_deg, trans_delta = transform_frame_deltas(transforms_all)
        transforms_all = lock_transform_motion_gated_soft_release(
            transforms_all, rot_delta_deg, trans_delta, **DEADBAND_LOCK_CONFIG)
        print(f'transform-deadband-lock-soft ON (STAGED): hysteresis-gated lock with a '
              f'{SOFT_RELEASE_RAMP_FRAMES}-frame cubic ease-out RELEASE ramp applied to the FULL '
              f'camera transform.')

    if args.transform_deadband_lock_slew:
        rot_delta_deg, trans_delta = transform_frame_deltas(transforms_all)
        transforms_all = lock_transform_motion_gated_slew_release(
            transforms_all, rot_delta_deg, trans_delta, **DEADBAND_LOCK_CONFIG)
        print(f'transform-deadband-lock-slew ON (STAGED): hysteresis-gated lock with a '
              f'slew-rate-limited RELEASE (max {SLEW_MAX_TRANS_PER_FRAME} trans-units/frame, '
              f'{SLEW_MAX_ROT_DEG_PER_FRAME} deg/frame, capped at {SLEW_MAX_RAMP_FRAMES} frames) '
              f'applied to the FULL camera transform.')

    if args.transform_deadband_lock_smoothstep:
        rot_delta_deg, trans_delta = transform_frame_deltas(transforms_all)
        transforms_all = lock_transform_motion_gated_smoothstep_release(
            transforms_all, rot_delta_deg, trans_delta, **DEADBAND_LOCK_CONFIG,
            ramp_frames=args.transform_deadband_ramp_frames)
        print(f'transform-deadband-lock-smoothstep ON (STAGED): hysteresis-gated lock with a '
              f'{args.transform_deadband_ramp_frames}-frame smoothstep RELEASE ramp applied to '
              f'the FULL camera transform.')

    if args.transform_damped_lock and not (args.transform_deadband_lock or
            args.transform_deadband_lock_soft or args.transform_deadband_lock_slew or
            args.transform_deadband_lock_smoothstep):
        rot_delta_deg, trans_delta = transform_frame_deltas(transforms_all)
        transforms_all = lock_transform_motion_gated_damped(
            transforms_all, rot_delta_deg, trans_delta, **DEADBAND_LOCK_CONFIG,
            damp_factor=DAMPED_LOCK_DAMP_FACTOR)
        print(f'transform-damped-lock ON: motion-gated EMA damping '
              f'(factor {DAMPED_LOCK_DAMP_FACTOR}) applied to the FULL camera transform.')

    gaze_dim1 = gaze_dim4 = None
    if args.gaze_calib:
        track_d = np.load(TRACK_NPZ)
        gaze_dim1, gaze_dim4 = compute_gaze_eye_pose_dims(
            track_d['landmarks_mp'], track_d['global_rot'], track_d['valid'], track_d['eyelid'])
        print(f'gaze-calib ON: horizontal gaze wired to eye_pose dims 1(R)/4(L). '
              f'dim1 range=[{gaze_dim1.min():+.3f},{gaze_dim1.max():+.3f}] '
              f'dim4 range=[{gaze_dim4.min():+.3f},{gaze_dim4.max():+.3f}] '
              f'(vertical/roll dims 0,2,3,5 unchanged/zero).')

    blink_dim0 = blink_dim3 = None
    if args.blink_calib:
        track_d = np.load(TRACK_NPZ)
        blink_dim0, blink_dim3 = compute_blink_eye_pose_dims(track_d['eyelid'], track_d['valid'])
        print(f'blink-calib ON: eyelid closure wired to eye_pose dims 0(R)/3(L). '
              f'dim0 range=[{blink_dim0.min():+.3f},{blink_dim0.max():+.3f}] '
              f'dim3 range=[{blink_dim3.min():+.3f},{blink_dim3.max():+.3f}].')

    if args.freeze_noise:
        print('freeze-noise ON (DIAGNOSTIC): model.upsampler called with randomize_noise=False '
              "at every call site (both tall up_a/up_b crops), using the model's registered "
              'fixed noise buffers instead of a fresh draw per call.')

    # render camera: production is model.cam_params ({focal 12/12, size [512,512]}).
    # tall path overrides focal_y + size; head stays pixel-identical in the centre
    # band, canvas_pad extra rows appear top & bottom.
    if tall:
        cam_params = {'focal_x': FOCAL, 'focal_y': focal_y_tall, 'size': [canvas_h, CANVAS_W]}
    else:
        cam_params = model.cam_params
    # compositor warp: the tall render shifts the head DOWN by canvas_pad rows,
    # so the row translation drops by s_cal*canvas_pad to land it identically.
    M = np.array([[s_cal, 0, t_cal[0]], [0, s_cal, t_cal[1] - s_cal * canvas_pad]], dtype=np.float64)

    frames = list(range(n)) if frame_iter is None else [f for f in frame_iter if 0 <= f < n]

    contact_panels = {}
    n_black, n_frozen_suspect = 0, 0
    prev_avatar_rgb = None
    meta_rows = []

    for f in frames:
        if f % 20 == 0:
            print(f'  frame {f}/{n} ...')
        expression = torch.zeros(1, 100, device=device)
        expression[:, :50] = torch.from_numpy(expression_all[f]).to(device)
        pose_params = torch.cat([
            torch.zeros(3, device=device),
            torch.from_numpy(jaw_all[f]).to(device),
        ])[None].float()
        eye_pose = torch.zeros(1, 6, device=device)
        if args.gaze_calib:
            eye_pose[0, 1] = float(gaze_dim1[f])
            eye_pose[0, 4] = float(gaze_dim4[f])
        if args.blink_calib:
            eye_pose[0, 0] = float(blink_dim0[f])
            eye_pose[0, 3] = float(blink_dim3[f])
        t_points = flame_model(shape_params=f_shape, expression_params=expression,
                                pose_params=pose_params, eye_pose_params=eye_pose)
        if args.jaw_mesh_expand:
            t_points = displace_jaw_vertices(t_points, faces_t, jaw_idx_t, jaw_weight_t,
                                              jaw_expand_delta_m)
        t_transform = torch.from_numpy(transforms_all[f]).float().to(device)[None]

        with torch.no_grad():
            batch = {'f_image': f_image, 'f_planes': f_planes,
                     't_image': torch.zeros(1, 3, 512, 512, device=device),
                     't_points': t_points, 't_transform': t_transform}
            if not hasattr(model, '_gs_params'):
                model.forward_expression(batch)
                if args.jaw_gaussian_inflate:
                    op = model._gs_params['opacities'][0, jaw_idx_t]
                    sc = model._gs_params['scales'][0, jaw_idx_t]
                    print(f'jaw-gaussian-inflate: pre-inflate mesh-Gaussian opacities in '
                          f'selection: min={op.min().item():.4f} mean={op.mean().item():.4f} '
                          f'max={op.max().item():.4f}; scales: mean={sc.mean().item():.5f} '
                          f'max={sc.max().item():.5f}')
                    inflate_jaw_gaussian_scales(model._gs_params, jaw_idx_t, jaw_weight_t,
                                                 float(args.jaw_inflate_factor))
                if args.jaw_opacity_inflate:
                    op_before = model._gs_params['opacities'][0, jaw_idx_t].clone()
                    inflate_jaw_gaussian_opacity(model._gs_params, jaw_idx_t, jaw_weight_t,
                                                  float(args.jaw_opacity_factor))
                    op_after = model._gs_params['opacities'][0, jaw_idx_t]
                    print(f'jaw-opacity-inflate: opacities before: min={op_before.min().item():.4f} '
                          f'mean={op_before.mean().item():.4f} max={op_before.max().item():.4f}; '
                          f'after: min={op_after.min().item():.4f} mean={op_after.mean().item():.4f} '
                          f'max={op_after.max().item():.4f} '
                          f'(n at ceiling 1.0 = {int((op_after >= 0.999).sum().item())}/{len(jaw_idx_t)})')
            else:
                model._gs_params['xyz'][:, :5023] = t_points
            gs_params = model._gs_params
            black_bg = t_transform.new_zeros(1, NUM_CHANNELS)
            white_bg = t_transform.new_ones(1, NUM_CHANNELS)
            gen_black = render_gaussian(gs_params, t_transform, cam_params, bg_color=black_bg)['images']
            gen_white = render_gaussian(gs_params, t_transform, cam_params, bg_color=white_bg)['images']
            alpha = (1.0 - (gen_white - gen_black).mean(dim=1, keepdim=True)).clamp(0, 1)
            if tall:
                rgb = _upsample_tall(model, gen_black, canvas_h, freeze_noise=args.freeze_noise)
            else:
                rgb = model.upsampler(gen_black, randomize_noise=not args.freeze_noise)[:, :3].clamp(0, 1)
        avatar_rgb = (rgb[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        avatar_alpha_raw = (alpha[0, 0].cpu().numpy() * 255).round().astype(np.uint8)
        avatar_alpha, comb_cutoff_row, area_before, area_after = apply_floater_mask(avatar_alpha_raw)
        avatar_alpha, despeckle_area_removed = despeckle_silhouette(avatar_alpha)
        meta_rows.append(dict(frame=f, comb_cutoff_row=comb_cutoff_row,
                               area_before=area_before, area_after=area_after,
                               area_dropped=area_before - area_after,
                               despeckle_area_removed=despeckle_area_removed))

        # failure-mode checks (round-1 recurrence gate)
        frac_alpha_on = float((avatar_alpha > 32).mean())
        mean_rgb_where_alpha = float(avatar_rgb[avatar_alpha > 32].mean()) if frac_alpha_on > 0 else 0.0
        is_black = frac_alpha_on < 0.01 or mean_rgb_where_alpha < 3.0
        if is_black:
            n_black += 1
        if prev_avatar_rgb is not None:
            frame_diff = float(np.abs(avatar_rgb.astype(np.int16) - prev_avatar_rgb.astype(np.int16)).mean())
            if frame_diff < 0.05:  # essentially identical to previous frame -> possible frozen output
                n_frozen_suspect += 1
        prev_avatar_rgb = avatar_rgb

        avatar_bgr = cv2.cvtColor(avatar_rgb, cv2.COLOR_RGB2BGR)
        rgba = np.dstack([avatar_bgr, avatar_alpha])
        cv2.imwrite(str(frames_out_dir / f'{f:05d}_avatar.png'), rgba)

        driver_bgr = cv2.imread(str(FRAMES_DIR / f'{f:05d}.png'))
        dh, dw = driver_bgr.shape[:2]
        warped_bgr = cv2.warpAffine(avatar_bgr, M, (dw, dh), flags=cv2.INTER_LINEAR)
        warped_alpha = cv2.warpAffine(avatar_alpha, M, (dw, dh), flags=cv2.INTER_LINEAR)
        a = (warped_alpha.astype(np.float32) / 255.0)[..., None]
        naive_composite = (driver_bgr.astype(np.float32) * (1 - a) + warped_bgr.astype(np.float32) * a).astype(np.uint8)
        cv2.imwrite(str(frames_out_dir / f'{f:05d}_naive_composite.png'), naive_composite)

        if f in CONTACT_SHEET_FRAMES:
            panel = naive_composite.copy()
            in_crown_pop = CROWN_POP_WINDOW[0] <= f <= CROWN_POP_WINDOW[1]
            label = f'f{f:04d}' + (' [crown-pop window]' if in_crown_pop else '')
            cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            if is_black:
                cv2.putText(panel, 'BLACK/EMPTY!', (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            contact_panels[f] = panel

    print(f'\nDone. {len(frames)} frames rendered to {frames_out_dir}')
    print(f'Failure-mode scan: {n_black}/{len(frames)} frames flagged black/empty; '
          f'{n_frozen_suspect} consecutive-frame-pairs flagged near-identical (frozen-output suspect).')

    meta_keys = list(meta_rows[0].keys())
    meta_arrs = {k: np.array([r[k] for r in meta_rows]) for k in meta_keys}
    # framing info alongside the per-frame arrays, so the overlay side can
    # reconstruct M without re-deriving it. blend_v3 does not read this -- it
    # rebuilds M from fit3d_results.npz via load_M(canvas_h).
    meta_out = dict(canvas_h=canvas_h, canvas_w=CANVAS_W, canvas_pad=canvas_pad,
                    focal_x=FOCAL, focal_y=(FOCAL if not tall else focal_y_tall),
                    s_cal=s_cal, t_cal=t_cal, M=M, frames=np.array(frames),
                    motion_adaptive_smooth=bool(args.motion_adaptive_smooth),
                    gaze_calib=bool(args.gaze_calib), blink_calib=bool(args.blink_calib), **meta_arrs)
    full_run = frame_iter is None
    meta_path = OUT_DIR / 'render_gaga_frames_meta.npz' if full_run \
        else frames_out_dir / 'render_gaga_frames_meta.npz'
    np.savez(meta_path, **meta_out)
    area_dropped = meta_arrs['area_dropped']
    print(f'Floater-mask area_dropped (px, comb removed per frame): mean={area_dropped.mean():.1f} '
          f'median={np.median(area_dropped):.1f} min={area_dropped.min()} max={area_dropped.max()}')
    despeckle_removed = meta_arrs['despeckle_area_removed']
    print(f'Despeckle area_removed (px, non-largest components zeroed per frame): '
          f'mean={despeckle_removed.mean():.1f} median={np.median(despeckle_removed):.1f} '
          f'min={despeckle_removed.min()} max={despeckle_removed.max()}')
    print(f'Saved meta -> {meta_path}')

    # contact sheet is a production-gate artifact -- only for a full-clip run.
    if not full_run:
        print('(subset run -- skipping the production contact sheet)')
        return 0

    # ---- assemble contact sheet: 4 rows x 3 cols (12 frames) ----
    ordered = [f for f in CONTACT_SHEET_FRAMES if f in contact_panels]
    cell_h, cell_w = 200, 267
    cols = 3
    rows = (len(ordered) + cols - 1) // cols
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)
    for i, f in enumerate(ordered):
        r, c = divmod(i, cols)
        cell = cv2.resize(contact_panels[f], (cell_w, cell_h))
        sheet[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell
    header = np.full((30, cols * cell_w, 3), 255, dtype=np.uint8)
    cv2.putText(header, 'Step 3 contact sheet: 4 named hard frames + 8 spanning the yaw range (naive composite, pre-blend_v3)',
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    full_sheet = np.concatenate([header, sheet], axis=0)
    cv2.imwrite(str(OUT_DIR / 'render_gaga_contact_sheet.png'), full_sheet)
    print(f'Saved contact sheet: {OUT_DIR / "render_gaga_contact_sheet.png"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
