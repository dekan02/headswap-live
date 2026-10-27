#!/usr/bin/env python
"""
Phase 3D-4: v3/blend_v3.py -- PLAN_3D.md Stage 7 (COMPOSITE).

====================================================================================
2026-09 PRODUCTION MERGE: whole-head erasure + static-plate recompose
(replaces the branch_a/b/c "paste avatar over driver, patch mismatches"
architecture that lived here through commit ee2f772)
====================================================================================
This module previously pasted the avatar render directly over the driver
frame and patched the seams after the fact (collar-cut distance-taper,
render-canvas edge taper, un-premultiply-with-brightness-cap, and three
named hide-mask/diffusion-fill branches). That approach went through many
band-aid fixes (Fix #1-#7, branch_b fill ramp, orange-rim un-premult cap)
and each fix for its own target defect tended to create a new visible
artifact at the boundary between corrected and uncorrected territory.

The architecture below is ported from the `v3/debug_out/3d4/fix3_verify/`
diagnostic investigation's winning prototype (`wholehead_cp_r8.build_r8`,
"round 8" whole-head erasure, plus `_track5_fullhead_lib.build_fullhead`,
"Variant F" full-head LAB harmonization). It is a genuinely different
method, not a patch on the old one:

  1. Erase the driver's ENTIRE head via a jaw-polygon (MediaPipe mandible
     landmarks, offset outward) + hair-dilate mask, feathered at the
     boundary (`We`).
  2. Fill the erased region with a STATIC background plate image (the same
     plate for every frame -- the driver never moves relative to the camera
     in this clip -- derived as a per-pixel temporal median over all clip
     frames with only the driver's head+neck excluded via BiSeNet, NOT an
     empty-room capture; driver torso/shoulders are legitimately baked in
     by construction) instead of any driver pixel.
  3. Paste the avatar's own rendered RGB directly on top of that plated
     background, using the avatar's own alpha (`a_av`), with a series of
     validated fixes for avatar-renderer-side defects (dark matte-fringe
     rim, canvas-edge clip, avatar/driver neck-color seam, soft-gray halo).
  4. On top of that, apply ONE uniform LAB whitening-transform correction to
     the avatar's ENTIRE head+neck footprint (not just the seam) so the
     avatar's tone matches the driver's face/neck key throughout, blended
     into the driver near the jaw seam with a gradient-guarded distance
     ramp (`w_smooth`) and an L*-high-frequency lock (avatar's own texture
     preserved away from the seam, driver's texture takes over at the seam)
     -- "Variant F", the option confirmed via full-video review as the
     best-looking of the F/F.1/F.2 family.

Two of Variant F's inputs (the per-frame overhang-past-jaw depth used to
size the seam ramp, and the per-frame avatar-head/driver-face LAB color
statistics) are only meaningful once temporally smoothed across the WHOLE
245-frame clip (a centered rolling median needs neighbours on both sides).
That is fundamentally different from the old branch-based method, where
every frame was 100% independent and trivially parallel. `precompute_stats()`
below is a new, explicit Phase 1 batch pass (serial, one call per frame,
GPU-bound on the BiSeNet parse) that computes and caches those two
smoothed arrays to an .npz; Phase 2 (the actual per-frame composite,
`composite_frame_v3`) reads the cached arrays and is once again
per-frame-independent / embarrassingly parallel, same as before.

Everything under `# ---- ported from wholehead_cp_r8.py ----` and
`# ---- ported from _track5_fullhead_lib.py / _track4_variantC(2)_lib.py ----`
below is copied logic (constants, control flow, comments explaining WHY),
adapted only to read from this module's existing production path constants
instead of the diagnostic tree's hardcoded paths. `v3/debug_out/3d4/
fix3_verify/wholehead_cp_r8.py` and the `_track4_*`/`_track5_*`/`_track6_*`
files are NOT imported from here and are NOT modified -- they stay exactly
as the diagnostic/prototype reference they always were.
"""
import hashlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_HEADSWAP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HEADSWAP_ROOT / "compositing"))
sys.path.insert(0, str(_HEADSWAP_ROOT / "v3"))

from parse import (FaceParser, HEAD_LABELS, CLOTH, BACKGROUND,  # noqa: E402
                    HAIR, HAT, NECK, NECK_L, L_EAR, R_EAR, EAR_R, EYE_G, SKIN, mask_for)

# ---------------------------------------------------------------------------
# production paths
# ---------------------------------------------------------------------------
FIT3D_NPZ = _HEADSWAP_ROOT / "v3" / "debug_out" / "3d2" / "fit3d_results.npz"
TRACK_NPZ = _HEADSWAP_ROOT / "v3" / "debug_out" / "3d1" / "track_results.npz"
FRAMES_DIR = _HEADSWAP_ROOT / "offline_demo" / "frames" / "test"
AVATAR_FRAMES_DIR = _HEADSWAP_ROOT / "v3" / "debug_out" / "3d3" / "render_gaga_frames"
OUT_DIR = _HEADSWAP_ROOT / "v3" / "debug_out" / "3d4"
OUT_FRAMES_DIR = OUT_DIR / "blend_v3_frames"

# Static background plate used to fill the erased driver-head region (build_r8's
# STATIC plate, `bgplate/t1c_plate_sub.png` in the fix3_verify diagnostic tree).
# That file lives under the diagnostic tree, not as a checked-in production
# asset -- copied verbatim (same bytes) to a production location here. See the
# porting report for this decision; override with a different path if the
# real production asset location should differ.
PLATE_PNG = _HEADSWAP_ROOT / "v3" / "assets" / "bgplate_t1c_plate_sub.png"

# Task 42: presence-gate threshold for static_plate()'s optional face check
# (see static_plate() docstring). Deliberately low/conservative -- biased
# toward over-rejecting, since a false reject only costs a recapture while a
# false accept ships the exact plate-contamination bug this gate exists to
# close. Calibrated against the 4 real (contaminated) live sessions found on
# disk in v3/live/session_data/, which all measured ~0.061 SKIN-class
# fraction -- this threshold sits >10x below that.
FACE_PRESENT_SKIN_FRAC_THRESHOLD = 0.005

STATS_CACHE_NPZ = OUT_DIR / "blend_v3_precompute_stats.npz"

RENDER_CANVAS_PX = 512
H, W = 720, 1280
REG = (460, 174, 1280, 720)  # erased/plated region -- (x0, y0, x1, y1)

# ---------------------------------------------------------------------------
# ---- ported from wholehead_cp_r8.py (class ids + jaw polyline) ----
# ---------------------------------------------------------------------------
FACE_SKIN_SET = tuple(l for l in HEAD_LABELS if l not in (HAIR, HAT))   # SKIN + features + ears
HAIR_SET = (HAIR, HAT)

# MediaPipe 468-topology mandible polyline (23 pts), track_results.npz's own indexing.
JAW_IDX = [127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
           377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356]

JAW_OUT_PX = 6.0
HAIR_DILATE = 22
SIDE_MARGIN = 55
TOP_MARGIN = 55
FACE_LINK_PX = 70
MIN_CC_PX = 60
FEATHER_SIGMA = 4.0
JAW_FEATHER_PX = 4.0
AV_BLUR = 5.5

# FIX A3 (avatar<->driver neck colour mismatch bar)
AV_SUPPORT_EPS = 2.0 / 255.0
AV_RGB_PAD = 3
EDGE_BAND_PX = 6
NECK_BLUR = 15

# FIX B4 (ear/hinge hair coverage, contour boundary)
UPPER_MARGIN = 18
EAR_ZONE_PX = 55
EAR_CLOSE_PX = 15
EAR_HAIR_DILATE = 20
EAR_HAIRTONE_LUMA = 110
EAR_NEAR_HAIR_PX = 10
CORE_ERODE = 8
MORPH_CLOSE_OPEN_PX = 7

# FIX A4 (renderer's own fixed bottom crop edge).
# Production avatar render canvas height (width stays RENDER_CANVAS_PX = 512).
# 768 = the below-jaw-coverage render (render_gaga.py default); a real avatar
# neck exists below the jaw. --canvas-h 512 on render_gaga.py + load_M(512) is
# the byte-identical legacy path.
RENDER_CANVAS_H = 768
CROP_EDGE_RAMP_PX = 15.0

# FIX HALO (dark matte-fringe rim baked into the avatar renderer's own RGB)
HALO_ABS_DARK_LUMA = 70
HALO_ALPHA_MIN = 120
HALO_EDGE_BAND = 10

# dehalo_avatar_rgb's cv2.inpaint (Telea) call is the single largest per-frame
# cost in the pipeline (~440-475ms/frame wall-clock, vs ~700ms/frame for
# everything else combined -- see v3/REALTIME_FEASIBILITY_REPORT.md). Full
# 245-frame ablation (2026-09-15, throwaway diagnostic scripts
# v3/_diag_dehalo_ablation.py + v3/_diag_dehalo_fullrange.py, blend_v3.py
# never edited) found skipping it changes under 0.31% of pixels in every
# frame, confined to a 1-5px silhouette-edge band; the single worst pixel
# across all 245 frames (f164, diff=625/765 max possible) is a subtle,
# localized discoloration only visible at 8x zoom directly on that pixel --
# not a visible defect at normal viewing scale anywhere in the clip. Default
# is now OFF for the ~40% per-frame speed win; flip back to True (or pass
# dehalo_inpaint=True to composite_frame_v3) if future footage shows a
# visible case this validation didn't catch.
DEHALO_INPAINT_DEFAULT = False

# FIX EDGE-CLIP (renderer's own 512x512 canvas hard-edge clip)
EDGE_CLIP_ALPHA_MIN = 200
EDGE_CLIP_RGB_MAX = 5

# candidate D (soft-gray-halo fix) -- validated default, DO NOT change.
HALO_BAND_THRESH_DEFAULT = 0.15

# Candidate D-soft (Task hairline-flicker, opt-in, default None -- byte-
# identical to the existing hard `alpha01 >= HALO_BAND_THRESH_DEFAULT` gate
# when unset). Root cause (diagnosed via _diag_hairline_flicker_* /
# _diag_partA/_partC scripts under v3/): at low-alpha silhouette-perimeter
# pixels (head-top/hairline against background), the renderer's own alpha01
# has genuine, harmless, continuous frame-to-frame noise of ~0.02-0.09 --
# NOT a rendering defect (confirmed: alpha01 itself and a_av immediately
# after morph_sdf_alpha's binarize+SDF step both stay smooth across every
# sampled flip; the value never approaches the SDF's own 0.5 threshold
# either). When that noise straddles HALO_BAND_THRESH_DEFAULT (0.15), the
# hard `>=` gate converts a few-hundredths float wobble into a full
# 100%->0%->100% one-frame a_av snap -- a real, uncorrelated flicker
# (confirmed via isolated single-frame spike-and-revert events, not smooth
# drift), visible as a ~90/255 mean luminance pop at this specific
# high-contrast (hairline vs. plain background) boundary (10/10 sampled
# events, correlation(|d a_av|,|d luma|)=0.98 at this boundary).
#
# HALO_BAND_SMOOTHSTEP_DEFAULT, when set to a (lo, hi) tuple, replaces the
# hard cutoff with `smoothstep(lo, hi, alpha01)` (0 at/below lo, 1 at/above
# hi, cubic ease between) -- a narrow ramp straddling 0.15 collapses the
# snap into a small, continuous a_av change instead. Trade-off, quantified
# on the ORIGINAL Candidate-D validation frames (f26,32,45,78,120 --
# _D_default_visual_confirm.py), 44 tracked hairline spike-and-revert events
# under the hard gate (f115-185): ramp[0.10,0.20] -> 1/44 residual spike,
# 4.06% of halo-target pixels partially reintroduced (mean~0.08, max~0.2874);
# ramp[0.12,0.18] -> 4/44 residual spikes, 2.24% reintroduced (max~0.2763);
# ramp[0.08,0.22] -> 0/44 residual spikes, 6.20% reintroduced (max~0.2922).
# Visually confirmed (whole-avatar + 3x hairline-band crops, all 5 halo
# frames, all 3 candidates): no perceptible halo difference vs. the hard
# gate at any candidate; per-pixel diff is real but small/localized
# (max_pixel_channel_diff 60-143, a few hundred to ~1100 px/frame with
# L1>10 out of ~1M). Byte-identical to the hard gate's own current behavior
# when this stays None (verified).
#
# CLOSED, NOT SHIPPED: full-clip side-by-side playback review (production
# vs. ramp[0.08,0.22], the strongest/0-residual-spike candidate) found the
# candidate perceptually indistinguishable from production -- the
# human-visible head-boundary jitter/"swimming" against the wall during the
# f115-185 still window is UNCHANGED on playback. This metric (isolated
# single-frame a_av spike-and-revert events) is real but does not correspond
# to the perceptible defect; some other, still-unidentified mechanism drives
# the visible jitter. Do not ship this candidate or change
# HALO_BAND_THRESH_DEFAULT on the strength of this diagnostic thread -- kept
# here only as an inert, off-by-default opt-in for any future reuse.
HALO_BAND_SMOOTHSTEP_DEFAULT = None

# Avatar render canvas height seen by build_r8 / composite_frame_v3. crop_edge_fix
# uses it to place the canvas-bottom ramp; at 768 the ramp lands below the driver
# frame so the avatar neck is no longer force-zeroed at the old ~larynx row.
# Overridable per-call for a legacy 512 render.
RENDER_CANVAS_H_DEFAULT = RENDER_CANVAS_H

# ---------------------------------------------------------------------------
# Below-jaw alpha taper.  Multiplies a_av by a smooth 1->0 ramp anchored to the
# per-frame jaw sign line, over [start, start+depth] px below the jaw.  Paired
# with the tall (768) render canvas: without it the tall canvas's ~solid
# below-jaw skirt composites the low-quality generic-FLAME deep neck at full
# opacity (pale smear over the shirt on turns, low-grade shimmer over the plate
# on the frontal tail).  The taper bounds the exposed avatar-neck area to ~depth
# px below the jaw (kills the smear by removing area, not by adding an edge),
# has no high-frequency transition (the warm Variant-F handoff band is feathered
# out, no fringe), and is anchored to the jaw line so it tracks the mandible per
# frame.  A fixed-depth hard clip was tested and rejected (warm collar fringe +
# floating line -- project_belowjaw_clip_rejected).
# ---------------------------------------------------------------------------
NECK_TAPER_DEFAULT = True
NECK_TAPER_START_PX_DEFAULT = 8.0        # begin the fade this many px below the jaw sign line
NECK_TAPER_DEPTH_PX_DEFAULT = 35.0       # fade a_av 1->0 over this many px below the start.
                                         # Tuned on the 245-frame review clip: turn-R neck dLab
                                         # vs driver 18.0 -> 5.8, frontal shimmer FtF 0.96 -> 0.69,
                                         # tail dLab drift +0.43 -> +0.05/s, no fringe. Deeper
                                         # values regress back toward the raw skirt.
NECK_TAPER_EASE_DEFAULT = "smoothstep"   # "linear" | "smoothstep" -- fade curve shape

# ---------------------------------------------------------------------------
# Contact shadow (opt-in, default off).  The jaw/neck boundary reads as a flat
# cutout partly because nothing darkens the driver's real neck where the
# avatar's chin/jaw would naturally cast a contact shadow onto it -- this is
# independent of NECK_TAPER's color-mismatch problem (which is about color
# ACCURACY of the exposed avatar-neck skirt, not the presence/absence of a
# shadow cue) and does not change avatar coverage, NECK_TAPER's own
# depth/shape, or a_av at all. It multiplicatively darkens `plated` (the
# driver/plate background BEFORE the avatar is laid on top) in a band
# anchored to the same jaw_taper_depth_field neck_taper uses, positioned to
# start right around where neck_taper has already faded the avatar to ~0
# (START_PX = neck_taper's own start+depth) and fading back out over
# DEPTH_PX beyond that -- so it only ever darkens the real driver/plate neck
# pixels neck_taper just exposed, never the avatar itself (visible only
# where a_av<1, by construction of the final compose). Spatially gated to a
# dilation of `support` (the avatar's own alpha footprint) so it cannot
# leak into unrelated background far from the head.
# ---------------------------------------------------------------------------
CONTACT_SHADOW_DEFAULT = False
CONTACT_SHADOW_START_PX_DEFAULT = NECK_TAPER_START_PX_DEFAULT + NECK_TAPER_DEPTH_PX_DEFAULT  # 43.0
CONTACT_SHADOW_DEPTH_PX_DEFAULT = 25.0     # fade the darkening back out over this many px past START
CONTACT_SHADOW_STRENGTH_DEFAULT = 0.35     # peak darkening multiplier (0=no shadow, 1=black)
CONTACT_SHADOW_X_PAD_PX = 40               # dilation radius of `support` gating the shadow's lateral extent

# ---------------------------------------------------------------------------
# AA-band content gate.  At silhouette anti-aliasing pixels (AABAND_AV_LO <
# a_av < AABAND_AV_HI) where BiSeNet classifies the *driver* pixel underneath
# as BACKGROUND, the globally LAB-matched, skin-toned av_rgb_canvas bleeds into
# what should be pure background at fractional alpha (dE ~20 vs the real driver
# there, along most of the head/jaw/hairline perimeter).  This zeroes a_av at
# exactly those pixels (a_av -> a_av * 0) and is identity everywhere else:
# outside the AA band the gate is 1.0 by construction, so opaque avatar pixels
# (incl. BiSeNet mis-classified interior islands) are bit-identical to before.
# Validated statically (fix3_verify _gatefix2: dE 22.75 -> 3.99 at target px,
# frac<6 0.06 -> 0.86, zero regression on the opaque-pixel and out-of-band
# criteria) and in a full 245-frame motion review (every silhouette-edge jitter
# metric <= baseline, no new flicker, no conflict with the below-jaw taper --
# both only ever reduce a_av).  Applied after neck_taper / halo_band_thresh so
# the AA-band test sees the final a_av.
# ---------------------------------------------------------------------------
AABAND_CONTENT_GATE_DEFAULT = True
AABAND_AV_LO = 0.05
AABAND_AV_HI = 0.9

# ---------------------------------------------------------------------------
# AA-band gate relaxations (Part A, live-turned-pose hairline notch, opt-in --
# all three default to the exact production gate above; each is a strict
# narrowing of `delete`, never a widening, so with all three left at their
# defaults the gate is byte-identical to before this block existed).
#
# The gate was validated offline on the wholehead_cp_r8 245-frame review clip,
# whose driver frames are stationary-ish and whose BiSeNet background calls
# are stable frame to frame. On live turned poses, a single frame's BiSeNet
# call can misclassify a handful of true, thin hair strands at the very top
# of the head silhouette as BACKGROUND for one frame (no majority vote, no
# neighboring-frame consistency check exists in the live per-snapshot path),
# so the hard per-frame gate zeroes real avatar alpha there and produces a
# dark wispy notch along the hairline edge -- a false positive the gate's own
# design doc doesn't anticipate (see block above), not a flaw in the gate's
# original true-background case.
AABAND_TEMPORAL_HYSTERESIS_DEFAULT = False   # require BACKGROUND in this frame
                                              # AND the supplied previous-frame
                                              # sl before deleting (candidate 2)
AABAND_HAIRLINE_EXEMPT_ROW_PX_DEFAULT = 0    # exempt this many rows below the
                                              # avatar's own top-of-silhouette
                                              # row from the gate entirely
                                              # (candidate 3, narrower scope)
AABAND_EAR_EXEMPT_PX_DEFAULT = 0             # candidate 4 (opt-in, default off --
                                              # strictly narrows `delete`, so 0 is
                                              # byte-identical to before this option
                                              # existed): dilate the DRIVER's own
                                              # BiSeNet ear label (L_EAR/R_EAR/EAR_R)
                                              # by this many px and exempt it from
                                              # the gate entirely. Diagnosed via
                                              # task_crescent_f86 (Task F): at turned
                                              # poses (f~72-92 in the offline demo
                                              # clip) the driver's own ear detection
                                              # is intermittent/partial, so the gate's
                                              # BACKGROUND test spuriously fires along
                                              # the ear rim and zeroes genuine partial
                                              # avatar alpha there (the ear-rim SDF
                                              # falloff from av_alpha_morph_sdf),
                                              # producing a pale crescent. This is a
                                              # SEPARATE, upstream contributor to that
                                              # crescent (the SDF falloff itself is a
                                              # core-stage baseline, unaffected by
                                              # this flag) -- causal test (aaband
                                              # gate OFF entirely) showed only a
                                              # partial reduction, not elimination.

# ---------------------------------------------------------------------------
# Candidate C -- soft alpha retention (Task V27, opt-in, default off --
# byte-identical to the existing hard `a_av[delete]=0` gate when unset).
#
# V25 (Candidate B, single-frame threshold-crossing exemption) was rejected:
# alpha-history-based, so it can only ever restore a *transition's first
# frame*, and produces a motion-correlated moving-halo artifact at the
# crown/temple silhouette during turns (26,339/29,046 crossing events are
# genuine multi-frame fringe transitions, not one-frame HOT1-like blips).
#
# V26 (numeric simulation only, no code change) found that a fixed-fraction
# retention (e.g. always keep 50% of a_av_pre at delete-fired pixels) cannot
# satisfy both ends of the population: any fraction >=25% measurably tints
# the confirmed-real low-alpha bleed controls (a_av_pre 0.05-0.15), while
# nothing short of ~100% closes the HOT1/HOT2-type near-full-alpha hard-zero
# discontinuity (HOT1 f191 a_av_pre=0.8636 hard-zeroes a (244,242,237) flash
# ~223-236 dE from both neighbors; even 75% retention leaves ~65-78 dE). An
# a_av_pre-DEPENDENT linear ramp closed both ends with measured zero cost to
# the low-alpha population and full closure of HOT1/HOT2 in that simulation.
#
# This candidate is alpha-based, not history-based like B: it looks only at
# THIS frame's own a_av_pre value at each delete-fired pixel, so (unlike B)
# it treats every frame of a persistent multi-frame event identically -- it
# is not limited to a transition's first frame.
# ---------------------------------------------------------------------------
AABAND_CANDIDATE_C_RETENTION_DEFAULT = False  # opt-in; False == byte-identical
                                               # to the existing hard gate
AABAND_CANDIDATE_C_LO = 0.2   # a_av_pre at/below this retains 0 (matches the
                               # hard gate exactly for confirmed low-alpha
                               # bleed controls, a_av_pre 0.05-0.15)
AABAND_CANDIDATE_C_HI = 0.8   # a_av_pre at/above this retains 100% (matches
                               # full retention for HOT1/HOT2, a_av_pre
                               # 0.86-0.90); linear ramp in between

# ---------------------------------------------------------------------------
# RVM lateral matte  (opt-in, STAGED -- not in the production call path).
# Robust Video Matting supplies a temporally-recurrent outer-silhouette alpha
# that tracks the true head / jaw / cheek boundary on turned frames, where
# BiSeNet's per-frame argmax leaves a ~2k-px lateral wedge un-erased (turn-L /
# turn-R) and its lateral hinge boundary snaps up to 33 px frame-to-frame.
# When enabled, this augments `erase_bin` with the RVM silhouette but ONLY at
# or above the jaw line: the below-jaw neck / collar region is owned by
# neck_taper + the AA-band gate and is excluded here by construction
# (`jaw_taper_depth_field >= -margin`, the same unclamped jaw field neck_taper
# uses -- continuous past the mandible x-endpoints, no step).  That sidesteps
# the full-silhouette RVM version's failure mode (naive inpaint of the exposed
# neck -> collar colour-bleed) structurally, not by tuning.  The change touches
# only `erase_bin` -> `We`; `a_av` (hence neck_taper AND the AA-band gate) is
# computed independently and is not reachable from here.  Below-jaw pixel
# identity is guaranteed by an explicit crossfade of the RVM-augmented `We` back
# to the production `We` across the JAW_FEATHER_PX band at the continuous jaw
# line (seam_ramp's clamped field can't cover the jaw corners on its own).
# Cached matte: matting_experiment/out/rvm_pha.npz (245 frames, in-order
# recurrent; RTX 2060 ~13 ms/frame -- see VERDICT_rvm_matting.md).
# ---------------------------------------------------------------------------
RVM_LATERAL_MATTE_DEFAULT = False
RVM_PHA_NPZ = _HEADSWAP_ROOT / "v3" / "debug_out" / "3d4" / "matting_experiment" / "out" / "rvm_pha.npz"
RVM_MATTE_THRESH = 0.5
RVM_ABOVE_JAW_MARGIN_PX = 0.0    # RVM erase allowed only where jaw_taper_depth_field >= -this
_RVM_PHA_CACHE = None

# ---------------------------------------------------------------------------
# RVM hair-alpha blend (Priority 2, 2026-09-28 demo2 critique, approved
# architecture: "BiSeNet spatial gate + RVM soft alpha"). A DIFFERENT mechanism
# than rvm_lateral_matte above (which augments erase_bin GEOMETRY above the
# jaw only, from RVM's outer silhouette threshold): this feathers the AVATAR's
# own compositing alpha (a_av) at the hair silhouette, replacing its hard
# stair-step edge with the driver's real strand-level soft alpha, without ever
# inventing coverage a_av didn't already have (multiplicative-only, so it
# structurally cannot reopen a coverage-gap dead end like av_alpha_dilate /
# crown_temple_dilate did -- see those constants' docstrings).
#
# Mechanism: `hair` (this frame's BiSeNet driver hair/hat mask, already
# computed above) is dilated by ROI_DILATE_PX to bound where RVM is allowed to
# act (never neck/faceskin/cloth, even inside that dilation) and eroded by
# CORE_ERODE_PX to mark the confident interior, left untouched ("force to 1.0"
# is a no-op there since a_av is already ~1 in real hair core). The RING in
# between (roi & ~core, excluding faceskin/neck/cloth) is where a_av is
# multiplied by RVM's per-pixel driver-frame alpha -- since a_av is already
# near-binary there and RVM is smoothly graded across several px, the product
# closely tracks RVM's own soft curve, replacing the hard edge with strand-
# level tapering. Only ever shrinks a_av relative to baseline; the erase
# geometry beneath (HAIR_DILATE=22px margin) is far wider than this ring, so
# no driver-hair leak-through is possible.
#
# Cached matte: built by task_rvm_hair_blend_build.py, a single in-order
# sequential (recurrent-state) RVM pass over all demo2 driver frames, uint8-
# quantized to keep the cache file size reasonable.
RVM_HAIR_BLEND_DEFAULT = False
RVM_HAIR_BLEND_ROI_DILATE_PX = 20   # driver-hair dilation defining the "is this a hair
                                     # zone at all" gate -- wide on purpose: measured
                                     # (f260) up to ~15-20px divergence between the raw
                                     # BiSeNet driver-hair label edge and the avatar's
                                     # OWN alpha boundary at the temples/sides (they only
                                     # coincide closely at the crown), so a tight dilation
                                     # here would miss the real edge exactly where it
                                     # matters. See RVM_HAIR_BLEND_BAND_PX below for what
                                     # actually locates the ring within this zone.
RVM_HAIR_BLEND_BAND_PX = 10   # ring is a band straddling a_av's OWN edge (distance-
                                # transform from a_av<0.02, same mechanism as the
                                # validated edge_rgb_dilate v2), not the driver-hair
                                # label boundary -- this is what fixes the misalignment.
RVM_HAIR_PHA_DEMO2_NPZ = _HEADSWAP_ROOT / "v3" / "debug_out" / "3d4" / "matting_experiment" / "out" / "rvm_pha_demo2.npz"
_RVM_PHA_DEMO2_CACHE = None

# ---------------------------------------------------------------------------
# Faceskin above-jaw dilate  (opt-in, STAGED -- not in the production call path).
# Task edge-artifacts trace (2026-09-17, task_edge_artifacts_verifyvideo_pixel_trace.md)
# re-confirmed ALPHA_BOUNDARY_DIAGNOSIS.md's Task A/B mechanism on test2: `cand`'s
# `faceskin` term carries NO outward dilation (unlike `hair`, dilated by HAIR_DILATE
# into hair_ext before entering `cand`), so BiSeNet's few-px under-segmentation at
# the skin/background edge leaves a persistent gap between erase_bin's boundary and
# the driver's true head outline -- a thin un-erased driver rim (Artifact A,
# crown/temple) and, where it runs through the excluded neck/cloth terms, a longer
# higher-contrast leak (Artifact B, ear/neck/collar).
#
# This was already prototyped once (project_coverage_gap_fix_attempt1, 2026-09-10,
# v3/debug_out/3d4/alpha_boundary_diag/COVERAGE_GAP_FIX_ATTEMPT_1.md): dilating
# faceskin closes the gap cleanly ABOVE the jaw (-40..-50% longest driver-survivor
# run on turned frames, zero collateral -- no brow/beard erosion, frontal frames
# untouched) but is a structural dead end AT/BELOW the jaw: `We = min(We_soft,
# seam_ramp)` and seam_ramp -> 0 at/below the jaw polyline regardless of how wide
# erase_bin is grown there -- widening cannot reach the ear/neck hinge (Artifact B)
# without touching seam_ramp itself, which is the deliberate, PARKED hand-off
# boundary to Variant F's w_smooth LAB harmonization (project_jawneck_seam_parked:
# six prior compositing-side attempts at that exact seam failed/regressed; "do not
# attempt a 7th"). So this flag is scoped to close Artifact A only, by design --
# it is NOT expected to close Artifact B.
#
# Implementation mirrors rvm_lateral_matte's crossfade template exactly, with the
# augmentation source swapped from an RVM silhouette to a dilated-faceskin `cand`
# recompute: augment erase_bin with the wider candidate only where
# jaw_taper_depth_field >= -margin (strictly at/above the jaw, continuous past the
# mandible x-endpoints -- same unclamped field neck_taper uses), then crossfade the
# recomputed We back to the production We across JAW_FEATHER_PX at that line. Below
# the line lat==0 and the result is pixel-identical to production, exactly like the
# rvm_lateral_matte guarantee.
# ---------------------------------------------------------------------------
FACESKIN_DILATE_ABOVE_JAW_DEFAULT = False
FACESKIN_DILATE_PX = 11    # matches attempt#1's validated "fix11" (~HAIR_DILATE/2)

# ---------------------------------------------------------------------------
# 3D mesh rim injection (opt-in, feature-flagged -- PARTIAL P0 mitigation,
# Round AB / 2026-09-12 cheek/mouth-corner leak investigation).
#
# The P0 leak: at AA-band silhouette pixels where BiSeNet misclassifies the
# driver frame underneath as BACKGROUND, aaband_content_gate (above) zeroes
# a_av there -- correct for genuine background bleed, but wrong at pixels
# where the avatar's own geometry genuinely occupies that space (a true
# occlusion-boundary edge of the FLAME mesh, not driver background). Every
# 2D-signal candidate investigated for telling these two cases apart failed:
# 2D alpha-contour geometry at any angle/y-bound restriction (Rounds
# S,U,V,W,X) and BiSeNet's own P(skin) confidence (Round Y) are both
# 2D-indistinguishable between the leak and the hairline/plain-background
# false-positive cases at this yaw range.
#
# Round Z found that the avatar's TRUE 3D mesh silhouette -- FLAME vertices +
# fit3d pose, projected via render_gaga.py's own camera math, restricted to
# front/back-facing-edge occlusion-boundary edges AND further restricted to
# the projected mesh's own 2D convex hull (excludes interior self-occlusion
# folds, e.g. the nose bridge occluding the far cheek) -- is the first signal
# that achieves a clean 0px false-positive rate against the hairline and
# plain-background test cases. It is a PARTIAL signal, not a complete fix:
# it only recovers the upper-cheek portion of the leak (~16%/9%/0% of the
# leak mask at the three validation frames) because rim vertices inside the
# validated cheek search box thin out before reaching the jaw/chin at wider
# yaw -- that gap is UNSOLVED and deliberately not attempted here. It also
# legitimately overlaps the separately-tracked "wedge" cheek-silhouette
# artifact, because that location is also a true 3D rim -- not a regression
# introduced by this change and not something this patch disambiguates.
#
# Mechanism: wherever the 3D rim mask (dilated 5x5, Round Z's own overlap-
# check kernel), restricted to Round Z/W's validated cheek-quadrant search
# box, is true AND the avatar's raw render alpha is genuinely high there
# (alpha01 >= HALO_BAND_THRESH_DEFAULT -- the existing, already-validated
# "real avatar silhouette, not fringe" floor; not a new number), the AA-band
# content gate is exempted at that pixel: `delete` reverts to False (trust
# avatar) instead of True, inside aaband_content_gate's own
# `trans & sl==BACKGROUND` test only -- nowhere else in the pipeline.
#
# Heavy (torch + FLAME) 3D machinery is imported and loaded lazily, only
# when rim3d_injection=True is actually passed -- default OFF, zero
# cost/behaviour change to every existing call path.
# ---------------------------------------------------------------------------
RIM3D_INJECTION_DEFAULT = True
RIM3D_HULL_DIST_PX = 12.0        # Round Z's validated outer-hull-vs-interior-fold cutoff
RIM3D_DILATE_PX = 5              # Round Z's validated overlap-check dilation kernel
RIM3D_ALPHA_MIN = HALO_BAND_THRESH_DEFAULT   # reuse the existing validated "real avatar content" floor
RIM3D_IDENTITY_IMAGE = _HEADSWAP_ROOT / "avatar" / "reference_photos" / "headswap_target.jpg"
RIM3D_FOCAL = 12.0
# Round Z/W's validated cheek-quadrant search box (MediaPipe-landmark-anchored,
# same indices/paddings as quadrant_box_W in the Round Z/W diagnostics). The
# rim injection is restricted to this box on purpose -- Round Z never tested,
# and this round does not attempt to extend to, forehead/chin/temple geometry
# outside it.
RIM3D_NOSE_IDX = 1
RIM3D_LOWER_EYELID_IDX = 145
RIM3D_MOUTH_CORNER_IDX = 291
RIM3D_MOUTH_CORNER_IDX_2 = 61   # the OTHER mouth corner -- needed only for the jaw-ext mouth exclusion box below
RIM3D_Y_PAD_TOP = 15
RIM3D_Y_PAD_BOT = 10
RIM3D_X_MARGIN = 40

# ---------------------------------------------------------------------------
# rim3d jaw/chin extension (Round AE / 2026-09-13). Investigated WHY the
# cheek rim thins out before the jaw/chin (task: mesh topology density,
# back-face culling, self-occlusion, or projection distortion at high yaw).
# Diagnostic (v3/debug_out/3d4/round_ae_rimjaw/diag1_rim_density.py) found
# it is NONE of those: the raw count of front/back-facing-edge silhouette
# vertices projecting into the mouth-to-jaw/chin band is stable, ~123-130
# points, at every one of f46/80/100/150 -- topology density and back-face
# culling are not the bottleneck, and there is no vertex dropout.
#
# The actual cause is two compounding, purely 2D-restriction bugs in the
# EXISTING cheek mechanism, not a rendering/geometry limitation:
#   1. `_rim3d_cheek_quad_box`'s y1 = mouth_y - RIM3D_Y_PAD_BOT hard-excludes
#      everything below the mouth corner -- the jaw/chin is never even
#      considered, regardless of what the raw rim signal contains there.
#   2. Round Z's convex-hull-distance test (RIM3D_HULL_DIST_PX=12), tuned for
#      the cheek, is the wrong reference for the jaw/chin: the jaw/chin
#      contour is legitimately CONCAVE relative to the overall head+hair
#      hull (dominated by the hairline/forehead/cheek extent), so real jaw
#      rim points sit far outside any hull-distance threshold that also
#      excludes interior folds -- median distance-to-hull for the same
#      candidate points is ~35px at f46/f80 and jumps to ~95px at f100/f150
#      (worse at higher yaw: the visible cheek bulges further past the
#      chin's true position as the head turns, exactly the "projection
#      distortion at high yaw" candidate from the task brief).
#
# Fix: for the mouth-to-jaw/chin band specifically, replace the convex-hull
# reference with the same 2D jaw curve build_r8 already computes and trusts
# elsewhere (`jaw_offset` + `jaw_sign_field`'s per-column signed distance) --
# a reference that actually follows the jaw's concave contour instead of a
# hull that cannot. A raw column-wise jaw-line distance test alone pulls in
# the mouth/lip self-occlusion fold (chin-adjacent, so also close to the jaw
# curve in image space) -- confirmed visually
# (round_ae_rimjaw/diag2_jawcurve_filter.py, f80: dense false-positive
# cluster right at the lips). An explicit mouth-corner exclusion box
# (landmarks 61 + 291, padded 30px -- RIM3D_MOUTH_EXCLUDE_PAD_PX) is
# subtracted from the candidate set to remove this fold; the original
# assumption that it was harmless via the downstream BACKGROUND gate alone
# was checked directly and found WRONG at moderate yaw (real, if small,
# a_av change inside the raw mouth/lip region before this box was added) --
# see round_ae_rimjaw/diag3_regression_check.py history for the full
# before/after numbers.
#
# THREE FINDINGS from full validation (composite_frame_v3, full-frame diff,
# direct crop inspection at f46/80/100/150 -- round_ae_rimjaw/diag3_regression_check.py),
# all more nuanced than the initial geometric prototype suggested:
#
# 1. The `aaband_content_gate` BACKGROUND-mislabel mechanism this apparatus
#    exempts pixels FROM barely fires in the jaw/chin region for this
#    footage -- only a handful of real pixels per frame (tens, not thousands;
#    full-frame effect is 0.002-0.011% at f46/80/100/150). The dominant
#    VISIBLE defect at the jaw/ear boundary across all four frames (a thin
#    light/background-colored gap, clearest at f100/f150) is a SEPARATE,
#    genuine a_av==0 avatar-alpha coverage hole -- the same failure class as
#    the explicitly out-of-scope Cord Segment 2 -- and is NOT touched by this
#    flag at all (no gate is over-firing there to relax).
#
# 2. Where the mechanism DOES fire, at moderate yaw (f46/f80) it produces a
#    small, genuine, visually-confirmed improvement: a jagged
#    background-colored notch in the jaw/beard silhouette is filled with
#    real avatar content (round_ae_rimjaw/f00046_residual_diff_zoom.png,
#    f00080_residual_diff_zoom.png).
#
# 3. At high yaw (f100/f150) the SAME mechanism produces a small but real,
#    visually-confirmed ARTIFACT instead: 1-3 isolated skin-toned specks
#    appear detached from the true head silhouette
#    (f00100_residual_diff_zoom.png). Root cause is NOT a dilation gap
#    (widening RIM3D_DILATE_PX for the jaw case only enlarged the same
#    floating patch -- f00100_residual_diff_zoom2.png) and NOT a weak-alpha
#    false hit (the artifact pixels have alpha01 = 0.99, stronger than the
#    genuine f46/f80 fixes at 0.23-0.59) -- it is a genuine FLAME
#    mesh-to-driver reprojection/registration mismatch at extreme yaw: a
#    real, confidently-rendered avatar vertex projects to the wrong 2D
#    location relative to the true silhouette. This IS the "projection
#    distortion at high yaw" candidate from the original task brief, just
#    manifesting as point-level mis-registration rather than vertex dropout
#    (diag1 already ruled out dropout: silhouette-flip vertex counts are
#    stable across yaw). A connectivity gate (RIM3D_JAW_LINK_PX, requiring an
#    exempted component to touch already-visible avatar content, mirroring
#    `keep_linked_cc`) suppresses most but not all of it -- f100 improved
#    from 3 disconnected specks to 1 small connected protrusion; this is a
#    real mesh-calibration issue, not something a masking heuristic can fully
#    close.
#
# Net: correctly implemented and a genuine (if small) improvement at moderate
# yaw, with a small residual cosmetic artifact at high yaw that does not meet
# Round AB's "zero measured false positives" bar. Shipped OPT-IN
# (default False) rather than on by default -- flip it on per-shot if the
# moderate-yaw benefit is wanted and the high-yaw protrusion is checked to be
# acceptable, or revisit FLAME calibration at high yaw before defaulting on.
# ---------------------------------------------------------------------------
RIM3D_JAW_EXT_DEFAULT = False
RIM3D_JAW_Y_MARGIN_PX = 40   # how far past the lowest jaw landmark to search (catches the neck hinge)
RIM3D_JAW_LINE_DIST_PX = 20.0  # column-wise |jaw_sign_field| tolerance (looser than the cheek hull's
                                # 12px -- the jaw landmark curve itself carries more positional noise
                                # than the mesh-vs-hull geometry the cheek test measures)
RIM3D_MOUTH_EXCLUDE_PAD_PX = 30  # explicit exclusion box around both mouth corners (see finding above);
                                   # 15px left a residual false positive from the lower-lip/chin-crease
                                   # fold dipping below corner level at f46/f80 -- widened after re-check
RIM3D_JAW_LINK_PX = 1  # connectivity requirement (see build_r8's jaw_rim_mask filtering below): an
                         # exempted jaw-rim component must touch the region that is ALREADY visible
                         # avatar content (not currently deleted), dilated by this margin. At high yaw
                         # (f100/f150) some kept jaw-curve rim points project to a small isolated patch
                         # of real (alpha01-passing) avatar geometry that sits detached from the actual
                         # visible head silhouette -- a genuine FLAME reprojection/registration
                         # mismatch at extreme angles, not a dilation-gap issue (widening the dilation
                         # kernel just enlarged the same floating patch -- tried and rejected, see
                         # round_ae_rimjaw/f00100_residual_diff_zoom2.png). Exempting it produced a
                         # disconnected floating skin-toned speck in the background -- a real, visually
                         # confirmed regression caught only by directly viewing the crop, not by the
                         # near-zero pixel-count diff alone. Requiring the exempted component to be
                         # connected to already-visible content (matching the `keep_linked_cc` pattern
                         # already used for erase_bin elsewhere in this file) suppresses exactly these
                         # disconnected islands while preserving the genuine f46/f80 silhouette-edge
                         # gap closures (which DO touch the real jaw line).

_RIM3D_STATE = None  # lazy: dict(device, flame_model, f_shape, FACES, EDGE_V,
                     #             EDGE_FACE_A, EDGE_FACE_B, jaw_all, expression_all,
                     #             transforms_all, s_cal, t_cal, y_flip)

# ---------------------------------------------------------------------------
# Avatar alpha footprint dilation (opt-in, feature-flagged -- f44 coverage-gap
# fix, Round AC / 2026-09-12). Root cause (established before this round, and
# distinct from the P0 leak rim3d_injection addresses above): at f44 the
# cheek/jaw coverage gap is NOT erase_bin failing to reach far enough --
# widening erase_bin was tried and failed -- it's a_av's OWN footprint being
# narrower than the driver's real head at the affected rows (the avatar
# render's alpha silhouette undershoots the true head boundary there). A
# small dilation of a_av directly closes that gap without touching the erase
# side at all.
#
# Applied immediately after a_av is computed (post-blur/av_clamp), BEFORE
# crop_edge_fix / neck_taper / halo_band_thresh / aaband_content_gate, so the
# widened footprint flows through every downstream a_av consumer exactly as a
# wider native render would have -- including, deliberately, the AA-band
# content gate (a wider a_av shifts more pixels into its transition band) and
# the halo_band_thresh floor. This is the known, disclosed risk: verify no
# hairline leakage and no measurable wedge (f76-84) regression before
# shipping (see Round AC validation notes).
# ---------------------------------------------------------------------------
AV_ALPHA_DILATE_DEFAULT = True   # Round AD 2026-09-13: enabled for final render --
# validated at f44 (closes the ghost line, visually confirmed) and f76/80/84 (the
# wedge's own worst-case range): full-frame changed-pixel fraction 0.838-0.887%,
# stays under the documented <1% bound at all four frames, no visible wedge
# regression by direct crop inspection. See v3/debug_out/3d4/round_ad_avdilate/.
AV_ALPHA_DILATE_PX = 3   # smallest kernel that visibly closes the f44 line; not swept further

# ---------------------------------------------------------------------------
# Avatar-alpha morphological-SDF smoothing (opt-in, per-clip-tuned -- test.mp4
# staircase-fix task). Root cause: GAGAvatar's alpha channel is computed as a
# coverage-difference estimate (1-(gen_white-gen_black).mean()) that never
# passes through the RGB path's neural upsampler/refiner, so it is inherently
# more jagged at the silhouette boundary than RGB -- confirmed NOT caused by
# despeckle_silhouette's square-kernel dilate (byte-identical across square/
# elliptical/soft/no-despeckle variants) and NOT fixable by simply widening
# AV_BLUR's sigma (halo scales ~linearly with sigma while the staircase's
# geometric wavelength survives even at sigma=20). Replaces the plain
# `cv2.GaussianBlur(alpha01, (0,0), AV_BLUR)` step with binary morphological
# open+close (removes boundary jaggedness at <=morph_radius scale directly,
# as a SHAPE operation) followed by a distance-transform-derived smooth ramp
# whose width is a free parameter independent of morph_radius -- decouples
# "how much shape-smoothing" from "how wide the halo is", unlike a single
# Gaussian sigma. Radius/ramp chosen from a 4-candidate sweep (r6/r10 @
# ramp5.5, r10 @ ramp8) validated against 7 turn-window frames: halo width
# reduced at temple/ear/jaw, staircase visibly smoothed, edge-position
# stability preserved, and -- after catching and fixing a stale-cache bug in
# the ear-notch verification script (_BUILD_R8_CORE_CACHE isn't monkeypatch-
# aware) -- confirmed NOT to clip real ear anatomy: the ground-truth RGB
# boundary at the flagged ear/jaw location is smooth with no anatomical
# notch, and r6 tracks that true smooth boundary more faithfully than
# smaller radii (r3/r4), not less. Default False (byte-identical when off);
# this is a per-clip offline-demo setting for test.mp4 (whole-clip, no frame
# window -- applies uniformly wherever av_alpha_morph_sdf=True), not
# validated as a general production default.
AV_ALPHA_MORPH_SDF_DEFAULT = False
AV_ALPHA_MORPH_SDF_RADIUS = 6
AV_ALPHA_MORPH_SDF_RAMP = 5.5


def morph_sdf_alpha(alpha01, morph_radius=AV_ALPHA_MORPH_SDF_RADIUS, ramp_sigma=AV_ALPHA_MORPH_SDF_RAMP):
    """Binary open+close (removes boundary jaggedness at <=morph_radius scale)
    then a distance-transform-based smooth ramp of width ~ramp_sigma, fully
    decoupled from morph_radius. Returns a float32 [0,1] alpha field, same
    shape as alpha01. Drop-in replacement for
    `cv2.GaussianBlur(alpha01, (0,0), AV_BLUR)` in _build_r8_core."""
    binary = (alpha01 >= 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * morph_radius + 1, 2 * morph_radius + 1))
    smoothed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, k)
    inside_dist = cv2.distanceTransform(smoothed, cv2.DIST_L2, 5)
    outside_dist = cv2.distanceTransform(1 - smoothed, cv2.DIST_L2, 5)
    signed = np.where(smoothed > 0, inside_dist, -outside_dist).astype(np.float32)
    a = np.clip(0.5 + signed / (2.0 * ramp_sigma), 0.0, 1.0).astype(np.float32)
    return a

# ---------------------------------------------------------------------------
# Crown/temple avatar-support dilation (opt-in, feature-flagged -- test2.mp4
# registration-shift task). Root-caused (diagnosis-only task, not this one):
# the crown/temple avatar-support gap that widens at high yaw is NOT a
# registration bug in M/T -- overlaying tracked vs. predicted landmarks at
# f65/81/138/143/151/233 shows the per-frame T-solve fits tracked landmarks
# essentially exactly at every yaw (visually indistinguishable dots, 5-7px
# RMS residual, flat vs. yaw). The gap is specifically where MediaPipe's
# landmark set has NO points at all -- crown, temple, hairline, upper ear --
# so nothing anchors the avatar's own rendered silhouette there, and that
# silhouette (a frozen single-frontal-photo billboard, per
# project_billboard_scoped_costing / project_gaga_render_side_diag; mesh
# Gaussians confirmed inert) foreshortens differently from the true head as
# yaw increases. This is an inherent projection limitation of the current
# avatar architecture, not a fixable registration step -- see
# project_silhouette_refit_global_shift_confirmed for the same architectural
# wall reached from a different angle.
#
# Structurally modeled on rim3d_injection (landmark-anchored quad box(es),
# opt-in flag, independent/additive of other gates) -- but NOT the same
# underlying mechanism: rim3d_injection restores real avatar alpha that the
# AA-band content gate over-deletes (alpha01 already high, just gated away).
# Here alpha01 is genuinely ~0 in the gap -- the render placed nothing
# there at all -- so the applicable mechanism is AV_ALPHA_DILATE's approach
# (grayscale-dilate the continuous a_av field so the AA falloff stays
# smooth), scoped to this region.
#
# Sizing: the measured yaw-vs-gap relationship on test2.mp4 (f65=9.8deg,
# f81=11.1deg, f138=22.0deg, f143=30.1deg, f151=35.0deg, f233=-0.0deg) is
# noisy and NON-monotonic (f81's gap exceeds f138's despite lower yaw) --
# a yaw-conditioned lookup table would necessarily under-serve some angles
# to avoid over-serving others. head_parse/hair/erase_bin/support are
# already computed for THIS exact frame at this point in build_r8, so
# measuring the true gap directly (per side, per frame) is strictly more
# accurate than proxying it through yaw, and self-calibrates to whatever
# the real gap is rather than an offline table's approximation of it.
# CROWN_TEMPLE_DILATE_MAX_PX is a safety ceiling only, set just above the
# worst single-frame gap actually measured (f151, ~40px raw extent).
# ---------------------------------------------------------------------------
CROWN_TEMPLE_DILATE_DEFAULT = False
CROWN_TEMPLE_BROW_IDX_L = 105   # left eyebrow (MediaPipe canonical index)
CROWN_TEMPLE_BROW_IDX_R = 334   # right eyebrow (MediaPipe canonical index)
CROWN_TEMPLE_Y_PAD = 15         # small pad below the brow row so the box's lower edge
                                # doesn't hard-clip right at the brow itself
CROWN_TEMPLE_GAP_PCTL = 90      # 90th percentile (not max) of the per-frame measured
                                # true_head-vs-support gap distance, per side -- avoids
                                # a single outlier pixel driving the whole kernel size
CROWN_TEMPLE_DILATE_MAX_PX = 45  # safety ceiling; see docstring above
CROWN_TEMPLE_SOLID_THRESH = 0.5  # a_av level counted as "real avatar content already
                                  # here" for sizing the dilation kernel (see below --
                                  # measuring against raw `support` undersized it)

# ---------------------------------------------------------------------------
# Erase-feather local-ROI fix (opt-in, feature-flagged -- cord artifact,
# Round AH 2026-09-12 -> Round AI -> Round AJ 2026-09-13 final form).
#
# History (why this ended up as a hardcoded local ROI, not a general rule):
#   Round AG: cord = real driver content leaking through wherever
#     0 < We < 1 (erase_bin's feather ring) AND a_av == 0 (avatar contributes
#     nothing). Round AH clamped We -> 1 wherever that ring occurs -- but
#     that ring is not a local patch, it traces the ENTIRE head/hairline
#     silhouette, so the blanket version flattened real hairline detail
#     everywhere (~14-19k px changed at hairline frames, required 0) and
#     clamped over real driver motion. Rejected.
#   Round AI found the visible cord is actually TWO segments sharing the
#     a_av==0 signal: (1) a genuine 0<We<1 partial leak, fixable by forcing
#     We->1; (2) a segment where We is ALREADY 1.0 (deep in erase_bin's
#     interior) and av_rgb_canvas is literally (0,0,0) -- a true data hole,
#     not a weight problem; no We value fixes it. Tried a general
#     shape/geometry discriminator (morphological opening, "thin cord vs
#     thick ear") to scope segment 1 -- confirmed TWICE not to work: the
#     0<We<1 ring's width is set by feather_sigma's blur radius and is
#     roughly constant everywhere, so it can't distinguish "cord" from
#     "ordinary silhouette AA" by shape at all.
#   Round AJ (this fix): given the deadline, stop looking for a general
#     rule. Use a hardcoded local ROI (EFC_ROI_*) -- explicit canvas-space
#     coordinates validated directly (see round_aj_task1_survey /
#     roi_candidate_contactsheet.png) to contain the cord's full path
#     across the whole 245-frame clip. This is an explicit, disclosed
#     trade of generality for certainty: it is correct for THIS clip's
#     geometry only and would need re-validating (or dropping) on any
#     other clip.
#
# erase_feather_clamp (segment 1 only): closes the cord artifact's upper
# segment (genuine 0<We<1 partial leak) inside EFC_ROI. The lower
# ear-to-jaw segment is NOT fixed by this -- av_rgb_canvas is (0,0,0)
# there, a genuine avatar-render coverage gap, architecturally out of
# reach for a compositing weight change (see project_jawneck_seam_parked).
# In poses where the ear is mostly hair-occluded, a small sliver of real
# ear pixels may also be clamped as a side effect of the same
# a_av==0-driven mechanism -- visually minor (confirmed via direct crop
# inspection at f100/f150) but not zero, as an earlier validation pass
# incorrectly claimed. Segment 2 (letting driver content show in the
# coverage gap) was tested and rejected -- it exposes the driver's real
# dark hair as a large black patch at some poses (see
# f080_black_patch_zoom.png) and is not present in this shipped version.
# Nowhere outside EFC_ROI is touched, regardless of a_av/We values there.
# ---------------------------------------------------------------------------
ERASE_FEATHER_CLAMP_DEFAULT = True
EFC_ROI = (380, 650, 540, 700)   # (y0, y1, x0, x1) canvas-space, THIS CLIP ONLY -- see history above

# erase_feather_clamp_ramp (Task 2, this clip's turn-window trade-off): the
# hard ROI clamp above is a net win at rest but a net loss during the
# f74-90 turn where the ear rotates into profile -- clamp ON there recreates
# the rectangular seg1-mask-shaped white patch/neck block (see
# scratch_task2_step1b_seg1mask.py's contour overlay), while clamp OFF
# reveals real skin there instead of cord. Outside that window (f<=70,
# f>=94) OFF instead introduces a new pale rim-light "ghost cord" sliver
# along the ear/hairline that ON does not have -- so neither ON nor OFF is
# a per-clip-wide improvement; per-frame coverage/deviation metrics do NOT
# separate the two regimes (tested across 17 frames, f74 vs f94 are
# numerically indistinguishable despite opposite visual verdicts), so this
# is a frame-index ramp, not a threshold gate. Optional (ry0, ry1, rx0, rx1
# -> ...) no: format is (down_start, down_end, up_start, up_end) frame
# indices; clamp strength s(f) is 1.0 (full clamp, identical to
# erase_feather_clamp_ramp=None) for f<=down_start, linearly 1->0 over
# (down_start, down_end), 0.0 for down_end<=f<=up_start, linearly 0->1 over
# (up_start, up_end), 1.0 for f>=up_end. s multiplies seg1's effective
# target (We -> We*(1-s) + 1.0*s) instead of the hard We->1.0 assignment.
# Opt-in (default None == old hard-clamp behavior, byte-identical) --
# see erase_feather_clamp's branch below for the s==1.0 fast path that
# guarantees this. Per-clip offline-demo setting for test.mp4: the demo
# recipe uses erase_feather_clamp_ramp=(70, 78, 86, 94), i.e. down-ramp
# f70-78, clamp OFF f78-86, up-ramp f86-94 -- tuned to this clip's f74-90
# ear-turn window, not a general production default.
ERASE_FEATHER_CLAMP_RAMP_DEFAULT = None

# jaw_seam_xmin_taper (Task E, this clip's f78-86 pale/dark vertical strip):
# jaw_sign_field's outside_cols clamp forces a hard-positive value for every
# column past the tracked jaw polyline's x_min, while the boundary column
# itself (x == x_min) keeps its real, small, unclamped signed-distance value
# -- seam_ramp (4px feather) sees that single-column step as a near-vertical
# notch, reported as a thin pale/dark strip that tracks x_min through f74-90
# (see debug_out/task_strip_f78/part1_repro.py: x_min matches the notch's x
# location 9/9 sampled frames, within 1px). A causal ablation (part1_causal.py)
# confirms removing the clamp entirely is NOT safe -- it also flips ~50% of
# the wider neck/ear ROI back to raw, un-erased driver pixels, matching this
# clip's already-rejected jaw_taper_continuous/B/B' history. This flag instead
# tapers ONLY the columns strictly left of x_min (x_min side only; columns
# >= x_min are always byte-identical) outward from the boundary column's own
# value to the existing clamp value over `taper_px` columns -- see
# jaw_sign_field's own xmin_taper_px docstring for the formula. Format:
# (f0, f1, f2, f3, taper_px) -- taper strength ramps 0->1 over (f0,f1), full
# 1 over [f1,f2], ramps 1->0 over (f2,f3), 0 outside [f0,f3] (same 4-point-
# ramp convention as erase_feather_clamp_ramp above). Opt-in (default None ==
# old hard-clamp behavior, byte-identical to A3 at every frame). Per-clip
# offline-demo candidate for test.mp4: (70, 76, 88, 94, 10) covers the f78-86
# strip window with edge ramps, taper_px=10 -- not a general production value.
JAW_SEAM_XMIN_TAPER_DEFAULT = None

# Task 2c: opt-in fix for jaw_taper_depth_field's out-of-tracked-range sign-flip
# discontinuity (see jaw_taper_depth_field_continuous's docstring). Default False
# == old behavior, byte-identical. NOT used by the test.mp4 offline-demo
# recipe (that recipe's ear-side discontinuity is instead handled by
# NECK_TAPER_EAR_PROTECT_DEFAULT) -- kept unused as validated diagnostic
# history, not wired into any current call path.
JAW_TAPER_CONTINUOUS_DEFAULT = False

# neck_taper_oor_gate (opt-in, default False == byte-identical). Fixes the
# rectangular "pale box" beside the ear on turned-head frames (demo2 f260-290,
# f715-720). jaw_taper_depth_field extends the jaw FLAT past the tracked jaw
# polyline's x-range, so every row below the polyline's end-row in those columns
# reads as "deep neck" and neck_taper fades the avatar's valid ear/neck-side alpha
# to 0 (vertical edge = polyline x-end, horizontal edge = its end-row). The driver
# there is already erased (We>0), so the loss shows as a near-white plate box.
# The gate pushes the taper's signed depth to ~+1 (== at the jaw: no fade) at
# columns outside the polyline x-range, weighted by We, with a soft RAMP_PX edge.
# jaw_taper_continuous does NOT cure this (f280 box 4051 -> 4006 px).
NECK_TAPER_OOR_GATE_DEFAULT = False
NECK_TAPER_OOR_RAMP_PX = 12


def _neck_taper_oor_gate(d_taper, We, jaw_off):
    xj = np.clip(jaw_off[:, 0], 0, W - 1).astype(np.float64)
    xmin, xmax = xj.min(), xj.max()
    cols = np.arange(W, dtype=np.float64)
    r = float(NECK_TAPER_OOR_RAMP_PX)
    out_r = np.clip((cols - xmax + r / 2) / r, 0.0, 1.0)
    out_l = np.clip((xmin - cols + r / 2) / r, 0.0, 1.0)
    w = np.maximum(out_r, out_l)[None, :].astype(np.float32)
    m = np.clip(We, 0.0, 1.0).astype(np.float32) * w
    return np.where(d_taper < 0, d_taper * (1.0 - m) + 1.0 * m, d_taper).astype(np.float32)
# driver_neck_own (opt-in, default False == byte-identical). DRIVER-NECK OWNERSHIP
# (demo2 jaw/neck seam): below a tracked beard/jaw contour the driver keeps its OWN
# neck/collar -- BOTH the avatar alpha and the erase weight We are multiplied to 0 there
# (a_av alone would leave the erased driver neck replaced by the static plate). Contour =
# jaw_off (MediaPipe mandible polyline) via jaw_taper_depth_field, after trimming the steep
# far-cheek leg at the chin-side end (that leg is the face-edge outline, not the beard-bottom
# contour, and printed a vertical pale stripe). Applied only inside the trimmed polyline
# x-range (10px ramps both ends); outside it baseline (taper/OOR gate) is untouched.
# a_av: smoothstep feather DRIVER_NECK_OWN_FEATHER_PX centred on the contour.
# We: switches off at the TOP of the feather (a_av~1 hides it) so no plate leaks into the
# feather band.
DRIVER_NECK_OWN_DEFAULT = False
DRIVER_NECK_OWN_FEATHER_PX = 6.0
_DNO_RAMP_PX = 10.0
# dno_ear_carry (opt-in, default False == byte-identical). Hysteresis for the chin-end choice in
# _dno_trim_chin_leg: when <=30 ear px are visible, reuse the last valid ear centroid x instead of
# the y-heuristic (which flips the trimmed end, e.g. demo2 f479->480, f587->588). Causal (no
# lookahead), so live-compatible; state is per-process -- call dno_ear_carry_reset() at clip start.
DNO_EAR_CARRY_DEFAULT = False
_DNO_EAR = {"on": DNO_EAR_CARRY_DEFAULT, "last": None}


def dno_ear_carry_reset():
    _DNO_EAR["last"] = None


def _dno_trim_chin_leg(jaw_off, sl):
    p = jaw_off.astype(np.float64)
    ear = np.isin(sl, [L_EAR, R_EAR, EAR_R])
    ex = np.where(ear)[1].mean() if ear.sum() > 30 else None
    if _DNO_EAR["on"]:
        if ex is not None: _DNO_EAR["last"] = float(ex)
        else: ex = _DNO_EAR["last"]
    if ex is None:
        ear_hi = p[np.argmax(p[:, 0]), 1] < p[np.argmin(p[:, 0]), 1]
    else:
        ear_hi = abs(ex - p[:, 0].max()) < abs(ex - p[:, 0].min())
    q = p[np.argsort(p[:, 0])]
    if ear_hi:   # chin end is the left end
        while len(q) > 4 and abs(q[1, 1] - q[0, 1]) > 1.2 * max(abs(q[1, 0] - q[0, 0]), 1e-3):
            q = q[1:]
    else:
        while len(q) > 4 and abs(q[-1, 1] - q[-2, 1]) > 1.2 * max(abs(q[-1, 0] - q[-2, 0]), 1e-3):
            q = q[:-1]
    return np.round(q).astype(np.int32)


def _driver_neck_own(a_av, We, jaw_off, sl):
    jt = _dno_trim_chin_leg(jaw_off, sl)
    d = jaw_taper_depth_field(jt)
    FW = float(DRIVER_NECK_OWN_FEATHER_PX)
    t = np.clip((d + FW / 2.0) / FW, 0.0, 1.0)
    w = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
    tW = np.clip((d - FW / 2.0) / 4.0, 0.0, 1.0)
    wW = (tW * tW * (3.0 - 2.0 * tW)).astype(np.float32)
    lo, hi = float(jt[:, 0].min()), float(jt[:, 0].max())
    xs = np.arange(a_av.shape[1], dtype=np.float32)
    inr = np.clip(np.minimum(xs - lo, hi - xs) / _DNO_RAMP_PX, 0.0, 1.0)[None, :]
    w = (1.0 - inr * (1.0 - w)).astype(np.float32)
    wW = (1.0 - inr * (1.0 - wW)).astype(np.float32)
    return (a_av * w).astype(np.float32), (We * wW).astype(np.float32)


# wall_sliver_clean (opt-in, default False == byte-identical). Removes the thin unerased
# driver ear/cheek-edge strip left on the wall just screen-left of the avatar jaw/ear (demo2
# f0-60, f100, f223, f265/275, f651). Diagnosis: a_av==0, We partial (mean 0.23-0.66), labels
# BACKGROUND/FACESKIN, i.e. driver pixels outside the erase mask. Candidate pixels V =
# (a_av<.05)&(We<.9)&(|drv-plate|_L1>45)&(<=30px from avatar core a_av>.5)&screen-left half,
# excluding NECK/NECK_L/CLOTH, rows in [avatar-core top+40, jaw polyline y max]. Keep only
# connected components (area>=30) whose mean We>0.1: the driver neck that driver_neck_own
# zeroes on purpose has We==0, so it is never touched (a Y limit alone does NOT protect it).
# Kept comps reach <=6px into the We==0 tail, then soft mask = dilate(5) -> gaussian(4) x1.6
# gain x (1-a_av); output blended toward plate_stat. Post-composite, exact for a_av==0.
WALL_SLIVER_CLEAN_DEFAULT = False
_WSC_MAX_DIST_PX = 30.0
_WSC_DIFF_L1 = 45.0
_WSC_MIN_AREA = 30
_WSC_MIN_MEAN_WE = 0.1
_WSC_TAIL_REACH_PX = 6
_WSC_DILATE_PX = 5
_WSC_SIGMA_PX = 4.0
_WSC_GAIN = 1.6
# EAR-labelled px within this many px of the avatar core are the ear lobe in the jaw/neck pocket: cleaning them
# turns the pocket into a large white plate notch, so they stay driver (f400-440).
_WSC_EAR_KEEP_PX = 20.0


def _wall_sliver_clean_mask(d):
    a = d["a_av"]; We = d["We_eff"]; sl = d["sl"]
    diff = np.abs(d["drv"].astype(np.float32) - d["plate_stat"].astype(np.float32)).sum(2)
    core = a > 0.5
    if not core.any():
        return np.zeros(a.shape, bool)
    ys, xs = np.where(core)
    V0 = (a < 0.05) & (We < 0.9) & (diff > _WSC_DIFF_L1)
    V0[:, int(np.median(xs)):] = False
    V = V0 & (We > 0.0)
    dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
    G = ~np.isin(sl, [NECK, NECK_L, CLOTH]) & (dist <= _WSC_MAX_DIST_PX)
    G[: int(ys.min()) + 40] = False
    G[int(d["jaw_off"][:, 1].max()):] = False
    V &= G; V0 &= G
    V = cv2.morphologyEx(V.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n_, lab_, st, _ = cv2.connectedComponentsWithStats(V, connectivity=8)
    keep = np.zeros(V.shape, bool)
    for i in range(1, n_):
        m = lab_ == i
        if st[i, 4] >= _WSC_MIN_AREA and We[m].mean() > _WSC_MIN_MEAN_WE:
            keep |= m
    if keep.any():
        k = 2 * _WSC_TAIL_REACH_PX + 1
        keep |= V0 & cv2.dilate(keep.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    return keep & ~(np.isin(sl, [L_EAR, R_EAR, EAR_R]) & (dist < _WSC_EAR_KEEP_PX))


def _wall_sliver_clean(out, d):
    V = _wall_sliver_clean_mask(d)
    if not V.any():
        return out
    m = cv2.GaussianBlur(cv2.dilate(V.astype(np.uint8), np.ones((_WSC_DILATE_PX, _WSC_DILATE_PX), np.uint8)).astype(np.float32),
                         (0, 0), _WSC_SIGMA_PX)
    m = np.clip(m * _WSC_GAIN, 0, 1) * (1 - d["a_av"])
    pl = d["plate_stat"].astype(np.float32)
    return (out * (1 - m[..., None]) + pl * m[..., None]).astype(np.uint8)


# How many px past the tracked jaw polyline's own x-range the continuous-field
# correction is blended in over (see jaw_taper_depth_field_continuous). Chosen
# to match the scope scan's own out-of-range search margin -- wide enough to
# cover the diagnosed jump's effective reach, narrow enough not to touch
# distant columns (shoulders/hair) where the original field already saturates
# correctly.
_JAW_TAPER_MARGIN_PX = 150.0

# Task 2c-revised: opt-in neck_taper strength ramp (Candidate C1). Default None
# == full neck_taper strength at every frame, byte-identical to current
# production. When set to (down_start, down_end, up_start, up_end), the
# neck_taper fade field `tt` is scaled by a 1->0->1 schedule (reusing
# _efc_ramp_strength's exact shape) so the ear-side jaw-taper discontinuity is
# suppressed only inside the dip window (s->0 == neck_taper=OFF there) while
# every other frame keeps s=1.0 (== neck_taper=ON, unchanged). Superseded by
# NECK_TAPER_EAR_PROTECT_DEFAULT below (Candidate C3) and NOT used by the
# test.mp4 offline-demo recipe -- kept unused, as validated dead-end/
# superseded-candidate history, not wired into any current call path.
NECK_TAPER_RAMP_DEFAULT = None

# Task 2c-revised (round 3): opt-in ear-protected neck_taper (Candidate C3).
# Default None == byte-identical to current production. C1 (neck_taper_ramp)
# was rejected because tt_eff=0 for the WHOLE frame inside its window, which
# also disables the taper's below-jaw neck job, not just its (buggy) ear-side
# job -- at f82 that exposed a hard-outlined neck bulge current production's
# feathered taper normally hides. This kwarg instead gates the suppression
# spatially to the ear label itself: tt_eff = tt * (1 - p(f) * ear_soft),
# where ear_soft is the BiSeNet ear label dilated NECK_TAPER_EAR_PROTECT_
# DILATE_PX then Gaussian-feathered by NECK_TAPER_EAR_PROTECT_FEATHER_SIGMA
# (Step 1 of the round-3 task confirmed this label+dilation covers the real
# visible ear-side wash at every checked frame in f60-88), and p(f) reuses
# _efc_ramp_strength's schedule inverted (0 outside the window -> ear_soft
# term vanishes and tt_eff==tt exactly; ->1 inside the window -> ear pixels
# get neck_taper=OFF while non-ear pixels, including the whole below-jaw
# neck, keep tt unchanged regardless of p).
# Per-clip offline-demo setting for test.mp4: the demo recipe uses
# neck_taper_ear_protect=(-1, 0, 244, 245), i.e. p(f)=1.0 (ear-side
# suppression fully ON) for effectively the whole 245-frame clip (Recipe A2,
# adopted after the Task 2c-revised round-3 review) -- not a general
# production default.
NECK_TAPER_EAR_PROTECT_DEFAULT = None
NECK_TAPER_EAR_PROTECT_DILATE_PX = 14
NECK_TAPER_EAR_PROTECT_FEATHER_SIGMA = 8.0
# Saturate ear_soft as clip(ear_soft / x, 0, 1) so the protection plateaus at 1.0 across the
# whole ear instead of a Gaussian tail (<1 at the outer rim leaves a 1-ear_soft alpha step where the
# jaw-line taper field jumps 0->1). None = off (byte-identical). Module global, read at call time
# like the DILATE/FEATHER globals above; the pitch-compensation path sets it per frame (k<1).
NECK_TAPER_EAR_SOFT_SATURATE = None
NECK_TAPER_EAR_SOFT_SATURATE_XMIN = 0    # saturation only applied at columns >= this (0 = whole frame)
AABAND_EAR_EXEMPT_XMIN = 0               # ear exemption only at columns >= this (0 = whole frame)
AABAND_EAR_EXEMPT_FEATHER_SIGMA = 0.0    # >0: exemption fades out (gate = ear weight) instead of a hard square edge

# Round 9 Task 2: opt-in fix for the f10-38 "double neck" strip -- confirmed
# (project_task2c3r8 round 8, Part 2) to be class (iv) driver content left
# visible where BOTH the avatar's own rendered alpha AND a_av_final are ~0 and
# BiSeNet mislabels the pixel BACKGROUND, so erase_bin/We never touch it (drv
# shows through raw or through `plated` almost unchanged). Default None ==
# byte-identical to production. When set to a 4-tuple ramp window (same
# (down_start, down_end, up_start, up_end) shape _efc_ramp_strength consumes,
# reused inverted exactly like NECK_TAPER_EAR_PROTECT_DEFAULT's `p`), pushes
# We_eff toward 1.0 (full plate) ONLY at pixels the per-frame strip mask
# selects, feathered, inside build_fullhead -- never touches a_av/av_rgb_canvas,
# so it cannot affect real avatar coverage, only which of {drv, plate_stat}
# shows through where the avatar already contributes nothing.
#
# Round 9 Part 1 correction: the raw 2-condition mask (a_av<thresh & drv
# sum<thresh, inside the jaw-anchored box) alone is NOT strip-specific -- at
# f39-60 it is 40-96% BiSeNet CLOTH (the real shirt collar) and, even after
# gating to sl==BACKGROUND only, a static dark V-neck notch's rim (mislabeled
# BACKGROUND at its own edge, NOT the strip) still fires steadily from f44-60
# at a near-constant location/size, wholly disconnected from the ear/jaw
# hairline the real strip always touches. Two refinements were required
# before this was safe to use as a compositing gate: (1) sl==BACKGROUND
# (drops the CLOTH collar-body contamination), (2) a per-connected-component
# proximity filter that keeps only components touching within
# DRIVER_LEAK_PLATE_NEAR_JAW_PX of the jaw line (drops the disconnected
# collar-notch rim, which never comes within ~70px of the jaw). Validated
# (Round 9 Part 1b crops) clean f10-38; f39-60's collar-notch contamination
# is fully excluded by (2) even though it still passes (1) at some frames.
# Per-clip offline-demo setting for test.mp4: the demo recipe uses
# driver_leak_plate=(11, 15, 38, 41) -- up-ramp f11-15, full strength (plate
# leak-fix ON) f15-38, down-ramp f38-41 -- tuned to this clip's f10-38
# double-neck strip, not a general production default.
DRIVER_LEAK_PLATE_DEFAULT = None
DRIVER_LEAK_PLATE_AV_THRESH = 0.1      # a_av_final< this -> avatar truly absent
DRIVER_LEAK_PLATE_DRV_SUM_THRESH = 360  # drv BGR sum< this -> dark/strip-like pixel
DRIVER_LEAK_PLATE_NEAR_JAW_PX = 60      # component must touch within this of jaw_y(x_max)
DRIVER_LEAK_PLATE_DILATE_PX = 3
DRIVER_LEAK_PLATE_FEATHER_SIGMA = 2.0
# Search-box rule (Round 8 Part 1a), anchored to the tracked jaw polyline's
# rightmost (ear-side, in this clip's pose) extent x_max and its jaw_y there:
# x in [x_max-10, x_max+80], y in [jaw_y(x_max)-20, jaw_y(x_max)+120]. This box
# alone is NOT sufficient (see the correction above) -- sl==BACKGROUND plus the
# near-jaw component filter are both required on top of it.
DRIVER_LEAK_PLATE_BOX_X_LO = -10
DRIVER_LEAK_PLATE_BOX_X_HI = 80
DRIVER_LEAK_PLATE_BOX_Y_LO = -20
DRIVER_LEAK_PLATE_BOX_Y_HI = 120


def driver_leak_plate_box(jaw_off):
    """Per-frame search box for the driver-leak strip mask -- see
    DRIVER_LEAK_PLATE_DEFAULT's docstring for the rule and its validation."""
    xj = np.clip(jaw_off[:, 0], 0, W - 1).astype(np.float64)
    yj = jaw_off[:, 1].astype(np.float64)
    o = np.argsort(xj)
    xj_u, idx = np.unique(xj[o], return_index=True)
    yj_u = yj[o][idx]
    x_max = float(xj_u[-1])
    jaw_y_at_xmax = float(yj_u[-1])
    x0 = int(round(x_max + DRIVER_LEAK_PLATE_BOX_X_LO))
    x1 = int(round(x_max + DRIVER_LEAK_PLATE_BOX_X_HI))
    y0 = int(round(jaw_y_at_xmax + DRIVER_LEAK_PLATE_BOX_Y_LO))
    y1 = int(round(jaw_y_at_xmax + DRIVER_LEAK_PLATE_BOX_Y_HI))
    return max(0, y0), min(H, y1), max(0, x0), min(W, x1)


def _driver_leak_plate_seed(jaw_off, drv, a_av, sl):
    """Round 9 base seed: avatar-absent (a_av_final<thresh) AND dark/strip-like
    driver pixel (drv BGR sum<thresh) AND BiSeNet BACKGROUND, inside the
    jaw-anchored search box, THEN restricted to connected components that
    touch within DRIVER_LEAK_PLATE_NEAR_JAW_PX of the jaw line at x_max (drops
    the disconnected collar-notch rim -- see DRIVER_LEAK_PLATE_DEFAULT's
    Round 9 correction note). This is deliberately narrow/dark-pixel-only;
    Round 10 layers a tightly-scoped extension on top (see
    driver_leak_plate_mask) to also catch the pale tan-skin wedge Round 9's
    mask under-covered, WITHOUT reusing Round 9's rejected wide region-growing
    (see DRIVER_LEAK_PLATE_EXT_* docstring: that approach reached the collar
    notch at f43 and was dropped as a design direction entirely)."""
    y0, y1, x0, x1 = driver_leak_plate_box(jaw_off)
    region = np.zeros((H, W), bool)
    region[y0:y1, x0:x1] = True
    drv_sum = drv.astype(np.float32).sum(axis=2)
    raw = (a_av < DRIVER_LEAK_PLATE_AV_THRESH) & \
          (drv_sum < DRIVER_LEAK_PLATE_DRV_SUM_THRESH) & \
          (sl == BACKGROUND) & region
    if not raw.any():
        return raw

    xj = np.clip(jaw_off[:, 0], 0, W - 1).astype(np.float64)
    yj = jaw_off[:, 1].astype(np.float64)
    o = np.argsort(xj)
    xj_u, idx = np.unique(xj[o], return_index=True)
    jaw_y_at_xmax = float(yj[o][idx][-1])
    near_jaw_y1 = jaw_y_at_xmax + DRIVER_LEAK_PLATE_NEAR_JAW_PX

    n_cc, labels = cv2.connectedComponents(raw.astype(np.uint8), connectivity=8)
    keep = np.zeros((H, W), bool)
    for c in range(1, n_cc):
        comp = labels == c
        ys = np.where(comp)[0]
        if ys.min() <= near_jaw_y1:
            keep |= comp
    return keep


# Round 10 Task: Round 9's auto region-growing (radius 110px from the jaw
# endpoint, minus CLOTH) was DROPPED as a design direction -- at f43 it
# exploded from a 2px residual seed to 1119px covering the entire clean
# V-neck collar-notch triangle, and at f38/f42 it ran extra pixels down the
# already-clean neck edge past the visible stubble (see task2c3r10 Part 1c/1d
# STOP report). Do not re-attempt wide/large-radius growth.
#
# Replacement: a hard frame cutoff plus a small, tightly-boxed extension.
# - f>=DRIVER_LEAK_PLATE_TAIL_CUTOFF_FRAME: mask forced EMPTY, no seed, no
#   extension. This is the frame range (f41-43) where Round 9's seed itself
#   was already down to 2-18px and any growth reached the notch -- rather
#   than trying to salvage a shrinking, fragile detection near the tail, that
#   range is simply excluded and the residual (already small, 2-34px/frame
#   pre-cutoff) is accepted as OPEN/reported, not compositor-fixed.
# - Extension only fires when the seed itself has >=DRIVER_LEAK_PLATE_EXT_MIN_SEED_PX
#   pixels (skips the near-zero tail frames where growth is riskiest even
#   before the hard cutoff) and is spatially locked to the seed's OWN bbox
#   expanded DRIVER_LEAK_PLATE_EXT_BBOX_Y_PAD px vertically (not the full
#   140px search box), plus a small DRIVER_LEAK_PLATE_EXT_RADIUS_PX distance
#   cap from the nearest seed pixel -- both far tighter than Round 9's
#   110px/full-box growth, specifically to avoid reaching the notch or
#   running along the clean neck edge. Validated per-frame with crops
#   (task2c3r10 Part 2/3) before being left enabled; DRIVER_LEAK_PLATE_USE_EXTENSION
#   flips to False (seed-only fallback) if any frame fails that check.
DRIVER_LEAK_PLATE_TAIL_CUTOFF_FRAME = 41  # mask forced empty at f>=this
DRIVER_LEAK_PLATE_EXT_MIN_SEED_PX = 30    # skip extension if seed smaller than this
DRIVER_LEAK_PLATE_EXT_RADIUS_PX = 12      # R: max distance (px) from nearest seed pixel
DRIVER_LEAK_PLATE_EXT_BBOX_Y_PAD = 16     # seed bbox vertical expansion (px, both directions)
DRIVER_LEAK_PLATE_EXT_AV_THRESH = 0.10
DRIVER_LEAK_PLATE_EXT_WE_THRESH = 0.90
DRIVER_LEAK_PLATE_EXT_DIFF_THRESH = 45.0
DRIVER_LEAK_PLATE_USE_EXTENSION = True     # set False (seed-only fallback) if validation fails


def driver_leak_plate_mask(f, jaw_off, drv, a_av, sl, We_eff, plate_stat_):
    """Round 10 mask: Round 9's `_driver_leak_plate_seed` (dark-pixel strip
    only) plus, when enabled and the seed is large enough, a small tan-wedge
    extension using the colour-agnostic visible-leak reference V
    (a_av<thresh & We_eff<thresh & |drv-plate_stat|sum>thresh), capped to a
    small radius from the seed and to the seed's own (vertically padded)
    bbox -- see the constants block above for the full rationale and why the
    wide region-growing tried in Round 9/task2c3r10 Part 1 was dropped
    instead of extended. Hard-empty for f>=DRIVER_LEAK_PLATE_TAIL_CUTOFF_FRAME."""
    if f >= DRIVER_LEAK_PLATE_TAIL_CUTOFF_FRAME:
        return np.zeros((H, W), bool)

    seed = _driver_leak_plate_seed(jaw_off, drv, a_av, sl)
    n_seed = int(seed.sum())
    if not DRIVER_LEAK_PLATE_USE_EXTENSION or n_seed < DRIVER_LEAK_PLATE_EXT_MIN_SEED_PX:
        return seed

    ys, xs = np.where(seed)
    y0e = max(0, ys.min() - DRIVER_LEAK_PLATE_EXT_BBOX_Y_PAD)
    y1e = min(H, ys.max() + DRIVER_LEAK_PLATE_EXT_BBOX_Y_PAD + 1)
    x0, x1 = xs.min(), xs.max() + 1
    bbox_mask = np.zeros((H, W), bool)
    bbox_mask[y0e:y1e, x0:x1] = True

    dist = cv2.distanceTransform((~seed).astype(np.uint8), cv2.DIST_L2, 5)
    diff = np.abs(drv.astype(np.float32) - plate_stat_.astype(np.float32)).sum(axis=2)
    V = (a_av < DRIVER_LEAK_PLATE_EXT_AV_THRESH) & \
        (We_eff < DRIVER_LEAK_PLATE_EXT_WE_THRESH) & \
        (diff > DRIVER_LEAK_PLATE_EXT_DIFF_THRESH)
    ext = V & (dist <= DRIVER_LEAK_PLATE_EXT_RADIUS_PX) & bbox_mask
    return seed | ext


def _efc_ramp_strength(f, ramp):
    """Frame-index clamp-strength schedule for erase_feather_clamp_ramp.
    Returns 1.0 (full clamp, current default behavior) when ramp is None."""
    if ramp is None:
        return 1.0
    down_start, down_end, up_start, up_end = ramp
    if f <= down_start:
        return 1.0
    if f < down_end:
        return 1.0 - (f - down_start) / float(down_end - down_start)
    if f <= up_start:
        return 0.0
    if f < up_end:
        return (f - up_start) / float(up_end - up_start)
    return 1.0


# ---------------------------------------------------------------------------
# seg2_pushpull_fill (Round AO): synthetic skin-color extrapolation for the
# cord artifact's lower ear-to-jaw gap -- the pixels inside EFC_ROI where
# We>=0.999 (fully inside the erased/plated head) but a_av<=0 (the avatar
# renderer has literally no usable content there; see erase_feather_clamp's
# comment above for segment 1/2 terminology). Two earlier attempts to use
# the RENDERER's own content here were both rejected: Round AJ's hard
# reveal produced a large black patch, and Round AN's proportional
# (unthresholded) alpha blend still produced a visible dark smudge at this
# gap's highest-raw-alpha pixels -- confirmed via direct pixel comparison
# (on-disk AND a fresh float re-render, both near-black at raw alpha~=0.98,
# e.g. [57,70,88]/[3.5,5.9,4.8] vs. the driver's real skin tone [74,89,143]
# at the same spot) that the underlying rendered content is intrinsically
# dark/non-skin, not thin/grazing real skin. No compositing-weight change
# on the renderer's own output can recover plausible skin at these exact
# pixels, ruling out Option 1 entirely for this gap.
#
# seg2_pushpull_fill instead SYNTHESIZES a color fill via cv2.inpaint
# (Telea fast marching), sourcing ONLY from nearby pixels with solid,
# trusted avatar alpha (a_av >= SEG2_REAL_SKIN_ALPHA_MIN) -- the same dark
# low-alpha fringe already ruled out as a real-skin source is explicitly
# masked out of the inpaint source image (alongside the gap itself), so the
# fill cannot silently pull that dark content back in through the back
# door just because it happens to be the nearest unmasked pixel.
# ---------------------------------------------------------------------------
SEG2_PUSHPULL_FILL_DEFAULT = False
# NOTE: this is NOT the same "dark content" concern Round AN tested -- that
# was about the RAW pre-threshold alpha (never entering a_av at all). Here,
# a_av is already POST halo_band_thresh/gate -- any a_av>0 pixel already
# cleared that bar and is legitimate avatar skin, exactly as the task spec's
# "a_av>0, real rendered skin" says. An earlier draft used 0.5 here by
# mistake, which excluded large amounts of legitimate lower-alpha skin from
# the inpaint source and produced a much BIGGER black area than segment 2's
# actual gap (visually confirmed on f020/f080 -- see round AO report).
SEG2_REAL_SKIN_ALPHA_MIN = 1e-3
SEG2_INPAINT_RADIUS_PX = 15
SEG2_INPAINT_METHOD = cv2.INPAINT_TELEA   # NS also tested -- see project notes; Telea chosen


def _seg2_pushpull_fill(av_rgb_canvas, a_av, We, roi=EFC_ROI,
                         real_skin_alpha_min=SEG2_REAL_SKIN_ALPHA_MIN,
                         radius=SEG2_INPAINT_RADIUS_PX, method=SEG2_INPAINT_METHOD):
    """Returns (av_rgb_canvas', a_av') with segment 2 (We>=0.999 & a_av<=0,
    inside `roi`) filled with an inpainted skin-color extrapolation and a_av
    forced to 1.0 there. No-op outside `roi` / where there's no such gap."""
    ry0, ry1, rx0, rx1 = roi
    roi_mask = np.zeros(a_av.shape, bool)
    roi_mask[ry0:ry1, rx0:rx1] = True
    gap_mask = (a_av <= 0.0) & roi_mask & (We >= 0.999)
    if not gap_mask.any():
        return av_rgb_canvas, a_av
    # Exclude the untrusted low-alpha fringe from the inpaint's source
    # pixels too, but ONLY within the composited-head footprint (We>=0.999)
    # -- gated the same way gap_mask itself is. Without the We gate this
    # swept in EFC_ROI's background/hair pixels outside the erased head
    # silhouette (We==0 there, never part of a_av's domain at all), which
    # are dark for unrelated reasons and dragged in as inpaint source from
    # far outside the gap -- visually confirmed on f020: real skin sits
    # immediately either side of the true (thin) gap, yet the un-gated
    # version painted a much wider black column than the true gap because
    # the inpaint source ended up dominated by ROI-boundary background.
    untrusted = roi_mask & (We >= 0.999) & (~gap_mask) & (a_av < real_skin_alpha_min)
    inpaint_mask = ((gap_mask | untrusted).astype(np.uint8)) * 255
    # av_rgb_canvas/av_rgb_real is the renderer's alpha-PREMULTIPLIED color
    # (RGB = alpha * unpremultiplied_color -- see render_gaga.py's black-bg
    # extraction trick; never divided back out before storage). At the
    # partial-alpha ring bordering the gap (a_av ~0.4-0.8, exactly the
    # inpaint's nearest source pixels) this understates true skin brightness
    # by roughly that same factor -- un-premultiplying here recovers a much
    # more plausible skin tone (e.g. [91,113,160] vs. the raw [0,12,33] at
    # one f020 boundary pixel, close to the driver's own real skin tone a
    # few pixels further out). Left untouched at a_av<=eps (already 0, no
    # information to recover) and everywhere outside the inpaint mask.
    unprem = np.where(a_av[..., None] > 1e-3,
                       av_rgb_canvas / np.maximum(a_av[..., None], 1e-3),
                       av_rgb_canvas)
    src = np.clip(unprem, 0, 255).astype(np.uint8)
    filled = cv2.inpaint(src, inpaint_mask, radius, method)
    av_rgb_canvas = np.where(gap_mask[..., None], filled.astype(np.float32), av_rgb_canvas)
    a_av = np.where(gap_mask, 1.0, a_av).astype(np.float32)
    return av_rgb_canvas, a_av


# ---------------------------------------------------------------------------
# edge_rgb_dilate (Priority-1 candidate, re-opened per explicit user
# instruction 2026-09-28 despite project_halo_ghostline_closed.md's prior
# verdict).
#
# v1 (superseded, see git history if needed) gated the fringe band by
# `a_av >= trust_min` (an alpha threshold) on the theory that the dark ring
# is a low-alpha premultiplied-against-black fringe. TESTED against the
# actual production demo2 pipeline (task_edge_rgb_dilate_test.py, f0-350):
# mean|diff| vs baseline was only 0.03-0.05 across the whole frame (<0.4% of
# pixels touched) and the wide-silhouette / hairline-zoom crops showed the
# dark rim completely unchanged. Root-caused via a direct pixel scan across
# a jaw-silhouette row (f260, row 431): the renderer emits near-black RGB
# not just at low alpha but ALL THE WAY to a_av==1.0 right at the boundary
# (e.g. x=488 a_av=0.591 rgb=[0,1,0] ... x=493 a_av=1.000 rgb=[18,20,24] ...
# x=494 a_av=1.000 rgb=[108,112,117] -- the real skin/hair color only
# starts 1-2px further in). So an alpha-threshold "trust" gate can't
# distinguish this from legitimate dark hair at the same alpha -- v1 either
# left the true dark-matte pixels untouched (trust_min<=1.0 always admits
# some of them as "trusted" once a_av crosses the threshold) or would have
# to set trust_min so high it swallows real anti-aliased edges everywhere.
#
# v2 (this version) gates by DISTANCE from the outer silhouette boundary
# instead of by alpha: any pixel within `band_px` of the a_av<0.02
# background (by distance transform, regardless of its OWN a_av or
# brightness) is fringe and gets re-filled; the "trusted" source region is
# everything beyond that band with a_av >= core_min. This matches the
# renderer's actual failure mode (dark content baked in for the first few
# px inward from the edge, independent of alpha) and the user's literal
# request ("push skin/hair RGB outwards ... via pixel dilation"). Same
# class-aware (hair vs. non-hair) inpaint-source split as v1 to avoid the
# previously-documented skin<->hair color-bleed failure mode. a_av is NOT
# modified -- only the RGB that a_av multiplies.
# ---------------------------------------------------------------------------
EDGE_RGB_DILATE_DEFAULT = False
EDGE_RGB_DILATE_BAND_PX = 6
EDGE_RGB_DILATE_CORE_MIN = 0.9
EDGE_RGB_DILATE_RADIUS_PX = 9


def _edge_rgb_dilate(av_rgb_canvas, a_av, alpha01, hair, support,
                      band_px=EDGE_RGB_DILATE_BAND_PX, core_min=EDGE_RGB_DILATE_CORE_MIN,
                      radius=EDGE_RGB_DILATE_RADIUS_PX):
    """Returns av_rgb_canvas' with the outer `band_px` of the silhouette
    (measured by distance transform from the a_av<0.02 background, NOT by
    each pixel's own alpha) re-filled with class-aware (hair vs. non-hair)
    straight-color inpainting sourced from the trusted interior. No-op
    where there's no silhouette. a_av is NOT modified."""
    bg = (a_av < 0.02) | (~support)
    interior = (~bg).astype(np.uint8)
    if not interior.any():
        return av_rgb_canvas
    dist = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
    edge_band = (dist > 0) & (dist <= band_px) & (alpha01 > 1e-3) & support
    if not edge_band.any():
        return av_rgb_canvas
    trusted = (dist > band_px) & (a_av >= core_min) & support
    src = np.clip(av_rgb_canvas, 0, 255).astype(np.uint8)
    out = av_rgb_canvas.copy()
    for cls_mask in (hair, ~hair):
        cls_edge = edge_band & cls_mask
        if not cls_edge.any():
            continue
        cls_trusted = trusted & cls_mask
        if not cls_trusted.any():
            continue
        # Fill mask = this class's fringe, PLUS every pixel that is either
        # untrusted or of the other class -- so inpaint's search can never
        # cross the hair/skin boundary for its source color.
        fill_mask = (cls_edge | (~cls_trusted)).astype(np.uint8) * 255
        filled = cv2.inpaint(src, fill_mask, radius, cv2.INPAINT_TELEA)
        out = np.where(cls_edge[..., None], filled.astype(np.float32), out)
    return out


def _rim3d_cheek_quad_box(f, lm):
    """Round Z/W's validated cheek-quadrant search box for frame f. Ported
    verbatim (method + constants) from quadrant_box_W in the Round Z/W
    diagnostics."""
    lower_lid_y = float(lm[f, RIM3D_LOWER_EYELID_IDX, 1])
    mouth_y = float(lm[f, RIM3D_MOUTH_CORNER_IDX, 1])
    nose_x = float(lm[f, RIM3D_NOSE_IDX, 0])
    y0 = max(0, int(lower_lid_y) + RIM3D_Y_PAD_TOP)
    y1 = min(H, int(mouth_y) - RIM3D_Y_PAD_BOT)
    if y1 <= y0:
        y0, y1 = min(y0, y1), max(y0, y1) + 20
    x0 = max(0, int(nose_x) - RIM3D_X_MARGIN)
    box = np.zeros((H, W), bool)
    box[y0:y1, x0:W] = True
    return box


def _rim3d_load():
    """Lazily load the FLAME identity + mesh edge topology needed for the 3D
    rim injection. Only reached when rim3d_injection=True. Reproduces
    render_gaga.py's own FLAME-forward/camera setup (read-only reuse of the
    already-cached tracking result, no new tracking work) -- see Round Z."""
    global _RIM3D_STATE
    if _RIM3D_STATE is not None:
        return _RIM3D_STATE
    import torch
    from collections import defaultdict
    gaga_root = _HEADSWAP_ROOT / "third_party" / "GAGAvatar"
    if str(gaga_root) not in sys.path:
        sys.path.insert(0, str(gaga_root))
    import inference as offline_inference
    from core.libs.flame_model import FLAMEModel
    from core.libs.GAGAvatar_track.engines import CoreEngine as TrackEngine

    device = "cuda" if torch.cuda.is_available() else "cpu"
    track_engine = TrackEngine(focal_length=RIM3D_FOCAL, device=device)
    feature_data = offline_inference.get_tracked_results(str(RIM3D_IDENTITY_IMAGE), track_engine)
    del track_engine
    if device == "cuda":
        torch.cuda.empty_cache()
    f_shape = feature_data["shapecode"].float().to(device)[None]

    flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=5.0, no_lmks=True).to(device)
    FACES = (flame_model.faces.cpu().numpy() if hasattr(flame_model, "faces")
             else flame_model.faces_tensor.long().cpu().numpy())

    track = np.load(str(TRACK_NPZ))
    fit3d = np.load(str(FIT3D_NPZ))
    jaw_all = track["jaw"]; expression_all = track["expression"]
    transforms_all = fit3d["transform"]
    s_cal, t_cal = float(fit3d["s_cal"]), fit3d["t_cal"]

    edge_faces = defaultdict(list)
    for fi, (a, b, c) in enumerate(FACES):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_faces[key].append(fi)
    manifold = [(e, fs) for e, fs in edge_faces.items() if len(fs) == 2]
    EDGE_V = np.array([e for e, fs in manifold], dtype=np.int64)
    EDGE_FACE_A = np.array([fs[0] for e, fs in manifold], dtype=np.int64)
    EDGE_FACE_B = np.array([fs[1] for e, fs in manifold], dtype=np.int64)

    _RIM3D_STATE = dict(device=device, flame_model=flame_model, f_shape=f_shape, FACES=FACES,
                         EDGE_V=EDGE_V, EDGE_FACE_A=EDGE_FACE_A, EDGE_FACE_B=EDGE_FACE_B,
                         jaw_all=jaw_all, expression_all=expression_all,
                         transforms_all=transforms_all, s_cal=s_cal, t_cal=t_cal, y_flip=1.0)
    return _RIM3D_STATE


def _rim3d_silhouette_points(f, render_canvas_h):
    """Shared FLAME-forward + projection step for both the cheek (Round Z) and
    jaw/chin (Round AE) rim masks: this frame's FLAME vertices -> front/back-
    facing silhouette edges -> projected driver-frame pixel coordinates.
    Returns (px_sil, py_sil, hull) where px_sil/py_sil are the in-bounds
    silhouette-flip vertex projections (candidate rim points, BEFORE any
    hull-distance or jaw-line filtering) and hull is the projected mesh's own
    2D convex hull (all in-bounds vertices, not just silhouette ones)."""
    import torch
    st = _rim3d_load()
    device = st["device"]
    canvas_pad = (int(render_canvas_h) - RENDER_CANVAS_PX) // 2
    focal_y = RIM3D_FOCAL * float(RENDER_CANVAS_PX) / float(render_canvas_h)

    expression = torch.zeros(1, 100, device=device)
    expression[:, :50] = torch.from_numpy(st["expression_all"][f]).to(device)
    pose_params = torch.cat([
        torch.zeros(3, device=device),
        torch.from_numpy(st["jaw_all"][f]).to(device),
    ])[None].float()
    eye_pose = torch.zeros(1, 6, device=device)
    with torch.no_grad():
        verts_t = st["flame_model"](shape_params=st["f_shape"], expression_params=expression,
                                     pose_params=pose_params, eye_pose_params=eye_pose)
    verts = verts_t[0].detach().cpu().numpy()

    R = st["transforms_all"][f, :, :3]; T = st["transforms_all"][f, :, 3]
    y_flip = st["y_flip"]
    FACES = st["FACES"]
    v0 = verts[FACES[:, 0]]; v1 = verts[FACES[:, 1]]; v2 = verts[FACES[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    face_normals = np.cross(v1 - v0, v2 - v0)
    c_raw = centroids @ R + T
    n_raw = face_normals @ R
    c_view = np.stack([-c_raw[:, 0], -y_flip * c_raw[:, 1], c_raw[:, 2]], axis=1)
    n_view = np.stack([-n_raw[:, 0], -y_flip * n_raw[:, 1], n_raw[:, 2]], axis=1)
    view_dir = c_view / (np.linalg.norm(c_view, axis=1, keepdims=True) + 1e-9)
    facing_sign = np.sum(n_view * view_dir, axis=1) > 0.0
    flips = facing_sign[st["EDGE_FACE_A"]] != facing_sign[st["EDGE_FACE_B"]]
    sil_edges = st["EDGE_V"][flips]
    sil_idx = np.unique(sil_edges.reshape(-1)) if len(sil_edges) else np.array([], dtype=np.int64)

    empty = (np.array([]), np.array([]), None)
    if len(sil_idx) == 0:
        return empty

    p_raw = verts @ R + T
    p_view = np.stack([-p_raw[:, 0], -y_flip * p_raw[:, 1], p_raw[:, 2]], axis=1)
    ndc_x = RIM3D_FOCAL * p_view[:, 0] / p_view[:, 2]
    ndc_y = focal_y * p_view[:, 1] / p_view[:, 2]
    render_px = (ndc_x + 1.0) * 0.5 * RENDER_CANVAS_PX
    render_py = (ndc_y + 1.0) * 0.5 * float(render_canvas_h)
    px_all = st["s_cal"] * render_px + st["t_cal"][0]
    py_all = st["s_cal"] * render_py + (st["t_cal"][1] - st["s_cal"] * canvas_pad)

    px, py = px_all[sil_idx], py_all[sil_idx]
    inb = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    if not inb.any():
        return empty

    inb_all = (px_all >= 0) & (px_all < W) & (py_all >= 0) & (py_all < H)
    hull_pts = np.stack([px_all[inb_all], py_all[inb_all]], axis=1).astype(np.float32)
    hull = cv2.convexHull(hull_pts)
    return px[inb], py[inb], hull


def _rim3d_mesh_boundary_mask(f, render_canvas_h):
    """Round Z's exact method: restrict `_rim3d_silhouette_points`'s candidates
    to the projected mesh's own 2D convex hull (excludes interior self-
    occlusion folds, e.g. the nose bridge) -> dilated point mask in
    driver-frame pixel space. Returns an (H, W) bool mask (NOT yet restricted
    to the cheek quadrant box -- callers intersect that separately, same as
    Round Z's own overlap check did)."""
    mask = np.zeros((H, W), bool)
    px, py, hull = _rim3d_silhouette_points(f, render_canvas_h)
    if hull is None:
        return mask
    dist_to_hull = np.array([abs(cv2.pointPolygonTest(hull, (float(x), float(y)), True))
                              for x, y in zip(px, py)])
    keep = dist_to_hull < RIM3D_HULL_DIST_PX
    ix = px[keep].astype(int); iy = py[keep].astype(int)
    if len(ix) == 0:
        return mask
    mask[iy, ix] = True
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((RIM3D_DILATE_PX, RIM3D_DILATE_PX), np.uint8)) > 0
    return mask


def _rim3d_jaw_mask(f, render_canvas_h, lm):
    """Round AE jaw/chin extension: restrict `_rim3d_silhouette_points`'s SAME
    candidate set to points that sit close to the actual jaw curve (per-
    column |jaw_sign_field| distance -- the reference build_r8's own erase
    geometry already trusts for this contour), instead of Round Z's cheek-
    tuned convex-hull-distance test, which is structurally wrong for the
    jaw/chin's concave geometry (see the RIM3D_JAW_* comment block above).
    Restricted to the mouth-corner-to-jaw/chin band only (`quad_box`'s cheek
    region above the mouth is untouched/unaffected by this function).
    Explicitly excludes a padded box around both mouth corners -- the jaw
    curve passes close enough to the lips in image space that the raw
    distance test alone pulls in the mouth/lip self-occlusion fold (see the
    RIM3D_JAW_* comment block above; confirmed as a REAL, not just
    geometric, false positive via diag3_regression_check.py)."""
    mask = np.zeros((H, W), bool)
    px, py, hull = _rim3d_silhouette_points(f, render_canvas_h)
    if hull is None:
        return mask
    jaw_off = jaw_offset(lm[f, JAW_IDX, :2])
    d_signed = jaw_sign_field(jaw_off)
    ix_all = np.clip(px.astype(int), 0, W - 1)
    iy_all = np.clip(py.astype(int), 0, H - 1)
    dist_to_jaw = np.abs(d_signed[iy_all, ix_all])
    c1 = lm[f, RIM3D_MOUTH_CORNER_IDX, :2]
    c2 = lm[f, RIM3D_MOUTH_CORNER_IDX_2, :2]
    mx0 = min(c1[0], c2[0]) - RIM3D_MOUTH_EXCLUDE_PAD_PX
    mx1 = max(c1[0], c2[0]) + RIM3D_MOUTH_EXCLUDE_PAD_PX
    my0 = min(c1[1], c2[1]) - RIM3D_MOUTH_EXCLUDE_PAD_PX
    my1 = max(c1[1], c2[1]) + RIM3D_MOUTH_EXCLUDE_PAD_PX
    in_mouth_box = (px >= mx0) & (px <= mx1) & (py >= my0) & (py <= my1)
    keep = (dist_to_jaw < RIM3D_JAW_LINE_DIST_PX) & (~in_mouth_box)
    ix = px[keep].astype(int); iy = py[keep].astype(int)
    if len(ix) == 0:
        return mask
    mask[iy, ix] = True
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((RIM3D_DILATE_PX, RIM3D_DILATE_PX), np.uint8)) > 0
    return mask


def _rim3d_jaw_quad_box(f, lm):
    """Round AE: the mouth-corner-to-jaw/chin search band. Starts exactly
    where the cheek quad box (`_rim3d_cheek_quad_box`) stops (`mouth_y -
    RIM3D_Y_PAD_BOT`) and extends down to the lowest jaw landmark plus a
    margin (catches the neck hinge). Same x-range as the cheek box (same
    side of the face, restricted away from the nose)."""
    mouth_y = float(lm[f, RIM3D_MOUTH_CORNER_IDX, 1])
    nose_x = float(lm[f, RIM3D_NOSE_IDX, 0])
    jaw_y_max = float(lm[f, JAW_IDX, 1].max())
    y0 = max(0, int(mouth_y) - RIM3D_Y_PAD_BOT)
    y1 = min(H, int(jaw_y_max) + RIM3D_JAW_Y_MARGIN_PX)
    if y1 <= y0:
        return np.zeros((H, W), bool)
    x0 = max(0, int(nose_x) - RIM3D_X_MARGIN)
    box = np.zeros((H, W), bool)
    box[y0:y1, x0:W] = True
    return box


def _crown_temple_quad_boxes(f, lm):
    """Landmark-anchored candidate region for crown_temple_dilate, split into
    left/right halves at the nose so each side's dilation can be sized
    independently (the measured gap is directional/asymmetric, not uniform --
    see the CROWN_TEMPLE_* block above). Each box spans the full canvas width
    on its side, from the top of the canvas down to that side's eyebrow row
    (+ small pad) -- i.e. "everything at or above brow level," the region
    with no MediaPipe landmark coverage at all. Intersected by callers with
    erase_bin/true_head/support so the box itself only bounds candidate
    location, not shape."""
    nose_x = float(lm[f, RIM3D_NOSE_IDX, 0])
    brow_y_l = float(lm[f, CROWN_TEMPLE_BROW_IDX_L, 1])
    brow_y_r = float(lm[f, CROWN_TEMPLE_BROW_IDX_R, 1])
    y1_l = min(H, max(1, int(brow_y_l) + CROWN_TEMPLE_Y_PAD))
    y1_r = min(H, max(1, int(brow_y_r) + CROWN_TEMPLE_Y_PAD))
    x_mid = int(round(nose_x))
    box_l = np.zeros((H, W), bool)
    box_l[0:y1_l, 0:max(1, x_mid)] = True
    box_r = np.zeros((H, W), bool)
    box_r[0:y1_r, x_mid:W] = True
    return box_l, box_r


def dehalo_avatar_rgb(rgba):
    """Strip the dark rim from the avatar's native RGBA. Returns a new RGBA array;
    alpha channel is always identical to the input. RGB is unchanged wherever no
    anomaly is flagged. Ported verbatim from wholehead_cp_r8.dehalo_avatar_rgb."""
    alpha = rgba[..., 3].astype(np.float32)
    mask = (alpha > 0).astype(np.uint8)
    if mask.sum() < 100:
        return rgba
    dist_in = cv2.distanceTransform(mask * 255, cv2.DIST_L2, 5)
    safe_a = np.clip(alpha, 1, 255) / 255.0
    unpremult = np.clip(rgba[..., :3].astype(np.float32) / safe_a[..., None], 0, 255).astype(np.uint8)

    near_edge = (mask > 0) & (dist_in <= HALO_EDGE_BAND + 4)
    ref = cv2.inpaint(unpremult, near_edge.astype(np.uint8) * 255, 15, cv2.INPAINT_TELEA)

    luma_actual = 0.114 * unpremult[..., 0] + 0.587 * unpremult[..., 1] + 0.299 * unpremult[..., 2]
    edge_band = (mask > 0) & (alpha >= HALO_ALPHA_MIN) & (dist_in <= HALO_EDGE_BAND)
    anomaly = edge_band & (luma_actual < HALO_ABS_DARK_LUMA)
    if not anomaly.any():
        return rgba

    anomaly_d = cv2.dilate(anomaly.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    replacement = np.clip(ref.astype(np.float32) * safe_a[..., None], 0, 255).astype(np.uint8)
    out = rgba.copy()
    out[..., :3][anomaly_d] = replacement[anomaly_d]
    return out


def fix_canvas_edge_clip(rgba):
    """Zero alpha at pixels flagged as canvas-edge render-clip artifacts. Returns a new
    RGBA array; RGB is always unchanged (only alpha is ever modified, only ever lowered
    to 0). Ported verbatim from wholehead_cp_r8.fix_canvas_edge_clip."""
    alpha = rgba[..., 3]
    rgb = rgba[..., :3].astype(np.int32)
    h, w = alpha.shape
    cand = (alpha >= EDGE_CLIP_ALPHA_MIN) & (rgb.max(axis=2) <= EDGE_CLIP_RGB_MAX)
    if not cand.any():
        return rgba
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    touches_edge = np.zeros(n, bool)
    for i in range(1, n):
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if x0 == 0 or y0 == 0 or x0 + cw == w or y0 + ch == h:
            touches_edge[i] = True
    flag = touches_edge[labels]
    if not flag.any():
        return rgba
    out = rgba.copy()
    out[..., 3][flag] = 0
    return out


def load_M(canvas_h=RENDER_CANVAS_H):
    """Calibrated render-canvas<->driver-frame similarity transform (pure
    scale+translate, per fit3d.py's design). Ported from wholehead_cp_r8.load_M,
    reading this module's own FIT3D_NPZ (same file build_r8 reads).

    A canvas_h > 512 render (render_gaga.py's tall below-jaw path) centres the
    head identically but pushes it DOWN by canvas_pad = (canvas_h - 512)//2 rows,
    so the row translation drops by s_cal * canvas_pad to land it back.
    canvas_h=512 -> canvas_pad=0 -> the exact legacy M."""
    fit = np.load(str(FIT3D_NPZ))
    s_cal, t_cal = float(fit["s_cal"]), fit["t_cal"]
    canvas_pad = (int(canvas_h) - RENDER_CANVAS_PX) // 2
    return np.array([[s_cal, 0, t_cal[0]],
                     [0, s_cal, t_cal[1] - s_cal * canvas_pad]], np.float64)


def static_plate(parser=None, reg=None):
    """Ported from wholehead_cp_r8.static_plate, reading this module's own PLATE_PNG.

    Task 42: if `parser` (a compositing.parse.FaceParser instance) is given,
    the loaded plate crop is checked for a face using the same BiSeNet
    presence-gate v3/live/pipeline.py's calibrate_plate() applies to live
    calibration bursts -- so a stale/wrong/re-captured-with-someone-in-it
    PLATE_PNG raises instead of silently compositing a contaminated plate
    (the root cause Task 41 traced for test2.mp4, reached here via a bad
    file instead of a missing one). Skipped (backward compatible) when
    parser is None, since dozens of archived diagnostic scripts under
    v3/debug_out/ call this with no args and never load a parser.

    KNOWN LIMITATION, not silently assumed solved: this only catches a
    clearly-visible face in a normal seated capture pose -- the case all 4
    real contaminated sessions found on disk (Task 42) exhibited. It has
    NOT been validated against a ducked, turned-away, or partially-out-of-
    frame operator; SKIN-class coverage from a BiSeNet model trained on
    frontal/near-frontal faces is unverified in those poses. Flagged as
    future work, not shipped as complete coverage.
    """
    ps = cv2.imread(str(PLATE_PNG))
    assert ps is not None, PLATE_PNG
    if parser is not None:
        skin_frac = float((parser.parse(ps) == SKIN).mean())
        if skin_frac > FACE_PRESENT_SKIN_FRAC_THRESHOLD:
            raise ValueError(
                f"PLATE_PNG appears to contain a person (SKIN-class pixel "
                f"fraction {skin_frac:.4f} > {FACE_PRESENT_SKIN_FRAC_THRESHOLD}) "
                f"-- recapture an empty background. {PLATE_PNG}")
    rx0, ry0, rx1, ry1 = REG if reg is None else reg   # reg: per-clip plate crop (must match the reg passed to composite/precompute)
    pf = np.zeros((H, W, 3), np.float32)
    pf[ry0:ry1, rx0:rx1] = ps.astype(np.float32)
    return pf


def jaw_offset(jaw_xy):
    """MediaPipe mandible polyline pushed out JAW_OUT_PX along its per-vertex outward
    normal. Ported verbatim from wholehead_cp_r8.jaw_offset."""
    p = jaw_xy.astype(np.float64)
    t = np.gradient(p, axis=0)
    t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-6)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    ctr = p.mean(0)
    sgn = np.sign(((p - ctr) * n).sum(1)); sgn[sgn == 0] = 1.0
    n *= sgn[:, None]
    return np.round(p + JAW_OUT_PX * n).astype(np.int32)


def jaw_sign_field(jaw_off, xmin_taper_px=None):
    """Signed distance to the jaw polyline with the sign from a per-column jaw height.
    Ported verbatim from wholehead_cp_r8.jaw_sign_field.

    xmin_taper_px (opt-in, default None = byte-identical to the original):
    the outside_cols clamp below forces d_signed hard-positive for EVERY column
    past the tracked polyline's x-endpoints, which is what keeps the temple/ear/
    forehead region unambiguously erase-eligible (see this file's causal ablation
    in debug_out/task_strip_f78/part1_causal_*: removing the clamp entirely flips
    ~50% of the neck/ear ROI back to raw driver pixels, not just the target strip).
    But because the boundary column (x == xj_u[0]) uses the real, unclamped,
    typically-small |d|, and the very next column jumps straight to the clamped
    value, seam_ramp (JAW_FEATHER_PX=4px wide) sees a near-discontinuous step at
    that single boundary column -- the reported f78-86 pale/dark vertical strip at
    x_min. xmin_taper_px, when set, tapers the xj_u[0]-side columns outward from
    the boundary column's own (real) value to the existing clamp value over
    xmin_taper_px columns, so the notch closes locally while every column at or
    beyond xj_u[0]-xmin_taper_px, and every column >= xj_u[0], is untouched."""
    xj = np.clip(jaw_off[:, 0], 0, W - 1).astype(np.float64)
    yj = jaw_off[:, 1].astype(np.float64)
    o = np.argsort(xj); xj, yj = xj[o], yj[o]
    xj_u, idx = np.unique(xj, return_index=True)
    yj_u = yj[idx]
    cols = np.arange(W, dtype=np.float64)
    jaw_y = np.interp(cols, xj_u, yj_u, left=yj_u[0], right=yj_u[-1])
    yy = np.arange(H, dtype=np.float64)[:, None]
    above = yy < jaw_y[None, :]
    curve = np.zeros((H, W), np.uint8)
    cv2.polylines(curve, [jaw_off], isClosed=False, color=1, thickness=1)
    d = cv2.distanceTransform((1 - curve).astype(np.uint8), cv2.DIST_L2, 5)
    d_signed = np.where(above, d, -d).astype(np.float32)
    outside_cols = (cols < xj_u[0]) | (cols > xj_u[-1])
    if outside_cols.any():
        d_signed[:, outside_cols] = np.abs(d_signed[:, outside_cols]) + 1.0
    if xmin_taper_px is not None and xmin_taper_px > 0:
        x_min = int(round(xj_u[0]))
        lo = max(0, x_min - int(xmin_taper_px))
        if lo < x_min:
            d_boundary = d_signed[:, x_min:x_min + 1]
            dist = (x_min - cols[lo:x_min]).astype(np.float32)
            blend = np.clip(dist / float(xmin_taper_px), 0.0, 1.0)[None, :]
            far_val = d_signed[:, lo:x_min]
            d_signed[:, lo:x_min] = ((1.0 - blend) * d_boundary + blend * far_val).astype(np.float32)
    return d_signed


def jaw_taper_depth_field(jaw_off):
    """Signed distance to the jaw polyline for the neck_taper code path ONLY.

    Identical to jaw_sign_field EXCEPT it omits the outside_cols `|d|+1` clamp.
    That clamp forces d_signed hard-positive for every column past the mandible
    polyline's x-endpoints, i.e. a ~215-unit vertical step at x = poly x-min /
    x-max on every frame. jaw_sign_field's other consumers (seam_ramp / We) mask
    that step via `np.minimum(We_soft, seam_ramp)` and the erase geometry, but
    neck_taper multiplies a_av *directly* by f(d_signed) with no such mask, so
    the step printed a moving vertical collar seam wherever a jaw endpoint swept
    across still-exposed avatar skirt (yaw-recovery f44-63; see the tall-canvas
    seam diagnosis, 2026-09-10). Dropping the clamp lets np.interp's flat
    left=/right= extrapolation carry the jaw line straight past its endpoints, so
    d_signed stays continuous and the taper fades smoothly on and beyond the jaw
    corners. jaw_sign_field itself is left byte-identical on purpose."""
    xj = np.clip(jaw_off[:, 0], 0, W - 1).astype(np.float64)
    yj = jaw_off[:, 1].astype(np.float64)
    o = np.argsort(xj); xj, yj = xj[o], yj[o]
    xj_u, idx = np.unique(xj, return_index=True)
    yj_u = yj[idx]
    cols = np.arange(W, dtype=np.float64)
    jaw_y = np.interp(cols, xj_u, yj_u, left=yj_u[0], right=yj_u[-1])
    yy = np.arange(H, dtype=np.float64)[:, None]
    above = yy < jaw_y[None, :]
    curve = np.zeros((H, W), np.uint8)
    cv2.polylines(curve, [jaw_off], isClosed=False, color=1, thickness=1)
    d = cv2.distanceTransform((1 - curve).astype(np.uint8), cv2.DIST_L2, 5)
    return np.where(above, d, -d).astype(np.float32)


def jaw_taper_depth_field_continuous(jaw_off):
    """jaw_taper_continuous=True variant of jaw_taper_depth_field (Task 2c).
    At any column outside the tracked jaw polyline's own x-range, the
    original function's zero-set for the distance transform is still just
    the polyline itself -- so `d` there is the distance to the polyline's
    nearest ENDPOINT (large, not ~0), while `above`/`below` is decided from
    the flat-extrapolated `jaw_y` line. The sign flip at row jaw_y[col] then
    jumps by ~2*d (confirmed: +27.2 -> -27.0 at f78 col 605, an out-of-range
    ear-side column) instead of the ~0 a genuine curve crossing gives. This
    printed as the ear-top step + pale wash whenever that jump landed on
    exposed ear skin during a yaw turn (f74/78/82; see project_task2b/2c
    diagnosis) -- a purely geometric bug, unrelated to erase_feather_clamp.

    Candidate A (drawing the flat-extended line as the zero-set for EVERY
    column, in-range included) was tried first per Task 2c's instructions
    but measured as changing the neck-taper proto at 245/245 frames with
    diffs up to max=250 far outside the f70-94 turn window (e.g. f125, f160,
    f200) -- it silently trades the true 2D polyline distance for a 1D
    per-column curve everywhere, not just at the diagnosed bug location.
    That fails the "don't change the neck materially" constraint.

    A first Candidate B attempt (hard column cutoff: use the polyline-based
    `d_orig` in-range, a pure per-column vertical distance `jaw_y[col]-row`
    out-of-range) measured WORSE than Candidate A (245/245 frames, diffs up
    to max=252, near-universal). Root cause: outside the tracked range
    `jaw_y` is FLAT (constant at the endpoint's row), so a pure vertical
    distance treats the ENTIRE horizontal line at that row -- across the
    whole image width, including the shoulders/hair far from the actual
    jaw -- as "at the jaw" (d~=0), whereas the true polyline-endpoint
    distance correctly grows large and saturates the neck_taper multiplier
    there. That is a real algorithmic mismatch, not just an over-broad
    column mask.

    This is Candidate B': the vertical-distance correction is blended
    against the original polyline-endpoint distance with a column-distance
    ramp, `_JAW_TAPER_MARGIN_PX`, so it is active (weight 1) exactly at the
    tracked range's own edge (where the original bug's jump lives) and
    fades OFF (weight 0, byte-identical to the original field) by that many
    pixels past the edge -- confining any change to a narrow band around
    the real jaw endpoint instead of the whole out-of-range half-image."""
    d_orig = jaw_taper_depth_field(jaw_off)
    xj = np.clip(jaw_off[:, 0], 0, W - 1).astype(np.float64)
    yj = jaw_off[:, 1].astype(np.float64)
    o = np.argsort(xj); xj, yj = xj[o], yj[o]
    xj_u, idx = np.unique(xj, return_index=True)
    yj_u = yj[idx]
    x_lo, x_hi = float(xj_u[0]), float(xj_u[-1])
    cols = np.arange(W, dtype=np.float64)
    jaw_y = np.interp(cols, xj_u, yj_u, left=yj_u[0], right=yj_u[-1])
    yy = np.arange(H, dtype=np.float64)[:, None]
    d_vert = (jaw_y[None, :] - yy).astype(np.float32)  # signed vertical distance, matches `above` sign
    dist_outside = np.maximum(0.0, np.maximum(x_lo - cols, cols - x_hi))
    w = np.clip(1.0 - dist_outside / _JAW_TAPER_MARGIN_PX, 0.0, 1.0).astype(np.float32)  # 1 at edge, 0 past margin
    d_hybrid = d_orig * (1.0 - w[None, :]) + d_vert * w[None, :]
    return d_hybrid.astype(np.float32)


# ---------------------------------------------------------------------------
# Floating-glasses-fragment fix (task_test2_artifact_diag.txt item 4). EYE_G
# (BiSeNet's glasses label) is already part of FACE_SKIN_SET, so real glasses
# pixels ARE part of `cand` -- but in a profile pose the lens/temple can
# protrude past the visible face contour far enough that it is a genuinely
# SEPARATE connected component in the 2D image (real background pixels sit
# between it and the rest of the face), not just a topologically-thin bridge.
# keep_linked_cc's dilate-and-touch test (FACE_LINK_PX=70) then drops that
# island as unlinked noise, and since the avatar has no glasses geometry
# either (a_av==0 there, legitimately -- this identity doesn't wear them),
# the real, un-erased lens is left floating next to the head with nothing
# covering it. Confirmed pixel-by-pixel on f100: EYE_G detected (91px,
# x=548-574 y=342-357), both erase_bin and a_av OFF at those exact pixels.
#
# OFF by default: only fires when BiSeNet actually detects EYE_G content in
# a dropped, glasses-dominated component -- a driver frame with no glasses
# (every existing validated render, including the offline production clip)
# has eye_g.any()==False and this block is a no-op.
GLASSES_ERASE_EXTEND_DEFAULT = False
GLASSES_MIN_FRAGMENT_PX = 20     # below this, more likely a stray BiSeNet
                                  # misclassification blip than real glasses
GLASSES_EYEG_FRAC_THRESH = 0.5   # component must be EYE_G-majority -- keeps
                                  # this scoped to glasses specifically, not a
                                  # general re-widening of keep_linked_cc's
                                  # link radius (which would resurrect any
                                  # dropped noise island, not just glasses)
GLASSES_DILATE_PX = 5            # small fatten so a thin lens-rim/temple
                                  # outline erases as a solid patch instead of
                                  # a sparse one-pixel-wide line


def _rvm_pha_frame(f):
    """Lazy-load the cached RVM alpha-matte stack (matting_experiment output) and
    return frame `f` as a float32 (H, W) array. The stack is the full in-order
    recurrent pass -- one load, cached for the process. STAGED helper: only
    reached when build_r8 is called with rvm_lateral_matte=True."""
    global _RVM_PHA_CACHE
    if _RVM_PHA_CACHE is None:
        assert RVM_PHA_NPZ.exists(), f"RVM matte cache missing: {RVM_PHA_NPZ}"
        _RVM_PHA_CACHE = np.load(str(RVM_PHA_NPZ))["pha"]
    return _RVM_PHA_CACHE[f].astype(np.float32)


def _rvm_pha_frame_demo2(f):
    """Lazy-load the cached RVM alpha-matte stack for the demo2 driver clip
    (Priority 2 hair-alpha blend, 2026-09-28), built by
    task_rvm_hair_blend_build.py as a single in-order sequential (recurrent-
    state) pass over all demo2 frames, uint8-quantized on disk. Returns frame
    `f` as float32 (H, W) in [0, 1]. STAGED helper: only reached when build_r8
    is called with rvm_hair_blend=True."""
    global _RVM_PHA_DEMO2_CACHE
    if _RVM_PHA_DEMO2_CACHE is None:
        assert RVM_HAIR_PHA_DEMO2_NPZ.exists(), f"RVM demo2 matte cache missing: {RVM_HAIR_PHA_DEMO2_NPZ}"
        _RVM_PHA_DEMO2_CACHE = np.load(str(RVM_HAIR_PHA_DEMO2_NPZ))["pha_u8"]
    return _RVM_PHA_DEMO2_CACHE[f].astype(np.float32) / 255.0


def _rvm_hair_blend_apply(a_av, hair, faceskin, neck, cloth, f,
                           roi_dilate_px=RVM_HAIR_BLEND_ROI_DILATE_PX,
                           band_px=RVM_HAIR_BLEND_BAND_PX):
    """Priority-2 hair-alpha blend (see constants block above for the full
    mechanism writeup, and the ROI-misalignment fix note on
    RVM_HAIR_BLEND_ROI_DILATE_PX/BAND_PX). The "is this a hair zone" gate is
    a wide dilation of the driver's BiSeNet hair label (hair_zone); the actual
    ring is located by a_av's OWN edge via distance transform (same mechanism
    as edge_rgb_dilate v2), restricted to hair_zone. a_av is multiplied by
    RVM's per-pixel driver-frame soft alpha inside the ring -- monotonic
    non-increasing, so this can only shrink a_av relative to baseline, never
    invent coverage. No-op if this frame has no hair label at all (e.g. fully
    bald driver frame, defensive only)."""
    if not hair.any():
        return a_av
    hair_zone = cv2.dilate(hair.astype(np.uint8), np.ones((roi_dilate_px, roi_dilate_px), np.uint8)) > 0
    hair_zone = hair_zone & (~faceskin) & (~neck) & (~cloth)
    if not hair_zone.any():
        return a_av
    av_support = a_av >= 0.02
    dist = cv2.distanceTransform(av_support.astype(np.uint8), cv2.DIST_L2, 5)
    ring = (dist > 0) & (dist <= band_px) & hair_zone
    if not ring.any():
        return a_av
    rvm = _rvm_pha_frame_demo2(f)
    return np.where(ring, a_av * rvm, a_av).astype(np.float32)


def keep_linked_cc(cand, faceskin):
    """Ported verbatim from wholehead_cp_r8.keep_linked_cc."""
    link = cv2.dilate(faceskin.astype(np.uint8), np.ones((FACE_LINK_PX, FACE_LINK_PX), np.uint8)) > 0
    n_cc, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
    keep = np.zeros((H, W), bool); dropped = 0
    for i in range(1, n_cc):
        comp = lab == i
        a = int(comp.sum())
        if a >= MIN_CC_PX and (comp & link).any():
            keep |= comp
        else:
            dropped += a
    if not keep.any():
        keep = cand.copy()
    return keep, dropped


# ---------------------------------------------------------------------------
# BiSeNet parse cache (Task 49 follow-up: dedicated profiling found
# parser.parse() costing ~110ms/frame, from 4 calls/frame -- precompute_stats
# and composite_frame_v3 each call build_r8 once per live snapshot (Phase 1
# stats vs Phase 2 render use different neck_taper/seg2_pushpull_fill/etc.
# kwargs, so the two build_r8 calls are NOT redundant as a whole), and each
# build_r8 call parses BOTH the driver frame and the avatar render (2 parses
# x 2 build_r8 calls = 4/frame). The driver PNG and avatar PNG on disk are
# read via a bare cv2.imread with no pre-parse pixel modification in either
# call site (see build_r8 below: `sl = parser.parse(drv)` runs directly on
# the freshly-loaded driver image, `alab = parser.parse(rgba[..., :3])` runs
# on rgba after only fix_canvas_edge_clip/dehalo -- both applied identically
# by both call sites, since pipeline.py passes the SAME build_kwargs to both
# bv3.precompute_stats() and bv3.composite_frame_v3() for a given snapshot).
# So within one process_snapshot() call, both build_r8 invocations parse
# byte-identical pixel content for the same frame index -- the second parse
# of each image is pure redundant computation, not a different result.
#
# Keyed on (path, mtime_ns, size) rather than pixel content so a cache hit
# costs one os.stat(), not a hash over a 720p image, and a hit is only ever
# returned for a file that has not changed since it was cached (the live
# pipeline reuses a single-slot filename per FRAME_IDX -- see pipeline.py's
# `FRAME_IDX = 0  # single reusable slot` -- so mtime is exactly what
# changes between snapshots and exactly what stays fixed within one). Keyed
# by path (not by path+signature) in the dict itself so a live session's two
# fixed filenames (driver/avatar) never grow past 2 entries; a stale-content
# entry for the same path is simply overwritten, never silently reused,
# because the stored signature is checked before returning it.
#
# The offline batch path (blend_v3.main()) reuses this for free too --
# Phase 1 (precompute_stats over all frames) and Phase 2 (composite_frame_v3
# over all frames) read the same unchanging frame files in two full passes,
# so Phase 2 hits this cache for every frame. That grows the cache to one
# entry per distinct frame file ever seen (2x n_frames for a full clip);
# _PARSE_CACHE_MAX_ENTRIES bounds this so a long offline run can't retain an
# unbounded amount of label-map memory -- once past the cap the cache is
# cleared and rebuilt, which only costs a cache-miss recompute, never a
# wrong answer.
_PARSE_CACHE = {}  # (path, variant) -> ((mtime_ns, size), label_map)
_PARSE_CACHE_MAX_ENTRIES = 16


def _cached_parse(parser, img, path, variant=None):
    """variant folds in any kwarg that can change the PIXELS passed to
    parser.parse() for this same on-disk path (e.g. edge_clip_fix/halo_fix
    applied to the avatar rgba before its parse) -- (path, mtime, size)
    alone only proves the FILE hasn't changed, not that both call sites
    preprocessed it identically. Both build_r8 call sites currently use
    build_r8's own defaults for these (neither precompute_stats' stat_bkw
    nor pipeline.py's build_kwargs override them), so this is currently a
    no-op guard rather than an observed real difference -- kept so a future
    caller passing a different edge_clip_fix/halo_fix can't silently get a
    stale cross-variant cache hit."""
    try:
        st = os.stat(path)
        sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        return parser.parse(img)
    key = (path, variant)
    cached = _PARSE_CACHE.get(key)
    if cached is not None and cached[0] == sig:
        return cached[1]
    result = parser.parse(img)
    if len(_PARSE_CACHE) >= _PARSE_CACHE_MAX_ENTRIES:
        _PARSE_CACHE.clear()
    _PARSE_CACHE[key] = (sig, result)
    return result


# ---------------------------------------------------------------------------
# hinge_leak_fix (Step 25): driver ear/jaw-hinge residue during the f053-081
# turn. Where the driver pixel survives into the composite (a_av<0.5, We<0.5,
# composite ~= driver, driver != plate) inside a fixed hinge box, the erase
# geometry never claimed it (erase_bin==0, or erase_bin==1 but seam_ramp==0
# below the jaw sign line). BiSeNet labels of the same residue flip between
# skin/background/ear over time, so the target is defined by PROVENANCE, not
# by label: per frame, on the BASELINE render b (fix off),
#   C0 = zone & (|drv-plate_stat|_L1 > fg_l1) & (|proto-drv|_L1 < vis_l1)
#        & (a_av < a_av_max) & (We < we_max)
#   T  = 8-connected components of C0 with >= min_px pixels
# then re-render with erase_bin |= T and seam_ramp = 1 on T only (so We takes
# its feathered We_soft value there). faceskin, a_av and every pixel outside T
# are untouched. There is no distance-to-faceskin term and no label term.
# TUNED TO THIS SPECIFIC CLIP (test.mp4): the zone box and frame window are
# the f053-081 hinge geometry of that clip's driver turn and must be
# re-derived for any new driving video. Default None = OFF, byte-identical.
# Params: dict(window=(f0,f1), zone=(x0,y0,w,h), fg_l1, vis_l1, a_av_max, we_max, min_px).
HINGE_LEAK_FIX_DEFAULT = None
HINGE_LEAK_FIX_TEST_MP4 = dict(window=(53, 81), zone=(497, 409, 100, 120),
                               fg_l1=45, vis_l1=30, a_av_max=0.5, we_max=0.5, min_px=6)


def _hinge_leak_target(b, p):
    """Step-25 target mask T from baseline build_fullhead dict `b` (see block above)."""
    We = np.asarray(b["We"]); a = np.asarray(b["a_av"])
    drv = np.asarray(b["drv"]).astype(np.float32)
    pl = np.asarray(b["plate_stat"]).astype(np.float32)
    pr = np.asarray(b["proto"]).astype(np.float32)
    x0, y0, zw, zh = p["zone"]
    zone = np.zeros(We.shape, bool); zone[y0:y0 + zh, x0:x0 + zw] = True
    c0 = (zone & (np.abs(drv - pl).sum(-1) > p["fg_l1"]) & (np.abs(pr - drv).sum(-1) < p["vis_l1"])
          & (a < p["a_av_max"]) & (We < p["we_max"]))
    ncc, ccl = cv2.connectedComponents(c0.astype(np.uint8), connectivity=8)
    keep = np.bincount(ccl.ravel(), minlength=ncc) >= p["min_px"]; keep[0] = False
    return keep[ccl]


_BUILD_R8_CORE_CACHE = {"key": None, "core": None}


def _build_r8_core(f, parser, M, plate_stat, lm, ear_boost=True,
                    feather_sigma=FEATHER_SIGMA, crop_edge_fix=True,
                    halo_fix=DEHALO_INPAINT_DEFAULT, av_clamp=False,
                    edge_clip_fix=True, render_canvas_h=RENDER_CANVAS_H_DEFAULT,
                    avatar_frames_dir=None, frames_dir=None,
                    rvm_lateral_matte=RVM_LATERAL_MATTE_DEFAULT,
                    rvm_hair_blend=RVM_HAIR_BLEND_DEFAULT,
                    faceskin_dilate_above_jaw=FACESKIN_DILATE_ABOVE_JAW_DEFAULT,
                    av_alpha_dilate=AV_ALPHA_DILATE_DEFAULT,
                    av_alpha_morph_sdf=AV_ALPHA_MORPH_SDF_DEFAULT,
                    reg=REG,
                    glasses_erase_extend=GLASSES_ERASE_EXTEND_DEFAULT,
                    jaw_seam_xmin_taper=JAW_SEAM_XMIN_TAPER_DEFAULT,
                    sl_override=None, hinge_T=None,
                    drv_img=None, avatar_rgba_img=None):
    """build_r8's frame-load / erase-geometry / avatar-alpha computation --
    every step that does NOT depend on neck_taper or any of the other
    strictly-tail-only flags (halo_band_thresh, aaband_content_gate,
    crown_temple_dilate, av_edge_fix, erase_feather_clamp, seg2_pushpull_fill,
    rim3d_*). Split out of build_r8 (Task 51, build_r8-dedup) because
    precompute_stats and composite_frame_v3/build_fullhead each call build_r8
    once per frame with IDENTICAL args except neck_taper -- this function is
    the ~90% of build_r8's cost (BiSeNet-dependent masks, jaw distance field,
    morphology chain, both warpAffine calls, both GaussianBlur calls) that
    was previously computed twice per frame for no reason. See build_r8's
    own docstring for the memoizing wrapper built on top of this."""
    # frames_dir is an explicit per-call argument (not the module-level
    # FRAMES_DIR global) precisely so concurrent callers -- e.g. two live
    # PipelineState instances/threads in the same process -- can never race
    # on each other's driver-frame path; see the cross-process/cross-thread
    # scratch-path collision fix in pipeline.py's module docstring, and
    # AVATAR_FRAMES_DIR's identical existing pattern just below.
    # Task 52 (live disk-round-trip elimination): the live pipeline has the
    # driver frame and avatar RGBA already in memory (it only wrote them to
    # PNG so this function could read them back via frames_dir/
    # avatar_frames_dir) -- drv_img/avatar_rgba_img let a caller pass those
    # arrays directly and skip the imwrite-then-imread round trip entirely.
    # Offline callers that never pass these keep reading from disk exactly
    # as before.
    if drv_img is not None:
        drv = drv_img
    else:
        _drv_dir = Path(frames_dir) if frames_dir is not None else FRAMES_DIR
        _drv_path = _drv_dir / f"{f:05d}.png"
        drv = cv2.imread(str(_drv_path))
    if avatar_rgba_img is not None:
        rgba = avatar_rgba_img
        # _av_path is also used below as _cached_parse's disk-signature key;
        # a path that can never exist on disk makes os.stat() raise and
        # _cached_parse fall through to a fresh parser.parse() call, which is
        # correct here regardless -- with drv_img/avatar_rgba_img supplied,
        # _build_r8_core already only runs once per live snapshot (see the
        # _BUILD_R8_CORE_CACHE memoization in build_r8), so there is nothing
        # for _cached_parse's own disk-mtime cache to usefully dedup.
        _av_path = Path(f"<in-memory avatar id={id(avatar_rgba_img)}>")
    else:
        _av_dir = Path(avatar_frames_dir) if avatar_frames_dir is not None else AVATAR_FRAMES_DIR
        _av_path = _av_dir / f"{f:05d}_avatar.png"
        rgba = cv2.imread(str(_av_path), cv2.IMREAD_UNCHANGED)
    assert drv is not None and rgba is not None, f"missing inputs f{f}"
    if edge_clip_fix:
        rgba = fix_canvas_edge_clip(rgba)
    if halo_fix:
        rgba = dehalo_avatar_rgb(rgba)

    # Task B1 (live glasses/thin-feature detection fix, opt-in): the live
    # pipeline can precompute a higher-quality segmentation label map by
    # parsing the native camera frame directly, before pipeline.py's
    # _ensure_size letterbox-downscales it to this module's fixed W x H
    # working canvas -- see pipeline.py's process_snapshot for how
    # sl_override is built and remapped into this canvas's coordinate
    # system. None (the offline/default path) is fully unaffected: `sl`
    # is computed exactly as before from the on-disk driver frame.
    sl = sl_override if sl_override is not None else _cached_parse(parser, drv, str(_drv_path))
    head_parse = mask_for(sl, HEAD_LABELS)
    hair = mask_for(sl, HAIR_SET)
    faceskin = mask_for(sl, FACE_SKIN_SET)
    ear = mask_for(sl, (L_EAR, R_EAR, EAR_R))
    neck = (sl == NECK) | (sl == NECK_L)
    cloth = (sl == CLOTH)
    drv_gray = cv2.cvtColor(drv, cv2.COLOR_BGR2GRAY).astype(np.float32)

    hair_ext = cv2.dilate(hair.astype(np.uint8), np.ones((HAIR_DILATE, HAIR_DILATE), np.uint8)) > 0

    if ear_boost and ear.any():
        ear_zone = cv2.dilate(ear.astype(np.uint8), np.ones((EAR_ZONE_PX, EAR_ZONE_PX), np.uint8)) > 0
        hair_ear = cv2.morphologyEx(hair.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((EAR_CLOSE_PX, EAR_CLOSE_PX), np.uint8))
        hair_ear = cv2.dilate(hair_ear, np.ones((EAR_HAIR_DILATE, EAR_HAIR_DILATE), np.uint8)) > 0
        near_hair = cv2.dilate(hair.astype(np.uint8), np.ones((EAR_NEAR_HAIR_PX, EAR_NEAR_HAIR_PX), np.uint8)) > 0
        hairtoned = (drv_gray < EAR_HAIRTONE_LUMA) | near_hair
        hair_boost = hair_ear & ear_zone & hairtoned
    else:
        hair_boost = np.zeros((H, W), bool)

    jaw_off = jaw_offset(lm[f, JAW_IDX, :2])
    reg_mask = np.zeros((H, W), bool); rx0, ry0, rx1, ry1 = reg; reg_mask[ry0:ry1, rx0:rx1] = True

    if jaw_seam_xmin_taper is not None:
        f0, f1, f2, f3, taper_px = jaw_seam_xmin_taper
        if f <= f0 or f >= f3:
            xmin_taper_w = 0.0
        elif f < f1:
            xmin_taper_w = (f - f0) / max(1, (f1 - f0))
        elif f <= f2:
            xmin_taper_w = 1.0
        else:
            xmin_taper_w = 1.0 - (f - f2) / max(1, (f3 - f2))
        if xmin_taper_w > 0.0:
            d_signed_plain = jaw_sign_field(jaw_off)
            d_signed_taper = jaw_sign_field(jaw_off, xmin_taper_px=taper_px)
            d_signed = (d_signed_plain + xmin_taper_w * (d_signed_taper - d_signed_plain)).astype(np.float32)
        else:
            d_signed = jaw_sign_field(jaw_off)
    else:
        d_signed = jaw_sign_field(jaw_off)
    seam_ramp = np.clip((d_signed + 0.5 * JAW_FEATHER_PX) / JAW_FEATHER_PX, 0.0, 1.0).astype(np.float32)
    if hinge_T is not None:   # hinge_leak_fix: T pixels take We_soft undiminished
        seam_ramp = np.where(hinge_T, np.float32(1.0), seam_ramp).astype(np.float32)
    head_hair = head_parse | hair_ext | hair_boost
    upper_region = cv2.dilate(head_hair.astype(np.uint8), np.ones((UPPER_MARGIN, UPPER_MARGIN), np.uint8)) > 0
    cand = (hair_ext | hair_boost | faceskin) & upper_region & (~neck) & (~cloth) & reg_mask

    erase_bin, dropped = keep_linked_cc(cand, faceskin)
    erase_prod_src = erase_bin   # pre-morphology production erase geometry (RVM branch below)

    k_co = np.ones((MORPH_CLOSE_OPEN_PX, MORPH_CLOSE_OPEN_PX), np.uint8)
    erase_bin_u8 = cv2.morphologyEx(erase_bin.astype(np.uint8), cv2.MORPH_CLOSE, k_co)
    erase_bin_u8 = cv2.morphologyEx(erase_bin_u8, cv2.MORPH_OPEN, k_co)
    erase_bin = erase_bin_u8 > 0
    if hinge_T is not None:   # hinge_leak_fix: inject provenance target into the final erase_bin
        erase_bin = erase_bin | hinge_T

    We_soft = cv2.GaussianBlur(erase_bin.astype(np.float32), (0, 0), feather_sigma)
    core = cv2.erode(erase_bin.astype(np.uint8), np.ones((CORE_ERODE, CORE_ERODE), np.uint8)) > 0
    We_soft = np.maximum(We_soft, core.astype(np.float32))
    We = np.minimum(We_soft, seam_ramp).astype(np.float32)

    if rvm_lateral_matte:
        # Augment the erase geometry with RVM's recurrent outer silhouette, but
        # ONLY at/above the jaw line, then splice the result back so the below-jaw
        # region stays PIXEL-IDENTICAL to production.
        #  * above_jaw is keyed off jaw_taper_depth_field -- the UNCLAMPED jaw
        #    distance (continuous past the mandible x-endpoints, the same field
        #    neck_taper uses). seam_ramp is built from jaw_sign_field, whose
        #    outside_cols clamp is hard-positive past those endpoints, so seam_ramp
        #    alone does NOT re-zero a morphology/feather spill past the jaw
        #    corners -- hence the explicit crossfade here.
        #  * We is recomputed from the augmented erase_bin, then crossfaded back to
        #    the production We across the JAW_FEATHER_PX band at that continuous
        #    line: below it We == production exactly; above it RVM coverage wins.
        #  * a_av (hence BOTH neck_taper and the AA-band gate) is derived further
        #    down from alpha_canvas, never from erase_bin / We -- untouched.
        pha_f = _rvm_pha_frame(f)
        rvm_sil = (pha_f >= RVM_MATTE_THRESH) & reg_mask & (~cloth)
        if rvm_sil.any():
            n_cc, cc_lab = cv2.connectedComponents(rvm_sil.astype(np.uint8), connectivity=8)
            if n_cc > 2:
                cc_area = np.bincount(cc_lab.ravel()); cc_area[0] = 0
                rvm_sil = cc_lab == int(cc_area.argmax())
            dt_jaw = jaw_taper_depth_field(jaw_off)
            above_jaw = dt_jaw >= -float(RVM_ABOVE_JAW_MARGIN_PX)
            eb_aug = erase_prod_src | (rvm_sil & above_jaw & (~erase_prod_src))
            au8 = cv2.morphologyEx(eb_aug.astype(np.uint8), cv2.MORPH_CLOSE, k_co)
            au8 = cv2.morphologyEx(au8, cv2.MORPH_OPEN, k_co)
            eb_aug_m = au8 > 0
            wsa = cv2.GaussianBlur(eb_aug_m.astype(np.float32), (0, 0), feather_sigma)
            cra = cv2.erode(eb_aug_m.astype(np.uint8), np.ones((CORE_ERODE, CORE_ERODE), np.uint8)) > 0
            wsa = np.maximum(wsa, cra.astype(np.float32))
            We_aug = np.minimum(wsa, seam_ramp).astype(np.float32)
            # lat = 0 for dt_jaw <= 0 (at/below the jaw line -> We stays exactly
            # the production value: below-jaw pixel-identical), ramping to 1 over
            # the JAW_FEATHER_PX band ABOVE the line. The RVM contribution only
            # ever adds erase weight strictly above the jaw.
            lat = np.clip(dt_jaw / JAW_FEATHER_PX, 0.0, 1.0).astype(np.float32)
            We = (We * (1.0 - lat) + np.maximum(We, We_aug) * lat).astype(np.float32)
            erase_bin = np.where(lat > 0.0, eb_aug_m, erase_bin)

    if faceskin_dilate_above_jaw:
        # Same crossfade template as rvm_lateral_matte above, with the
        # augmentation source swapped for a dilated-faceskin cand recompute
        # (Task A "mask-short" fix -- see the FACESKIN_DILATE_ABOVE_JAW_DEFAULT
        # docstring above for why this is scoped to strictly above the jaw
        # line only). Stacks on top of rvm_lateral_matte if both are enabled,
        # since it reads/writes the current erase_bin/We (whatever state they
        # are already in), not the pre-augmentation erase_prod_src/We_soft.
        faceskin_dil = cv2.dilate(faceskin.astype(np.uint8),
                                   np.ones((FACESKIN_DILATE_PX, FACESKIN_DILATE_PX), np.uint8)) > 0
        cand_fsd = (hair_ext | hair_boost | faceskin_dil) & upper_region & (~neck) & (~cloth) & reg_mask
        erase_fsd, _ = keep_linked_cc(cand_fsd, faceskin_dil)
        dt_jaw = jaw_taper_depth_field(jaw_off)
        above_jaw = dt_jaw >= -float(RVM_ABOVE_JAW_MARGIN_PX)
        eb_aug = erase_bin | (erase_fsd & above_jaw & (~erase_bin))
        au8 = cv2.morphologyEx(eb_aug.astype(np.uint8), cv2.MORPH_CLOSE, k_co)
        au8 = cv2.morphologyEx(au8, cv2.MORPH_OPEN, k_co)
        eb_aug_m = au8 > 0
        wsa = cv2.GaussianBlur(eb_aug_m.astype(np.float32), (0, 0), feather_sigma)
        cra = cv2.erode(eb_aug_m.astype(np.uint8), np.ones((CORE_ERODE, CORE_ERODE), np.uint8)) > 0
        wsa = np.maximum(wsa, cra.astype(np.float32))
        We_aug = np.minimum(wsa, seam_ramp).astype(np.float32)
        # lat = 0 at/below the jaw line -> We stays exactly the (possibly
        # rvm_lateral_matte-augmented) value already in `We`: below-jaw
        # pixel-identical. Ramps to 1 over JAW_FEATHER_PX strictly above.
        lat = np.clip(dt_jaw / JAW_FEATHER_PX, 0.0, 1.0).astype(np.float32)
        We = (We * (1.0 - lat) + np.maximum(We, We_aug) * lat).astype(np.float32)
        erase_bin = np.where(lat > 0.0, eb_aug_m, erase_bin)

    if glasses_erase_extend:
        eye_g = (sl == EYE_G)
        if eye_g.any():
            n_cc_g, lab_g = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
            glasses_add = np.zeros((H, W), bool)
            for i in range(1, n_cc_g):
                comp = lab_g == i
                if (comp & erase_bin).any():
                    continue  # already erased via the ordinary link test
                area = int(comp.sum())
                if area < GLASSES_MIN_FRAGMENT_PX:
                    continue
                if float((comp & eye_g).sum()) / area < GLASSES_EYEG_FRAC_THRESH:
                    continue  # not glasses-dominated -- leave the drop as-is
                glasses_add |= comp
            if glasses_add.any():
                glasses_add = cv2.dilate(glasses_add.astype(np.uint8),
                                          np.ones((GLASSES_DILATE_PX, GLASSES_DILATE_PX), np.uint8)) > 0
                glasses_add &= reg_mask & (~neck) & (~cloth)
                erase_bin = erase_bin | glasses_add
                We_g = cv2.GaussianBlur(erase_bin.astype(np.float32), (0, 0), feather_sigma)
                core_g = cv2.erode(erase_bin.astype(np.uint8), np.ones((CORE_ERODE, CORE_ERODE), np.uint8)) > 0
                We_g = np.maximum(We_g, core_g.astype(np.float32))
                We = np.minimum(We_g, seam_ramp).astype(np.float32)

    alab = _cached_parse(parser, rgba[..., :3], str(_av_path), variant=(edge_clip_fix, halo_fix))
    # avatar's own BiSeNet labels, warped into canvas coords with the same M
    # as av_rgb_real/alpha_canvas below -- nearest-neighbor since it's a label
    # map, not a continuous field. Lets precompute_stats's avatar_stats_mask=
    # "faceskin" option restrict the avatar "from" stats to face-skin-only
    # pixels (excluding hair/ears mixed into support&alpha01>=0.5) without a
    # second parser.parse() call -- alab is already computed above regardless.
    alab_canvas = cv2.warpAffine(alab.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST)
    aa = rgba[..., 3].copy().astype(np.float32); aa[alab == CLOTH] = 0.0
    alpha_canvas = cv2.warpAffine(aa, M, (W, H), flags=cv2.INTER_LINEAR)
    alpha01 = np.clip(alpha_canvas / 255.0, 0.0, 1.0).astype(np.float32)
    support = alpha01 > AV_SUPPORT_EPS
    if av_alpha_morph_sdf:
        a_av = morph_sdf_alpha(alpha01)
    else:
        a_av = cv2.GaussianBlur(alpha01, (0, 0), AV_BLUR)
    if av_clamp:
        a_av = a_av * support.astype(np.float32)
    if av_alpha_dilate:
        # Round AC f44 fix -- widen the avatar's own alpha footprint directly
        # (see AV_ALPHA_DILATE_* block above). Grayscale dilation on the
        # continuous a_av field, not a binary mask op, so the AA falloff at
        # the new outer edge stays smooth.
        k = np.ones((AV_ALPHA_DILATE_PX, AV_ALPHA_DILATE_PX), np.uint8)
        a_av = cv2.dilate(a_av, k)
    av_rgb_real = cv2.warpAffine(rgba[..., :3], M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)

    if crop_edge_fix:
        # render_canvas_h is the avatar render's true canvas height (512 prod,
        # 768 for the taller below-jaw-coverage render). With the tall render's
        # M this row lands below the driver frame, so the ramp is a no-op and
        # the avatar neck is no longer force-zeroed at the old ~larynx row.
        crop_edge_row = float(M[1, 1] * (render_canvas_h - 1) + M[1, 2])
        yy = np.arange(H, dtype=np.float32)[:, None]
        edge_ramp = np.clip((crop_edge_row - yy) / CROP_EDGE_RAMP_PX, 0.0, 1.0).astype(np.float32)
        a_av = a_av * edge_ramp

    if rvm_hair_blend:
        a_av = _rvm_hair_blend_apply(a_av, hair, faceskin, neck, cloth, f)

    return dict(drv=drv, We=We, d_signed=d_signed, jaw_off=jaw_off,
                erase_bin=erase_bin, hair=hair, hair_ext=hair_ext, hair_boost=hair_boost,
                faceskin=faceskin, head_parse=head_parse, a_av=a_av, alpha01=alpha01,
                support=support, sl=sl, drv_gray=drv_gray, dropped=int(dropped),
                av_rgb_real=av_rgb_real, alab_canvas=alab_canvas)


def _build_r8_tail(core, f, lm, plate_stat, av_edge_fix=True,
                    halo_band_thresh=HALO_BAND_THRESH_DEFAULT,
                    halo_band_smoothstep=HALO_BAND_SMOOTHSTEP_DEFAULT,
                    render_canvas_h=RENDER_CANVAS_H_DEFAULT,
                    neck_taper=NECK_TAPER_DEFAULT, neck_taper_start_px=NECK_TAPER_START_PX_DEFAULT,
                    neck_taper_depth_px=NECK_TAPER_DEPTH_PX_DEFAULT, neck_taper_ease=NECK_TAPER_EASE_DEFAULT,
                    aaband_content_gate=AABAND_CONTENT_GATE_DEFAULT,
                    rim3d_injection=RIM3D_INJECTION_DEFAULT,
                    rim3d_jaw_ext=RIM3D_JAW_EXT_DEFAULT,
                    crown_temple_dilate=CROWN_TEMPLE_DILATE_DEFAULT,
                    erase_feather_clamp=ERASE_FEATHER_CLAMP_DEFAULT,
                    erase_feather_clamp_ramp=ERASE_FEATHER_CLAMP_RAMP_DEFAULT,
                    jaw_taper_continuous=JAW_TAPER_CONTINUOUS_DEFAULT,
                    neck_taper_oor_gate=NECK_TAPER_OOR_GATE_DEFAULT,
                    driver_neck_own=DRIVER_NECK_OWN_DEFAULT,
                    neck_taper_ramp=NECK_TAPER_RAMP_DEFAULT,
                    neck_taper_ear_protect=NECK_TAPER_EAR_PROTECT_DEFAULT,
                    seg2_pushpull_fill=SEG2_PUSHPULL_FILL_DEFAULT,
                    edge_rgb_dilate=EDGE_RGB_DILATE_DEFAULT,
                    aaband_av_lo=None, aaband_av_hi=None,
                    aaband_temporal_hysteresis=AABAND_TEMPORAL_HYSTERESIS_DEFAULT,
                    aaband_sl_prev=None,
                    aaband_hairline_exempt_row_px=AABAND_HAIRLINE_EXEMPT_ROW_PX_DEFAULT,
                    aaband_ear_exempt_px=AABAND_EAR_EXEMPT_PX_DEFAULT,
                    aaband_delete_exempt_override=None,
                    aaband_candidate_c_retention=AABAND_CANDIDATE_C_RETENTION_DEFAULT,
                    aaband_candidate_c_lo=None, aaband_candidate_c_hi=None,
                    contact_shadow=CONTACT_SHADOW_DEFAULT,
                    contact_shadow_start_px=CONTACT_SHADOW_START_PX_DEFAULT,
                    contact_shadow_depth_px=CONTACT_SHADOW_DEPTH_PX_DEFAULT,
                    contact_shadow_strength=CONTACT_SHADOW_STRENGTH_DEFAULT):
    """The neck_taper-onward tail of build_r8 -- everything that either
    depends on neck_taper or is cheap enough (elementwise ops on masks/
    fields _build_r8_core already computed, no new BiSeNet/distanceTransform/
    warpAffine/blur work) that caching it separately isn't worth the
    complexity. See _build_r8_core's docstring."""
    drv = core["drv"]; support = core["support"]; d_signed = core["d_signed"]
    jaw_off = core["jaw_off"]; sl = core["sl"]; alpha01 = core["alpha01"]
    av_rgb_real = core["av_rgb_real"]; head_parse = core["head_parse"]
    hair = core["hair"]; erase_bin = core["erase_bin"]; We = core["We"]
    a_av = core["a_av"]

    if neck_taper:
        # smooth 1->0 fade of the avatar alpha below the jaw (see NECK_TAPER
        # constants). Anchored to the jaw sign line: 1.0 at/above -start (jaw --
        # untouched), 0.0 at/below -(start+depth) (deep neck -- avatar gone). No
        # step edge -> the Variant-F warm handoff band is alpha-feathered out
        # instead of terminating in a line.
        # Uses jaw_taper_depth_field (NOT the shared d_signed): the latter's
        # outside_cols clamp is a hard vertical step at the mandible x-endpoints
        # that has no mask on this path and printed a moving collar seam on yaw
        # sweeps. The unclamped field fades smoothly past the jaw corners.
        d_taper = (jaw_taper_depth_field_continuous(jaw_off) if jaw_taper_continuous
                   else jaw_taper_depth_field(jaw_off))
        if neck_taper_oor_gate:
            d_taper = _neck_taper_oor_gate(d_taper, We, jaw_off)
        _depth = max(float(neck_taper_depth_px), 1.0)
        tt = np.clip(((-d_taper) - float(neck_taper_start_px)) / _depth, 0.0, 1.0)
        if str(neck_taper_ease).lower() == "smoothstep":
            tt = tt * tt * (3.0 - 2.0 * tt)
        # Candidate C1 (Task 2c-revised): scale the taper strength by a
        # frame-index schedule. s=1.0 (ramp=None, or f outside the dip
        # window) reproduces `tt` unchanged -- byte-identical to before this
        # kwarg existed. s=0.0 (dip floor) makes tt_eff==0 everywhere, i.e.
        # this frame behaves exactly like neck_taper=False.
        s = _efc_ramp_strength(f, neck_taper_ramp)
        tt_eff = tt if s >= 1.0 else (np.zeros_like(tt) if s <= 0.0 else tt * s)

        # Candidate C3 (Task 2c-revised, round 3): spatially gate the
        # suppression to the ear label instead of the whole frame. p=0.0
        # (neck_taper_ear_protect=None, or f outside its window) skips this
        # block entirely -- tt_eff stays exactly what C1 (or plain neck_taper)
        # produced above, byte-identical. p>0.0 only softens tt_eff where
        # ear_soft>0 (the dilated+feathered ear label); the below-jaw neck,
        # where ear_soft==0, keeps tt_eff unchanged regardless of p.
        p = 1.0 - _efc_ramp_strength(f, neck_taper_ear_protect) if neck_taper_ear_protect is not None else 0.0
        if p > 0.0:
            ear = mask_for(sl, (L_EAR, R_EAR, EAR_R))
            if ear.any():
                ear_d = cv2.dilate(ear.astype(np.uint8),
                                    np.ones((NECK_TAPER_EAR_PROTECT_DILATE_PX, NECK_TAPER_EAR_PROTECT_DILATE_PX), np.uint8))
                ear_soft = cv2.GaussianBlur(ear_d.astype(np.float32), (0, 0), NECK_TAPER_EAR_PROTECT_FEATHER_SIGMA)
                ear_soft = np.clip(ear_soft, 0.0, 1.0)
                if NECK_TAPER_EAR_SOFT_SATURATE:
                    _sat = np.clip(ear_soft / float(NECK_TAPER_EAR_SOFT_SATURATE), 0.0, 1.0)
                    if NECK_TAPER_EAR_SOFT_SATURATE_XMIN > 0:
                        _sat[:, :NECK_TAPER_EAR_SOFT_SATURATE_XMIN] = ear_soft[:, :NECK_TAPER_EAR_SOFT_SATURATE_XMIN]
                    ear_soft = _sat
                tt_eff = tt_eff * (1.0 - p * ear_soft)

        a_av = (a_av * (1.0 - tt_eff)).astype(np.float32)

    if halo_band_thresh is not None:
        if halo_band_smoothstep is not None:
            # Candidate D-soft (opt-in, see HALO_BAND_SMOOTHSTEP_DEFAULT's
            # docstring): replaces the hard `>=` cutoff with a narrow cubic
            # ramp straddling it, so renderer alpha01 noise near the
            # boundary produces a small continuous a_av change instead of a
            # full 0/1 snap. halo_band_thresh itself is otherwise unused in
            # this branch (kept as the outer None-check / opt-out switch).
            _hb_lo, _hb_hi = halo_band_smoothstep
            _t = np.clip((alpha01 - _hb_lo) / max(_hb_hi - _hb_lo, 1e-6), 0.0, 1.0)
            _ramp = (_t * _t * (3.0 - 2.0 * _t)).astype(np.float32)
            a_av = a_av * _ramp
        else:
            a_av = a_av * (alpha01 >= halo_band_thresh).astype(np.float32)

    if aaband_content_gate:
        # zero a_av at AA-band silhouette pixels sitting over BiSeNet BACKGROUND
        # (see AABAND_* constants). Hard gate; identity outside the band.
        # a_av_pre: the raw a_av value at entry to this gate block, BEFORE
        # this block's own hard-delete (or Candidate C's soft retention)
        # applies -- captured here, not re-derived later, so Candidate C's
        # ramp below is never computed from an already-attenuated value
        # (no recursive attenuation). trans/delete's own condition is
        # unchanged, still computed from this same a_av, exactly as before
        # Candidate C existed.
        _ear_w = None
        a_av_pre = a_av
        _av_lo = AABAND_AV_LO if aaband_av_lo is None else aaband_av_lo
        _av_hi = AABAND_AV_HI if aaband_av_hi is None else aaband_av_hi
        trans = (a_av > _av_lo) & (a_av < _av_hi)
        delete = trans & (sl == BACKGROUND)

        # Candidate 2 (opt-in, default off): temporal hysteresis -- only
        # delete where the SAME pixel was also BiSeNet BACKGROUND in the
        # supplied previous-frame label map. A single-frame misclassified
        # hair strand won't agree with the prior frame at the same pixel
        # (head moves little frame to frame at video framerate), so this
        # strictly narrows `delete` versus the per-frame-only case above;
        # with aaband_sl_prev=None (no prior frame available, e.g. the
        # very first snapshot) this is a no-op and delete is unchanged.
        if aaband_temporal_hysteresis and aaband_sl_prev is not None:
            delete = delete & (aaband_sl_prev == BACKGROUND)

        # Candidate 3 (opt-in, default off): exempt a band of rows just
        # below the avatar's own top-of-silhouette row (the hairline) from
        # the gate entirely, on the theory that the false-positive case the
        # gate targets (skin-toned bleed onto true background) mostly shows
        # up lower on the head/jaw perimeter, not at the very crown where a
        # thin hair strand is most likely to get one-frame misclassified.
        if aaband_hairline_exempt_row_px > 0:
            _occ_rows = np.where(alpha01 > 0.02)[0]
            if _occ_rows.size:
                _top_row = int(_occ_rows.min())
                _exempt = np.zeros((H, W), bool)
                _exempt[_top_row:_top_row + int(aaband_hairline_exempt_row_px), :] = True
                delete = delete & (~_exempt)

        # Candidate 4 (opt-in, default off, see AABAND_EAR_EXEMPT_PX_DEFAULT):
        # exempt the driver's own (dilated) ear label from the gate entirely.
        if aaband_ear_exempt_px > 0:
            ear = mask_for(sl, (L_EAR, R_EAR, EAR_R))
            if ear.any():
                ear_d = cv2.dilate(ear.astype(np.uint8),
                                    np.ones((int(aaband_ear_exempt_px), int(aaband_ear_exempt_px)), np.uint8)) > 0
                if AABAND_EAR_EXEMPT_XMIN > 0:
                    ear_d[:, :AABAND_EAR_EXEMPT_XMIN] = False
                if AABAND_EAR_EXEMPT_FEATHER_SIGMA > 0:
                    _ear_w = np.clip(cv2.GaussianBlur(ear_d.astype(np.float32), (0, 0),
                                                       AABAND_EAR_EXEMPT_FEATHER_SIGMA), 0.0, 1.0)
                else:
                    delete = delete & (~ear_d)

        if rim3d_injection:
            # Round AB P0 mitigation (cheek) + Round AE jaw/chin extension --
            # see the RIM3D_* block above for the full method. Exempts only
            # pixels the true 3D mesh rim marks as genuine avatar geometry
            # with high underlying render alpha; everywhere else `delete` is
            # unchanged from production.
            quad_box = _rim3d_cheek_quad_box(f, lm)
            rim_mask = _rim3d_mesh_boundary_mask(f, render_canvas_h) & quad_box
            if rim3d_jaw_ext:
                jaw_box = _rim3d_jaw_quad_box(f, lm)
                jaw_rim_mask = _rim3d_jaw_mask(f, render_canvas_h, lm) & jaw_box
                # Connectivity gate (see RIM3D_JAW_LINK_PX comment above): only keep
                # jaw-rim components touching content that's ALREADY visible (not
                # currently deleted) -- drops isolated floating patches from FLAME
                # reprojection mismatches at extreme yaw, keeps genuine silhouette
                # gap closures.
                already_visible = cv2.dilate((support & (~delete)).astype(np.uint8),
                                              np.ones((RIM3D_JAW_LINK_PX, RIM3D_JAW_LINK_PX), np.uint8)) > 0
                n_cc, cc_lab = cv2.connectedComponents(jaw_rim_mask.astype(np.uint8), connectivity=8)
                jaw_rim_linked = np.zeros((H, W), bool)
                for i in range(1, n_cc):
                    comp = cc_lab == i
                    if (comp & already_visible).any():
                        jaw_rim_linked |= comp
                rim_mask = rim_mask | jaw_rim_linked
            exempt = rim_mask & (alpha01 >= RIM3D_ALPHA_MIN)
            delete = delete & (~exempt)

        # V21-REDO diagnostic hook (verify_V21b, uncommitted): subtracts an
        # externally-supplied boolean mask from the already-fully-computed
        # `delete` mask, at the last possible point before the hard-zero --
        # applied AFTER every existing exemption above, and touches nothing
        # else (sl/a_av/We/plated/proto upstream of this line are untouched).
        # Default None: byte-identical to production. Not a candidate for
        # commit/tag; diagnostic-only per this round's authorization.
        if aaband_delete_exempt_override is not None:
            delete = delete & (~aaband_delete_exempt_override)

        if aaband_candidate_c_retention:
            # Candidate C (Task V27, opt-in, default off -- see the
            # AABAND_CANDIDATE_C_* constants block above for the full
            # rationale/validation history). Replaces ONLY the hard-zero
            # below: instead of gate=0 at every delete-fired pixel, retain
            # a_av_pre * r, where r is a linear ramp of a_av_pre itself
            # (0 at/below _c_lo, 1 at/above _c_hi). `delete`'s own trigger
            # condition above is completely unchanged -- this only changes
            # what happens to a_av where delete is True. Byte-identical to
            # the untouched gate=np.where(...) path when False (default).
            _c_lo = AABAND_CANDIDATE_C_LO if aaband_candidate_c_lo is None else aaband_candidate_c_lo
            _c_hi = AABAND_CANDIDATE_C_HI if aaband_candidate_c_hi is None else aaband_candidate_c_hi
            r = np.clip((a_av_pre - _c_lo) / max(_c_hi - _c_lo, 1e-6), 0.0, 1.0).astype(np.float32)
            retained = (a_av_pre * r).astype(np.float32)
            a_av = np.where(delete, retained, a_av).astype(np.float32)
        else:
            gate = np.where(delete, 0.0 if _ear_w is None else _ear_w, 1.0).astype(np.float32)
            a_av = (a_av * gate).astype(np.float32)

    if crown_temple_dilate:
        # See the CROWN_TEMPLE_* constants block above for the full mechanism
        # rationale (not rim3d_injection's alpha-restore case -- alpha01 is
        # genuinely ~0 in this gap, so this grows a_av like av_alpha_dilate
        # does, just scoped to the crown/temple box(es) and sized per-side
        # per-frame from the actual measured gap instead of a fixed kernel).
        # Placed LAST, after halo_band_thresh/aaband_content_gate: both of
        # those gate a_av off the RAW (undilated) alpha01, which is ~0 in
        # this exact gap by definition -- applying this any earlier just
        # gets immediately re-zeroed by those gates, a no-op (confirmed:
        # an earlier placement right after av_alpha_dilate measurably made
        # the tracked gap metric WORSE, not better, once halo_band_thresh
        # zeroed the growth back out and support/erase_bin bookkeeping in
        # the validation script picked up the discrepancy).
        #
        # Sized against a_av ITSELF (post-gating), not the raw `support`
        # mask: `support` is alpha01>AV_SUPPORT_EPS (~0.008), an extremely
        # low bar that halo_band_thresh (0.15) and aaband_content_gate
        # already trim back hard right at that boundary -- measuring the
        # gap relative to `support` sized the dilation kernel too small
        # (the "source" values within that radius were themselves mostly
        # already gated to ~0, so the growth had nothing solid to
        # propagate). Measuring relative to where a_av is actually solid
        # right now correctly accounts for whatever the earlier gates
        # already trimmed.
        true_head_ct = head_parse | hair
        box_l, box_r = _crown_temple_quad_boxes(f, lm)
        av_solid = a_av > CROWN_TEMPLE_SOLID_THRESH
        dist_from_solid = None
        for box in (box_l, box_r):
            gap_side = box & erase_bin & true_head_ct & (~av_solid)
            if not gap_side.any():
                continue
            if dist_from_solid is None:
                dist_from_solid = cv2.distanceTransform((~av_solid).astype(np.uint8), cv2.DIST_L2, 5)
            px = float(np.percentile(dist_from_solid[gap_side], CROWN_TEMPLE_GAP_PCTL))
            px = int(np.clip(round(px), 1, CROWN_TEMPLE_DILATE_MAX_PX))
            k_ct = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
            grown = cv2.dilate(a_av, k_ct)
            a_av = np.where(gap_side, np.maximum(a_av, grown), a_av).astype(np.float32)

    if av_edge_fix:
        drv_local = cv2.GaussianBlur(drv.astype(np.float32), (0, 0), NECK_BLUR)
        din = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 5)
        w_edge = np.clip((EDGE_BAND_PX - din) / EDGE_BAND_PX, 0.0, 1.0).astype(np.float32)
        w_edge *= support.astype(np.float32)
        w_edge *= (d_signed < JAW_FEATHER_PX * 2.0).astype(np.float32)
        we3 = w_edge[..., None]
        av_rgb_canvas = av_rgb_real * (1.0 - we3) + drv_local * we3
        pad = cv2.dilate(av_rgb_canvas, np.ones((AV_RGB_PAD, AV_RGB_PAD), np.uint8))
        av_rgb_canvas = np.where(support[..., None], av_rgb_canvas, pad)
    else:
        av_rgb_canvas = av_rgb_real

    if erase_feather_clamp:
        # erase_feather_clamp (segment 1 only): closes the cord artifact's
        # upper segment (genuine 0<We<1 partial leak) inside EFC_ROI. The
        # lower ear-to-jaw segment is NOT fixed by this -- av_rgb_canvas is
        # (0,0,0) there, a genuine avatar-render coverage gap,
        # architecturally out of reach for a compositing weight change (see
        # project_jawneck_seam_parked). In poses where the ear is mostly
        # hair-occluded, a small sliver of real ear pixels may also be
        # clamped as a side effect of the same a_av==0-driven mechanism --
        # visually minor (confirmed via direct crop inspection at
        # f100/f150) but not zero, as an earlier validation pass
        # incorrectly claimed. Segment 2 (letting driver content show in
        # the coverage gap) was tested and rejected -- it exposes the
        # driver's real dark hair as a large black patch at some poses (see
        # f080_black_patch_zoom.png) and is not present in this shipped
        # version.
        roi = np.zeros((H, W), bool)
        ry0, ry1, rx0, rx1 = EFC_ROI
        roi[ry0:ry1, rx0:rx1] = True
        no_avatar = (a_av <= 0.0) & roi
        seg1 = no_avatar & (We > 0.0) & (We < 1.0)
        s = _efc_ramp_strength(f, erase_feather_clamp_ramp)
        if s >= 1.0:
            # exact original expression -- guarantees erase_feather_clamp_ramp=None
            # (or any ramp whose window excludes this frame) is byte-identical.
            We = np.where(seg1, 1.0, We).astype(np.float32)
        elif s <= 0.0:
            pass  # clamp fully off at this frame -- leave We untouched (== OFF)
        else:
            We = np.where(seg1, We * (1.0 - s) + s, We).astype(np.float32)

    if seg2_pushpull_fill:
        av_rgb_canvas, a_av = _seg2_pushpull_fill(av_rgb_canvas, a_av, We)

    if edge_rgb_dilate:
        av_rgb_canvas = _edge_rgb_dilate(av_rgb_canvas, a_av, alpha01, hair, support)

    if driver_neck_own:
        a_av, We = _driver_neck_own(a_av, We, jaw_off, sl)

    Wc = We[..., None]; Aa = a_av[..., None]
    plated = drv.astype(np.float32) * (1.0 - Wc) + plate_stat * Wc

    if contact_shadow:
        # See CONTACT_SHADOW_* constants block above. Darkens `plated` only
        # (never av_rgb_canvas / a_av), so it is only ever visible through
        # (1 - Aa) below -- exactly the real driver/plate neck pixels
        # neck_taper's own fade already exposes, never the avatar itself.
        d_cs = jaw_taper_depth_field(jaw_off)
        depth_below = np.clip(-d_cs, 0.0, None)
        into_shadow = np.clip(depth_below - float(contact_shadow_start_px), 0.0, None)
        _depth_cs = max(float(contact_shadow_depth_px), 1.0)
        falloff = np.clip(1.0 - into_shadow / _depth_cs, 0.0, 1.0)
        falloff = falloff * falloff  # ease out, matches neck_taper's smoothstep feel
        x_gate = cv2.dilate(support.astype(np.uint8),
                             np.ones((2 * CONTACT_SHADOW_X_PAD_PX + 1, 2 * CONTACT_SHADOW_X_PAD_PX + 1),
                                     np.uint8)) > 0
        shadow = (float(contact_shadow_strength) * falloff * x_gate.astype(np.float32)).astype(np.float32)
        plated = plated * (1.0 - shadow[..., None])

    proto = np.clip(plated * (1.0 - Aa) + av_rgb_canvas * Aa, 0, 255).astype(np.uint8)

    return dict(drv=drv, proto=proto, We=We, d_signed=d_signed, jaw_off=jaw_off,
                erase_bin=erase_bin, hair=hair, hair_ext=core["hair_ext"], hair_boost=core["hair_boost"],
                faceskin=core["faceskin"], head_parse=head_parse, a_av=a_av, alpha01=alpha01,
                support=support, sl=sl, drv_gray=core["drv_gray"], dropped=core["dropped"],
                av_rgb_canvas=av_rgb_canvas, av_rgb_real=av_rgb_real, plate_stat=plate_stat,
                alab_canvas=core["alab_canvas"])


def build_r8(f, parser, M, plate_stat, lm, av_edge_fix=True, ear_boost=True,
             feather_sigma=FEATHER_SIGMA, crop_edge_fix=True, halo_fix=DEHALO_INPAINT_DEFAULT, av_clamp=False,
             edge_clip_fix=True, halo_band_thresh=HALO_BAND_THRESH_DEFAULT,
             halo_band_smoothstep=HALO_BAND_SMOOTHSTEP_DEFAULT,
             render_canvas_h=RENDER_CANVAS_H_DEFAULT, avatar_frames_dir=None, frames_dir=None,
             neck_taper=NECK_TAPER_DEFAULT, neck_taper_start_px=NECK_TAPER_START_PX_DEFAULT,
             neck_taper_depth_px=NECK_TAPER_DEPTH_PX_DEFAULT, neck_taper_ease=NECK_TAPER_EASE_DEFAULT,
             aaband_content_gate=AABAND_CONTENT_GATE_DEFAULT,
             rvm_lateral_matte=RVM_LATERAL_MATTE_DEFAULT,
             rvm_hair_blend=RVM_HAIR_BLEND_DEFAULT,
             faceskin_dilate_above_jaw=FACESKIN_DILATE_ABOVE_JAW_DEFAULT,
             rim3d_injection=RIM3D_INJECTION_DEFAULT,
             rim3d_jaw_ext=RIM3D_JAW_EXT_DEFAULT,
             av_alpha_dilate=AV_ALPHA_DILATE_DEFAULT,
             av_alpha_morph_sdf=AV_ALPHA_MORPH_SDF_DEFAULT,
             crown_temple_dilate=CROWN_TEMPLE_DILATE_DEFAULT,
             erase_feather_clamp=ERASE_FEATHER_CLAMP_DEFAULT,
             erase_feather_clamp_ramp=ERASE_FEATHER_CLAMP_RAMP_DEFAULT,
             jaw_taper_continuous=JAW_TAPER_CONTINUOUS_DEFAULT,
             neck_taper_oor_gate=NECK_TAPER_OOR_GATE_DEFAULT,
             driver_neck_own=DRIVER_NECK_OWN_DEFAULT,
             neck_taper_ramp=NECK_TAPER_RAMP_DEFAULT,
             neck_taper_ear_protect=NECK_TAPER_EAR_PROTECT_DEFAULT,
             seg2_pushpull_fill=SEG2_PUSHPULL_FILL_DEFAULT,
             edge_rgb_dilate=EDGE_RGB_DILATE_DEFAULT,
             reg=REG,
             glasses_erase_extend=GLASSES_ERASE_EXTEND_DEFAULT,
             jaw_seam_xmin_taper=JAW_SEAM_XMIN_TAPER_DEFAULT,
             sl_override=None, hinge_T=None,
             aaband_av_lo=None, aaband_av_hi=None,
             aaband_temporal_hysteresis=AABAND_TEMPORAL_HYSTERESIS_DEFAULT,
             aaband_sl_prev=None,
             aaband_hairline_exempt_row_px=AABAND_HAIRLINE_EXEMPT_ROW_PX_DEFAULT,
             aaband_ear_exempt_px=AABAND_EAR_EXEMPT_PX_DEFAULT,
             aaband_delete_exempt_override=None,
             aaband_candidate_c_retention=AABAND_CANDIDATE_C_RETENTION_DEFAULT,
             aaband_candidate_c_lo=None, aaband_candidate_c_hi=None,
             contact_shadow=CONTACT_SHADOW_DEFAULT,
             contact_shadow_start_px=CONTACT_SHADOW_START_PX_DEFAULT,
             contact_shadow_depth_px=CONTACT_SHADOW_DEPTH_PX_DEFAULT,
             contact_shadow_strength=CONTACT_SHADOW_STRENGTH_DEFAULT,
             drv_img=None, avatar_rgba_img=None):
    """Whole-head erasure + static-plate recompose. Ported verbatim (control flow and
    constants unchanged) from wholehead_cp_r8.build_r8 -- see that file's module
    docstring for the full history of each named fix. Production defaults here match
    the validated wholehead_cp_r8 defaults exactly: av_edge_fix=True, ear_boost=True,
    feather_sigma=FEATHER_SIGMA, crop_edge_fix=True, av_clamp=False,
    edge_clip_fix=True, halo_band_thresh=0.15 (candidate D, validated via the full
    245-frame regression in the fix3_verify session -- do not change). halo_fix now
    defaults to DEHALO_INPAINT_DEFAULT=False (see that constant's docstring for the
    245-frame dehalo-ablation validation behind this change) -- pass halo_fix=True
    to restore the (visually near-identical, ~440ms/frame slower) inpaint pass.

    Task 51 (build_r8-dedup): precompute_stats and composite_frame_v3/build_fullhead
    each call this once per live snapshot with IDENTICAL args except neck_taper (in
    the live pipeline specifically -- FRAME_IDX is fixed at 0 and both calls share
    the same parser/M/plate_stat/lm/sl_override objects, so the two calls are
    otherwise indistinguishable). This function is now a thin wrapper: the ~90% of
    the work that can't change between such calls (_build_r8_core) is memoized in
    _BUILD_R8_CORE_CACHE keyed on every argument that CAN affect it; only the cheap
    neck_taper-onward tail (_build_r8_tail) always re-runs. A call whose core-
    relevant args genuinely differ (different frame, different flags) simply misses
    the cache and recomputes core from scratch -- this changes only whether prior
    work is reused, never the output."""
    core_kwargs = dict(ear_boost=ear_boost, feather_sigma=feather_sigma, crop_edge_fix=crop_edge_fix,
                        halo_fix=halo_fix, av_clamp=av_clamp, edge_clip_fix=edge_clip_fix,
                        render_canvas_h=render_canvas_h, avatar_frames_dir=avatar_frames_dir,
                        frames_dir=frames_dir, rvm_lateral_matte=rvm_lateral_matte,
                        rvm_hair_blend=rvm_hair_blend,
                        faceskin_dilate_above_jaw=faceskin_dilate_above_jaw,
                        av_alpha_dilate=av_alpha_dilate, av_alpha_morph_sdf=av_alpha_morph_sdf, reg=reg,
                        glasses_erase_extend=glasses_erase_extend,
                        jaw_seam_xmin_taper=jaw_seam_xmin_taper, sl_override=sl_override,
                        drv_img=drv_img, avatar_rgba_img=avatar_rgba_img, hinge_T=hinge_T)
    # drv_img/avatar_rgba_img are ndarrays (unhashable) -- key on identity like
    # parser/M/plate_stat/lm above, not on the excluded-from-sort dict entry.
    key = (id(parser), id(M), id(plate_stat), id(lm), f, id(sl_override),
           id(drv_img), id(avatar_rgba_img),
           tuple(sorted((k, v) for k, v in core_kwargs.items()
                        if k not in ("sl_override", "drv_img", "avatar_rgba_img", "hinge_T"))),
           None if hinge_T is None else hashlib.md5(np.packbits(hinge_T).tobytes()).hexdigest())
    cached = _BUILD_R8_CORE_CACHE
    if cached["key"] == key:
        core = cached["core"]
    else:
        core = _build_r8_core(f, parser, M, plate_stat, lm, **core_kwargs)
        cached["key"] = key
        cached["core"] = core
    return _build_r8_tail(core, f, lm, plate_stat, av_edge_fix=av_edge_fix,
                           halo_band_thresh=halo_band_thresh,
                           halo_band_smoothstep=halo_band_smoothstep,
                           render_canvas_h=render_canvas_h,
                           neck_taper=neck_taper, neck_taper_start_px=neck_taper_start_px,
                           neck_taper_depth_px=neck_taper_depth_px, neck_taper_ease=neck_taper_ease,
                           aaband_content_gate=aaband_content_gate,
                           rim3d_injection=rim3d_injection, rim3d_jaw_ext=rim3d_jaw_ext,
                           crown_temple_dilate=crown_temple_dilate,
                           erase_feather_clamp=erase_feather_clamp,
                           erase_feather_clamp_ramp=erase_feather_clamp_ramp,
                           jaw_taper_continuous=jaw_taper_continuous,
                           neck_taper_oor_gate=neck_taper_oor_gate,
                           driver_neck_own=driver_neck_own,
                           neck_taper_ramp=neck_taper_ramp,
                           neck_taper_ear_protect=neck_taper_ear_protect,
                           seg2_pushpull_fill=seg2_pushpull_fill,
                           edge_rgb_dilate=edge_rgb_dilate,
                           aaband_av_lo=aaband_av_lo, aaband_av_hi=aaband_av_hi,
                           aaband_temporal_hysteresis=aaband_temporal_hysteresis,
                           aaband_sl_prev=aaband_sl_prev,
                           aaband_hairline_exempt_row_px=aaband_hairline_exempt_row_px,
                           aaband_ear_exempt_px=aaband_ear_exempt_px,
                           aaband_delete_exempt_override=aaband_delete_exempt_override,
                           aaband_candidate_c_retention=aaband_candidate_c_retention,
                           aaband_candidate_c_lo=aaband_candidate_c_lo,
                           aaband_candidate_c_hi=aaband_candidate_c_hi,
                           contact_shadow=contact_shadow,
                           contact_shadow_start_px=contact_shadow_start_px,
                           contact_shadow_depth_px=contact_shadow_depth_px,
                           contact_shadow_strength=contact_shadow_strength)


# ---------------------------------------------------------------------------
# ---- ported from _track4_variantC_lib.py / _track4_variantC2_lib.py ----
# (seam-blend gradient guard + guided-filter weight smoothing, reused
# verbatim by Variant F -- these constants/functions are the "unchanged
# mechanism" build_fullhead's docstring refers to). ----
# ---------------------------------------------------------------------------
GAMMA = 0.03
LF_SIGMA = 15.0
LF_BLUR_DOWNSAMPLE = 4  # Task 49 step 2: see _fast_lowfreq_blur docstring
OVERHANG_EPS = 1.0
GF_RADIUS = 8
GF_EPS = 1e-3
EPS_STD = 1e-3


def _fast_lowfreq_blur(img, sigma, downsample=LF_BLUR_DOWNSAMPLE):
    """Approximates cv2.GaussianBlur(img, (0,0), sigma) via downsample ->
    blur at sigma/downsample -> upsample back to the original size.

    Task 49 step 2: dedicated call-site profiling found this exact blur
    (avatar_low/driver_low, sigma=LF_SIGMA=15.0, on the full 720x1280x3 LAB
    canvas, called 4x/frame across build_fullhead + precompute_stats) is
    91.4% of ALL GaussianBlur cost in the pipeline (~244ms/frame of the
    ~267ms/frame total) -- by far the single largest concrete operation in
    composite_frame_v3. A sigma=15 blur is already a broad low-pass filter;
    approximating it at 1/downsample resolution (sigma'=sigma/downsample,
    still >=3 for downsample=4) and upsampling with bilinear interpolation
    reproduces the same broad low-frequency signal at a fraction of the
    pixel count the true full-res blur would touch, since a filter this
    wide has no fine detail to lose in the first place.

    Used at TWO call sites with different downstream sensitivity:
    build_fullhead subtracts this from the un-blurred image to get a
    per-pixel high-frequency residual (avatar_high/L_high_driver) that is
    added BACK into the final reconstruction -- so any approximation error
    here is only in the LOW-frequency term the residual trick is designed
    to discard, not in the high-frequency detail that is preserved exactly
    via the un-blurred original minus this. precompute_stats only reads
    mean/std of this over a masked region, an even more approximation-
    tolerant use. Both call sites verified byte-for-byte / visually against
    the pre-change output before this was shipped (see commit message)."""
    h, w = img.shape[:2]
    dh = max(1, round(h / downsample))
    dw = max(1, round(w / downsample))
    small = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
    small_blur = cv2.GaussianBlur(small, (0, 0), sigma / downsample)
    return cv2.resize(small_blur, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Region-aware harmonization weight (collar/shoulder seam fix).
# Variant F's w_smooth LAB harmonization crossfade (below) recolors the
# avatar's neck toward the driver's LOCAL LAB stats even where the two
# genuinely disagree (avatar neck carries baked billboard-source lighting;
# driver neck/collar does not) -- root-caused as the collar/shoulder band,
# project_collar_seam_diag. region_weight_field scopes w_smooth down to zero
# by REGION_HARMONIZE_DEPTH_PX below the jaw, anchored to the SAME
# (unclamped) jaw_taper_depth_field neck_taper already uses -- not a new
# coordinate system. Continuous (smoothstep), so it cannot itself relocate
# the seam to a new hard edge (the failure mode of the earlier hard-clip /
# hard-gate attempts on unrelated fixes). Below the taper depth, the LAB
# blend collapses to avatar_matched_low exactly -- a global mean/std tone
# match to the driver that PRESERVES the avatar's own local shading
# structure, not raw uncorrected avatar color (that "avatar_matched_low IS
# already a global-tone-match" mechanism is why the diagnosis's Experiment A
# region-aware-weight and Experiment B global-tone-match candidates turned
# out to be the same operation, differing only in decay depth).
#
# Validated (project_collar_seam_regionfix_verified.md): 8-frame spot-check +
# diff-heatmaps confirm the band closes without trading it for the larger,
# uncontrolled mismatch a hard harmonize-off produces; full 245-frame mean-
# L*a*b* temporal check shows no flicker regression vs production; targeted
# full-clip playback review (every fast-yaw transition, 3 speech/jaw-
# articulation-with-minimal-yaw segments, boundary-position tracking, patch-
# size-over-time) found no hard-edge/traveling-wall artifact, no size/shape
# runaway, and no visible difference from production at normal viewing.
# DEPTH_PX=10.0 ("Experiment B", aggressive) was the validated candidate --
# an initial 35-70px guess (matched to neck_taper's own depth / overhang_px
# scale) was a near-no-op; the raw w_smooth crossfade is already small within
# the shallow depth range where a_av (post neck_taper) survives, so only a
# considerably shorter decay actually separates from production there.
# ---------------------------------------------------------------------------
REGION_HARMONIZE_START_PX = NECK_TAPER_START_PX_DEFAULT   # same anchor as neck_taper
REGION_HARMONIZE_DEPTH_PX = 10.0                           # validated "Experiment B" decay


def region_weight_field(d_taper, start_px=REGION_HARMONIZE_START_PX,
                         depth_px=REGION_HARMONIZE_DEPTH_PX):
    """1.0 at/above the jaw (depth<=start_px), smoothstep down to 0.0 by
    depth=start_px+depth_px. Anchored to jaw_taper_depth_field (unclamped),
    the same field neck_taper's own alpha fade uses."""
    depth = np.clip(-d_taper, 0.0, None)
    _depth = max(float(depth_px), 1.0)
    tt = np.clip((depth - float(start_px)) / _depth, 0.0, 1.0)
    tt = tt * tt * (3.0 - 2.0 * tt)
    return (1.0 - tt).astype(np.float32)


def _gradient_guard(drv):
    gray = cv2.cvtColor(drv, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    G = np.sqrt(gx * gx + gy * gy)
    G = cv2.GaussianBlur(G, (0, 0), 2.0)
    return G


def _to_lab(img_u8):
    f = img_u8.astype(np.float32) / 255.0
    return cv2.cvtColor(f, cv2.COLOR_BGR2LAB)


def _lab_to_bgr_u8(lab):
    bgr = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR)
    return np.clip(bgr * 255.0, 0, 255)


# ---------------------------------------------------------------------------
# Hair neutral-grey recolor (Candidate H1, opt-in). Diagnostic investigation
# (task_hair_diag/partH_candidateG, partH_candidateH scratch scripts) found
# raw avatar hair pasted over the composite (G0/G1: dilated+feathered raw
# avatar RGB) flattened the crown silhouette (a square dilation kernel
# flattens a curved contour) and painted a black/off-colour rim (the feather
# band extends several px past a_av_final's true edge into un-premultiplied
# avatar pixels build_fullhead's own real compositing never shows). H1
# instead RECOLORS build_fullhead's own final `proto` pixels in place --
# never pastes raw avatar pixels, never dilates outward. Low-frequency LAB
# colour (LF_SIGMA-blurred, matching Variant F's own decomposition) is pulled
# toward a fixed neutral-grey target; high-frequency strand detail is kept
# exactly as build_fullhead already rendered it. The blend weight is
# build_fullhead's own real final alpha (a_av) times two 2px INWARD-only
# tapers (on the true alpha edge and on the hair-label boundary) -- it can
# never be nonzero where build_fullhead's own compositing doesn't already
# show real avatar hair, by construction.
#
# Applied entirely after build_r8()'s own core computation, reading only
# build_fullhead's already-computed proto/a_av/alab_canvas/support -- it
# cannot affect (and is intentionally not part of) _BUILD_R8_CORE_CACHE's
# cached path.
#
# Constants measured on test.mp4's avatar (partH1_raw_reference_look.py /
# partH4_candidates_H0_H1.py): TARGET_L=105.93 is the median raw avatar hair
# brightness over f0-244; target (a,b)=(+1.90,+2.90) is HALF the raw avatar's
# own taupe chroma (+3.80,+5.81) -- this is Candidate H1 (the user's chosen
# "neutral grey" variant; H0, the unscaled-chroma variant, is not used here).
# Unlike Candidate G's raw-pixel design, this low-frequency target is a true
# constant -- no per-frame gain is applied to it.
HAIR_RECOLOR_H1_DEFAULT = False
HAIR_RECOLOR_H1_TARGET_L = 105.93
HAIR_RECOLOR_H1_TARGET_A = 1.90
HAIR_RECOLOR_H1_TARGET_B = 2.90
HAIR_RECOLOR_H1_TAPER_PX = 2.0


HAIR_FACE_GATE_OPEN_PX = 11       # opening kernel that severs thin bridges between hair-label islands and real hair
HAIR_FACE_GATE_MIN_FRAC = 0.25    # keep opened components >= this fraction of the largest one
HAIR_FACE_GATE_RECON_PX = 7       # keep original hair pixels within this many px of a kept component


def _hair_face_gate(hair_mask):
    """Opt-in (hair_recolor_h1_face_gate): drop small isolated BiSeNet HAIR-label islands
    (brow / nose / lip patches the parser mislabels on the avatar face during head turns)
    so H1 does not grey them. Opens the mask to sever thin bridges, keeps components
    >= HAIR_FACE_GATE_MIN_FRAC of the largest, and restores original hair pixels within
    HAIR_FACE_GATE_RECON_PX of those (so the true hair edge is unchanged)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (HAIR_FACE_GATE_OPEN_PX, HAIR_FACE_GATE_OPEN_PX))
    op = cv2.morphologyEx(hair_mask.astype(np.uint8), cv2.MORPH_OPEN, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(op)
    if n <= 1:
        return hair_mask
    keep_ids = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= HAIR_FACE_GATE_MIN_FRAC * st[1:, cv2.CC_STAT_AREA].max()]
    kept = np.isin(lab, keep_ids).astype(np.uint8)
    r = HAIR_FACE_GATE_RECON_PX
    near = cv2.dilate(kept, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    return hair_mask & near


def _hair_recolor_h1(proto, a_av, alab_canvas, support, face_gate=False,
                      target_l=HAIR_RECOLOR_H1_TARGET_L,
                      target_a=HAIR_RECOLOR_H1_TARGET_A,
                      target_b=HAIR_RECOLOR_H1_TARGET_B,
                      taper_px=HAIR_RECOLOR_H1_TAPER_PX):
    """Candidate H1 hair recolor -- see the constants block above for the
    full derivation. hair_mask uses the SAME fixed definition as every other
    hair-labelled region in this module (support & alab_canvas in HAIR_SET).
    A 3x3 morphological closing is used ONLY as the input to the hair-
    boundary taper's distance transform (smooths small internal holes in the
    noisy per-pixel BiSeNet hair label so they don't locally depress the
    weight inside the hair interior); the actual compositing gate below still
    uses the ORIGINAL (unclosed) hair_mask, so this cannot grow the true
    outer silhouette boundary. Skin-labelled pixels are hard-zeroed
    regardless of weight."""
    hair_mask = support & np.isin(alab_canvas, HAIR_SET)
    if face_gate and hair_mask.any():
        hair_mask = _hair_face_gate(hair_mask)
    if not hair_mask.any():
        return proto

    hair_mask_closed = cv2.morphologyEx(hair_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                         np.ones((3, 3), np.uint8)) > 0
    dist_hair = cv2.distanceTransform(hair_mask_closed.astype(np.uint8), cv2.DIST_L2, 5)
    taper_hair = np.clip(dist_hair / max(taper_px, 1e-6), 0.0, 1.0).astype(np.float32)

    dist_edge = cv2.distanceTransform((a_av > 0.02).astype(np.uint8), cv2.DIST_L2, 5)
    taper_alpha = np.clip(dist_edge / max(taper_px, 1e-6), 0.0, 1.0).astype(np.float32)

    w = (a_av.astype(np.float32) * taper_alpha * taper_hair)
    w = np.where(hair_mask, w, 0.0).astype(np.float32)
    w = np.where(np.isin(alab_canvas, FACE_SKIN_SET), 0.0, w).astype(np.float32)

    if not (w > 1e-6).any():
        return proto

    # NOTE: intentionally NOT this module's own _to_lab/_lab_to_bgr_u8 helpers
    # (float32 convention, L in [0,100], a/b unshifted around 0) -- the
    # validated constants above (TARGET_L=105.93, chroma=(+1.90,+2.90)) were
    # measured and approved under OpenCV's uint8 LAB convention instead
    # (cv2.cvtColor called directly on a uint8 BGR array: L in [0,255], a/b in
    # [0,255] centred on 128), exactly as partH4_candidates_H0_H1.py's
    # build_variant() does it. Mixing the two conventions (this function's
    # first wired version) pushed L=105.93 out of the float32 convention's
    # [0,100] gamut and blew the recolored hair out to near-white -- see
    # task_hair_diag/wire_verify_h1 root-cause notes. This block reproduces
    # the original script's conversion path bit-for-bit.
    proto_lab = cv2.cvtColor(proto, cv2.COLOR_BGR2LAB).astype(np.float32)
    proto_low = _fast_lowfreq_blur(proto_lab, LF_SIGMA)
    target_low = proto_low.copy()
    target_low[..., 0] = target_l
    target_low[..., 1] = 128.0 + target_a
    target_low[..., 2] = 128.0 + target_b

    new_lab = proto_lab + w[..., None] * (target_low - proto_low)
    new_lab = np.clip(new_lab, 0, 255).astype(np.uint8)
    recolored = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)
    # Copy the original pixel through exactly wherever w==0, rather than
    # round-tripping it through BGR<->LAB (that round trip is not bit-exact
    # and would otherwise flag hundreds of thousands of untouched pixels as
    # "changed" -- see the scratch validation's changed_outside fix).
    mask3 = (w > 1e-6)[..., None]
    return np.where(mask3, recolored, proto.astype(np.float32)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Plate-vs-driver mismatch gate (opt-in, STAGED -- not in the production call
# path). Tasks 24-27 (project_task24_ghostline_root_cause through
# project_task27_gating_viability_split_verdict) traced the f36 ghost-line /
# f80 white-patch defects to static_plate() (a fixed 245-frame temporal
# median) disagreeing with the real driver background wherever We substitutes
# plate content for driver content. Task27 measured that "mismatch" (|plate -
# drv|) and "seam-hiding benefit" (how much the frame changes if plate is
# disabled, |P1-P0|) are nearly the SAME signal within legitimate plate use
# (class B, spearman rho 0.99+ at both f36 and f80) -- so a gate can only ever
# help where class A (the visible defect) and class B (legit use) actually
# separate in mismatch value. Task27 found they do NOT separate at f36
# (interleaved, same mismatch range -- gating this frame type is a dead end,
# do not enable this gate for that defect class) but DO separate cleanly at
# f80 (class A median mismatch 165 vs class B median 1, two orders of
# magnitude apart). This gate is scoped to that f80-type case only.
#
# gate(mismatch) is a smoothstep ramp from 1 (keep We as-is) to 0 (fully
# suppress plate substitution, fall back to raw driver) between
# [threshold - width/2, threshold + width/2]. Defaults (threshold=55,
# width=70 -> ramp spans mismatch in [20, 90]) are chosen directly from
# task27's measured f80 percentiles: class B's mismatch is almost entirely
# below 13 (5-95pct [0,13]) and class A's is almost entirely above 114
# (5-95pct [113.7,226.3]) -- the ramp sits entirely in the gap between the
# two populations so it neither touches class B's typical range nor leaves
# class A only partially suppressed. A smooth ramp (not a hard cutoff) is
# used because a hard step in We would itself be a new visible edge -- the
# same reasoning behind every other feathered field in this module
# (neck_taper, region_weight_field, w_smooth's guided filter).
#
# TASK 28 RESULT (project_task28_mismatch_gate_negative_leaktrough,
# v3/debug_out/3d_silhouette/task28_report.txt): implemented and tested,
# NEGATIVE. Suppressing We has no clean background to fall back to -- it
# falls back to raw `drv`, so every pixel this gate touches trades a
# plate-disagreement defect for the driver's real, un-swapped anatomy
# leaking through (confirmed visually at f80: a leaked ear-shaped blob
# replaces the white patch; at f36: a dark seam replaces the ghost arc).
# Also fires broadly at every frame tested, not narrowly at f80, touching
# thousands of legitimately-active plate pixels per frame. Do not enable
# without a real background-fill fallback (out of scope here) or a much
# more targeted trigger than frame-wide mismatch.
# ---------------------------------------------------------------------------
PLATE_MISMATCH_GATE_DEFAULT = False
PLATE_GATE_THRESHOLD_DEFAULT = 55.0   # gray-diff |plate_stat - drv|, ramp center
PLATE_GATE_WIDTH_DEFAULT = 70.0       # ramp spans [threshold-width/2, threshold+width/2]


def _plate_mismatch_gate_field(plate_stat_, drv, threshold, width):
    mismatch = np.abs(plate_stat_.astype(np.float32).mean(axis=2) - drv.astype(np.float32).mean(axis=2))
    low = threshold - width / 2.0
    high = threshold + width / 2.0
    t = np.clip((mismatch - low) / max(high - low, 1e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return (1.0 - smooth).astype(np.float32), mismatch


# ---------------------------------------------------------------------------
# Avatar/background sharpness matching (live-pipeline follow-up to the
# aspect-ratio/REG fixes, task_test2_bugfix_report.txt). Diagnosed in
# task_test2_artifact_diag.txt item 4: the avatar render is pin-sharp
# (GAGAvatar's Gaussian splats have no lens model) while a real camera's
# background carries its own optical blur (depth-of-field falloff, sensor
# noise floor, compression) -- the avatar visibly looks "pasted on top of"
# rather than "filmed within" the scene. This is an avatar-APPEARANCE-only
# adjustment (a softening pass on the avatar's own RGB before it is alpha-
# composited) -- it never touches static_plate(), We/erase geometry, or any
# alpha value, so it cannot reopen the a_av==0 coverage gap or interact with
# the erase-candidate mask the way REG/reg does.
#
# OFF by default: only meaningful for a real live camera feed with genuine
# optical blur; the offline production clip's own camera and every existing
# validated render are unaffected unless this is explicitly turned on, and
# no offline call site passes it.
AVATAR_BLUR_MATCH_DEFAULT = False
AVATAR_BLUR_MAX_SIGMA = 2.5     # hard cap: never soften the avatar more than
                                 # this even if the background is very blurry --
                                 # avoids an obviously-mushy face if the estimate
                                 # is thrown off by a low-texture/low-light frame.
AVATAR_BLUR_BISECT_STEPS = 12    # ~1/1000px sigma precision on the search below
AVATAR_BLUR_MARGIN_PX = 60      # background sample = a ring this many px around
                                 # the avatar's own silhouette (local depth/focus,
                                 # not some unrelated part of the frame)
AVATAR_BLUR_MIN_AVATAR_AREA_PX = 400  # skip (sigma=0) if the visible avatar
                                        # silhouette this frame is too small to
                                        # measure a stable Laplacian variance on


def _estimate_avatar_blur_sigma(plated_bg, av_rgb, support_mask,
                                 max_sigma=AVATAR_BLUR_MAX_SIGMA,
                                 bisect_steps=AVATAR_BLUR_BISECT_STEPS,
                                 margin_px=AVATAR_BLUR_MARGIN_PX,
                                 min_avatar_area_px=AVATAR_BLUR_MIN_AVATAR_AREA_PX):
    """Per-frame Laplacian-variance blur match: measures how much blurrier the
    real background (`plated_bg` -- what's actually visible behind the avatar,
    i.e. drv/plate_stat already combined by We) is than the avatar's own sharp
    render (`av_rgb`), then returns the Gaussian sigma that would bring the
    avatar down to roughly the same sharpness. Returns 0.0 (no softening) if
    the avatar silhouette is too small to measure, or if the background is
    already as sharp as (or sharper than) the avatar -- e.g. a high-quality
    webcam with a sharp background should never get artificial blur added.

    Both regions are cropped to a local bounding box around the avatar's own
    silhouette (not sampled from some unrelated corner of the frame) so the
    comparison reflects this subject's actual local depth-of-field, and both
    crops are small (a few hundred px per side at most), so this is cheap:
    one Laplacian + a handful of small Gaussian-blur/Laplacian passes for the
    search below, not a full-frame operation.
    """
    if not support_mask.any():
        return 0.0
    ys, xs = np.where(support_mask)
    ay0, ay1, ax0, ax1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if (ay1 - ay0) * (ax1 - ax0) < min_avatar_area_px:
        return 0.0
    av_gray = cv2.cvtColor(np.clip(av_rgb[ay0:ay1, ax0:ax1], 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    av_var = float(cv2.Laplacian(av_gray, cv2.CV_64F).var())
    if av_var <= 0.0:
        return 0.0

    m8 = support_mask.astype(np.uint8)
    outer = cv2.dilate(m8, np.ones((margin_px, margin_px), np.uint8)) > 0
    bys, bxs = np.where(outer)
    by0, by1, bx0, bx1 = bys.min(), bys.max() + 1, bxs.min(), bxs.max() + 1
    bg_gray = cv2.cvtColor(np.clip(plated_bg[by0:by1, bx0:bx1], 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    bg_var = float(cv2.Laplacian(bg_gray, cv2.CV_64F).var())

    if bg_var >= av_var:
        return 0.0  # background already at least as sharp as the avatar

    # Bisection, not a linear step search: Laplacian variance falls off very
    # steeply over the first ~1px of Gaussian sigma (confirmed empirically --
    # a fixed 0.25px step search overshot to the same 0.75px answer regardless
    # of how blurry the background actually was, landing the avatar BLURRIER
    # than its target background on every test frame). ~12 bisection steps on
    # a small crop is still cheap and converges to within ~1/1000px of the
    # sigma that actually matches bg_var, instead of coarsely overshooting it.
    lo, hi = 0.0, max_sigma
    hi_var = float(cv2.Laplacian(cv2.GaussianBlur(av_gray, (0, 0), hi), cv2.CV_64F).var())
    if hi_var > bg_var:
        return max_sigma  # even the max softening isn't enough -- clamp
    for _ in range(bisect_steps):
        mid = (lo + hi) / 2.0
        mid_var = float(cv2.Laplacian(cv2.GaussianBlur(av_gray, (0, 0), mid), cv2.CV_64F).var())
        if mid_var > bg_var:
            lo = mid
        else:
            hi = mid
    return round(hi, 3)


# ---------------------------------------------------------------------------
# Bare-neck chroma-only harmonization (Priority 4, 2026-09-28 demo2 critique
# follow-up; re-requested 2026-10-01 as a driver<->avatar colour-matching /
# white-balance step). NOTE: Variant F above ALREADY performs a colour-
# matching/white-balance transfer -- see `avatar_matched_low` a few lines up
# in this same function: a per-frame LAB mean/std whitening transform mapping
# the avatar's ENTIRE rendered head onto the driver's face stats (smoothed
# across the whole clip by precompute_stats). That is real and already in
# production (demo2 uses it with the improved avatar_stats_mask="faceskin").
# What it does NOT correct is the frame-local mismatch right at the jaw-to-
# neck transition: the avatar's rendered bare-neck/collar chroma vs. the
# driver's REAL bare-neck chroma (project_demo2_critique_p3p4_reconfirmed:
# L* delta 8-16 at rest, up to 39 at under-chin-shadow outlier frames --
# naively matching L* there reproduces the already-rejected neck-shadow-
# contamination failure, project_color_lighting_closed). This mechanism
# matches ONLY a*/b* (chroma), leaving L* untouched, feathered by a cosine
# falloff from the jaw line -- ported verbatim from
# task_demo2_neck_color.py's validated `CAND["color"]`/"FIX" preset (see that
# file + project_demo2_critique_p3p4_reconfirmed.md for the original
# build+validation: mean|dL*|=0.057-0.066, max|dL*|=1-2 across f0/175/260/
# 350/500 incl. the f500 shadow outlier -- confirms genuinely chroma-only).
#
# Needs a CAUSAL EMA of (driver bare-neck, avatar near-jaw) a*/b* stats across
# frames (alpha=0.15) -- same "must be called in-order, state carried
# forward" requirement as RVM's recurrent state / the motion-adaptive filter
# elsewhere in this module. State lives in a plain dict the caller owns
# (default: the module-level _BARE_NECK_EMA_STATE, reset via
# reset_bare_neck_chroma_state()) so a harness replaying frames out of order
# can warm it up explicitly, exactly as task_demo2_neck_color.py's own
# warm() helper already did.
# ---------------------------------------------------------------------------
BARE_NECK_CHROMA_MATCH_DEFAULT = False
BARE_NECK_CHROMA_EMA_ALPHA = 0.15
BARE_NECK_CHROMA_RISE_PX = 180.0
BARE_NECK_CHROMA_GAIN = (1.0, 0.6)     # (a*, b*) blend gain -- validated "FIX" preset
BARE_NECK_CHROMA_OFFSET = (2.5, 0.0)   # (a*, b*) additive offset -- validated "FIX" preset
BARE_NECK_CHROMA_BEARD_B = 5.0         # extra b* (yellow) boost in low-chroma (beard-shadow) zone
BARE_NECK_CHROMA_BEARD_A = 3.0         # extra a* (red) boost, same zone
BARE_NECK_CHROMA_MIN_PIXELS = 300      # minimum bare-neck/near-jaw sample size to trust this frame's stats

_BARE_NECK_EMA_STATE = {}


def reset_bare_neck_chroma_state(state=None):
    """Clears the causal EMA state -- call before a fresh in-order render
    pass (e.g. switching clips) so stats don't leak across them."""
    (state if state is not None else _BARE_NECK_EMA_STATE).clear()


def _bare_neck_chroma_match(av_rgb_canvas, drv, a_av, support, sl, jaw_off, f,
                             state=None, ema_alpha=BARE_NECK_CHROMA_EMA_ALPHA,
                             rise_px=BARE_NECK_CHROMA_RISE_PX, gain=BARE_NECK_CHROMA_GAIN,
                             offset=BARE_NECK_CHROMA_OFFSET, beard_b=BARE_NECK_CHROMA_BEARD_B,
                             beard_a=BARE_NECK_CHROMA_BEARD_A, min_px=BARE_NECK_CHROMA_MIN_PIXELS):
    """See the constants block above for the mechanism writeup. Returns a
    (possibly) recoloured av_rgb_canvas; identity wherever the cosine-falloff
    weight is ~0, so the LAB round-trip's non-bit-exactness doesn't flag
    untouched pixels as changed (same guard _hair_recolor_h1 uses)."""
    if state is None:
        state = _BARE_NECK_EMA_STATE
    dt = jaw_taper_depth_field(jaw_off).astype(np.float32)   # >0 above jaw, <0 below
    hair = cv2.GaussianBlur(np.isin(sl, (HAIR, HAT)).astype(np.float32), (0, 0), 3)
    chin_y = int(jaw_off[:, 1].max())
    yy = np.arange(H)[:, None]
    bare = (np.isin(sl, (NECK, NECK_L)) | ((sl == SKIN) & (yy > chin_y))) & (a_av < 0.05) & (dt < -10) & (dt > -160)
    near = (a_av > 0.9) & support & (dt >= 0) & (dt < rise_px) & (hair < 0.1)
    lab = _to_lab(np.clip(av_rgb_canvas, 0, 255).astype(np.uint8)).astype(np.float32)

    st = state.setdefault("stats", {})
    done = state.setdefault("done", set())
    if f not in done:
        done.add(f)
        if bare.sum() > min_px and near.sum() > min_px:
            dl = _to_lab(drv)[bare].reshape(-1, 3)
            al = lab[near].reshape(-1, 3)
            raw = dict(dm=dl[:, 1:].mean(0), ds=dl[:, 1:].std(0), am=al[:, 1:].mean(0), as_=al[:, 1:].std(0))
            for k, v in raw.items():
                st[k] = v if k not in st else (1 - ema_alpha) * st[k] + ema_alpha * v
    if not st:
        return av_rgb_canvas

    tt = np.clip(dt / rise_px, 0, 1)
    wgt = (0.5 * (1 + np.cos(np.pi * tt))) * (dt >= 0) * (1 - hair) * support.astype(np.float32)
    wgt = cv2.GaussianBlur(wgt.astype(np.float32), (0, 0), 4)
    if not (wgt > 1e-6).any():
        return av_rgb_canvas

    lab2 = lab.copy()
    for i, c in enumerate((1, 2)):
        r = np.clip(st["ds"][i] / (st["as_"][i] + 1e-3), 0.8, 1.3)
        tgt_c = st["dm"][i] + (lab[..., c] - st["am"][i]) * r + offset[i]
        lab2[..., c] = lab[..., c] * (1 - wgt * gain[i]) + tgt_c * (wgt * gain[i])

    lowc = np.clip(1 - np.hypot(lab[..., 1], lab[..., 2]) / 16.0, 0, 1)
    lab2[..., 2] += beard_b * lowc * wgt
    lab2[..., 1] += beard_a * lowc * wgt

    av_new = _lab_to_bgr_u8(lab2).astype(np.float32)   # L* (channel 0) untouched -- chroma-only by construction
    mask3 = (wgt > 1e-6)[..., None]
    return np.where(mask3, av_new, av_rgb_canvas.astype(np.float32)).astype(np.uint8)


def _build_fullhead_impl(f, parser, M, plate_stat, lm, overhang_px,
                    av_head_mean_s, av_head_std_s, dr_face_mean_s, dr_face_std_s,
                    gamma=GAMMA, gf_radius=GF_RADIUS, gf_eps=GF_EPS,
                    region_harmonize_start_px=REGION_HARMONIZE_START_PX,
                    region_harmonize_depth_px=REGION_HARMONIZE_DEPTH_PX,
                    seg2_pushpull_fill=SEG2_PUSHPULL_FILL_DEFAULT,
                    edge_rgb_dilate=EDGE_RGB_DILATE_DEFAULT,
                    plate_mismatch_gate=PLATE_MISMATCH_GATE_DEFAULT,
                    plate_gate_threshold=PLATE_GATE_THRESHOLD_DEFAULT,
                    plate_gate_width=PLATE_GATE_WIDTH_DEFAULT,
                    avatar_blur_match=AVATAR_BLUR_MATCH_DEFAULT,
                    contact_shadow=CONTACT_SHADOW_DEFAULT,
                    contact_shadow_start_px=CONTACT_SHADOW_START_PX_DEFAULT,
                    contact_shadow_depth_px=CONTACT_SHADOW_DEPTH_PX_DEFAULT,
                    contact_shadow_strength=CONTACT_SHADOW_STRENGTH_DEFAULT,
                    driver_leak_plate=DRIVER_LEAK_PLATE_DEFAULT,
                    bare_neck_chroma_match=BARE_NECK_CHROMA_MATCH_DEFAULT,
                    hair_recolor_h1=HAIR_RECOLOR_H1_DEFAULT, hair_recolor_h1_face_gate=False, **kwargs):
    """"Variant F" full-head LAB harmonization, layered on top of build_r8's whole-head
    erasure output. Ported verbatim from _track5_fullhead_lib.build_fullhead -- the
    uniform LAB whitening-transform correction applies to the avatar's ENTIRE rendered
    head+neck footprint (no spatial gating), while the seam-side blend mechanism
    (w_smooth gradient-guard + guided filter, L*-high-frequency lock) is the SAME
    mechanism Variant C/C.2 used, reused verbatim on top of the harmonized avatar.
    `av_head_*_s` / `dr_face_*_s` are this frame's SMOOTHED per-channel LAB mean/std
    (from `precompute_stats`'s Phase-1 cache), not raw per-frame values.

    w_smooth is additionally scoped by region_weight_field (collar/shoulder seam
    fix, project_collar_seam_diag / project_collar_seam_regionfix_verified) before
    use in both the low-freq LAB blend and the high-freq L* blend -- see that
    function's docstring and the constants above for the validation chain.

    hair_recolor_h1=True applies Candidate H1 (see that constants block and
    _hair_recolor_h1's docstring above) to this function's own final `proto`:
    recolors hair-labelled pixels toward a fixed neutral-grey LAB target
    (TARGET_L=105.93, target a,b=(+1.90,+2.90)) while keeping build_fullhead's
    own strand detail, weighted by this function's own a_av with 2px inward
    tapers on the alpha edge and hair-label boundary -- no outward dilation,
    no raw avatar pixels pasted in. Default False: byte-identical output when
    omitted. Applied after build_r8()'s own core computation on already-
    computed proto/a_av/alab_canvas/support, so it is NOT part of
    _BUILD_R8_CORE_CACHE's cache key (it cannot affect that cached path)."""
    d = build_r8(f, parser, M, plate_stat, lm, av_edge_fix=False, **kwargs)
    drv = d["drv"]; support = d["support"]; d_signed = d["d_signed"]
    av_rgb_real = d["av_rgb_real"]; a_av = d["a_av"]; We = d["We"]; plate_stat_ = d["plate_stat"]
    jaw_off = d["jaw_off"]; sl = d["sl"]

    denom = max(float(overhang_px), OVERHANG_EPS)
    w = np.clip((-d_signed) / denom, 0.0, 1.0).astype(np.float32)
    w *= support.astype(np.float32)
    w *= (d_signed < 0).astype(np.float32)

    G = _gradient_guard(drv)
    guard = np.exp(-gamma * G).astype(np.float32)
    w_guided = w * guard
    guide = drv if drv.dtype == np.uint8 else np.clip(drv, 0, 255).astype(np.uint8)
    w_smooth = cv2.ximgproc.guidedFilter(guide=guide, src=w_guided, radius=gf_radius, eps=gf_eps)
    w_smooth = np.clip(w_smooth, 0.0, 1.0).astype(np.float32)

    d_taper = jaw_taper_depth_field(jaw_off)
    region_weight = region_weight_field(d_taper, region_harmonize_start_px, region_harmonize_depth_px)
    w_eff = (w_smooth * region_weight).astype(np.float32)

    avatar_lab = _to_lab(np.clip(av_rgb_real, 0, 255).astype(np.uint8))
    driver_lab = _to_lab(drv)
    avatar_low = _fast_lowfreq_blur(avatar_lab, LF_SIGMA)
    driver_low = _fast_lowfreq_blur(driver_lab, LF_SIGMA)
    avatar_high = avatar_lab[..., 0] - avatar_low[..., 0]
    L_high_driver = driver_lab[..., 0] - driver_low[..., 0]

    avatar_matched_low = np.empty_like(avatar_low)
    for c in range(3):
        avatar_matched_low[..., c] = dr_face_mean_s[c] + dr_face_std_s[c] * (
            (avatar_low[..., c] - av_head_mean_s[c]) / (av_head_std_s[c] + EPS_STD)
        )

    we3 = w_eff[..., None]
    low_blend = avatar_matched_low * (1.0 - we3) + driver_low * we3
    L_high = avatar_high * (1.0 - w_eff) + L_high_driver * w_eff
    L_final = np.clip(low_blend[..., 0] + L_high, 0, 100)
    recon_lab = np.stack([L_final, low_blend[..., 1], low_blend[..., 2]], axis=-1)
    recon_rgb = _lab_to_bgr_u8(recon_lab)

    av_rgb_canvas = np.where(support[..., None], recon_rgb, av_rgb_real)
    pad = cv2.dilate(av_rgb_canvas, np.ones((AV_RGB_PAD, AV_RGB_PAD), np.uint8))
    av_rgb_canvas = np.where(support[..., None], av_rgb_canvas, pad)

    if seg2_pushpull_fill:
        # Applied HERE (build_fullhead's own av_rgb_canvas/a_av), not inside
        # build_r8 -- build_fullhead discards build_r8's av_rgb_canvas
        # entirely (recon_rgb/av_rgb_real above is a fresh reconstruction),
        # so filling only at the build_r8 level would leave a_av flipped to
        # 1.0 with no corresponding RGB fill, reintroducing the dark-smudge
        # defect this is meant to fix. `d["a_av"]`/`We` above are still the
        # pre-fill values (this function's own seg2_pushpull_fill kwarg is
        # NOT forwarded into build_r8's **kwargs -- see the two separate
        # named parameters), so gap detection here is still valid.
        av_rgb_canvas, a_av = _seg2_pushpull_fill(av_rgb_canvas, a_av, We)

    if edge_rgb_dilate:
        # Applied HERE (build_fullhead's own reconstructed av_rgb_canvas),
        # same reasoning as seg2_pushpull_fill above -- build_r8's own
        # edge_rgb_dilate kwarg is a separate, unforwarded parameter (see
        # build_fullhead's **kwargs split), so this operates on the
        # LAB-harmonized recon_rgb, not build_r8's raw av_rgb_real.
        av_rgb_canvas = _edge_rgb_dilate(av_rgb_canvas, a_av, d["alpha01"], d["hair"], support)

    if bare_neck_chroma_match:
        # Same reasoning as seg2_pushpull_fill/edge_rgb_dilate above: operates
        # on build_fullhead's own LAB-harmonized av_rgb_canvas, not build_r8's
        # raw av_rgb_real -- this kwarg is intentionally NOT forwarded into
        # build_r8's **kwargs.
        av_rgb_canvas = _bare_neck_chroma_match(av_rgb_canvas, drv, a_av, support, sl, jaw_off, f)

    We_eff = We
    plate_gate = None
    plate_mismatch = None
    if plate_mismatch_gate:
        plate_gate, plate_mismatch = _plate_mismatch_gate_field(
            plate_stat_, drv, plate_gate_threshold, plate_gate_width)
        We_eff = We * plate_gate

    if driver_leak_plate is not None:
        # s(f): 0 outside the window (byte-identical), ramps 0->1 over the
        # first 4 frames entering, 1.0 fully inside, ramps 1->0 over the last
        # 4 frames leaving -- same trick NECK_TAPER_EAR_PROTECT_DEFAULT uses
        # (1 - _efc_ramp_strength), just inverted to "off outside" instead of
        # "on outside" since this fix is opt-in per-window, not always-on.
        s = 1.0 - _efc_ramp_strength(f, driver_leak_plate)
        if s > 0.0:
            strip_mask = driver_leak_plate_mask(f, jaw_off, drv, a_av, sl, We_eff, plate_stat_)
            if strip_mask.any():
                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * DRIVER_LEAK_PLATE_DILATE_PX + 1, 2 * DRIVER_LEAK_PLATE_DILATE_PX + 1))
                mask_d = cv2.dilate(strip_mask.astype(np.uint8), k)
                mask_soft = np.clip(
                    cv2.GaussianBlur(mask_d.astype(np.float32), (0, 0), DRIVER_LEAK_PLATE_FEATHER_SIGMA),
                    0.0, 1.0) * s
                # Nudges We_eff toward 1.0 (full plate) only inside the soft
                # mask; identity (We_eff unchanged) wherever mask_soft==0.
                # Never touches a_av/av_rgb_canvas, so real avatar coverage
                # (Aa) is unaffected -- this can only change which of
                # {drv, plate_stat} shows through where the avatar already
                # contributes nothing.
                We_eff = (We_eff + mask_soft * (1.0 - We_eff)).astype(np.float32)

    Wc = We_eff[..., None]; Aa = a_av[..., None]
    plated = drv.astype(np.float32) * (1.0 - Wc) + plate_stat_ * Wc

    avatar_blur_sigma_applied = 0.0
    if avatar_blur_match:
        avatar_blur_sigma_applied = _estimate_avatar_blur_sigma(plated, av_rgb_canvas, support)
        if avatar_blur_sigma_applied > 0.0:
            av_rgb_canvas = cv2.GaussianBlur(av_rgb_canvas, (0, 0), avatar_blur_sigma_applied)

    if contact_shadow:
        # Same mechanism as build_r8's own contact_shadow (see CONTACT_SHADOW_*
        # constants block) -- duplicated here rather than reused from build_r8's
        # `d` because build_fullhead discards d["proto"]/d["plated"] entirely and
        # recomputes its own `plated` above (Variant F's plate_mismatch_gate-scaled
        # We_eff, not build_r8's We). Darkens only this `plated`, never
        # av_rgb_canvas/a_av.
        d_cs = jaw_taper_depth_field(jaw_off)
        depth_below = np.clip(-d_cs, 0.0, None)
        into_shadow = np.clip(depth_below - float(contact_shadow_start_px), 0.0, None)
        _depth_cs = max(float(contact_shadow_depth_px), 1.0)
        falloff = np.clip(1.0 - into_shadow / _depth_cs, 0.0, 1.0)
        falloff = falloff * falloff
        x_gate = cv2.dilate(support.astype(np.uint8),
                             np.ones((2 * CONTACT_SHADOW_X_PAD_PX + 1, 2 * CONTACT_SHADOW_X_PAD_PX + 1),
                                     np.uint8)) > 0
        shadow = (float(contact_shadow_strength) * falloff * x_gate.astype(np.float32)).astype(np.float32)
        plated = plated * (1.0 - shadow[..., None])

    proto = np.clip(plated * (1.0 - Aa) + av_rgb_canvas * Aa, 0, 255).astype(np.uint8)

    out = dict(d)
    out["proto"] = proto; out["av_rgb_canvas"] = av_rgb_canvas
    out["w_edge"] = w; out["w_guided"] = w_guided; out["w_smooth"] = w_smooth
    out["region_weight"] = region_weight; out["w_eff"] = w_eff
    out["grad_mag"] = G; out["overhang_used"] = float(overhang_px)
    out["We_eff"] = We_eff; out["plate_gate"] = plate_gate; out["plate_mismatch"] = plate_mismatch
    out["avatar_blur_sigma"] = avatar_blur_sigma_applied; out["plated"] = plated
    if hair_recolor_h1:
        out["proto"] = _hair_recolor_h1(out["proto"], out["a_av"], out["alab_canvas"], out["support"],
                                        face_gate=hair_recolor_h1_face_gate)
    return out


def build_fullhead(f, *args, hinge_leak_fix=HINGE_LEAK_FIX_DEFAULT, **kwargs):
    """hinge_leak_fix (default None = OFF, byte-identical): see the HINGE_LEAK_FIX_*
    block. Inside its frame window, renders the baseline first (same args), derives
    the Step-25 target T from it, and re-renders with T injected into erase_bin /
    seam_ramp via the private hinge_T kwarg. Outside the window, or if T is empty,
    the baseline result is returned as-is. All other args: see _build_fullhead_impl."""
    if hinge_leak_fix is None or kwargs.get("hinge_T") is not None:
        return _build_fullhead_impl(f, *args, **kwargs)
    w0, w1 = hinge_leak_fix["window"]
    if not (w0 <= f <= w1):
        return _build_fullhead_impl(f, *args, **kwargs)
    b = _build_fullhead_impl(f, *args, **kwargs)
    T = _hinge_leak_target(b, hinge_leak_fix)
    if not T.any():
        return b
    return _build_fullhead_impl(f, *args, hinge_T=T, **kwargs)


# ---------------------------------------------------------------------------
# ---- Phase 1: batch precompute (new -- see module docstring) ----
# ---- ported from _track4_variantA_overhang.py / _track5_fullhead_colorstats.py ----
# ---------------------------------------------------------------------------
def _rolling_median_smooth(x, window):
    n = len(x)
    half = window // 2
    xp = np.pad(x, (half, half), mode="edge")
    out = np.empty_like(x, dtype=np.float64)
    for i in range(n):
        out[i] = np.median(xp[i:i + window])
    return out


def precompute_stats(parser, M, plate_stat, lm, n_frames, force=False, verbose=True,
                     stats_cache=None, build_kwargs=None, reg=REG,
                     avatar_stats_mask="wholehead"):
    """Phase 1 (serial, one build_r8 call per frame -- GPU-bound on the BiSeNet parse):
    computes and caches to STATS_CACHE_NPZ the two arrays Variant F needs that are only
    meaningful smoothed across the WHOLE clip:
      * overhang_smooth[f]  -- centered rolling-median (window=5), from
        _track4_variantA_overhang.py's exact algorithm (max(-d_signed) over
        support & (d_signed<0) & (a_av>=0.95), per frame, raw then smoothed).
      * av_mean_s/av_std_s/dr_mean_s/dr_std_s[f, c]  -- per-frame LAB mean/std of the
        avatar's full rendered head (support & alpha01>=0.5, falling back to `support`
        alone if that's <200px) and the driver's face+neck skin (faceskin|NECK|NECK_L
        within REG, falling back to faceskin&REG if <200px, then REG alone if <50px),

    avatar_stats_mask: "wholehead" (default, validated/shipped behavior, byte-identical
        to pre-existing callers) or "faceskin" (opt-in candidate, NOT shipped/committed):
        additionally restricts the avatar "from" mask to face-skin-only pixels (via the
        avatar's own BiSeNet parse, core["alab_canvas"], intersected with FACE_SKIN_SET),
        falling back to the wholehead mask if that intersection is <200px. Rationale:
        the avatar "from" mask mixes hair/ears/skin while the driver "to" target
        (faceskin|NECK|NECK_L) is skin-only -- this asymmetry biases the per-frame LAB
        whitening transform away from the true skin-color mapping whenever the avatar's
        hair/ear color differs from its own skin (which it always does). Measured on
        test2.mp4 (3 frames, live-style real-camera proxy): closes 37-72% of the
        local-forehead ΔE gap that whole-head-only harmonization leaves behind under
        non-studio lighting, without touching the already-good whole-head-average ΔE.
        Default stays "wholehead" (byte-identical, unchanged for all other callers);
        "faceskin" is committed to the test.mp4 offline-demo recipe specifically
        (v3/render_demo_final.py), not to production as a whole.
        from _track5_fullhead_colorstats.py's exact algorithm, then smoothed with the
        SAME data-driven window-width decision procedure that script used: start at
        window=5, widen to window=9 only if the raw single-frame jump statistics show
        window=5 doesn't sufficiently damp full-head/full-face sample noise (see that
        script's own gate: max raw jump <=8.0 on 8 tracked series, AND the worst-jump
        channel's smoothed max-jump < 0.6x its raw max-jump). NOTE: on this exact clip
        the diagnostic run recorded in fix3_verify/EDGE_CLIP_FIX_evidence/
        track5_fullhead/final_window.txt landed on window=9 (window=5 was NOT enough --
        driver-L-std's raw max single-frame jump of 1.48 barely smoothed to 1.47 at
        window=5); this function re-derives that decision from the data rather than
        hardcoding either 5 or 9, but production numbers should match window=9 exactly
        if the clip/inputs are unchanged (see the porting report).

    Only ONE build_r8() call per frame is needed for both arrays (support/alpha01/
    d_signed/av_rgb_real/drv/sl/faceskin all come from the same dict) -- twice as
    cheap as the two separate diagnostic scripts that measured them independently.
    """
    cache_npz = Path(stats_cache) if stats_cache is not None else STATS_CACHE_NPZ
    bkw = dict(build_kwargs or {})   # extra build_r8 kwargs (tall canvas: avatar_frames_dir, render_canvas_h)
    bkw.setdefault("reg", reg)       # keep the internal build_r8 call's erase geometry in sync with REG_MASK below
    ch = int(bkw.get("render_canvas_h", RENDER_CANVAS_H_DEFAULT))  # a 512 vs 768 render gives different stats
    reg_key = np.asarray(reg)
    if cache_npz.exists() and not force:
        z = np.load(str(cache_npz))
        cached_ch = int(z["render_canvas_h"]) if "render_canvas_h" in z else None
        cached_reg = z["reg"] if "reg" in z else np.asarray(REG)  # older caches predate the reg param -> assume default
        # older caches predate avatar_stats_mask -> assume "wholehead" (the only mode that existed then)
        cached_avmask = str(z["avatar_stats_mask"]) if "avatar_stats_mask" in z else "wholehead"
        if (int(z["n_frames"]) == n_frames and cached_ch == ch and np.array_equal(cached_reg, reg_key)
                and cached_avmask == avatar_stats_mask):
            if verbose:
                print(f"[precompute_stats] cache hit: {cache_npz} (window={int(z['final_window'])})")
            return (z["overhang_smooth"], z["av_mean_s"], z["av_std_s"],
                    z["dr_mean_s"], z["dr_std_s"])
        if verbose:
            print(f"[precompute_stats] cache at {cache_npz} has n_frames="
                  f"{int(z['n_frames'])}/canvas_h {cached_ch}/reg {cached_reg.tolist()}/"
                  f"avatar_stats_mask {cached_avmask} != "
                  f"{n_frames}/{ch}/{list(reg)}/{avatar_stats_mask}, recomputing")

    REG_MASK = np.zeros((H, W), bool)
    rx0, ry0, rx1, ry1 = reg
    REG_MASK[ry0:ry1, rx0:rx1] = True

    overhang_raw = np.zeros(n_frames, dtype=np.float32)
    av_mean = np.zeros((n_frames, 3)); av_std = np.zeros((n_frames, 3))
    dr_mean = np.zeros((n_frames, 3)); dr_std = np.zeros((n_frames, 3))

    # overhang_raw measures how far the avatar's OPAQUE skirt reaches past the
    # jaw -- a raw-geometry input to Variant F's below-jaw blend ramp (`denom`
    # in build_fullhead). It must be read off the UN-tapered alpha: neck_taper
    # is a Phase-2 cosmetic gate, and leaving it on here collapses `a_av>=0.95`
    # to a thin ~start-px band, which then steepens that ramp in Phase 2. The
    # validated d35 pipeline measured overhang un-tapered; keep it that way.
    stat_bkw = {k: v for k, v in bkw.items()
                if not k.startswith("neck_taper")}
    stat_bkw["neck_taper"] = False
    # seg2_pushpull_fill flips a_av 0->1 in the gap, which would leak into
    # overhang_raw's `a_av>=0.95` mask and falsely extend the measured
    # opaque-skirt depth -- keep stats computed on the un-filled signal,
    # exactly like neck_taper above.
    stat_bkw["seg2_pushpull_fill"] = False
    # edge_rgb_dilate only touches RGB (never a_av) but still changes fringe
    # pixel color within the alpha01>=0.5 stats mask -- keep stats computed
    # on the un-dilated signal too, same conservative reasoning as above.
    stat_bkw["edge_rgb_dilate"] = False
    # rvm_hair_blend only ever shrinks a_av in the hair-edge ring (never grows
    # it), but keep stats on the un-shrunk signal too -- same conservative
    # reasoning as edge_rgb_dilate/seg2_pushpull_fill above, and it changes
    # a_av (unlike edge_rgb_dilate) so it directly could affect overhang_raw's
    # a_av>=0.95 mask if the ring ever reaches that far.
    stat_bkw["rvm_hair_blend"] = False
    # driver_leak_plate / hair_recolor_h1 are build_fullhead-only kwargs; build_r8
    # rejects them (TypeError on a stats-cache miss when a caller forwards the
    # whole DEMO_RECIPE). They do not affect the stats, so drop them here only.
    for _k in ("driver_leak_plate", "hair_recolor_h1", "hinge_leak_fix",
               "chin_shirt_taper", "hair_recolor_h1_face_gate", "bare_neck_chroma_match",
               "wall_sliver_clean"):
        stat_bkw.pop(_k, None)

    for f in range(n_frames):
        d = build_r8(f, parser, M, plate_stat, lm, av_edge_fix=False, ear_boost=True,
                     crop_edge_fix=True, **stat_bkw)
        a_av = d["a_av"]; d_signed = d["d_signed"]; support = d["support"]
        drv = d["drv"]; av_rgb_real = d["av_rgb_real"]; alpha01 = d["alpha01"]
        sl = d["sl"]; faceskin = d["faceskin"]

        ov_mask = support & (d_signed < 0) & (a_av >= 0.95)
        overhang_raw[f] = float(-d_signed[ov_mask].min()) if ov_mask.any() else 0.0

        avatar_mask = support & (alpha01 >= 0.5)
        if avatar_mask.sum() < 200:
            avatar_mask = support
        if avatar_stats_mask == "faceskin":
            faceskin_only = avatar_mask & np.isin(d["alab_canvas"], FACE_SKIN_SET)
            if faceskin_only.sum() >= 200:
                avatar_mask = faceskin_only
        driver_mask = (faceskin | (sl == NECK) | (sl == NECK_L)) & REG_MASK
        if driver_mask.sum() < 200:
            driver_mask = faceskin & REG_MASK
            if driver_mask.sum() < 50:
                driver_mask = REG_MASK

        avatar_lab = _to_lab(np.clip(av_rgb_real, 0, 255).astype(np.uint8))
        driver_lab = _to_lab(drv)
        avatar_low = _fast_lowfreq_blur(avatar_lab, LF_SIGMA)
        driver_low = _fast_lowfreq_blur(driver_lab, LF_SIGMA)
        for c in range(3):
            av_mean[f, c] = float(avatar_low[..., c][avatar_mask].mean())
            av_std[f, c] = float(avatar_low[..., c][avatar_mask].std()) + EPS_STD
            dr_mean[f, c] = float(driver_low[..., c][driver_mask].mean())
            dr_std[f, c] = float(driver_low[..., c][driver_mask].std()) + EPS_STD

        if verbose and f % 40 == 0:
            print(f"  [precompute_stats] f{f}/{n_frames} overhang_raw={overhang_raw[f]:.1f}px")

    overhang_smooth = _rolling_median_smooth(overhang_raw, 5).astype(np.float32)

    av_mean_s5 = np.zeros_like(av_mean); av_std_s5 = np.zeros_like(av_std)
    dr_mean_s5 = np.zeros_like(dr_mean); dr_std_s5 = np.zeros_like(dr_std)
    for c in range(3):
        av_mean_s5[:, c] = _rolling_median_smooth(av_mean[:, c], 5)
        av_std_s5[:, c] = _rolling_median_smooth(av_std[:, c], 5)
        dr_mean_s5[:, c] = _rolling_median_smooth(dr_mean[:, c], 5)
        dr_std_s5[:, c] = _rolling_median_smooth(dr_std[:, c], 5)

    # n_frames==1 (the live snapshot tier's only case) makes every np.diff()
    # below a zero-size array -- there's no adjacent frame to jump against, so
    # skip the jump-detection gate entirely rather than crash on .max() of
    # empty; window=5 is a no-op smoothing choice for a single frame anyway.
    window5_ok = True
    for raw in [dr_mean[:, 0], dr_mean[:, 1], dr_mean[:, 2], dr_std[:, 0], dr_std[:, 1], dr_std[:, 2],
                av_mean[:, 0], av_std[:, 0]]:
        jumps = np.abs(np.diff(raw))
        if jumps.size and jumps.max() > 8.0:
            window5_ok = False

    raw_jumps = np.abs(np.diff(dr_std[:, 0]))
    smoothed_jumps = np.abs(np.diff(dr_std_s5[:, 0]))
    if window5_ok and (raw_jumps.size == 0 or smoothed_jumps.max() < raw_jumps.max() * 0.6):
        final_window = 5
        av_mean_s, av_std_s, dr_mean_s, dr_std_s = av_mean_s5, av_std_s5, dr_mean_s5, dr_std_s5
    else:
        final_window = 9
        av_mean_s = np.zeros_like(av_mean); av_std_s = np.zeros_like(av_std)
        dr_mean_s = np.zeros_like(dr_mean); dr_std_s = np.zeros_like(dr_std)
        for c in range(3):
            av_mean_s[:, c] = _rolling_median_smooth(av_mean[:, c], 9)
            av_std_s[:, c] = _rolling_median_smooth(av_std[:, c], 9)
            dr_mean_s[:, c] = _rolling_median_smooth(dr_mean[:, c], 9)
            dr_std_s[:, c] = _rolling_median_smooth(dr_std[:, c], 9)

    if verbose:
        print(f"[precompute_stats] final_window={final_window}  overhang_smooth "
              f"mean={overhang_smooth.mean():.1f} min={overhang_smooth.min():.1f} max={overhang_smooth.max():.1f}")

    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(cache_npz), n_frames=n_frames, final_window=final_window, render_canvas_h=ch, reg=reg_key,
             avatar_stats_mask=avatar_stats_mask,
             overhang_raw=overhang_raw, overhang_smooth=overhang_smooth,
             av_mean_raw=av_mean, av_std_raw=av_std, dr_mean_raw=dr_mean, dr_std_raw=dr_std,
             av_mean_s=av_mean_s, av_std_s=av_std_s, dr_mean_s=dr_mean_s, dr_std_s=dr_std_s)
    if verbose:
        print(f"[precompute_stats] wrote {cache_npz}")
    return overhang_smooth, av_mean_s, av_std_s, dr_mean_s, dr_std_s


def load_stats():
    """Load a previously-written Phase-1 cache (raises if it doesn't exist yet)."""
    z = np.load(str(STATS_CACHE_NPZ))
    return z["overhang_smooth"], z["av_mean_s"], z["av_std_s"], z["dr_mean_s"], z["dr_std_s"]


# ---------------------------------------------------------------------------
# Phase 2: per-frame composite (production entry point)
# ---------------------------------------------------------------------------
# Chin/shirt localized narrow neck-taper (opt-in). Default None == OFF, byte-
# identical. When set to a depth in px (validated: 8.0), the frame is composited
# twice -- once as normal, once with neck_taper start=0 / depth=<px> (smoothstep)
# -- and the narrow-taper result replaces the normal one only inside a feathered
# mask of the driver's BiSeNet CLOTH label (dilated CHIN_SHIRT_DILATE_PX,
# Gaussian sigma CHIN_SHIRT_FEATHER_SIGMA). Removes the pale smear where the
# avatar chin alpha fades over the blue shirt (f72-88); unlike a global narrow
# taper it leaves the cheek/ear/jaw silhouette identical to baseline (a global
# taper exposed a pale-orange cheek sliver at f82-83).
CHIN_SHIRT_TAPER_DEFAULT = None
CHIN_SHIRT_CLOTH_LABEL = 16
CHIN_SHIRT_DILATE_PX = 41
CHIN_SHIRT_FEATHER_SIGMA = 6.0


def composite_frame_v3(f, parser, M, plate_stat, lm, stats, return_debug=False, frame_tag=None,
                       wall_sliver_clean=WALL_SLIVER_CLEAN_DEFAULT,
                       render_canvas_h=RENDER_CANVAS_H_DEFAULT, avatar_frames_dir=None, frames_dir=None,
                       neck_taper=NECK_TAPER_DEFAULT,
                       neck_taper_start_px=NECK_TAPER_START_PX_DEFAULT,
                       neck_taper_depth_px=NECK_TAPER_DEPTH_PX_DEFAULT,
                       neck_taper_ease=NECK_TAPER_EASE_DEFAULT,
                       rim3d_injection=RIM3D_INJECTION_DEFAULT,
                       rim3d_jaw_ext=RIM3D_JAW_EXT_DEFAULT,
                       av_alpha_dilate=AV_ALPHA_DILATE_DEFAULT,
                       av_alpha_morph_sdf=AV_ALPHA_MORPH_SDF_DEFAULT,
                       crown_temple_dilate=CROWN_TEMPLE_DILATE_DEFAULT,
                       erase_feather_clamp=ERASE_FEATHER_CLAMP_DEFAULT,
                       erase_feather_clamp_ramp=ERASE_FEATHER_CLAMP_RAMP_DEFAULT,
                       jaw_taper_continuous=JAW_TAPER_CONTINUOUS_DEFAULT,
                       neck_taper_oor_gate=NECK_TAPER_OOR_GATE_DEFAULT,
                       driver_neck_own=DRIVER_NECK_OWN_DEFAULT,
                       neck_taper_ramp=NECK_TAPER_RAMP_DEFAULT,
                       neck_taper_ear_protect=NECK_TAPER_EAR_PROTECT_DEFAULT,
                       seg2_pushpull_fill=SEG2_PUSHPULL_FILL_DEFAULT,
                       edge_rgb_dilate=EDGE_RGB_DILATE_DEFAULT,
                       rvm_hair_blend=RVM_HAIR_BLEND_DEFAULT,
                       dehalo_inpaint=DEHALO_INPAINT_DEFAULT,
                       reg=REG,
                       avatar_blur_match=AVATAR_BLUR_MATCH_DEFAULT,
                       glasses_erase_extend=GLASSES_ERASE_EXTEND_DEFAULT,
                       faceskin_dilate_above_jaw=FACESKIN_DILATE_ABOVE_JAW_DEFAULT,
                       jaw_seam_xmin_taper=JAW_SEAM_XMIN_TAPER_DEFAULT,
                       sl_override=None, alpha_out=None,
                       aaband_content_gate=AABAND_CONTENT_GATE_DEFAULT,
                       aaband_av_lo=None, aaband_av_hi=None,
                       aaband_temporal_hysteresis=AABAND_TEMPORAL_HYSTERESIS_DEFAULT,
                       aaband_sl_prev=None,
                       aaband_hairline_exempt_row_px=AABAND_HAIRLINE_EXEMPT_ROW_PX_DEFAULT,
                       aaband_ear_exempt_px=AABAND_EAR_EXEMPT_PX_DEFAULT,
                       contact_shadow=CONTACT_SHADOW_DEFAULT,
                       contact_shadow_start_px=CONTACT_SHADOW_START_PX_DEFAULT,
                       contact_shadow_depth_px=CONTACT_SHADOW_DEPTH_PX_DEFAULT,
                       contact_shadow_strength=CONTACT_SHADOW_STRENGTH_DEFAULT,
                       driver_leak_plate=DRIVER_LEAK_PLATE_DEFAULT,
                       bare_neck_chroma_match=BARE_NECK_CHROMA_MATCH_DEFAULT,
                       hair_recolor_h1=HAIR_RECOLOR_H1_DEFAULT, hair_recolor_h1_face_gate=False,
                       hinge_leak_fix=HINGE_LEAK_FIX_DEFAULT,
                       halo_band_smoothstep=HALO_BAND_SMOOTHSTEP_DEFAULT,
                       chin_shirt_taper=CHIN_SHIRT_TAPER_DEFAULT,
                       drv_img=None, avatar_rgba_img=None):
    """Production per-frame composite: build_r8 (whole-head erasure + static-plate
    recompose, all validated production-default fix flags on, candidate D's
    halo_band_thresh=0.15 unchanged) with Variant F's full-head LAB harmonization +
    gradient-guarded seam blend layered on top exactly as build_fullhead does it.

    `stats` is the Phase-1 tuple from `precompute_stats`/`load_stats`
    (overhang_smooth, av_mean_s, av_std_s, dr_mean_s, dr_std_s), each indexed by
    frame `f`. Per-frame independent given `stats` -- safe to fan across worker
    processes exactly like the old branch-based composite_frame_v3 was.

    dehalo_inpaint defaults to DEHALO_INPAINT_DEFAULT=False (~440ms/frame faster,
    see that constant's docstring for the 245-frame validation) -- pass
    dehalo_inpaint=True to restore the cv2.inpaint dark-rim fix.

    hair_recolor_h1=True applies Candidate H1's hair-neutral-grey recolor to
    build_fullhead's own final proto -- see build_fullhead's docstring and
    the HAIR_RECOLOR_H1_* constants block for the full derivation. Default
    False: byte-identical output when omitted."""
    if chin_shirt_taper is not None:
        _kw = {k: v for k, v in locals().items()
               if k not in ('f', 'parser', 'M', 'plate_stat', 'lm', 'stats', 'chin_shirt_taper')}
        _kw['return_debug'] = True
        base, dbg = composite_frame_v3(f, parser, M, plate_stat, lm, stats, chin_shirt_taper=None, **_kw)
        cloth = (dbg['sl'] == CHIN_SHIRT_CLOTH_LABEL).astype(np.uint8)
        if not cloth.any():
            return (base, dbg) if return_debug else base
        _kw2 = dict(_kw, neck_taper=True, neck_taper_start_px=0.0,
                    neck_taper_depth_px=float(chin_shirt_taper), neck_taper_ease="smoothstep")
        narrow, _ = composite_frame_v3(f, parser, M, plate_stat, lm, stats, chin_shirt_taper=None, **_kw2)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CHIN_SHIRT_DILATE_PX, CHIN_SHIRT_DILATE_PX))
        m = np.clip(cv2.GaussianBlur(cv2.dilate(cloth, k).astype(np.float32), (0, 0),
                                     CHIN_SHIRT_FEATHER_SIGMA), 0.0, 1.0)[..., None]
        out = np.clip(base.astype(np.float32) * (1.0 - m) + narrow.astype(np.float32) * m, 0, 255).astype(np.uint8)
        return (out, dbg) if return_debug else out
    overhang_smooth, av_mean_s, av_std_s, dr_mean_s, dr_std_s = stats
    d = build_fullhead(f, parser, M, plate_stat, lm, overhang_smooth[f],
                        av_mean_s[f], av_std_s[f], dr_mean_s[f], dr_std_s[f],
                        ear_boost=True, crop_edge_fix=True, halo_fix=dehalo_inpaint, av_clamp=False,
                        edge_clip_fix=True, halo_band_thresh=HALO_BAND_THRESH_DEFAULT,
                        render_canvas_h=render_canvas_h,
                        avatar_frames_dir=avatar_frames_dir, frames_dir=frames_dir,
                        neck_taper=neck_taper, neck_taper_start_px=neck_taper_start_px,
                        neck_taper_depth_px=neck_taper_depth_px, neck_taper_ease=neck_taper_ease,
                        rim3d_injection=rim3d_injection, rim3d_jaw_ext=rim3d_jaw_ext,
                        av_alpha_dilate=av_alpha_dilate,
                        av_alpha_morph_sdf=av_alpha_morph_sdf,
                        crown_temple_dilate=crown_temple_dilate,
                        erase_feather_clamp=erase_feather_clamp,
                        erase_feather_clamp_ramp=erase_feather_clamp_ramp,
                        jaw_taper_continuous=jaw_taper_continuous,
                        neck_taper_oor_gate=neck_taper_oor_gate,
                        driver_neck_own=driver_neck_own,
                        neck_taper_ramp=neck_taper_ramp,
                        neck_taper_ear_protect=neck_taper_ear_protect,
                        seg2_pushpull_fill=seg2_pushpull_fill,
                        edge_rgb_dilate=edge_rgb_dilate,
                        rvm_hair_blend=rvm_hair_blend,
                        reg=reg,
                        avatar_blur_match=avatar_blur_match,
                        glasses_erase_extend=glasses_erase_extend,
                        faceskin_dilate_above_jaw=faceskin_dilate_above_jaw,
                        jaw_seam_xmin_taper=jaw_seam_xmin_taper,
                        sl_override=sl_override,
                        aaband_content_gate=aaband_content_gate,
                        aaband_av_lo=aaband_av_lo, aaband_av_hi=aaband_av_hi,
                        aaband_temporal_hysteresis=aaband_temporal_hysteresis,
                        aaband_sl_prev=aaband_sl_prev,
                        aaband_hairline_exempt_row_px=aaband_hairline_exempt_row_px,
                        aaband_ear_exempt_px=aaband_ear_exempt_px,
                        contact_shadow=contact_shadow,
                        contact_shadow_start_px=contact_shadow_start_px,
                        contact_shadow_depth_px=contact_shadow_depth_px,
                        contact_shadow_strength=contact_shadow_strength,
                        driver_leak_plate=driver_leak_plate,
                        bare_neck_chroma_match=bare_neck_chroma_match,
                        hair_recolor_h1=hair_recolor_h1, hair_recolor_h1_face_gate=hair_recolor_h1_face_gate,
                        hinge_leak_fix=hinge_leak_fix,
                        halo_band_smoothstep=halo_band_smoothstep,
                        drv_img=drv_img, avatar_rgba_img=avatar_rgba_img)
    out = d["proto"]
    if wall_sliver_clean:
        out = _wall_sliver_clean(out, d)
    if alpha_out is not None: alpha_out.append(d["a_av"])   # opt-in tap of the avatar alpha canvas (render_demo_final.turn_fix)
    if return_debug:
        debug = dict(
            erase_px=int(d["erase_bin"].sum()),
            dropped_px=int(d["dropped"]),
            we_mean=float(d["We"].mean()),
            a_av_support_px=int(d["support"].sum()),
            overhang_used=float(d["overhang_used"]),
            w_smooth_mean=float(d["w_smooth"].mean()),
            # kept for backward-compatible logging in run_blend_parallel.py; the
            # old branch_b/branch_c/collar concepts no longer exist in this
            # architecture, so these are always 0/None now.
            collar_y=None, collar_was_capped=False,
            hide_mask_px=int(d["erase_bin"].sum()),
            branch_b_px=0, branch_c_px=0,
            avatar_blur_sigma=float(d["avatar_blur_sigma"]),
            sl=d["sl"],
        )
        return out, debug
    return out


def main():
    M = load_M()
    z = np.load(str(TRACK_NPZ)); lm = z["landmarks_mp"]
    n_frames = len(list(FRAMES_DIR.glob('*.png')))
    print(f'{n_frames} frames.')

    OUT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    print('Loading FaceParser (BiSeNet)...')
    parser = FaceParser(device='cuda')
    plate_stat = static_plate(parser=parser)  # Task 42: reject a person-contaminated PLATE_PNG

    print('Phase 1: precomputing overhang/color-stats (serial, whole-clip)...')
    stats = precompute_stats(parser, M, plate_stat, lm, n_frames)

    print('Phase 2: per-frame composite...')
    meta_rows = []
    for f in range(n_frames):
        if f % 20 == 0:
            print(f'  frame {f}/{n_frames} ...')
        out, debug = composite_frame_v3(f, parser, M, plate_stat, lm, stats, return_debug=True)
        cv2.imwrite(str(OUT_FRAMES_DIR / f'{f:05d}.png'), out)
        meta_rows.append(dict(frame=f, erase_px=debug['erase_px'], dropped_px=debug['dropped_px'],
                               we_mean=debug['we_mean'], a_av_support_px=debug['a_av_support_px'],
                               overhang_used=debug['overhang_used'], w_smooth_mean=debug['w_smooth_mean']))

    keys = list(meta_rows[0].keys())
    arrs = {k: np.array([r[k] for r in meta_rows]) for k in keys}
    np.savez(OUT_DIR / 'blend_v3_frames_meta.npz', **arrs)
    print(f'Done. {n_frames} frames -> {OUT_FRAMES_DIR}')
    print(f'erase_px: mean={arrs["erase_px"].mean():.1f} max={arrs["erase_px"].max()}')
    print(f'we_mean: mean={arrs["we_mean"].mean():.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
