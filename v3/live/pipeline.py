#!/usr/bin/env python
"""
Snapshot-tier live pipeline (2026-09-15 feasibility follow-up).

Wires the REAL production code paths (fit3d.py's calibration/T-solve math,
render_gaga.py's per-frame Gaussian render + floater/despeckle fixes,
blend_v3.py's whole-head erase + Variant F composite) to run against ONE
live-captured frame at a time, instead of a pre-extracted 245-frame clip.

This module never writes to any PRODUCTION path (v3/debug_out/**, the
checked-in checkpoints, track_results.npz, fit3d_results.npz, the production
static plate). Per-snapshot scratch (temp driver/avatar PNGs, stats cache)
lives under v3/live/session_data/<session id>/_tmp/, git-ignored.

Cross-process collision fix (this task): this scratch dir used to be a
single fixed on-disk path (v3/live/_session_tmp/) shared by EVERY
PipelineState instance in EVERY process, protected only by an in-process
threading.RLock -- which does nothing across separate OS processes. Any
second process touching PipelineState (another server.py instance, or any
of this project's diagnostic scripts) could overwrite the scratch file
between another instance's write and its own immediate read-back inside
build_r8, causing either a crash (cv2.imread returns None) or, worse, a
silent composite built from the WRONG frame's data with no error at all.
Now every PipelineState instance gets its own scratch dir nested under its
own uuid-suffixed session_dir (self.driver_tmp_dir / self.avatar_tmp_dir /
self.stats_tmp_npz), so no two instances -- in the same process or
different ones -- can ever share a scratch path. reset_session_tmp() is
kept as a backward-compatible no-op for existing call sites; per-instance
cleanup is PipelineState.reset_tmp().

Calibration results (plate_stat, s_cal/t_cal/T_prev) are ALSO persisted,
separately, to v3/live/session_data/<session id>/*.npz (the same directory
_tmp/ now nests under) -- unlike _tmp/ this is never auto-deleted, so a
session's calibration survives the server process dying and can be
reloaded later via load_session_calibration() for diagnostics, instead of
every follow-up investigation having to reconstruct/approximate it from
scratch.

Two production flags are deliberately turned OFF for live snapshots, and this
is NOT a fix for any of the permanently-closed defects (see
task30_consolidated_status.txt) -- it's because their *implementations*, not
just their tuning, are hardwired to the pre-recorded production clip's own
data/geometry, and would silently apply WRONG data against a live different
person's frame if left on:
  * rim3d_injection: `_rim3d_load()` inside blend_v3.py reads the production
    TRACK_NPZ/FIT3D_NPZ arrays by frame index to get the 3D mesh silhouette
    for the exemption test -- at live frame index 0 that would use the
    PRE-RECORDED ACTOR's frame-0 pose, not the live user's. Not parameterized
    per-call, so there's no way to feed it the live pose without editing
    blend_v3.py itself, which is out of scope here.
  * erase_feather_clamp: EFC_ROI is a hardcoded pixel rectangle, documented in
    blend_v3.py itself as "THIS CLIP ONLY -- see history above". Applying it
    to a live frame would force full erase-weight in that fixed rectangle
    regardless of whether it's relevant to the live user's framing.
Every other production default (whole-head erase, static-plate recompose,
neck_taper, AA-band content gate, av_alpha_dilate, crop-edge fix, Variant F
LAB harmonization) runs exactly as production configures it -- including
dehalo_inpaint, which as of the 245-frame ablation validation (2026-09-15,
see blend_v3.DEHALO_INPAINT_DEFAULT's docstring) now defaults to False. This
pipeline never pins dehalo_inpaint explicitly, so it inherits that default
automatically: snapshots render ~440ms/frame faster for free, with no
per-call change needed here.
"""
import sys
import time
import shutil
import threading
import uuid
from pathlib import Path

import numpy as np
import cv2
import torch
import torchvision
import lightning
from scipy.optimize import least_squares

_HEADSWAP_ROOT = Path(__file__).resolve().parent.parent.parent
_GAGA_ROOT = _HEADSWAP_ROOT / "third_party" / "GAGAvatar"
_V3 = _HEADSWAP_ROOT / "v3"
sys.path.insert(0, str(_GAGA_ROOT))
sys.path.insert(0, str(_V3))
sys.path.insert(0, str(_HEADSWAP_ROOT / "compositing"))

import inference as offline_inference  # noqa: E402
from core.models import build_model  # noqa: E402
from core.libs.utils import ConfigDict  # noqa: E402
from core.libs.flame_model import FLAMEModel  # noqa: E402
from core.libs.flame_model.lbs import vertices2landmarks, batch_rodrigues  # noqa: E402
from core.data.loader_track import build_points_planes  # noqa: E402
from core.libs.utils_renderer import render_gaussian, NUM_CHANNELS  # noqa: E402
from core.libs.GAGAvatar_track.engines import CoreEngine as TrackEngine  # noqa: E402

import render_gaga as rg   # noqa: E402  -- reuse apply_floater_mask/despeckle_silhouette/_upsample_tall verbatim
import blend_v3 as bv3     # noqa: E402  -- reuse build_r8/build_fullhead/composite_frame_v3/precompute_stats verbatim
import fit3d as f3d        # noqa: E402  -- reuse project_landmarks_to_render_canvas/solve_T_least_squares/similarity_solve
import track as trk        # noqa: E402  -- reuse track_frame/crop_face/crop_warp_gpu/SmirkEncoder

IDENTITY_IMAGE = str(_HEADSWAP_ROOT / "avatar" / "reference_photos" / "headswap_target.jpg")
GAGA_CKPT = str(_GAGA_ROOT / "assets" / "GAGAvatar.pt")

H, W = bv3.H, bv3.W          # 720, 1280 -- driver-frame size the compositor's REG/geometry constants assume

# blend_v3.REG = (460, 174, 1280, 720) is NOT an arbitrary framing choice: it is
# the exact pixel bounding box of the checked-in production plate PNG
# (v3/assets/bgplate_t1c_plate_sub.png, shape (546, 820, 3) == REG's h/w), which
# blend_v3.static_plate() pastes into a zero canvas at exactly this rectangle.
# The offline path is structurally tied to that fixed-size asset and must keep
# using bv3.REG unchanged. A live session never touches that PNG -- its plate
# is built fresh each session from a real camera burst (see calibrate_plate
# below) -- so it has no such constraint and can safely use the full working
# frame, which also removes the top-of-head clip that REG's narrower box
# caused when it clipped the erase-candidate mask.
LIVE_REG = (0, 0, W, H)

CANVAS_H = 768                # matches blend_v3.RENDER_CANVAS_H_DEFAULT (tall below-jaw-coverage render)
CANVAS_W = 512

# Task B1 (diagnostic report: live-vs-offline glasses/thin-feature erasure
# gap). Root cause, confirmed empirically (see v3/debug_out/3d_silhouette's
# B1 report): _ensure_size's letterbox downscale to this module's fixed
# W x H working canvas happens BEFORE BiSeNet ever sees the frame, and
# BiSeNet's own parse() always resizes its input down to 512x512 again --
# so a native camera frame goes through two lossy downsamples (native ->
# 1280x720 INTER_AREA -> 512x512 INTER_LINEAR) before segmentation, versus
# offline's single native-resolution-frame -> 512x512 resize. Thin features
# like glasses rims lose disproportionately more signal across two
# downsamples than one.
#
# When True, process_snapshot() parses the RAW native frame directly
# (single resize straight to BiSeNet's 512x512 input, matching offline's
# one-resize path) instead of the already-letterboxed frame, then remaps
# the resulting label map into the 1280x720 working canvas using the exact
# same scale+pad transform _ensure_size applied to the RGB frame, so it
# stays pixel-aligned with drv/erase_bin/etc. This ONLY changes which
# labels feed blend_v3.build_r8's existing `sl` variable (via the new
# sl_override param) -- no erase/composite math in blend_v3.py is touched,
# and every other stage (tracking, fit3d, rendering, plate) is unaffected.
#
# Off by default until validated by its own regression suite -- see
# v3/live/_regress_b1_hires_glasses_parse.py.
LIVE_HIRES_GLASSES_PARSE_DEFAULT = False
FOCAL = 12.0
CANVAS_PAD = (CANVAS_H - 512) // 2
FOCAL_Y_TALL = FOCAL * 512.0 / CANVAS_H

# FRAME_IDX no longer needs to be globally unique (see cross-process
# collision fix in the module docstring) -- each PipelineState instance's
# driver_tmp_dir/avatar_tmp_dir is already its own isolated directory, so a
# fixed filename inside it can never collide with any other instance's.
FRAME_IDX = 0

# Persisted per-session calibration AND (as of this task) per-session
# scratch space, one subdirectory per PipelineState instance keyed by a
# fresh session id. The calibration files (plate_calibration.npz,
# pose_calibration.npz) are deliberately never auto-deleted: it's the fix
# for every prior live diagnostic task having to reconstruct/approximate
# plate+pose calibration from scratch because the real, in-memory-only
# calibration from that session's server process was already gone. The
# _tmp/ subdirectory nested inside each session_dir (see PipelineState
# .driver_tmp_dir/.avatar_tmp_dir/.stats_tmp_npz) IS safe to delete --
# see PipelineState.reset_tmp(). Git-ignored scratch data, not a
# production path.
SESSION_DATA_ROOT = Path(__file__).resolve().parent / "session_data"


class NotCalibratedError(RuntimeError):
    pass


class PipelineState:
    """Loads every model once; serializes all GPU work behind self.lock (single
    RTX 2060, one session -- no concurrent pipeline calls, matches the spike's
    explicit single-user scope)."""

    def __init__(self, device="cuda", hires_glasses_parse=LIVE_HIRES_GLASSES_PARSE_DEFAULT,
                 avatar_stats_mask="wholehead"):
        self.device = device
        self.hires_glasses_parse = hires_glasses_parse
        # opt-in candidate (NOT shipped default): "faceskin" restricts the
        # Variant F harmonization avatar "from" stats to face-skin-only
        # pixels instead of the whole rendered head (hair+skin+ears mixed).
        # See bv3.precompute_stats's avatar_stats_mask docstring -- validated
        # on test2.mp4 to close 37-72% of a real-room-lighting local-forehead
        # ΔE gap that whole-head-only harmonization leaves uncorrected.
        self.avatar_stats_mask = avatar_stats_mask
        self.lock = threading.RLock()
        self.calibrated_plate = False
        self.calibrated_pose = False
        self.plate_stat = None
        self.s_cal = None
        self.t_cal = None
        self.T_prev = None
        self.session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.session_dir = SESSION_DATA_ROOT / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        # Per-instance scratch space (cross-process collision fix): nested
        # under this instance's own uuid-suffixed session_dir, so no other
        # PipelineState -- in this process or any other -- can ever share
        # these paths. See module docstring.
        self.tmp_dir = self.session_dir / "_tmp"
        self.driver_tmp_dir = self.tmp_dir / "driver"
        self.avatar_tmp_dir = self.tmp_dir / "avatar"
        self.stats_tmp_npz = self.tmp_dir / "stats.npz"
        self._load_models()

    def _load_models(self):
        t0 = time.perf_counter()
        lightning.fabric.seed_everything(42)
        self.fabric = lightning.Fabric(accelerator=self.device, strategy="auto", devices=[0])
        self.fabric.launch()
        full_checkpoint = self.fabric.load(GAGA_CKPT)
        meta_cfg = ConfigDict(init_dict=full_checkpoint["meta_cfg"])
        model = build_model(model_cfg=meta_cfg.MODEL)
        model.load_state_dict(full_checkpoint["model"])
        model = self.fabric.setup(model)
        model.eval()
        self.model = model

        track_engine = TrackEngine(focal_length=FOCAL, device=self.device)
        feature_data = offline_inference.get_tracked_results(IDENTITY_IMAGE, track_engine)
        del track_engine
        torch.cuda.empty_cache()
        self.f_image = torchvision.transforms.functional.resize(
            feature_data["image"].to(self.device), (518, 518), antialias=True)[None]
        self.f_transform = feature_data["transform_matrix"].float().to(self.device)
        f_planes_cpu = build_points_planes(meta_cfg.DATASET.POINT_PLANE_SIZE, self.f_transform.cpu())
        self.f_planes = {k: v[None].to(self.device) for k, v in f_planes_cpu.items()}
        self.f_shape = feature_data["shapecode"].float().to(self.device)[None]

        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=5.0, no_lmks=True).to(self.device)
        self.faces_tensor = self.flame_model.get_faces()

        # fit3d.py's own identity loader -- gives beta_avatar / source_transform /
        # the 105-point mediapipe<->FLAME correspondence table, verbatim.
        (_flame2, self.beta_avatar, self.source_transform, self.landmark_indices,
         self.lmk_face_idx, self.lmk_b_coords) = f3d.load_flame_and_identity(self.device)
        self.source_T_np = self.source_transform[:3, 3].cpu().numpy()
        self.source_R = self.source_transform[:3, :3]

        # SMIRK encoder (v3/track.py's own loader, verbatim).
        self.smirk_encoder = trk.SmirkEncoder().to(self.device)
        checkpoint = torch.load(str(trk.SMIRK_CHECKPOINT), map_location=self.device, weights_only=False)
        checkpoint_encoder = {k.replace("smirk_encoder.", ""): v
                               for k, v in checkpoint.items() if "smirk_encoder" in k}
        self.smirk_encoder.load_state_dict(checkpoint_encoder)
        self.smirk_encoder.eval()

        self.parser = bv3.FaceParser(device=self.device)
        self.cam_params = {"focal_x": FOCAL, "focal_y": FOCAL_Y_TALL, "size": [CANVAS_H, CANVAS_W]}

        self.driver_tmp_dir.mkdir(parents=True, exist_ok=True)
        self.avatar_tmp_dir.mkdir(parents=True, exist_ok=True)

        # One-shot warmup render at a neutral pose so model._gs_params exists
        # before any real snapshot -- render_gaga.py's own pattern (only the
        # FIRST call does forward_expression; later calls mutate xyz in place).
        with torch.no_grad():
            expression = torch.zeros(1, 100, device=self.device)
            pose_params = torch.zeros(1, 6, device=self.device)
            eye_pose = torch.zeros(1, 6, device=self.device)
            t_points = self.flame_model(shape_params=self.f_shape, expression_params=expression,
                                         pose_params=pose_params, eye_pose_params=eye_pose)
            t_transform = self.source_transform[None]
            batch = {"f_image": self.f_image, "f_planes": self.f_planes,
                     "t_image": torch.zeros(1, 3, 512, 512, device=self.device),
                     "t_points": t_points, "t_transform": t_transform}
            self.model.forward_expression(batch)
        print(f"[pipeline] models loaded in {time.perf_counter() - t0:.2f}s, GPU warm.")
        print(f"[pipeline] session id {self.session_id}, calibration will persist to {self.session_dir}")

    # ------------------------------------------------------------------
    def _track(self, frame_bgr):
        """Real per-frame tracking path (v3/track.py's track_frame, verbatim)."""
        smirk_out, kpt = trk.track_frame(frame_bgr, self.smirk_encoder, self.device)
        if smirk_out is None:
            return None
        expr = smirk_out["expression_params"][0].detach().cpu().numpy()
        jaw = smirk_out["jaw_params"][0].detach().cpu().numpy()
        grot = smirk_out["pose_params"][0].detach().cpu().numpy()
        return dict(expression=expr, jaw=jaw, global_rot=grot, landmarks=kpt.astype(np.float32))

    def _flame_forward_and_landmarks(self, expr50, jaw3):
        """One FLAME forward at GLOBAL ROTATION ZERO (fit3d.py round-2 design --
        see its module docstring) with the live expression/jaw. Returns
        (t_points (1,5023,3) for rendering, P0 (105,3) numpy for the T-solve) --
        both fit3d.py and render_gaga.py compute this SAME quantity separately
        offline; here it's one call reused for both, since we're not batching."""
        expression = torch.zeros(1, 100, device=self.device)
        expression[:, :50] = torch.from_numpy(expr50).to(self.device)
        pose_params = torch.cat([torch.zeros(3, device=self.device),
                                  torch.from_numpy(jaw3).to(self.device)])[None].float()
        eye_pose = torch.zeros(1, 6, device=self.device)
        t_points = self.flame_model(shape_params=self.f_shape, expression_params=expression,
                                     pose_params=pose_params, eye_pose_params=eye_pose)
        P0 = vertices2landmarks(t_points, self.faces_tensor,
                                 self.lmk_face_idx[None], self.lmk_b_coords[None])
        return t_points, P0[0].detach().cpu().numpy()

    # ------------------------------------------------------------------
    def calibrate_plate(self, frames_bgr):
        """Static background plate for THIS session's room, built the same way
        blend_v3.static_plate() shapes its production PNG (zeros everywhere
        except the REG erase/plate rectangle) -- just computed fresh from a
        short live burst instead of loaded from a checked-in file, since the
        checked-in plate (PLATE_PNG, a per-pixel temporal median over the
        original "test" clip's frames with only the driver's head+neck
        excluded -- NOT an empty-room capture) has no relationship to the
        MacBook's actual background.

        Task 42: before Task 42, this only checked len(frames) >= 3 (see
        server.py's calibrate_plate handler) -- it never checked the burst
        actually showed an empty room. Verified against real session data
        (v3/live/session_data/): all 4 prior sessions on disk had the
        operator visibly in frame despite passing calibration. Now rejects
        with RuntimeError (-> HTTP 422 in server.py) if the already-loaded
        BiSeNet parser finds SKIN-class pixels above
        blend_v3.FACE_PRESENT_SKIN_FRAC_THRESHOLD in ANY burst frame (checked
        per-frame, not just on the post-median plate, so a person who is only
        transiently in frame during the burst still trips it) -- same gate
        blend_v3.static_plate() applies to PLATE_PNG.

        KNOWN LIMITATION, not silently assumed solved: this only catches a
        clearly-visible face in a normal seated pose, matching what all 4
        real contaminated sessions exhibited. It has NOT been validated
        against a ducked, turned-away, or partially-out-of-frame operator --
        BiSeNet's SKIN-class coverage in those poses is unverified, not
        confirmed working. Flagged as a known gap for future work."""
        with self.lock:
            rx0, ry0, rx1, ry1 = LIVE_REG
            crops = []
            for fr in frames_bgr:
                fr = _ensure_size(fr, W, H)
                crop = fr[ry0:ry1, rx0:rx1]
                skin_frac = float((self.parser.parse(crop) == bv3.SKIN).mean())
                if skin_frac > bv3.FACE_PRESENT_SKIN_FRAC_THRESHOLD:
                    raise RuntimeError(
                        f"Person detected in background (SKIN-class pixel fraction "
                        f"{skin_frac:.4f} > {bv3.FACE_PRESENT_SKIN_FRAC_THRESHOLD}) -- "
                        f"please step out of frame and retry")
                crops.append(crop.astype(np.float32))
            median_crop = np.median(np.stack(crops, axis=0), axis=0)
            pf = np.zeros((H, W, 3), np.float32)
            pf[ry0:ry1, rx0:rx1] = median_crop
            self.plate_stat = pf
            self.calibrated_plate = True
            np.savez(str(self.session_dir / "plate_calibration.npz"),
                     plate_stat=self.plate_stat, reg=np.array(LIVE_REG))

    def calibrate_plate_person_excluded(self, frames_bgr, min_obs_frac=0.0, mask_dilate_px=0):
        """Part B (opt-in, not called by calibrate_plate or process_snapshot --
        caller invokes this explicitly instead of calibrate_plate when they want
        it): same median-plate-substitute shape as calibrate_plate(), but masks
        out each burst frame's own BiSeNet HEAD_LABELS pixels (face/hair/ears/
        neck -- the same label set blend_v3.build_r8 uses for head_parse) BEFORE
        computing the per-pixel median, so a coverage-gap region that happens to
        sit behind the subject's hair in most burst frames isn't contaminated by
        that hair in the final plate. This mirrors how the checked-in PLATE_PNG
        asset itself was built (calibrate_plate's own docstring: "a per-pixel
        temporal median over the original 'test' clip's frames with only the
        driver's head+neck excluded"), not a naive uncontrolled median.

        Does NOT apply the Task 42 SKIN-presence gate -- unlike calibrate_plate,
        this method exists specifically for the case where a person IS present
        in every burst frame (no person-free burst available), so rejecting on
        SKIN presence would make it unusable by construction.

        KNOWN LIMITATION (OUT OF SCOPE, not touched by the two fixes below): a
        background pixel occupied by the subject in EVERY burst frame has no
        unmasked observation to fall back on. At exactly those pixels this
        falls back to the plain (contaminated) per-pixel median across all
        frames -- same gap calibrate_plate() has for a stationary, centered
        subject. Using a burst with real head-turn motion (spanning many
        frames of a clip, not just a handful in a row) narrows this gap
        since different pixels get exposed at different times, but does not
        close it where the subject never uncovers a pixel across the whole
        burst.

        Two opt-in fixes for a SEPARATE, previously-undiagnosed contamination
        source: pixels that are only MOSTLY (not always) occluded, where a
        small, biased sample of nominally-unmasked observations (thin hair /
        motion blur / edge misclassification slipping past HEAD_LABELS) still
        pulls the per-pixel median away from the true background. Both
        default to off (byte-identical to the original method):

        min_obs_frac (Fix 1, graduated fallback): if the fraction of burst
        frames unmasked at a pixel is > 0 but below this threshold, treat
        that pixel's masked sample as too small/unreliable and use the plain
        (naive) whole-burst median there instead -- the SAME fallback value
        already used for the fully-occluded (0 unmasked frames) case, just
        triggered earlier. Pixels at or above the threshold are unaffected.

        mask_dilate_px (Fix 2, tighter per-frame exclusion): dilates each
        frame's HEAD_LABELS mask by this many pixels before excluding it, to
        catch thin-hair/motion-blur pixels just outside BiSeNet's classified
        boundary. Trades sample size for cleanliness -- too much dilation
        starts excluding genuine background near the head, which can shrink
        n_unmasked enough to trip min_obs_frac's fallback more often, not
        less."""
        with self.lock:
            rx0, ry0, rx1, ry1 = LIVE_REG
            crops = []
            masks = []
            dilate_kernel = (np.ones((mask_dilate_px * 2 + 1, mask_dilate_px * 2 + 1), np.uint8)
                              if mask_dilate_px > 0 else None)
            for fr in frames_bgr:
                fr = _ensure_size(fr, W, H)
                crop = fr[ry0:ry1, rx0:rx1]
                sl = self.parser.parse(crop)
                person_mask = bv3.mask_for(sl, bv3.HEAD_LABELS)
                if dilate_kernel is not None:
                    person_mask = cv2.dilate(person_mask.astype(np.uint8), dilate_kernel) > 0
                crops.append(crop.astype(np.float32))
                masks.append(person_mask)
            stack = np.stack(crops, axis=0)                      # (N, h, w, 3)
            mask_stack = np.stack(masks, axis=0)                 # (N, h, w) True=person
            masked = np.where(mask_stack[..., None], np.nan, stack)
            with np.errstate(invalid="ignore"):
                median_crop = np.nanmedian(masked, axis=0)
            plain_median = np.median(stack, axis=0)
            nan_px = np.isnan(median_crop).any(axis=-1)
            n_fallback = int(nan_px.sum())

            n_unmasked = (~mask_stack).sum(axis=0)               # (h, w) int count, 0..N
            n_lowsample = 0
            if min_obs_frac > 0:
                required = min_obs_frac * mask_stack.shape[0]
                lowsample_px = (n_unmasked > 0) & (n_unmasked < required) & (~nan_px)
                n_lowsample = int(lowsample_px.sum())
                median_crop = np.where(lowsample_px[..., None], plain_median, median_crop)

            median_crop = np.where(nan_px[..., None], plain_median, median_crop)
            pf = np.zeros((H, W, 3), np.float32)
            pf[ry0:ry1, rx0:rx1] = median_crop
            self.plate_stat = pf
            self.calibrated_plate = True
            np.savez(str(self.session_dir / "plate_calibration.npz"),
                     plate_stat=self.plate_stat, reg=np.array(LIVE_REG))
            return dict(fallback_px=n_fallback, lowsample_px=n_lowsample,
                        total_px=int(mask_stack.shape[1] * mask_stack.shape[2]))

    def calibrate_pose(self, frame_bgr):
        """Bootstraps this session's (s_cal, t_cal) render-canvas<->driver-frame
        calibration from ONE reference frame, using fit3d.py's own bootstrap
        formula verbatim (fit3d.py:305-320): project this frame's zero-rotation
        FLAME landmarks under a GUESS translation (the avatar identity's own
        neutral camera placement, source_T), then solve the 2D similarity that
        maps those render-canvas points onto this frame's REAL mediapipe
        landmarks. That similarity is then treated as fixed for every
        subsequent snapshot this session, exactly as fit3d.py treats it fixed
        for every frame of the batch clip."""
        with self.lock:
            frame_bgr = _ensure_size(frame_bgr, W, H)
            tracked = self._track(frame_bgr)
            if tracked is None:
                raise RuntimeError("no face detected in the calibration frame")
            _t_points, P0 = self._flame_forward_and_landmarks(tracked["expression"], tracked["jaw"])
            Rg = batch_rodrigues(torch.from_numpy(tracked["global_rot"])[None].to(self.device))[0]
            R_cam = (Rg.T @ self.source_R).cpu().numpy()
            T_guess = self.source_T_np.copy()
            render_lmk_guess = f3d.project_landmarks_to_render_canvas(P0, R_cam, T_guess)
            Q0 = tracked["landmarks"][self.landmark_indices.numpy(), :2]
            s_t, t_t = f3d.similarity_solve(
                torch.from_numpy(render_lmk_guess)[None].float(),
                torch.from_numpy(Q0)[None].float())
            self.s_cal = float(s_t[0])
            self.t_cal = t_t[0].numpy()
            self.T_prev = T_guess.astype(np.float64)
            self.calibrated_pose = True
            np.savez(str(self.session_dir / "pose_calibration.npz"),
                     s_cal=self.s_cal, t_cal=self.t_cal, T_prev=self.T_prev)
            return dict(s_cal=self.s_cal, t_cal=self.t_cal.tolist())

    def _M(self):
        s_cal, t_cal = self.s_cal, self.t_cal
        return np.array([[s_cal, 0, t_cal[0]],
                          [0, s_cal, t_cal[1] - s_cal * CANVAS_PAD]], dtype=np.float64)

    # ------------------------------------------------------------------
    def process_snapshot(self, frame_bgr, aaband_content_gate=bv3.AABAND_CONTENT_GATE_DEFAULT,
                          aaband_av_lo=None, aaband_av_hi=None,
                          aaband_temporal_hysteresis=bv3.AABAND_TEMPORAL_HYSTERESIS_DEFAULT,
                          aaband_hairline_exempt_row_px=bv3.AABAND_HAIRLINE_EXEMPT_ROW_PX_DEFAULT):
        if not (self.calibrated_plate and self.calibrated_pose):
            raise NotCalibratedError("run plate + pose calibration before capturing a snapshot")
        with self.lock:
            timings = {}
            frame_native = frame_bgr
            frame_bgr = _ensure_size(frame_bgr, W, H)

            # Task 49 profiling note: torch.cuda.synchronize() is inserted before
            # every stage boundary below (not present before this task). Without
            # it, CPU-side time.perf_counter() only measures kernel LAUNCH time --
            # actual GPU compute for an async CUDA op can finish during the NEXT
            # stage's wall-clock window, silently attributing that stage's own
            # GPU cost to whichever stage happens to call .cpu()/.item() first
            # afterward. This makes the per-stage split trustworthy at a small
            # (~sub-ms typically) added sync-wait cost, which is itself part of
            # each stage's true cost anyway (the CPU would have to wait for it
            # eventually regardless of where the boundary line is drawn).
            t0 = time.perf_counter()
            tracked = self._track(frame_bgr)
            if self.device == "cuda":
                torch.cuda.synchronize()
            timings["track_ms"] = (time.perf_counter() - t0) * 1000
            if tracked is None:
                raise RuntimeError("no face detected in this snapshot")

            t0 = time.perf_counter()
            t_points, P0 = self._flame_forward_and_landmarks(tracked["expression"], tracked["jaw"])
            Rg = batch_rodrigues(torch.from_numpy(tracked["global_rot"])[None].to(self.device))[0]
            R_cam = (Rg.T @ self.source_R).cpu().numpy()
            Q = tracked["landmarks"][self.landmark_indices.numpy(), :2]
            target_canvas = (Q - self.t_cal) / self.s_cal
            T, _resid = f3d.solve_T_least_squares(P0, R_cam, target_canvas, self.T_prev)
            self.T_prev = T
            if self.device == "cuda":
                torch.cuda.synchronize()
            timings["fit3d_ms"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            transform = np.zeros((3, 4), dtype=np.float32)
            transform[:, :3] = R_cam
            transform[:, 3] = T
            t_transform = torch.from_numpy(transform).float().to(self.device)[None]
            with torch.no_grad():
                self.model._gs_params["xyz"][:, :5023] = t_points
                gs_params = self.model._gs_params
                black_bg = t_transform.new_zeros(1, NUM_CHANNELS)
                white_bg = t_transform.new_ones(1, NUM_CHANNELS)
                gen_black = render_gaussian(gs_params, t_transform, self.cam_params, bg_color=black_bg)["images"]
                gen_white = render_gaussian(gs_params, t_transform, self.cam_params, bg_color=white_bg)["images"]
                alpha = (1.0 - (gen_white - gen_black).mean(dim=1, keepdim=True)).clamp(0, 1)
                rgb = rg._upsample_tall(self.model, gen_black, CANVAS_H)
            avatar_rgb = (rgb[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            avatar_alpha_raw = (alpha[0, 0].cpu().numpy() * 255).round().astype(np.uint8)
            avatar_alpha, _cutoff, _ab, _aa = rg.apply_floater_mask(avatar_alpha_raw)
            avatar_alpha, _despeckle_removed = rg.despeckle_silhouette(avatar_alpha)
            avatar_bgr = cv2.cvtColor(avatar_rgb, cv2.COLOR_RGB2BGR)
            if self.device == "cuda":
                torch.cuda.synchronize()
            timings["render_ms"] = (time.perf_counter() - t0) * 1000

            # Task 49 profiling: blend_ms was previously one bucket covering scratch
            # PNG I/O + stats precompute + the actual compositor call. Split so a
            # compositor-specific bottleneck (if any) is visible on its own,
            # separate from disk I/O -- per this task's out-of-scope note, a real
            # compositor slowdown gets flagged as its own follow-up, not fixed here.
            # Task 52 (disk-round-trip elimination): build_r8 used to require
            # frame_bgr/avatar_bgr+alpha on disk (frames_dir/avatar_frames_dir
            # PNG lookups) purely because it's shared with the offline batch
            # pipeline, which really does read frames off disk. The live path
            # already has both arrays in memory -- writing them out and having
            # build_r8 immediately read them back cost an imwrite+imread round
            # trip (~450x720 BGR + BGRA PNG, twice/frame) for no reason, since
            # nothing else in this process ever reads driver_tmp_dir/
            # avatar_tmp_dir. build_r8/_build_r8_core now accept drv_img/
            # avatar_rgba_img overrides (see blend_v3.py) so the live path can
            # hand them the arrays directly and skip the disk trip entirely.
            t0 = time.perf_counter()
            avatar_rgba = np.dstack([avatar_bgr, avatar_alpha])
            timings["blend_io_ms"] = (time.perf_counter() - t0) * 1000

            lm = tracked["landmarks"][None]  # (1, 478, 3) -- this session's single live frame
            M = self._M()
            sl_override = None
            if self.hires_glasses_parse:
                t0 = time.perf_counter()
                sl_override = _hires_parse_labels(self.parser, frame_native, W, H)
                timings["hires_glasses_parse_ms"] = (time.perf_counter() - t0) * 1000

            # Part A candidate 2 (temporal hysteresis, opt-in, default off):
            # needs THIS frame's own sl to hand to the NEXT call as its
            # "previous frame" reference. Only computed when the flag is
            # on -- one extra BiSeNet call per snapshot, paid only by
            # whoever opts in. self._aaband_prev_sl persists across calls
            # on this PipelineState instance; None on the first snapshot
            # (falls back to no-op inside build_r8, see that gate's code).
            aaband_sl_prev = None
            sl_for_hysteresis = None
            if aaband_temporal_hysteresis:
                aaband_sl_prev = getattr(self, "_aaband_prev_sl", None)
                sl_for_hysteresis = sl_override if sl_override is not None else self.parser.parse(frame_bgr)

            # frames_dir/avatar_frames_dir are passed as explicit per-call
            # arguments (blend_v3.build_r8/composite_frame_v3), never via a
            # module-level bv3.FRAMES_DIR reassignment -- a prior version of
            # this code monkey-patched bv3.FRAMES_DIR for the duration of this
            # call, which is a shared-module-attribute race between any two
            # PipelineState instances/threads live in the same process (one
            # instance's snapshot could read back another's driver frame via
            # the globally-visible FRAMES_DIR value). See module docstring.
            build_kwargs = dict(render_canvas_h=CANVAS_H, avatar_frames_dir=str(self.avatar_tmp_dir),
                                 frames_dir=str(self.driver_tmp_dir),
                                 drv_img=frame_bgr, avatar_rgba_img=avatar_rgba,
                                 rim3d_injection=False, erase_feather_clamp=False, reg=LIVE_REG,
                                 sl_override=sl_override,
                                 aaband_content_gate=aaband_content_gate,
                                 aaband_av_lo=aaband_av_lo, aaband_av_hi=aaband_av_hi,
                                 aaband_temporal_hysteresis=aaband_temporal_hysteresis,
                                 aaband_sl_prev=aaband_sl_prev,
                                 aaband_hairline_exempt_row_px=aaband_hairline_exempt_row_px)
            t0 = time.perf_counter()
            stats = bv3.precompute_stats(self.parser, M, self.plate_stat, lm, n_frames=1,
                                          force=True, verbose=False, stats_cache=self.stats_tmp_npz,
                                          build_kwargs=build_kwargs, reg=LIVE_REG,
                                          avatar_stats_mask=self.avatar_stats_mask)
            timings["blend_stats_ms"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            composited_bgr = bv3.composite_frame_v3(
                FRAME_IDX, self.parser, M, self.plate_stat, lm, stats,
                render_canvas_h=CANVAS_H, avatar_frames_dir=str(self.avatar_tmp_dir),
                frames_dir=str(self.driver_tmp_dir),
                drv_img=frame_bgr, avatar_rgba_img=avatar_rgba,
                rim3d_injection=False, erase_feather_clamp=False, reg=LIVE_REG,
                avatar_blur_match=bv3.AVATAR_BLUR_MATCH_DEFAULT,
                glasses_erase_extend=bv3.GLASSES_ERASE_EXTEND_DEFAULT,
                sl_override=sl_override,
                aaband_content_gate=aaband_content_gate,
                aaband_av_lo=aaband_av_lo, aaband_av_hi=aaband_av_hi,
                aaband_temporal_hysteresis=aaband_temporal_hysteresis,
                aaband_sl_prev=aaband_sl_prev,
                aaband_hairline_exempt_row_px=aaband_hairline_exempt_row_px)
            timings["blend_composite_ms"] = (time.perf_counter() - t0) * 1000
            if aaband_temporal_hysteresis:
                self._aaband_prev_sl = sl_for_hysteresis
            timings["blend_ms"] = (timings["blend_io_ms"] + timings["blend_stats_ms"]
                                    + timings["blend_composite_ms"]
                                    + timings.get("hires_glasses_parse_ms", 0.0))
            timings["total_ms"] = (timings["track_ms"] + timings["fit3d_ms"]
                                    + timings["render_ms"] + timings["blend_ms"])
            return composited_bgr, timings

    def reset_tmp(self):
        """Wipe THIS instance's own scratch dir only (session-ID-scoped since
        the DRIVER_TMP_DIR/AVATAR_TMP_DIR cross-process collision fix -- see
        module docstring). Safe to call any time; process_snapshot() recreates
        driver_tmp_dir/avatar_tmp_dir on demand regardless."""
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
        self.driver_tmp_dir.mkdir(parents=True, exist_ok=True)
        self.avatar_tmp_dir.mkdir(parents=True, exist_ok=True)


def _ensure_size(frame_bgr, w, h):
    """Fit frame_bgr into exactly (w, h), the fixed working resolution every
    downstream stage (BiSeNet, fit3d's calibration, blend_v3's REG/geometry
    constants) assumes. getUserMedia can negotiate any resolution/aspect ratio
    depending on the camera, and an arbitrary input video may not match (w, h)
    either -- a plain resize to (w, h) would stretch non-uniformly whenever the
    input aspect ratio differs (confirmed: a 1620x1080 source against a
    1280x720 target distorts ~18.5% more on one axis than the other), which
    throws off fit3d's landmark-based calibration and every hardcoded pixel
    constant downstream. Scale uniformly (same factor both axes, whichever
    axis is the tighter fit) and letterbox/pillarbox-pad the rest with black,
    so the true aspect ratio is preserved and nothing is cropped. When the
    input already matches (w, h) -- the expected case for a session negotiated
    at this resolution -- this is a no-op, identical to the previous code."""
    src_h, src_w = frame_bgr.shape[:2]
    if src_w == w and src_h == h:
        return frame_bgr
    scale = min(w / src_w, h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=frame_bgr.dtype)
    x0, y0 = (w - new_w) // 2, (h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _hires_parse_labels(parser, frame_native, w, h):
    """Task B1 opt-in fix: parse the RAW native frame directly (one resize,
    straight to BiSeNet's own 512x512 input -- same as offline's single-
    resize path), then remap the resulting label map into the (h, w)
    working canvas using the EXACT scale+pad transform _ensure_size applies
    to the RGB frame, so it stays pixel-aligned with drv/erase_bin/etc.
    Labels are categorical -- remap with INTER_NEAREST, never INTER_AREA/
    INTER_LINEAR, and pad with 0 (BiSeNet's background class) to match
    _ensure_size's black-pixel padding, which also parses as background."""
    labels_native = parser.parse(frame_native)
    src_h, src_w = frame_native.shape[:2]
    if src_w == w and src_h == h:
        return labels_native
    scale = min(w / src_w, h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    resized = cv2.resize(labels_native, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((h, w), dtype=labels_native.dtype)
    x0, y0 = (w - new_w) // 2, (h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def reset_session_tmp():
    """Deprecated no-op, kept for backward compatibility with existing call
    sites (server.py, several diagnostic scripts). Scratch dirs are now
    session-ID-scoped per PipelineState instance (see that class's
    driver_tmp_dir/avatar_tmp_dir/reset_tmp()) instead of a single shared
    on-disk path, so there is no longer a global path to reset here -- each
    instance's tmp dir is freshly created under its own uuid-suffixed
    session_dir and never shared with any other process or instance."""
    pass


def list_sessions():
    """Session ids with persisted calibration under SESSION_DATA_ROOT, newest
    first (the id itself sorts chronologically -- see PipelineState.__init__)."""
    if not SESSION_DATA_ROOT.exists():
        return []
    return sorted((p.name for p in SESSION_DATA_ROOT.iterdir() if p.is_dir()), reverse=True)


def load_session_calibration(session_id_or_dir):
    """Load a previously-persisted session's plate/pose calibration for
    offline diagnostic/debugging use (e.g. reproducing a live composite
    without needing the original server process still running). Returns a
    dict with whichever of 'plate_stat'/'reg'/'s_cal'/'t_cal'/'T_prev' were
    actually calibrated+saved for that session; a piece that was never
    calibrated (or a session with no persisted data at all) is simply absent
    from the returned dict rather than raising."""
    session_dir = Path(session_id_or_dir)
    if not session_dir.is_dir():
        session_dir = SESSION_DATA_ROOT / session_id_or_dir
    out = {}
    plate_npz = session_dir / "plate_calibration.npz"
    if plate_npz.exists():
        with np.load(str(plate_npz)) as d:
            out["plate_stat"] = d["plate_stat"]
            out["reg"] = tuple(d["reg"].tolist())
    pose_npz = session_dir / "pose_calibration.npz"
    if pose_npz.exists():
        with np.load(str(pose_npz)) as d:
            out["s_cal"] = float(d["s_cal"])
            out["t_cal"] = d["t_cal"]
            out["T_prev"] = d["T_prev"]
    return out
