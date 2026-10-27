#!/usr/bin/env python
"""
Phase 3D-3: motion-adaptive per-channel filter over {rotation, T_z (scale-
equivalent), T_xy (translation), jaw, expression}, per PLAN_3D.md's Phase 3D-3
spec -- each channel independently opt-in with its own cutoff, gated on BOTH
this project's standing signal-level flip-rate/magnitude test AND a hard
lag/smear check on this clip's two real-motion events (the head turn, frames
28-45; the fast mouth-open, frames 116-118), reproducing the SS35 precedent
test explicitly rather than assuming it transfers.

*** RESULT (v3/_stabilize_gate_check.py, 2026-09-04): every one of the 5
channels was swept across 10 OneEuroFilter configs (the 6 candidates this
project has always tried first, plus 4 progressively gentler ones added this
round to map the tradeoff). ALL 10 clear the signal-level gate (flip-rate and
max-delta both improve over raw at every config tried -- that gate is easy to
clear). NONE clear the lag/smear gate at a config that still provides
meaningful jitter reduction: lag during the real turn/mouth-open events scales
down as the cutoff is loosened, but so does the jitter benefit, and even the
gentlest config tried (mincutoff=8.0, beta=0.05 -- barely more than raw) still
shows real, visible lag (rotation: 5.9deg out of a ~38deg turn even at the
MOST RESPONSIVE config tried, mincutoff=1.0; jaw: filtered lags the real
mouth-open event by an amount comparable to the entire real motion range at
every config up to mincutoff=3.0). See NOTES_3D0_findings.md's Phase 3D-3
section for the full per-channel numbers and the lag-gate overlay plots
(v3/debug_out/3d3/lag_gate_turn_window.png, lag_gate_jaw_window.png).

Per the brief's own hard-gate rule ("Any channel that shows visible lag or
smear on these frames gets shipped unfiltered, with the reason recorded"):
ALL FIVE CHANNELS SHIP RAW/UNFILTERED THIS ROUND. This is a real, if
unglamorous, result -- not a placeholder. The infrastructure below (OneEuroFilter,
per-channel config, apply_stabilization()) stays in place so a future channel-
specific approach (a predictive/lookahead filter, a differently-shaped adaptive
cutoff, or accepting a small amount of jitter reduction at a documented lag
cost for a specific channel) can be dropped in without re-plumbing anything --
but nothing is enabled by default, and CONFIG below says why for every channel.

Run in the GAGAvatar venv:
    third_party/GAGAvatar/venv/Scripts/python.exe v3/stabilize.py
"""
from pathlib import Path

import numpy as np

_HEADSWAP_ROOT = Path(__file__).resolve().parent.parent
TRACK_NPZ = _HEADSWAP_ROOT / "v3" / "debug_out" / "3d1" / "track_results.npz"
FIT3D_NPZ = _HEADSWAP_ROOT / "v3" / "debug_out" / "3d2" / "fit3d_results.npz"
OUT_DIR = _HEADSWAP_ROOT / "v3" / "debug_out" / "3d3"
FPS = 30.0


class OneEuroFilter:
    """Verbatim copy of this project's shipped class (Casiez et al. 2012) --
    same as compositing/_sweep_one_euro_filter.py, _sweep_one_euro_rotation.py,
    and v3/_stabilize_gate_check.py's own copy -- kept in sync by convention,
    not by import, to match the standing pattern in this repo's research
    scripts (each diagnostic is meant to stand alone / be re-runnable in
    isolation)."""
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()
        te = 1.0 / self.freq
        dx = (x - self.x_prev) / te
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def filter_series(series, mincutoff, beta, freq=FPS):
    f = OneEuroFilter(freq=freq, mincutoff=mincutoff, beta=beta)
    return np.stack([f.filter(series[i]) for i in range(len(series))])


class MotionAdaptiveOneEuroFilter:
    """OneEuroFilter whose BASELINE cutoff is driven by an external per-frame
    speed signal (here |d(yaw)/dt|, deg/frame) instead of being one fixed
    constant for the whole clip.

    Phase 3D-3 (CONFIG above) tried one constant (mincutoff, beta) for the
    entire clip and hit an unavoidable tradeoff: gentle enough to leave the
    turn responsive == too gentle to touch the at-rest shimmer; strong enough
    for the shimmer == visible lag entering/leaving the turn. That tradeoff is
    inherent to a CONSTANT cutoff, not to filtering per se -- it has to pick one
    operating point for two regimes that need opposite ones.

    This variant removes the single-constant constraint: cutoff(f) ramps from
    `cutoff_still` (heavy smoothing) up to `cutoff_moving` (effectively pass-
    through) as `drive[f]` crosses [drive_lo, drive_hi]. Because `drive` is an
    EXTERNAL, already-known-per-frame signal (yaw rate from the full clip's own
    tracking, not something inferred from the channel being filtered), the
    filter opens up as soon as drive crosses the threshold -- there is no
      detect-then-relax lag the way a signal's own derivative-adaptive cutoff
    (the `beta` term below, standard OneEuroFilter behaviour, kept as a small
    secondary safety net) would have.  This is an offline batch pipeline over a
    fixed, fully-tracked clip (same as every `np.gradient(yaw)` used elsewhere
    in this project's diagnostics), so `drive` may look one frame into the
    future at a turn's leading edge -- there is no live/causal constraint here.
    """
    def __init__(self, freq, cutoff_still, cutoff_moving, drive_lo, drive_hi,
                 beta=0.0, dcutoff=1.0):
        assert cutoff_moving > cutoff_still > 0 and drive_hi > drive_lo
        self.freq = freq
        self.cutoff_still = cutoff_still
        self.cutoff_moving = cutoff_moving
        self.drive_lo = drive_lo
        self.drive_hi = drive_hi
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, drive):
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()
        te = 1.0 / self.freq
        dx = (x - self.x_prev) / te
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        w = np.clip((drive - self.drive_lo) / (self.drive_hi - self.drive_lo), 0.0, 1.0)
        base_cutoff = self.cutoff_still + w * (self.cutoff_moving - self.cutoff_still)
        cutoff = base_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def filter_series_motion_adaptive(series, drive, cutoff_still, cutoff_moving,
                                   drive_lo, drive_hi, beta=0.0, dcutoff=1.0, freq=FPS):
    f = MotionAdaptiveOneEuroFilter(freq=freq, cutoff_still=cutoff_still,
                                     cutoff_moving=cutoff_moving, drive_lo=drive_lo,
                                     drive_hi=drive_hi, beta=beta, dcutoff=dcutoff)
    return np.stack([f.filter(series[i], drive[i]) for i in range(len(series))])


# ---------------------------------------------------------------------------
# Motion-adaptive re-attempt (2026-09-11, STAGED -- opt-in, default OFF).
# Flicker-source diagnosis (FLICKSRC_* this session) found the at-rest shimmer
# (tail f100-244, |dyaw/dt|~0) is driven by fit3d's per-frame-independent T
# solve (std-beta +0.45), raw expression (+0.30), and residual raw pose
# (+0.25) -- NOT by fit3d's rotation output (R_cam partial r=-0.10 controlling
# for raw global_rot). So: filter T + jaw + expression only; leave rotation /
# R_cam / global_rot completely untouched (the channel Phase 3D-3 already
# proved fails the lag gate, and the diagnosis says isn't the contributor
# anyway). drive_lo/drive_hi bracket the turn-onset ramp (turn windows peak
# ~1.8-4.6 deg/frame, the tail sits at ~0.0-0.3) so filtering is wide open
# well before a turn is underway.
# ---------------------------------------------------------------------------
MOTION_ADAPTIVE_CONFIG = {
    'T':          dict(cutoff_still=0.5, cutoff_moving=30.0, drive_lo=0.2, drive_hi=1.0, beta=0.0),
    'jaw':        dict(cutoff_still=0.5, cutoff_moving=30.0, drive_lo=0.2, drive_hi=1.0, beta=0.0),
    'expression': dict(cutoff_still=0.5, cutoff_moving=30.0, drive_lo=0.2, drive_hi=1.0, beta=0.0),
}


# Per-channel config. `enabled=False` for every channel this round -- see the
# module docstring for the lag-gate evidence. `mincutoff`/`beta` are recorded
# anyway (the best-lag candidate tried for that channel) so a future revisit
# has a documented starting point instead of guessing again from scratch.
CONFIG = {
    'rotation': dict(enabled=False, mincutoff=1.0, beta=0.005,
                      reason='Best candidate tried still lags the real turn by 5.9deg '
                             '(out of ~38deg) -- reproduces SS35 exactly on this clip\'s own signal.'),
    'T_z': dict(enabled=False, mincutoff=1.0, beta=0.005,
                reason='Best candidate lags the turn window by 0.28 FLAME units (33% of the '
                       'real T_z swing) and the jaw window by 54% of its swing -- T_z visibly '
                       'couples to both real motion events (per Step 2\'s own finding), and no '
                       'tested cutoff separates "noise" from that real coupled motion cleanly.'),
    'T_xy': dict(enabled=False, mincutoff=1.0, beta=0.005,
                 reason='Best candidate lags the turn window by MORE than the window\'s own raw '
                        'range (residual lag carried in from before frame 28) -- clear fail.'),
    'jaw': dict(enabled=False, mincutoff=1.0, beta=0.005,
                reason='On the actual fast mouth-open event (frames 116-118) the filtered signal '
                       'lags by an amount comparable to the ENTIRE real motion range even at the '
                       'most responsive config tried -- this is exactly the smear the lag gate '
                       'exists to catch, despite this channel having a "shipped default" in this '
                       'project\'s pre-3D history; that history\'s own gate apparently did not '
                       'test this specific window this hard. Flagged as a real discrepancy with '
                       'prior work, not silently overridden -- see NOTES_3D0_findings.md.'),
    'expression': dict(enabled=False, mincutoff=1.0, beta=0.005,
                        reason='Same fast-mouth-open window shows expression lagging the real '
                               'spike/recovery by ~40-70% of its own range at every cutoff tried.'),
}


def apply_stabilization(track_npz_path=TRACK_NPZ, fit3d_npz_path=FIT3D_NPZ,
                         motion_adaptive=False, ma_config=None):
    """Returns a dict of per-channel arrays, filtered where CONFIG[...]['enabled']
    is True, raw (pass-through) otherwise -- this round, that's every channel
    under the old constant-cutoff CONFIG.

    `motion_adaptive=True` (opt-in, default False) applies the yaw-rate-driven
    MotionAdaptiveOneEuroFilter (see above) to T / jaw / expression instead --
    takes precedence over the old constant-cutoff CONFIG entries for those same
    three channels (which stay False by default anyway, so there's no overlap
    in practice). `global_rot` is NEVER filtered by either path; R_cam is not
    part of this dict at all (blend_v3/render_gaga apply T on top of the raw
    fit3d rotation column, never touching it)."""
    d = np.load(track_npz_path)
    fit3d = np.load(fit3d_npz_path)
    out = {
        'global_rot': d['global_rot'].copy(),
        'jaw': d['jaw'].copy(),
        'expression': d['expression'].copy(),
        'T': fit3d['T'].copy(),  # (N,3) = (T_x, T_y, T_z)
    }
    for name in ('jaw', 'expression'):
        cfg = CONFIG[name]
        if cfg['enabled']:
            out[name] = filter_series(out[name], cfg['mincutoff'], cfg['beta'])
    if CONFIG['T_z']['enabled']:
        cfg = CONFIG['T_z']
        out['T'][:, 2] = filter_series(out['T'][:, 2:3], cfg['mincutoff'], cfg['beta'])[:, 0]
    if CONFIG['T_xy']['enabled']:
        cfg = CONFIG['T_xy']
        out['T'][:, :2] = filter_series(out['T'][:, :2], cfg['mincutoff'], cfg['beta'])
    if CONFIG['rotation']['enabled']:
        raise NotImplementedError(
            'rotation filtering is disabled by CONFIG and was never wired up beyond the '
            'gate-check diagnostic (quaternion route in _stabilize_gate_check.py) -- '
            'flip CONFIG["rotation"]["enabled"] only after porting that logic here.')

    if motion_adaptive:
        cfg_all = ma_config or MOTION_ADAPTIVE_CONFIG
        yaw_deg = np.degrees(d['global_rot'][:, 1].astype(np.float64))
        yaw_rate = np.abs(np.gradient(yaw_deg))
        out['T'] = filter_series_motion_adaptive(out['T'], yaw_rate, freq=FPS, **cfg_all['T'])
        out['jaw'] = filter_series_motion_adaptive(out['jaw'], yaw_rate, freq=FPS, **cfg_all['jaw'])
        out['expression'] = filter_series_motion_adaptive(out['expression'], yaw_rate, freq=FPS,
                                                            **cfg_all['expression'])
        out['_yaw_rate'] = yaw_rate  # diagnostic passthrough, not consumed by the renderer

    return out


# ---------------------------------------------------------------------------
# V46 full-transform zero-phase smoothing (2026-09-25, STAGED -- opt-in,
# default OFF, wired into render_gaga.py's --transform-smooth flag). See
# project memory project_v46_transform_smoothing_positive.md and
# debug_out/task2c3r4/verify_V46/FINDINGS_V46.md: a 3-frame zero-phase FIR
# (0.25/0.50/0.25, verbatim V39 kernel, no group delay by construction) on the
# FULL per-frame camera transform -- rotation (continuity-fixed quaternion,
# not Euler angles, to avoid V38's wraparound bug) AND translation T -- is a
# materially different candidate from CONFIG['rotation']'s already-rejected
# causal OneEuroFilter (which lagged a real ~38deg turn by 5.9deg). Validated
# PASS on a 40-frame calm/turn/mouth short window: raw-avatar silhouette
# shimmer down 33-56%, final-composite shimmer down 12-35%, landmark
# trajectory matches unfiltered to sub-pixel/sub-0.1deg with zero measured
# lag in all three windows. Full 245-frame regression + video review is the
# gate before this can default on.
# ---------------------------------------------------------------------------
def smooth_zero_phase_3tap(x):
    """Symmetric 3-tap zero-phase FIR (0.25/0.50/0.25), replicate-pad edges.
    Verbatim V39/V46 kernel. Works along axis 0 for any (N,) or (N,k) array."""
    n = len(x)
    xp = np.empty((n + 2,) + x.shape[1:], dtype=x.dtype)
    xp[1:-1] = x
    xp[0] = x[0]
    xp[-1] = x[-1]
    return 0.25 * xp[:-2] + 0.50 * xp[1:-1] + 0.25 * xp[2:]


def _fix_quat_continuity(quats):
    q = quats.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def smooth_full_transform_zero_phase(transform, T):
    """V46: zero-phase 3-tap smoothing of the FULL per-frame camera transform
    -- rotation (via continuity-fixed quaternion representation) and
    translation `T` -- with no group delay by construction. `transform` is
    fit3d_results.npz['transform']-shaped (N,3,4); `T` is the translation
    column to smooth on top of (production's motion-adaptive-filtered T if
    that's enabled upstream, else raw fit3d T -- this function doesn't care
    which, it just smooths whatever's passed in). Returns a new (N,3,4)
    array; does not mutate inputs. Rotation is always smoothed from
    `transform`'s own R_cam column, regardless of what T is passed."""
    from scipy.spatial.transform import Rotation
    R_raw = transform[:, :, :3]
    rot = Rotation.from_matrix(R_raw)
    quats = _fix_quat_continuity(rot.as_quat())
    quats_smooth = smooth_zero_phase_3tap(quats)
    quats_smooth /= np.linalg.norm(quats_smooth, axis=1, keepdims=True)
    R_smooth = Rotation.from_quat(quats_smooth).as_matrix().astype(R_raw.dtype)
    T_smooth = smooth_zero_phase_3tap(np.asarray(T, dtype=np.float64)).astype(transform.dtype)
    out = np.zeros_like(transform)
    out[:, :, :3] = R_smooth
    out[:, :, 3] = T_smooth
    return out


# ---------------------------------------------------------------------------
# Motion-gated camera deadband/pose lock (2026-09-26, STAGED -- opt-in,
# default OFF, wired into render_gaga.py's --transform-deadband-lock flag).
# V46's constant 3-tap zero-phase FIR (see project_v46_closed_imperceptible)
# only ever damps R_cam/T jitter by ~30%, and was full-clip regressed +
# blind-tested and closed as real-but-imperceptible -- a stronger constant
# filter is explicitly out of scope. This round's own Round-3 causal ablation
# (freezing R_cam/T outright during f115-185) measured 100% elimination of
# the deep-neck ROI jitter, not ~30% -- a genuinely different mechanism
# (hard lock during detected stillness, hysteresis-gated by frame-to-frame
# rotation/translation velocity) rather than a stronger/gentler version of
# the same always-on FIR kernel V46_CLOSED already ruled out.
# ---------------------------------------------------------------------------
def transform_frame_deltas(transform):
    """Per-frame rotation-angle delta (degrees, quaternion-geodesic) and
    translation delta (L2 norm) between consecutive frames. delta[0]=0."""
    from scipy.spatial.transform import Rotation
    R = transform[:, :, :3]
    T = transform[:, :, 3]
    rot = Rotation.from_matrix(R)
    q = _fix_quat_continuity(rot.as_quat())
    dot = np.clip(np.abs((q[1:] * q[:-1]).sum(axis=1)), 0.0, 1.0)
    rot_delta_deg = np.degrees(2 * np.arccos(dot))
    trans_delta = np.linalg.norm(T[1:] - T[:-1], axis=1)
    rot_delta_deg = np.concatenate([[0.0], rot_delta_deg])
    trans_delta = np.concatenate([[0.0], trans_delta])
    return rot_delta_deg, trans_delta


# Hysteresis thresholds: chosen from this clip's own rot/trans-delta
# distribution (see task_diag_hairline_flicker diagnostics) -- f115-185
# (the "held still" window) has rot_delta mean=0.37deg/frame, p90=0.72,
# max=1.04; the shallowest real turn (f15-44) has rot_delta mean=1.65,
# min-of-window well above 1.0. lo/hi give a hysteresis band between those
# two regimes so the lock doesn't chatter on borderline frames.
DEADBAND_LOCK_CONFIG = dict(rot_thresh_lo=0.6, trans_thresh_lo=0.04,
                             rot_thresh_hi=1.0, trans_thresh_hi=0.06)


def lock_transform_motion_gated(transform, rot_delta_deg, trans_delta,
                                 rot_thresh_lo, trans_thresh_lo,
                                 rot_thresh_hi, trans_thresh_hi):
    """Sample-and-hold hysteresis lock: once frame-to-frame rotation AND
    translation velocity both drop below the `_lo` thresholds, the transform
    is held constant (locked to the value at the frame the lock engaged,
    NOT continuously re-anchored) until EITHER velocity rises above its
    `_hi` threshold, at which point raw pass-through resumes immediately
    (no crossfade -- see render_gaga.py's --transform-deadband-lock
    docstring for why this is checked explicitly for snap artifacts at
    unlock). Returns a new (N,3,4) array; does not mutate the input."""
    n = transform.shape[0]
    out = transform.copy()
    locked = False
    for f in range(1, n):
        still = (rot_delta_deg[f] < rot_thresh_lo) and (trans_delta[f] < trans_thresh_lo)
        moving = (rot_delta_deg[f] > rot_thresh_hi) or (trans_delta[f] > trans_thresh_hi)
        if locked:
            if moving:
                locked = False
                out[f] = transform[f]
            else:
                out[f] = out[f - 1]  # held at the anchor value
        else:
            if still:
                locked = True
                out[f] = out[f - 1]  # lock engages, anchored to the previous frame's value
            else:
                out[f] = transform[f]
    return out


def _slerp(q0, q1, w):
    """Shortest-path quaternion slerp (xyzw), w in [0,1]. Handles the
    double-cover sign ambiguity itself (does not assume q0/q1 already share
    a hemisphere) and falls back to linear+renormalize for near-identical
    inputs to avoid a 0/0 from arccos near dot=1."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + w * (q1 - q0)
        return result / np.linalg.norm(result)
    theta0 = np.arccos(dot)
    theta = theta0 * w
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


# Default ramp length for the soft-release lock, in frames. 6-10 is the
# user-specified range; 8 (~267ms @30fps) picked as the midpoint.
SOFT_RELEASE_RAMP_FRAMES = 8


def lock_transform_motion_gated_soft_release(transform, rot_delta_deg, trans_delta,
                                              rot_thresh_lo, trans_thresh_lo,
                                              rot_thresh_hi, trans_thresh_hi,
                                              ramp_frames=SOFT_RELEASE_RAMP_FRAMES):
    """Same hysteresis-gated hard lock as lock_transform_motion_gated, EXCEPT
    the release (LOCKED -> moving-detected) is not an instant cut back to the
    live pose: it blends from the locked anchor pose to the live (per-frame,
    still-moving) tracked pose over `ramp_frames` frames using a cubic
    ease-out weight (w=0 at the anchor, w=1 once fully live), rotation via
    quaternion slerp (`_slerp`, shortest path) and translation via linear
    interpolation. ENGAGE (passthrough -> locked) stays instant (0.00px by
    construction, same as the hard-lock version) -- only RELEASE is smoothed.
    If a new `moving` condition is detected mid-ramp, the ramp is NOT
    restarted or extended -- it always completes in exactly `ramp_frames`
    frames from when release began, since the whole point is to bound the
    worst-case catch-up cost, not to keep re-anchoring. Returns a new
    (N,3,4) array; does not mutate the input."""
    from scipy.spatial.transform import Rotation
    n = transform.shape[0]
    R = transform[:, :, :3]
    T = transform[:, :, 3]
    quats = _fix_quat_continuity(Rotation.from_matrix(R).as_quat())

    out = transform.copy()
    out_quat = quats.copy()

    state = 'PASSTHROUGH'
    anchor_quat = anchor_T = None
    ramp_k = 0

    for f in range(1, n):
        still = (rot_delta_deg[f] < rot_thresh_lo) and (trans_delta[f] < trans_thresh_lo)
        moving = (rot_delta_deg[f] > rot_thresh_hi) or (trans_delta[f] > trans_thresh_hi)

        if state == 'LOCKED' and moving:
            state = 'RELEASING'
            ramp_k = 0
            anchor_quat = out_quat[f - 1].copy()
            anchor_T = out[f - 1, :, 3].copy()

        if state == 'LOCKED':
            out_quat[f] = out_quat[f - 1]
            out[f, :, :3] = out[f - 1, :, :3]
            out[f, :, 3] = out[f - 1, :, 3]
        elif state == 'RELEASING':
            ramp_k += 1
            w = 1.0 - (1.0 - min(ramp_k, ramp_frames) / ramp_frames) ** 3
            q_blend = _slerp(anchor_quat, quats[f], w)
            t_blend = (1 - w) * anchor_T + w * T[f]
            out_quat[f] = q_blend
            out[f, :, :3] = Rotation.from_quat(q_blend).as_matrix()
            out[f, :, 3] = t_blend
            if ramp_k >= ramp_frames:
                state = 'PASSTHROUGH'
        else:  # PASSTHROUGH
            if still:
                state = 'LOCKED'
                out_quat[f] = out_quat[f - 1]
                out[f, :, :3] = out[f - 1, :, :3]
                out[f, :, 3] = out[f - 1, :, 3]
            else:
                out_quat[f] = quats[f]
                out[f] = transform[f]

    return out.astype(transform.dtype)


# Empirically fit from the soft-release (fixed 8-frame cubic ease-out)
# causal-validation data (v3/_diag_soft_release_causal_validate.py run,
# 2026-09-26): step1_px ~= k_t * trans_drift_units + k_r * rot_drift_deg,
# least-squares over the 3 observed releases in f115-185 (f117/f127/f175).
# Used only to size default slew limits below; not applied at runtime.
#   k_t ~= 19.4 px per raw-translation-unit, k_r ~= 0.86 px per degree.
SLEW_MAX_TRANS_PER_FRAME = 0.023   # ~0.45px/frame budget @ k_t=19.4
SLEW_MAX_ROT_DEG_PER_FRAME = 0.35  # ~0.30px/frame budget @ k_r=0.86
SLEW_MAX_RAMP_FRAMES = 15


def lock_transform_motion_gated_slew_release(transform, rot_delta_deg, trans_delta,
                                              rot_thresh_lo, trans_thresh_lo,
                                              rot_thresh_hi, trans_thresh_hi,
                                              max_trans_per_frame=SLEW_MAX_TRANS_PER_FRAME,
                                              max_rot_deg_per_frame=SLEW_MAX_ROT_DEG_PER_FRAME,
                                              max_ramp_frames=SLEW_MAX_RAMP_FRAMES):
    """Same hysteresis-gated hard lock as lock_transform_motion_gated, EXCEPT
    RELEASE is neither an instant cut (lock_transform_motion_gated) nor a
    fixed-duration ramp (lock_transform_motion_gated_soft_release): it is a
    true slew-rate limiter. Each frame during RELEASE, the output moves at
    most `max_rot_deg_per_frame` degrees (quaternion-geodesic) and
    `max_trans_per_frame` units (L2) toward the CURRENT live tracked pose
    (a moving target, re-read every frame, not a frozen snapshot). Release
    duration is NOT fixed -- it scales automatically with however much drift
    accumulated during the lock: small drift catches up in 1-2 frames, large
    drift takes longer, up to `max_ramp_frames` after which the output is
    force-snapped to the live pose regardless of remaining distance (safety
    valve against a pathological unbounded release; expected to bind rarely,
    since drift keeps being logged so this can be checked post hoc). ENGAGE
    remains instant (0.00px by construction). Returns a new (N,3,4) array;
    does not mutate the input."""
    from scipy.spatial.transform import Rotation
    n = transform.shape[0]
    R = transform[:, :, :3]
    T = transform[:, :, 3]
    quats = _fix_quat_continuity(Rotation.from_matrix(R).as_quat())

    out = transform.copy()
    out_quat = quats.copy()

    state = 'PASSTHROUGH'
    ramp_k = 0

    for f in range(1, n):
        still = (rot_delta_deg[f] < rot_thresh_lo) and (trans_delta[f] < trans_thresh_lo)
        moving = (rot_delta_deg[f] > rot_thresh_hi) or (trans_delta[f] > trans_thresh_hi)

        if state == 'LOCKED' and moving:
            state = 'RELEASING'
            ramp_k = 0

        if state == 'LOCKED':
            out_quat[f] = out_quat[f - 1]
            out[f, :, :3] = out[f - 1, :, :3]
            out[f, :, 3] = out[f - 1, :, 3]
        elif state == 'RELEASING':
            ramp_k += 1
            prev_quat = out_quat[f - 1]
            prev_T = out[f - 1, :, 3]
            target_quat = quats[f]
            target_T = T[f]

            dot = np.clip(np.abs(np.dot(prev_quat, target_quat)), 0.0, 1.0)
            ang_deg = np.degrees(2 * np.arccos(dot))
            if ang_deg <= 1e-9 or ramp_k >= max_ramp_frames:
                new_quat = target_quat
            else:
                w = min(1.0, max_rot_deg_per_frame / ang_deg)
                new_quat = _slerp(prev_quat, target_quat, w)

            dvec = target_T - prev_T
            dist = np.linalg.norm(dvec)
            if dist <= max_trans_per_frame or dist == 0 or ramp_k >= max_ramp_frames:
                new_T = target_T
            else:
                new_T = prev_T + dvec * (max_trans_per_frame / dist)

            out_quat[f] = new_quat
            out[f, :, :3] = Rotation.from_quat(new_quat).as_matrix()
            out[f, :, 3] = new_T

            caught_up = (ang_deg <= max_rot_deg_per_frame) and (dist <= max_trans_per_frame)
            if caught_up or ramp_k >= max_ramp_frames:
                state = 'PASSTHROUGH'
        else:  # PASSTHROUGH
            if still:
                state = 'LOCKED'
                out_quat[f] = out_quat[f - 1]
                out[f, :, :3] = out[f - 1, :, :3]
                out[f, :, 3] = out[f - 1, :, 3]
            else:
                out_quat[f] = quats[f]
                out[f] = transform[f]

    return out.astype(transform.dtype)


SMOOTHSTEP_RELEASE_RAMP_FRAMES = 8


def lock_transform_motion_gated_smoothstep_release(transform, rot_delta_deg, trans_delta,
                                                     rot_thresh_lo, trans_thresh_lo,
                                                     rot_thresh_hi, trans_thresh_hi,
                                                     ramp_frames=SMOOTHSTEP_RELEASE_RAMP_FRAMES):
    """Same hysteresis-gated hard lock as lock_transform_motion_gated, and
    structurally identical to lock_transform_motion_gated_soft_release
    (fixed-duration RELEASE ramp, anchor pose frozen at release-trigger time,
    quaternion slerp for rotation + linear interp for T), EXCEPT the blend
    weight is smoothstep (w = 3t^2 - 2t^3, t = k/ramp_frames) instead of
    cubic ease-out (w = 1-(1-t)^3). Smoothstep has zero slope at BOTH t=0
    and t=1, so the release neither jumps at the moment of unlock (cubic
    ease-out's dw/dt|_{t=0} = 3/ramp_frames, its peak velocity, is exactly
    where the 2.66px f175 spike came from) nor bumps when it merges back
    into the live track. Interior velocity is higher than either endpoint by
    construction (peaks at t=0.5, dw/dt=1.5/ramp_frames -- 1.5x cubic
    ease-out's peak, so a slower motion window taken mid-ramp is possible;
    this is the tradeoff being tested, not assumed). ENGAGE stays instant.
    Returns a new (N,3,4) array; does not mutate the input."""
    from scipy.spatial.transform import Rotation
    n = transform.shape[0]
    R = transform[:, :, :3]
    T = transform[:, :, 3]
    quats = _fix_quat_continuity(Rotation.from_matrix(R).as_quat())

    out = transform.copy()
    out_quat = quats.copy()

    state = 'PASSTHROUGH'
    anchor_quat = anchor_T = None
    ramp_k = 0

    for f in range(1, n):
        still = (rot_delta_deg[f] < rot_thresh_lo) and (trans_delta[f] < trans_thresh_lo)
        moving = (rot_delta_deg[f] > rot_thresh_hi) or (trans_delta[f] > trans_thresh_hi)

        if state == 'LOCKED' and moving:
            state = 'RELEASING'
            ramp_k = 0
            anchor_quat = out_quat[f - 1].copy()
            anchor_T = out[f - 1, :, 3].copy()

        if state == 'LOCKED':
            out_quat[f] = out_quat[f - 1]
            out[f, :, :3] = out[f - 1, :, :3]
            out[f, :, 3] = out[f - 1, :, 3]
        elif state == 'RELEASING':
            ramp_k += 1
            t = min(ramp_k, ramp_frames) / ramp_frames
            w = 3 * t ** 2 - 2 * t ** 3
            q_blend = _slerp(anchor_quat, quats[f], w)
            t_blend = (1 - w) * anchor_T + w * T[f]
            out_quat[f] = q_blend
            out[f, :, :3] = Rotation.from_quat(q_blend).as_matrix()
            out[f, :, 3] = t_blend
            if ramp_k >= ramp_frames:
                state = 'PASSTHROUGH'
        else:  # PASSTHROUGH
            if still:
                state = 'LOCKED'
                out_quat[f] = out_quat[f - 1]
                out[f, :, :3] = out[f - 1, :, :3]
                out[f, :, 3] = out[f - 1, :, 3]
            else:
                out_quat[f] = quats[f]
                out[f] = transform[f]

    return out.astype(transform.dtype)


DAMPED_LOCK_DAMP_FACTOR = 0.9


def lock_transform_motion_gated_damped(transform, rot_delta_deg, trans_delta,
                                        rot_thresh_lo, trans_thresh_lo,
                                        rot_thresh_hi, trans_thresh_hi,
                                        damp_factor=DAMPED_LOCK_DAMP_FACTOR):
    """Same hysteresis STILL/MOVING classification as lock_transform_motion_gated,
    but instead of a 100% hard freeze during STILL windows, applies a
    continuous exponential-moving-average (EMA) damping filter: each STILL
    frame, the output moves only `(1 - damp_factor)` of the way from the
    previous OUTPUT toward the current raw tracked pose (rotation via
    quaternion slerp with weight `1 - damp_factor`, translation via linear
    interpolation with the same weight). At damp_factor=0.8 this attenuates
    frame-to-frame high-frequency jitter by ~80% while still letting ~20% of
    any real organic micro-sway (slow drift, not just noise) pass through
    every frame -- unlike the hard lock, which freezes at a single anchor
    value and lets ZERO of the real signal through until release. There is
    no separate RELEASING ramp/state: since the output never fully departs
    from the live signal (it just lags it, exponentially), transitioning
    back to MOVING is a straight pass-through with no frozen-anchor gap to
    ramp out of, so no snap-artifact risk to check the way the hard-lock/
    soft-release/slew/smoothstep variants required. Returns a new (N,3,4)
    array; does not mutate the input."""
    from scipy.spatial.transform import Rotation
    n = transform.shape[0]
    R = transform[:, :, :3]
    T = transform[:, :, 3]
    quats = _fix_quat_continuity(Rotation.from_matrix(R).as_quat())

    out = transform.copy()
    out_quat = quats.copy()
    w = 1.0 - damp_factor  # fraction of raw signal let through per STILL frame

    still_state = False
    for f in range(1, n):
        still = (rot_delta_deg[f] < rot_thresh_lo) and (trans_delta[f] < trans_thresh_lo)
        moving = (rot_delta_deg[f] > rot_thresh_hi) or (trans_delta[f] > trans_thresh_hi)
        if still_state:
            if moving:
                still_state = False
        else:
            if still:
                still_state = True

        if still_state:
            q_blend = _slerp(out_quat[f - 1], quats[f], w)
            t_blend = (1 - w) * out[f - 1, :, 3] + w * T[f]
            out_quat[f] = q_blend
            out[f, :, :3] = Rotation.from_quat(q_blend).as_matrix()
            out[f, :, 3] = t_blend
        else:
            out_quat[f] = quats[f]
            out[f] = transform[f]

    return out.astype(transform.dtype)


def main():
    print('CONFIG (Phase 3D-3 result -- all channels raw this round):')
    for name, cfg in CONFIG.items():
        print(f"  {name}: enabled={cfg['enabled']}  ({cfg['reason']})")
    result = apply_stabilization()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / 'stabilized_signals.npz', **result)
    print(f"\nSaved {OUT_DIR / 'stabilized_signals.npz'} (pass-through / raw for every channel)")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
