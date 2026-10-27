"""
Local FLAME jaw/ear-hinge peripheral Gaussian OPACITY inflation
(Phase 3, Pathway 3 -- "ear-to-jaw hinge" render-side fix, attempt #3).

Context: task3 (jaw_expand.py, commit 24a96d9) showed FLAME mesh vertex
position (xyz) is inert -- the mesh Gaussians' silhouette contribution
doesn't move with their center. task4 (jaw_inflate.py, commit 3c5b4d6)
showed their rasterizer 'scale' is NOT inert -- widening it does spread
low-alpha coverage further -- but measured the mesh Gaussians' own
opacity directly for the first time (mean 0.6%, max 12% in the jaw/ear-
hinge selection) and found that ceiling is what caps scale inflation's
usefulness: 3DGS alpha = opacity * gaussian_falloff(distance), so no
pixel can exceed roughly that Gaussian's own opacity regardless of how
wide its footprint is. Scale inflation only ever produced a translucent
haze, never opaque coverage.

This module tests the remaining renderer-level lever directly implicated
by that ceiling: the mesh Gaussians' OPACITY itself, for the same
jaw/chin/ear-hinge vertex selection (reused unchanged from
jaw_expand.compute_jaw_selection -- not re-derived). If raising opacity
toward 1.0 lets these Gaussians actually paint visible, opaque-ish
content into the a_av==0 hole, that would be a real fix. If it instead
produces overbright/double-exposure spots where an inflated Gaussian's
now-high opacity overlaps an already-high-opacity facial Gaussian, or
hard-edged popping (opacity, unlike scale, does not spatially smear --
a Gaussian's spatial falloff shape is unchanged, only its peak alpha
value moves), that is a different failure mode from task3/task4's and
should be reported as such, not folded into "still a haze."

gs_params['opacities'] (like 'scales', 'colors', 'rotations') is computed
ONCE by gs_generator_g on the first model.forward_expression() call and
cached on model._gs_params; only 'xyz' is overwritten per frame. This
module's inflation is therefore also a one-shot mutation, applied
immediately after that cache is built (same injection point as
jaw_inflate.inflate_jaw_gaussian_scales).
"""
import torch

JAW_OPACITY_FACTOR_DEFAULT = 5.0  # conservative: mean 0.006 -> ~0.03 at full weight


def inflate_jaw_gaussian_opacity(gs_params: dict, jaw_idx_t: torch.Tensor,
                                  jaw_weight_t: torch.Tensor, factor: float) -> torch.Tensor:
    """
    In-place multiply the mesh Gaussians' 'opacities' for the jaw/chin/
    ear-hinge selection, weighted by jaw_weight_t so the multiplier fades
    to 1.0 (no-op) at the selection boundary, then clamp to the valid
    [0,1] opacity range (unlike scale, which has no natural upper bound,
    opacity is a probability-like value the rasterizer expects in [0,1]
    -- sigmoid-activated at generation time).

    gs_params: model._gs_params dict (mutated in place). 'opacities' has
        shape (1, N_total, 1); the first 5023 entries along dim=1 are the
        mesh Gaussians, one per FLAME vertex, same ordering as jaw_idx_t.
    jaw_idx_t: (K,) int64 tensor, mesh-vertex indices (0-5022), from
        jaw_expand.compute_jaw_selection.
    jaw_weight_t: (K,) float32 tensor in (0,1], same selection.
    factor: opacity multiplier at full weight (w=1). Effective multiplier
        per vertex = 1 + (factor-1)*weight. A large factor (e.g. 200x)
        against opacities of ~0.006 will clamp to ~1.0 at full weight --
        used as the ablation "ceiling" test to check whether opacity
        itself has a further, non-obvious ceiling (e.g. rasterizer-side
        clipping, or a compositing order effect) once pushed to its max.

    Returns the mutated 'opacities' tensor (already mutated in place).
    """
    op = gs_params['opacities']
    mult = (1.0 + (factor - 1.0) * jaw_weight_t).to(op.dtype)  # (K,)
    jaw_op = op.index_select(1, jaw_idx_t)                     # (1,K,1)
    new_op = (jaw_op * mult.view(1, -1, 1)).clamp(0.0, 1.0)
    op.index_copy_(1, jaw_idx_t, new_op)
    return op
