"""
Local FLAME jaw/ear-hinge peripheral Gaussian scale inflation
(Phase 3, Pathway 3 -- "ear-to-jaw hinge" render-side fix, attempt #2).

Context: task3 (jaw_expand.py, commit 24a96d9) proved that nudging the
FLAME mesh vertices' xyz position does essentially nothing to the
rasterized silhouette -- the a_av==0 coverage hole at the jaw/ear hinge
survived a 40mm displacement (5x the requested max) unchanged. The
conclusion drawn there was that the "mesh Gaussians" (the first 5023
entries of model._gs_params, one per FLAME vertex, produced by
gs_generator_g) are near-inert -- but that diagnosis was inferred from
xyz being ineffective, not from directly inspecting the mesh Gaussians'
own opacity/scale parameters.

This module tests a different lever on the SAME mesh-Gaussian population:
instead of moving their centers, it inflates their spatial footprint
(the 'scales' parameter fed to the 3DGS rasterizer) for the jaw/chin/
ear-hinge subset only, reusing the exact vertex selection already
derived in jaw_expand.compute_jaw_selection (ibug68 jawline polyline,
12mm radius, Gaussian falloff). If a mesh Gaussian's opacity is
non-negligible, a larger scale makes it cover more screen pixels at
partial alpha even without moving its center -- which could close a
narrow rasterization gap that xyz displacement could not, PROVIDED the
opacity ceiling for these Gaussians is actually non-zero. If opacity is
truly ~0 (as task3's result suggests), no scale multiplier will help
either, since 3DGS alpha = opacity * gaussian_falloff is capped by
opacity regardless of footprint size -- this module's caller should log
gs_params['opacities'][jaw_idx] before/after to check that directly
rather than assume.

gs_params['scales'] (and 'opacities', 'colors', 'rotations') for the
mesh-Gaussian population are computed ONCE by gs_generator_g on the
first model.forward_expression() call and cached on model._gs_params;
only 'xyz' is overwritten per frame (see render_gaga.py). This module's
inflation is therefore also a one-shot mutation, applied immediately
after that cache is built, not a per-frame operation.
"""
import torch

JAW_INFLATE_FACTOR_DEFAULT = 1.2  # +20% scale at full selection weight


def inflate_jaw_gaussian_scales(gs_params: dict, jaw_idx_t: torch.Tensor,
                                 jaw_weight_t: torch.Tensor, factor: float) -> torch.Tensor:
    """
    In-place multiply the mesh Gaussians' 'scales' (all 3 axes) for the
    jaw/chin/ear-hinge selection, weighted by jaw_weight_t so the
    multiplier fades to 1.0 (no-op) at the selection boundary -- avoiding
    a hard seam in Gaussian footprint size between touched and untouched
    vertices.

    gs_params: model._gs_params dict (mutated in place). 'scales' has
        shape (1, N_total, 3); the first 5023 entries along dim=1 are the
        mesh Gaussians, one per FLAME vertex, in the same vertex ordering
        as jaw_idx_t (see models.py: gs_params_g comes first in the
        torch.cat that builds the full gs_params).
    jaw_idx_t: (K,) int64 tensor, mesh-vertex indices (0-5022), from
        jaw_expand.compute_jaw_selection.
    jaw_weight_t: (K,) float32 tensor in (0,1], same selection.
    factor: inflation multiplier at full weight (w=1). Effective
        multiplier per vertex = 1 + (factor-1)*weight, so the untouched
        boundary (w->0) is exactly 1x.

    Returns the mutated 'scales' tensor (for convenience; already
    mutated in place).
    """
    scales = gs_params['scales']
    mult = (1.0 + (factor - 1.0) * jaw_weight_t).to(scales.dtype)  # (K,)
    jaw_scales = scales.index_select(1, jaw_idx_t)                 # (1,K,3)
    scales.index_copy_(1, jaw_idx_t, jaw_scales * mult.view(1, -1, 1))
    return scales
