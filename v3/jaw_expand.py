"""
Local FLAME jaw mesh expansion (Phase 3D, "ear-to-jaw hinge" render-side fix).

Context: the compositor sees a_av == 0 (no rasterized Gaussian coverage at
all) in a band running from the ear down along the jaw at high yaw -- up to
~50px of missing coverage that no 2D compositing trick can fabricate, since
the pixels were never rasterized in the first place (see
project_gaga_render_side_diag / project_billboard_scoped_costing in memory).
This module closes that hole at the source: it nudges the FLAME mesh's own
jaw/chin/ear-hinge vertices outward along their per-frame surface normal,
before the displaced vertices are handed to the 3DGS rasterizer, so the
render itself covers a few more millimetres past the (billboard-limited)
silhouette.

Vertex selection is NOT a guessed index range. FLAME_masks.pkl (the
standard per-region vertex segmentation shipped with FLAME) has no 'jaw' or
'chin' key at all, and no jaw/chin vertex list previously existed anywhere
in this codebase (verified by grepping v3/*.py) -- the only jaw-related
constant on file, blend_v3.JAW_IDX, indexes the 2D MediaPipe 468-point face
mesh, not the 3D 5023-vertex FLAME mesh, and is unrelated to this module.

Instead this module uses the FLAME model's OWN embedded ibug68 landmark
mapping (`full_lmk_faces_idx`/`full_lmk_bary_coords`, already loaded onto
every FLAMEModel instance in this pipeline) to locate the true 3D jawline
polyline (ibug68 landmarks 0-16, ear-to-ear via the chin) on the template
mesh, then selects every template vertex within JAW_SEL_RADIUS_M of that
polyline, weighted by a Gaussian falloff of its distance to the polyline so
the effect fades smoothly to ~0 well before the forehead/cheek/scalp
(verified offline against FLAME_masks.pkl: at the shipped 12mm radius, the
264 selected vertices land in the face/ear/neck/scalp masks, matching the
jaw-chin-ear-hinge region; none fall in forehead, nose, lips, or eye
regions -- see task3_report.txt for the verification numbers).
"""
import numpy as np
import torch

# ibug68 landmarks 0-16 are the jawline (ear -> chin -> ear); FLAME's own
# `full_lmk_faces_idx`/`full_lmk_bary_coords` buffers are the with_eye
# variant but share the same first 68 (and therefore first 17) landmarks
# as the base ibug68 embedding (verified byte-identical offline).
_N_JAW_LMKS = 17

JAW_SEL_RADIUS_M = 0.012   # 12mm: template-vertex selection cutoff around the jaw polyline
JAW_SEL_SIGMA_M = 0.006    # Gaussian falloff sigma (radius = 2*sigma -> weight ~0.135 at the edge)
JAW_DISPLACE_MM_DEFAULT = 8.0  # outward displacement at full weight (w=1), mid of the requested 5-12mm range


def _point_seg_dists(v: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from every row of v (N,3) to the segment a->b (3,)."""
    ab = b - a
    denom = max(float(np.dot(ab, ab)), 1e-12)
    t = np.clip(((v - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(v - proj, axis=1)


def compute_jaw_selection(v_template: np.ndarray, faces: np.ndarray,
                           lmk_faces_idx: np.ndarray, lmk_bary_coords: np.ndarray,
                           radius_m: float = JAW_SEL_RADIUS_M,
                           sigma_m: float = JAW_SEL_SIGMA_M):
    """
    Derive the jaw/chin/ear-hinge vertex selection + per-vertex Gaussian
    weight from this FLAME instance's OWN template + landmark embedding
    (so it stays correct if the checkpoint's topology ever changes).

    v_template: (5023,3) float, FLAME template vertices (flame_model.v_template).
    faces: (9976,3) int, FLAME face topology (flame_model.get_faces()).
    lmk_faces_idx: (>=17,) int, flame_model.full_lmk_faces_idx[0] (or any ibug68-ordered variant).
    lmk_bary_coords: (>=17,3) float, flame_model.full_lmk_bary_coords[0].

    Returns (jaw_idx: (K,) int64 ndarray, jaw_weight: (K,) float32 ndarray in (0,1]).
    """
    tri = faces[lmk_faces_idx[:_N_JAW_LMKS]]                    # (17,3) vertex ids
    bary = lmk_bary_coords[:_N_JAW_LMKS]                        # (17,3)
    jaw_poly = (v_template[tri] * bary[:, :, None]).sum(axis=1)  # (17,3) 3D jawline polyline

    n = v_template.shape[0]
    mind = np.full(n, np.inf, dtype=np.float64)
    for i in range(len(jaw_poly) - 1):
        d = _point_seg_dists(v_template, jaw_poly[i], jaw_poly[i + 1])
        mind = np.minimum(mind, d)

    jaw_idx = np.where(mind < radius_m)[0].astype(np.int64)
    jaw_weight = np.exp(-0.5 * (mind[jaw_idx] / sigma_m) ** 2).astype(np.float32)
    return jaw_idx, jaw_weight


def displace_jaw_vertices(t_points: torch.Tensor, faces_t: torch.Tensor,
                           jaw_idx_t: torch.Tensor, jaw_weight_t: torch.Tensor,
                           delta_m: float) -> torch.Tensor:
    """
    Push the selected jaw vertices outward along the mesh's OWN per-frame
    (posed) surface normal -- so the displacement direction rotates with
    jaw articulation/head pose, not the static rest-pose normal.

    t_points: (1,5023,3) float tensor, POSED (post-LBS) FLAME vertices for
        this frame -- the tensor produced right before it is written into
        model._gs_params['xyz'][:, :5023] in render_gaga.py.
    faces_t: (9976,3) int64 tensor, same device (flame_model.get_faces()).
    jaw_idx_t: (K,) int64 tensor, from compute_jaw_selection.
    jaw_weight_t: (K,) float32 tensor, from compute_jaw_selection.
    delta_m: outward displacement in metres at full weight (w=1).

    Returns a NEW (1,5023,3) tensor; does not mutate t_points in place.
    """
    v = t_points[0]  # (5023,3)
    v0 = v.index_select(0, faces_t[:, 0])
    v1 = v.index_select(0, faces_t[:, 1])
    v2 = v.index_select(0, faces_t[:, 2])
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # unnormalized; magnitude ~ 2*face area

    vertex_normals = torch.zeros_like(v)
    vertex_normals.index_add_(0, faces_t[:, 0], face_normals)
    vertex_normals.index_add_(0, faces_t[:, 1], face_normals)
    vertex_normals.index_add_(0, faces_t[:, 2], face_normals)

    n_jaw = vertex_normals.index_select(0, jaw_idx_t)
    n_jaw = n_jaw / n_jaw.norm(dim=1, keepdim=True).clamp_min(1e-8)

    out = t_points.clone()
    offset = (delta_m * jaw_weight_t).unsqueeze(1) * n_jaw  # (K,3)
    out[0].index_add_(0, jaw_idx_t, offset)
    return out
