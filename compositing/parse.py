"""
Face/hair/neck parsing (PLAN.md Phase 3 step 3 + step 5 dependency). Wraps a
BiSeNet ONNX model exported for the standard CelebAMask-HQ 19-class scheme
(background/skin/brows/eyes/glasses/ears/earring/nose/mouth/lips/neck/
necklace/cloth/hair/hat -- this label layout is the public convention for
this model family, not project-specific logic). Model weights reused as an
ASSET from ../GAGAvatar/models (fresh inference code, per the "fresh code"
decision in NOTES.md -- no compositing logic imported from that repo).
"""

import os
from pathlib import Path

import cv2
import numpy as np

if hasattr(os, "add_dll_directory"):
    # onnxruntime's CUDAExecutionProvider needs cudnn64_9.dll on the process's
    # DLL search path -- confirmed directly (NOTES.md): with no system-wide
    # cuDNN install on this machine, ort.InferenceSession(..., providers=
    # ["CUDAExecutionProvider", ...]) silently fell back to CPUExecutionProvider
    # (no exception raised -- get_providers() just quietly showed CPU only,
    # with the real cause only visible in onnxruntime's own stderr warning).
    # torch's own cu121 wheel already bundles a matching cudnn64_9.dll under
    # its own lib/ directory -- point onnxruntime at that copy via
    # os.add_dll_directory (must happen before the first InferenceSession
    # construction, so it's done here at import time) instead of requiring a
    # second, separate cuDNN install just for onnxruntime.
    try:
        import torch
        _torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if _torch_lib.is_dir():
            os.add_dll_directory(str(_torch_lib))
    except ImportError:
        pass

import onnxruntime as ort

# BiSeNet's raw output is a (1, 19, 512, 512) float32 logit volume. The
# per-pixel argmax over the 19 channels was measured (v3 blend profiling,
# 2026-09) as ~27ms/frame on CPU -- larger than the ONNX inference itself --
# plus a 19MB device->host copy of the full logit volume. Doing the argmax on
# the GPU instead collapses that to a 256KB (512x512 uint8) copy and sub-ms
# compute. Falls back to the identical numpy path when torch/CUDA is absent.
# Verified byte-identical to np.argmax over all 245x2 parses of the test clip
# (float32 conv-head logits never tie in practice) -- see NOTES_3D0_findings.md.
try:
    import torch as _torch
    _ARGMAX_CUDA = _torch.cuda.is_available()
except Exception:
    _torch = None
    _ARGMAX_CUDA = False

MODEL_PATH = Path(__file__).resolve().parent.parent / "avatar" / "assets" / "bisenet_celebamask_512.onnx"

# CelebAMask-HQ 19-class label indices.
BACKGROUND, SKIN, L_BROW, R_BROW, L_EYE, R_EYE, EYE_G, L_EAR, R_EAR, EAR_R, \
    NOSE, MOUTH, U_LIP, L_LIP, NECK, NECK_L, CLOTH, HAIR, HAT = range(19)

HEAD_LABELS = (SKIN, L_BROW, R_BROW, L_EYE, R_EYE, EYE_G, L_EAR, R_EAR, EAR_R,
               NOSE, MOUTH, U_LIP, L_LIP, HAIR, HAT)
NECK_LABELS = (NECK, NECK_L)
# Plain skin-toned regions only (face skin + neck skin) -- excludes hair,
# brows, eyes, nose, mouth/lips, ears, and clothing, for sampling/applying a
# skin-tone-only color match without those other regions dragging the stats.
SKIN_LABELS = (SKIN,) + NECK_LABELS
# Core facial features only (no brows, no ears, no hair) -- used to bound a
# "don't distort this" region for the shape-aware avatar placement warp.
FACE_LABELS = (L_EYE, R_EYE, NOSE, MOUTH, U_LIP, L_LIP)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FaceParser:
    def __init__(self, device: str = "cuda:0"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in device else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(MODEL_PATH), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def parse(self, bgr: np.ndarray) -> np.ndarray:
        """Returns a (H, W) uint8 label map at the INPUT image's resolution."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
        x = x.transpose(2, 0, 1)[None].astype(np.float32)
        logits = self.session.run(None, {self.input_name: x})[0]
        if _ARGMAX_CUDA:
            labels_512 = (_torch.as_tensor(logits[0], device="cuda")
                          .argmax(dim=0).to(_torch.uint8).cpu().numpy())
        else:
            labels_512 = logits[0].argmax(axis=0).astype(np.uint8)
        return cv2.resize(labels_512, (w, h), interpolation=cv2.INTER_NEAREST)

    def parse_batch(self, bgr_list: list) -> list:
        """Batched equivalent of calling parse() once per image in bgr_list.

        Additive, non-breaking: runs one session.run() over a stacked
        [N,3,512,512] tensor instead of N separate single-image calls, then
        does the argmax over the batch dim (dim=1) on GPU (or numpy axis=1
        on CPU fallback) before resizing each frame's 512x512 label map back
        to that frame's own original (h, w) -- identical per-frame resize
        step to parse(). Returns a list of (H, W) uint8 label maps, one per
        input image, in the same order as bgr_list.
        """
        sizes = [bgr.shape[:2] for bgr in bgr_list]
        batch = np.empty((len(bgr_list), 3, 512, 512), dtype=np.float32)
        for i, bgr in enumerate(bgr_list):
            rgb = cv2.cvtColor(cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
            x = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
            batch[i] = x.transpose(2, 0, 1)
        logits = self.session.run(None, {self.input_name: batch})[0]
        if _ARGMAX_CUDA:
            labels_512_batch = (_torch.as_tensor(logits, device="cuda")
                                 .argmax(dim=1).to(_torch.uint8).cpu().numpy())
        else:
            labels_512_batch = logits.argmax(axis=1).astype(np.uint8)
        return [
            cv2.resize(labels_512_batch[i], (w, h), interpolation=cv2.INTER_NEAREST)
            for i, (h, w) in enumerate(sizes)
        ]


def mask_for(labels: np.ndarray, label_ids) -> np.ndarray:
    """Returns a bool (H, W) mask for the union of the given label ids."""
    out = np.zeros(labels.shape, dtype=bool)
    for lid in label_ids:
        out |= labels == lid
    return out
