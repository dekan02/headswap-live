#!/usr/bin/env python
"""
Snapshot-tier live server (2026-09-15). aiohttp only -- flask/fastapi/uvicorn/
websockets are not installed in the GAGAvatar venv, and adding a new
dependency to that venv is out of scope for this spike.

Serves the capture page over HTTPS (self-signed cert, see v3/live/certs/ --
`tailscale cert` isn't available on this account's tier) and exposes 4 JSON
routes, all serialized behind PipelineState's own lock (single GPU, single
session, no concurrency):

  GET  /api/health
  POST /api/calibrate_plate   {frames: [dataURL, ...]}   -- "step out of frame" burst
  POST /api/calibrate_pose    {frame: dataURL}           -- "look at the camera" single shot
  POST /api/snapshot          {frame: dataURL}           -- the actual swap, ~1.2s+

Run (from the GAGAvatar venv):
    third_party/GAGAvatar/venv/Scripts/python.exe v3/live/server.py
"""
import asyncio
import base64
import ssl
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as pl  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
CERT_DIR = Path(__file__).resolve().parent / "certs"
PORT = 8443

state: "pl.PipelineState | None" = None


def _decode_data_url(data_url: str) -> np.ndarray:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("could not decode frame (bad/empty image data)")
    return frame


def _encode_data_url(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("failed to encode result frame")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _err(message: str, status: int = 400):
    return web.json_response({"ok": False, "error": message}, status=status)


async def health(_request):
    return web.json_response({"ok": True, "calibrated_plate": state.calibrated_plate,
                               "calibrated_pose": state.calibrated_pose})


async def calibrate_plate(request):
    try:
        body = await request.json()
        frames = [_decode_data_url(d) for d in body.get("frames", [])]
        if len(frames) < 3:
            return _err("send at least 3 frames for the background calibration burst")
        # LIVE TEST SESSION OVERRIDE (temporary, not committed to prod default):
        # calibrate_plate_person_excluded(mask_dilate_px=3) replaces the naive
        # calibrate_plate() for this session -- validated small win at f80
        # (plate -0.3pp, composite hairline -1.0pp), negligible at f36, no
        # regression. min_obs_frac stays 0.0 (Fix 1 confirmed to backfire,
        # not enabled). To revert: change this back to
        # `state.calibrate_plate, frames`.
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: state.calibrate_plate_person_excluded(frames, mask_dilate_px=3))
        return web.json_response({"ok": True})
    except RuntimeError as e:
        # Task 42: state.calibrate_plate raises this when its BiSeNet
        # presence-gate finds a person in the burst (see pipeline.py).
        return _err(str(e), status=422)
    except Exception as e:
        traceback.print_exc()
        return _err(f"plate calibration failed: {e}", status=500)


async def calibrate_pose(request):
    try:
        body = await request.json()
        frame = _decode_data_url(body["frame"])
        result = await asyncio.get_event_loop().run_in_executor(None, state.calibrate_pose, frame)
        return web.json_response({"ok": True, **result})
    except RuntimeError as e:
        return _err(str(e), status=422)
    except Exception as e:
        traceback.print_exc()
        return _err(f"pose calibration failed: {e}", status=500)


async def snapshot(request):
    try:
        body = await request.json()
        frame = _decode_data_url(body["frame"])
        composited, timings = await asyncio.get_event_loop().run_in_executor(
            None, state.process_snapshot, frame)
        return web.json_response({"ok": True, "result": _encode_data_url(composited),
                                   "timings_ms": timings})
    except pl.NotCalibratedError as e:
        return _err(str(e), status=409)
    except RuntimeError as e:
        return _err(str(e), status=422)
    except Exception as e:
        traceback.print_exc()
        return _err(f"pipeline error: {e}", status=500)


async def index(_request):
    return web.FileResponse(STATIC_DIR / "index.html")


def build_app():
    app = web.Application(client_max_size=32 * 1024 * 1024)  # a few uncompressed 720p frames per calibration burst
    app.router.add_get("/", index)
    app.router.add_get("/api/health", health)
    app.router.add_post("/api/calibrate_plate", calibrate_plate)
    app.router.add_post("/api/calibrate_pose", calibrate_pose)
    app.router.add_post("/api/snapshot", snapshot)
    app.router.add_static("/static/", STATIC_DIR)
    return app


def main():
    global state
    print("[server] loading pipeline (one-time GPU warmup, ~10-15s) ...")
    # LIVE TEST SESSION OVERRIDE (temporary, not committed to prod default):
    # hires_glasses_parse=True enables the B1 fix (validated: real fix for
    # glasses leak on strong EYE_G-signal frames, ~28-34ms/frame cost, no
    # regressions found). To revert: drop this kwarg (default is False).
    state = pl.PipelineState(device="cuda", hires_glasses_parse=True)

    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(str(CERT_DIR / "cert.pem"), str(CERT_DIR / "key.pem"))

    app = build_app()
    print(f"[server] ready -- listening on https://0.0.0.0:{PORT}/ (self-signed cert, accept the browser warning once).")
    web.run_app(app, host="0.0.0.0", port=PORT, ssl_context=ssl_ctx)


if __name__ == "__main__":
    sys.exit(main())
