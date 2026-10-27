# HeadSwap live pipeline

Code of the snapshot-tier live pipeline from the HeadSwap thesis project: a browser captures a webcam frame, the server tracks the face (SMIRK), fits the 3D head pose (`fit3d`), renders the GAGAvatar Gaussian avatar, and composites it over the live frame (`blend_v3`).

This repository contains only the code on the live path. The offline batch pipeline, diagnostics, experiments and all model weights live elsewhere and are not included.

## Layout

```
v3/live/server.py      aiohttp HTTPS server and capture page (static/index.html)
v3/live/pipeline.py    PipelineState: plate calibration, pose calibration, per-snapshot processing
v3/track.py            SMIRK face tracking
v3/fit3d.py            3D landmark fit and similarity transform solve
v3/render_gaga.py      GAGAvatar render with floater and silhouette fixes
v3/blend_v3.py         whole-head erase and composite
v3/blink_calib.py, gaze_calib.py, jaw_*.py, stabilize.py   helpers imported by render_gaga
compositing/parse.py   BiSeNet face parsing (onnxruntime)
```

## What you must provide

These are not in the repository and are required before the server starts.

| Item | Expected location |
|---|---|
| GAGAvatar checkout with its weights (`assets/GAGAvatar.pt`, `assets/FLAME_with_eye.pt`) | `third_party/GAGAvatar` |
| SMIRK checkpoint `SMIRK_em1.pt` | `third_party/GAGAvatar/third_party/smirk/pretrained_models/` |
| BiSeNet CelebAMask-HQ ONNX model | `avatar/assets/bisenet_celebamask_512.onnx` |
| Identity image of the avatar | `avatar/reference_photos/headswap_target.jpg` |
| Self-signed TLS certificate | `v3/live/certs/cert.pem` and `key.pem` |

The FLAME model requires registering and accepting the licence at https://flame.is.tue.mpg.de. LAM and GAGAvatar weights have their own licences; the thesis use is non-commercial.

## Environment

Python 3.10 on Windows with an NVIDIA GPU (developed on an RTX 2060, CUDA 12.1).

```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-live.txt
```

GAGAvatar's own dependencies (including pytorch3d) are installed from its `environment.yml`.

## Run

```
openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=localhost" -keyout v3/live/certs/key.pem -out v3/live/certs/cert.pem
python v3/live/server.py
```

Open `https://<server-address>:8443/`, accept the certificate warning, then use the page to calibrate the background plate, calibrate the pose, and take snapshots.

## Status

Extracted from the main HeadSwap working tree. The code was copied as is, apart from replacing a hardcoded hostname in the server start-up message and normalising line endings. It has not been executed from this layout yet.
