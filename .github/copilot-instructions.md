## Purpose
Short, actionable guidance to help an AI coding agent be productive in this repository.

## Big picture
- This is a small Flask image-processing web app. The UI is in `static/index.html` + `static/main.js`.
- The web server is `app.py` (entry point). It accepts a multipart POST at `/process-image` with form field `file` and returns JSON with:
  - `token` (string)
  - `segments` (list of polylines; each is a list of [x,y] points)
  - `junctions` (list of [x,y])
  - `junction_midpoints` (list of [x,y])
  - `image_url` (relative path to generated result image)
- Core image logic lives in `centerline_extractor.py` (masking, morphology, distance transform, skeletonize, junction detection, segment tracing, simplification).

## Key files and what to look for
- `centerline_extractor.py`: the single most important file. Look for these magic constants and behaviors:
  - Color thresholds in BGR (`color_min_tong`, `color_max_tong`, `color_min_duong`, `color_max_duong`, `color_min_chu`) — change these to tune mask extraction.
  - Morphology kernels: opening/closing sizes and `kernel_bridge` used to connect near fragments.
  - Area filter for connected components: `stats[i, cv2.CC_STAT_AREA] > 500` (removes small objects).
  - Skeletonization uses `skimage.morphology.skeletonize` on a boolean binary mask.
  - Junction detection: counts neighbors in a 3x3 ROI (junction if neighbor count >= 3). Border pixels are added as junctions with a margin of 5.
  - Segment tracing: traces from junctions/endpoints, filters out segments with Euclidean length <= 10.
  - Grouping and simplification: `group_junctions(..., radius=5)` and `simplify_segment(..., epsilon=2.0)`.

- `app.py`: Flask endpoints. Main behavior:
  - Saves uploaded image to `temp.jpg` and calls `extract_centerline_and_junctions(temp.jpg)`.
  - Draws colored polylines on a copy of the original and writes `static/result.jpg` (or in a variant `static/output/img/<date>_<uuid>.jpg`). Note there are two similar app variants in the repo; prefer `app.py` at the repo root.

- `static/main.js` and `static/index.html`: the front-end expects JSON returned by `/process-image` and reads `segments` and `image_url`.

- `requirements.txt` and `Dockerfile`: use these to install dependencies or build the container. Dockerfile installs system libs needed for OpenCV (`libgl1`, `libglib2.0-0`).

- Tests: `test_app.py`, `test_mask.py` — use `pytest` to run existing tests.

## How to run locally (developer shortcuts)
1. Create a Python environment and install dependencies:
```powershell
python -m pip install -r requirements.txt
```
2. Start the server for development (auto-reload not configured):
```powershell
python app.py
```
3. Open `static/index.html` in a browser (server serves it at `/`) and use the upload form.

Notes:
- `centerline_extractor.extract_centerline_and_junctions` has a `debug` flag which opens OpenCV windows (requires a GUI). For headless / Docker runs, keep `debug=False`.
- Output mask and intermediate images are written to `output/` by the extractor. Check these files when tuning thresholds.

## Tests and validation
- Run unit tests with:
```powershell
pytest -q
```
- Tests expect the image pipeline to write/read masks in `output/`. If you change filenames or outputs, update tests accordingly.

## Editing tips and common edits
- To tune detection quality, change the BGR thresholds near the top of `centerline_extractor.py`.
- To join/disconnect nearby fragments, adjust `kernel_bridge` size or the `distanceTransform` threshold (currently `threshold(dist, 10, ...)`).
- To change the minimum accepted segment size, edit the length filter `if length > 10:`.
- When adding fields to the API JSON, update `static/main.js` to display the new data and adjust test expectations.

## Project conventions / patterns
- Minimal dependencies: image work uses `opencv-python`, `numpy`, `scikit-image` (skeletonize).
- I/O is primarily file-based for debugging: uploaded images saved to `temp.jpg`, masks and visualizations saved under `output/` and `static/result.jpg`.
- Polylines and points are returned as lists of [x,y] (ints). Clients expect that shape.

## Integration points and gotchas
- Front-end posts FormData field `file` to `/process-image` (see `static/main.js`).
- There are multiple copies/variants of app code in the tree — prefer the root `app.py` unless you intentionally switch to the variant that writes dated filenames under `static/output/img/`.
- OpenCV GUI calls (`cv2.imshow`) will fail in headless/Docker environments.

## If you change the image pipeline
- Update: `output/` artifacts, `static/result.jpg` (or per-token file saving), tests (`test_mask.py`), and the front-end if response shape changes.

---
If any of the sections above are unclear or you'd like me to include additional examples (for example: sample JSON payloads, sample mask images, or a small unit test template), tell me which part to expand and I'll iterate.