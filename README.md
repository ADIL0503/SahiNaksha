# SahiNaksha

**AI-Based Automated Urban Parcel Mapping and Cadastral Feature Extraction System using Drone Imagery** — SIH 2026, PS-26012.

SahiNaksha is a prototype for extracting building footprints from aerial imagery, validating the resulting geometry, and preparing it for GIS workflows. The repository separates the AI inference layer, geospatial utilities, training pipeline, and web dashboard.

## Architecture

```text
Aerial / Drone Image
        |
        v
   FastAPI Upload
        |
        v
 YOLO Segmentation
        |
        v
 Pixel Polygons -> GeoJSON features -> Validation
        |
        v
 React Dashboard / Human Inspection
```

## Repository

- `backend/` — FastAPI API, YOLO inference, georeferencing and geometry services.
- `training/` — GeoJSONL + georeferenced-raster preparation and YOLO segmentation training.
- `frontend/` — lightweight React/Vite demo dashboard.
- `docker-compose.yml` — local two-container demo stack.
- `.github/workflows/ci.yml` — backend and frontend CI.

## Run locally

### Backend

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --reload --port 8000
```

Check `http://localhost:8000/health`.

Set `SAHINAKSHA_MODEL_PATH` to the trained `.pt` file. The default is `backend/models/sahinaksha-building-seg.pt`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

The dashboard uploads an image, starts inference, and overlays returned pixel polygons for visual inspection.

### Docker

```bash
docker compose up --build
```

The web UI is exposed on port `5173` and the API on port `8000`. Mount the trained model at `backend/models/sahinaksha-building-seg.pt`.

## Training

Install the training dependencies:

```bash
python -m pip install -r training/requirements-yolo.txt
```

Prepare a dataset from a **georeferenced raster** and building footprints:

```bash
python training/prepare_yolo_dataset.py \
  --image /path/to/image.tif \
  --buildings /path/to/buildings.geojsonl.gz \
  --output training/dataset \
  --split train
```

Train:

```bash
python training/train_yolo.py \
  --data training/sahinaksha.yaml \
  --model yolo11n-seg.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 8
```

Validate generated labels:

```bash
python training/validate_yolo_dataset.py --dataset training/dataset --split train
```

## API

- `POST /api/upload` — upload an aerial image.
- `POST /api/predict/{job_id}?confidence=0.25` — run building segmentation and return detections, polygon features, image dimensions, and structural validation.
- `GET /health` — API and configured-model status.

## Important limitations

- The repository does **not** include trained model weights; they are intentionally excluded from Git.
- A model must be trained or supplied before the prediction endpoint can produce detections.
- The training pipeline requires georeferenced imagery for reliable alignment with building footprints. Do not infer pixel/map alignment from an ordinary JPEG.
- Building footprints are **not legal cadastral parcel boundaries**. Legal cadastral decisions require authoritative cadastral records and human review.
- Accuracy metrics must be measured on an unseen validation/test set; this repository does not claim a fabricated accuracy number.
