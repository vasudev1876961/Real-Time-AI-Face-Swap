# Real-Time AI Face Swap Camera Application

A modular, production-quality, real-time AI face-swap camera desktop application built with **Python 3.11+**, **OpenCV**, **ONNX Runtime**, and **PyQt6**.

> [!WARNING]
> **AI-GENERATED FACE TRANSFORMATION DISCLAIMER**
> Use this software solely with authorized, consenting subjects and legitimate, permitted target assets. Do not use for impersonation, deceptive media, non-consensual content generation, or unauthorized identity transformations.

> [!IMPORTANT]
> **NO MODEL TRAINING WITHIN APPLICATION**
> This application is strictly an **inference and capture engine**. It contains no training, fine-tuning, or weight modification pipelines. Model weights are loaded locally via clean, replaceable ONNX adapters.

---

## 1. Project Architecture

The application executes a low-latency frame processing pipeline designed to achieve 20–30 FPS on standard hardware:

```
                      ┌────────────────────────┐
                      │     Camera Capture     │ (OpenCV DirectShow / MSMF / V4L2)
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │ Face Detection / Track │ (Lightweight periodic detection & EMA smoothing)
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │ Standard Face Alignment│ (5-Point Umeyama Affine Similarity Transform)
                      └───────────┬────────────┘
                                  │
        ┌─────────────────────────┴────────────────────────┐
        │                                                  │
┌───────▼────────┐                               ┌─────────▼────────┐
│ Target DB /    │                               │ BaseFaceSwapper /│ (Local ONNX Inference
│ Embedding Cache│                               │ ONNXSwapper      │  with CUDA / CPU fallback)
└───────┬────────┘                               └─────────┬────────┘
        │                                                  │
        └─────────────────────────┬────────────────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │    Color Correction    │ (Reinhard Lab / Gain / Histogram matching)
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │ Face Mask & Blending   │ (Feathered convex hull / Alpha / Seamless clone)
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │   Post-Processing      │ (Unsharp sharpening & telemetry overlay)
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │  Live PyQt6 Rendering  │ (Preview, Captures, MP4 Recording)
                      └────────────────────────┘
```

---

## 2. Directory Structure

```
deepfake/
├── README.md                     # Comprehensive documentation and operational guide
├── LICENSE                       # MIT License
├── requirements.txt              # Core runtime dependencies (CPU baseline)
├── requirements-gpu.txt          # Optional GPU acceleration setup instructions
├── pyproject.toml                # Project packaging and pytest configuration
├── .env.example                  # Environment variable configuration template
├── .gitignore                    # Git tracking ignore definitions
│
├── configs/                      # Application and model configurations
│   ├── config.yaml               # Camera, pipeline, and rendering parameters
│   ├── models.yaml               # Model paths, providers, and input specifications
│   └── targets.yaml              # Target categories and asset directory mappings
│
├── models/                       # Local Pretrained ONNX Models
│   ├── face_analysis/            # Face recognition & identity embedding model
│   ├── face_detection/           # SCRFD / RetinaFace detection models
│   ├── face_swap/                # Replaceable ONNX face-swap model
│   └── enhancement/              # Optional face restoration models
│
├── datasets/                     # Managed Datasets
│   ├── raw/                      # Raw source images separated by category
│   │   ├── actresses/
│   │   ├── actors/
│   │   └── telugu_heroes/
│   ├── processed/                # Cropped and standardized face datasets
│   │   ├── actresses/
│   │   ├── actors/
│   │   └── telugu_heroes/
│   ├── metadata/                 # CSV catalogs and audit reports
│   │   ├── actresses.csv
│   │   ├── actors.csv
│   │   ├── telugu_heroes.csv
│   │   ├── migration_report.csv
│   │   ├── rejection_report.csv
│   │   └── duplicate_report.csv
│   └── rejected/                 # Non-destructively filtered images
│       ├── blurry/
│       ├── no_face/
│       ├── multiple_faces/
│       ├── corrupted/
│       ├── too_small/
│       └── duplicates/
│
├── faces/                        # Active Target Database (Loaded into UI)
│   ├── actresses/
│   │   └── <person_id>/
│   │       ├── reference.jpg     # Reference portrait image
│   │       ├── metadata.json     # Identity metadata (consent, language, etc.)
│   │       └── face.npy          # Optional / auto-generated embedding cache
│   ├── actors/
│   └── telugu_heroes/
│
├── src/                          # Application Source Code
│   ├── core/                     # Device detection and typed config loading
│   │   ├── device.py             # Execution provider selector (CUDA, DML, CPU)
│   │   └── config_loader.py      # Dataclass YAML configuration parser
│   ├── camera/                   # Low-latency camera backend and device manager
│   │   ├── camera_backend.py     # Threaded non-blocking capture engine
│   │   └── camera_manager.py     # Camera discovery, switching, and recovery
│   ├── detection/                # Face detection and landmark extraction
│   │   ├── face_detector.py      # ONNX/OpenCV detector with multi-backend fallback
│   │   └── face_landmarks.py     # 5-point ArcFace landmark alignment utilities
│   ├── tracking/                 # Frame-to-frame face tracking and EMA smoothing
│   │   └── face_tracker.py       # High-FPS tracker with dynamic re-detection
│   ├── alignment/                # Geometric transformations
│   │   └── face_alignment.py     # Umeyama affine transform and inverse warping
│   ├── models/                   # Pluggable model abstractions
│   │   ├── base_swapper.py       # Abstract Base Class for face swappers
│   │   ├── onnx_swapper.py       # Production ONNX Runtime swapper adapter
│   │   └── model_manager.py      # Model lifecycle coordinator
│   ├── processing/               # Visual processing and blending
│   │   ├── mask.py               # Morphological and Gaussian feathered masks
│   │   ├── color_correction.py   # Reinhard Lab, Gain, and Histogram color matching
│   │   ├── blending.py           # Alpha blending and Poisson seamless cloning
│   │   └── postprocess.py        # Detail sharpening and HUD telemetry rendering
│   ├── targets/                  # Target database management
│   │   ├── target_embedding.py   # Facial identity embedding extractor
│   │   ├── target_loader.py      # Target folder scanner and cache generator
│   │   └── target_manager.py     # Dynamic target selector and in-memory cache
│   ├── pipeline/                 # Real-time pipeline orchestrator
│   │   └── realtime_pipeline.py  # End-to-end frame transformation loop
│   ├── performance/              # Telemetry and profiling
│   │   └── metrics.py            # Latency breakdown and FPS statistics
│   └── utils/                    # Common utilities
│       ├── logger.py             # Structured application logger
│       └── image_utils.py        # Safe Unicode I/O, blur check, pHash
│
├── dataset_tools/                # Dataset Preparation and Migration Tools
│   ├── scan_dataset.py           # Recursive image audit and format check
│   ├── validate_faces.py         # Quality filter (blur, multi-face, occlusion)
│   ├── extract_faces.py          # 5-point aligned face extractor
│   ├── remove_duplicates.py      # Perceptual hash (pHash) duplicate detector
│   ├── create_metadata.py        # Metadata CSV generator
│   ├── migrate_dataset.py        # Safe legacy dataset migration (e.g. TFI-Faces)
│   └── prepare_dataset.py        # Comprehensive end-to-end dataset pipeline
│
├── ui/                           # PyQt6 Desktop User Interface
│   ├── main_window.py            # Main application window
│   ├── camera_widget.py          # Responsive video rendering surface
│   ├── target_selector.py        # Category filter and thumbnail preview
│   └── status_panel.py           # Live telemetry dashboard
│
├── tests/                        # Automated Pytest Suite
│   ├── test_camera.py
│   ├── test_detection.py
│   ├── test_alignment.py
│   ├── test_dataset.py
│   ├── test_models.py
│   ├── test_processing.py
│   └── test_pipeline.py
│
├── scripts/                      # Operational Scripts
│   ├── run_app.py                # Main desktop application launcher
│   ├── prepare_dataset.py        # Dataset preparation CLI
│   └── benchmark.py              # Performance throughput and latency benchmark
│
└── outputs/                      # Generated Captures and Recordings
    ├── captures/                 # Timestamped screenshot captures
    ├── recordings/               # Recorded MP4 videos and metadata JSON
    └── benchmarks/               # Performance benchmark CSV reports
```

---

## 3. Installation

### Prerequisites
* **Python**: 3.11 or 3.12 (64-bit)
* **OS**: Windows 10/11, Linux, or macOS

### Step 1: Clone Repository & Create Virtual Environment
```powershell
git clone <repository_url>
cd deepfake

python -m venv .venv
.venv\Scripts\activate  # On Linux/macOS: source .venv/bin/activate
```

### Step 2: Install Base Dependencies (CPU Mode)
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: (Optional) Enable NVIDIA CUDA GPU Acceleration
If you have an NVIDIA GPU with matching CUDA Toolkit and cuDNN installed:
```powershell
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
```

> [!NOTE]
> The application will automatically detect whether `CUDAExecutionProvider` or `CPUExecutionProvider` is available and select the best provider without manual configuration.

---

## 4. Model Placement

Place your local pretrained ONNX models into the `models/` directory:

```
models/
├── face_analysis/
│   └── face_recognition.onnx     # ArcFace / InsightFace recognition model (512-D)
├── face_detection/
│   └── scrfd_500m.onnx           # SCRFD / RetinaFace detection ONNX model
├── face_swap/
│   └── model.onnx                # Standard ONNX face swap model (e.g. INSwapper)
└── enhancement/
    └── face_enhancer.onnx        # (Optional) GFPGAN / CodeFormer model
```

### Graceful Fallback When Models Are Absent
If no model file is present in `models/face_swap/`, **the application will NOT crash**:
* The camera will open and display the live preview.
* Face detection, tracking, and telemetry HUD will operate normally.
* The UI status badge will clearly display: `AI Swapper: MODEL NOT FOUND`.
* Swapping is safely bypassed until a valid ONNX file is provided.

---

## 5. Pluggable Model Architecture & Replacing Models

The application implements a clean abstract interface in [`src/models/base_swapper.py`](file:///c:/Users/vasud/OneDrive/Desktop/deepfake/src/models/base_swapper.py):

```python
class BaseFaceSwapper(ABC):
    @abstractmethod
    def load(self, model_path: str, provider: str = "auto") -> bool: ...

    @abstractmethod
    def swap(self, aligned_face: np.ndarray, source_face: FaceData, target_face: TargetFace) -> np.ndarray: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...
```

### How to Replace the Swap Model:
1. **Using an Alternate ONNX Model**:
   Place your ONNX file in `models/face_swap/my_custom_model.onnx` and update `configs/models.yaml`:
   ```yaml
   face_swap:
     model_path: "models/face_swap/my_custom_model.onnx"
     model_type: "custom_onnx"
   ```
2. **Implementing a Custom Adapter**:
   If your future model requires a custom pre/post-processing format:
   - Create `src/models/custom_swapper.py` subclassing `BaseFaceSwapper`.
   - Implement `load()` and `swap()`.
   - Register it in `src/models/model_manager.py`.

No changes to the camera, UI, tracker, blending, or dataset tools are needed.

---

## 6. Target Face Database Management

Target identities are stored under `faces/`:

```
faces/
├── actresses/
│   └── actress_001/
│       ├── reference.jpg
│       ├── metadata.json
│       └── face.npy          # Generated automatically if missing
├── actors/
│   └── actor_001/
└── telugu_heroes/
    └── hero_001/
        ├── reference.jpg
        ├── metadata.json
        └── face.npy
```

### `metadata.json` Format:
```json
{
  "person_id": "hero_001",
  "display_name": "Telugu Hero 01",
  "category": "telugu_heroes",
  "language": "telugu",
  "region": "telangana_andhra",
  "consent_status": "unknown",
  "license_status": "unknown",
  "source": "local"
}
```

* Embeddings (`face.npy`) are **optional caches**. If absent, the target loader computes the embedding from `reference.jpg` and caches it on first load.

---

## 7. Dataset Management & Migration Tools

### 1. Safely Auditing and Migrating Legacy Datasets (e.g. `TFI-Faces`)
Audit folder names without modifying or deleting original files:
```powershell
# Dry run: Generates datasets/metadata/migration_report.csv
python -m dataset_tools.migrate_dataset --source "TFI-Faces" --target "datasets/raw"

# Apply: Copies files safely into datasets/raw/
python -m dataset_tools.migrate_dataset --source "TFI-Faces" --target "datasets/raw" --apply
```

### 2. Scanning Raw Datasets
```powershell
python -m dataset_tools.scan_dataset --input datasets/raw
```

### 3. Quality Validation & Rejection Filtering
```powershell
# Dry run report:
python -m dataset_tools.validate_faces --input datasets/raw --blur-thresh 40.0

# Apply (moves rejected files to datasets/rejected/<reason>/):
python -m dataset_tools.validate_faces --input datasets/raw --apply
```

### 4. Extracting & Aligning Face Crops
```powershell
python -m dataset_tools.extract_faces --input datasets/raw --output datasets/processed --size 512
```

### 5. Near-Duplicate Detection (pHash)
```powershell
python -m dataset_tools.remove_duplicates --input datasets/raw --thresh 4
```

### 6. End-to-End Dataset Pipeline Runner
```powershell
# Dry run:
python scripts/prepare_dataset.py

# Apply full pipeline:
python scripts/prepare_dataset.py --apply
```

---

## 8. Running the Application

### Launch GUI Desktop Application:
```powershell
python scripts/run_app.py
```
*or:*
```powershell
python src/main.py
```

### Command Line Options:
```powershell
# Override camera device index
python scripts/run_app.py --camera 1

# Force CPU or CUDA
python scripts/run_app.py --device cpu

# Headless smoke test
python scripts/run_app.py --headless-check
```

---

## 9. Performance Benchmarking

Benchmark pipeline latency across resolutions (640x480, 1280x720, 1920x1080):
```powershell
python scripts/benchmark.py --iterations 30
```
Results are saved to `outputs/benchmarks/benchmark_results.csv`.

---

## 10. Running Automated Tests

Run the complete test suite:
```powershell
python -m pytest tests/ -v
```

---

## 11. Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `AI Swapper: MODEL NOT FOUND` | Model file not present in `models/face_swap/` | Place a compatible ONNX swap model in `models/face_swap/model.onnx`. |
| `Could not open camera device` | Camera in use or index mismatch | Use the Camera dropdown in the UI and click **Switch Device**, or check Windows Camera Privacy Settings. |
| Low Frame Rate (< 15 FPS) | Running large resolution on CPU | Switch Profile to **Performance Mode** in the right tuning panel or enable CUDA GPU execution. |
| Face jitter on video | Movement noise | EMA landmark smoothing is enabled by default (`ema_alpha: 0.65` in `configs/config.yaml`). |

---

## 12. License

This project is licensed under the [MIT License](LICENSE).