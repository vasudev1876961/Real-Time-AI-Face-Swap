# Real-Time AI Face Swap Camera Application

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16%2B-005CED.svg?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-GUI-41CD52.svg?logo=qt&logoColor=white)](https://riverbankcomputing.com/software/pyqt/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-468p_FaceMesh-00897B.svg?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Google Colab](https://img.shields.io/badge/Google_Colab-Free_T4_GPU-F9AB00.svg?logo=googlecolab&logoColor=white)](notebooks/07_train_hero_model_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A modular, high-performance, real-time AI face transformation and custom hero model training ecosystem.**  
Engineered with **Python 3.11+**, **OpenCV**, **ONNX Runtime**, **PyTorch**, and **PyQt6**.

[Key Features](#-key-features) •
[Architecture](#-system-architecture) •
[Installation](#-installation) •
[Quickstart](#-quickstart--running-the-app) •
[Model Training](#-custom-hero-model-training-cloud-gpu--local) •
[Documentation](#-comprehensive-feature-index-phases-117)

</div>

---

> [!WARNING]
> **ETHICAL AI USAGE & CONSENT DISCLAIMER**  
> Use this software solely with authorized, consenting individuals and verified target assets. Do not use for non-consensual imagery, deceptive media, impersonation, or malicious synthesis. Always adhere to local digital rights and intellectual property laws.

---

## 🌟 Key Features

* **Real-Time 512×512 HD Neural Face Swapping**: 30+ FPS low-latency pipeline with asynchronous background inference and GFPGAN super-resolution enhancement.
* **Resolution-Invariant Anatomical Masking**: 468-point MediaPipe 3D face mesh projection with dynamic resolution scaling, active skin-edge boundary snapping, and curvature-adaptive signed distance field (SDF) feathering.
* **Physics-Grounded Relighting & Color Adaptation**: Reinhard Lab skin tone matching, 3D key-light vector tracking, directional cast shadows, and Subsurface Scattering (SSS) epidermal warmth synthesis.
* **Custom Hero Model Training Pipeline**: Complete Shared-Encoder + Hero-Decoder architecture with SSIM, Sobel Gradient Edge Loss, and VGG19 Perceptual Loss.
* **1-Click Google Colab GPU Workflow**: Train hero models in 5–8 minutes using free Google Colab T4 GPUs and export directly to production ONNX.
* **Multi-Identity Matrix & Spatial Binding**: Swap multiple faces simultaneously or bind distinct target identities across left-to-right spatial axes for co-streaming.
* **Virtual Camera & Network Streaming**: DirectShow virtual webcam output for OBS Studio, Zoom, Discord, and zero-config HTTP MJPEG network streaming (`http://localhost:8080/stream`).
* **Zero-Allocation Memory Architecture**: Pre-allocated frame buffer recycling (`FrameBufferPool`) preventing Python GC stalls during continuous 30+ FPS broadcasting.

---

## 🏛 System Architecture

The pipeline executes an asynchronous multi-stage transformation loop designed to decouple camera acquisition from heavy neural inference:

```
                      ┌────────────────────────┐
                      │     Camera Capture     │ (DirectShow MJPG / MSMF / V4L2)
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │ MediaPipe FaceMesh 468p│ (Sub-pixel landmarks & head pose estimation)
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │ Standard Face Alignment│ (5-Point Umeyama similarity transform)
                      └───────────┬────────────┘
                                  │
        ┌─────────────────────────┴────────────────────────┐
        │                                                  │
┌───────▼────────┐                               ┌─────────▼────────┐
│ Target DB /    │                               │ Persistent Worker│ (Asynchronous Swap +
│ Embedding Cache│                               │ In-Place Cache   │  512x512 GFPGAN Enhancer)
└───────┬────────┘                               └─────────┬────────┘
        │                                                  │
        └─────────────────────────┬────────────────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │ 512px Resolution Sync  │ (Crop, landmark, and inverse transform scaling)
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │ Color & Relighting     │ (Reinhard Lab / 3D Key-light / SSS Warmth)
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │ Anatomical Dense Mask  │ (Mesh contour projection + Curvature SDF feather)
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │ Multi-Band ROI Blender │ (Laplacian pyramid / Accelerated Vectorized Alpha)
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │ Live PyQt6 / VirtualCam│ (60 Hz Canvas, Captures, MP4 Video Recording)
                      └────────────────────────┘
```

---

## 📂 Directory Structure

```
deepfake/
├── README.md                     # Comprehensive technical documentation & user guide
├── LICENSE                       # MIT Open Source License
├── requirements.txt              # Core runtime dependencies
├── requirements-gpu.txt          # GPU acceleration guide (CUDA / DirectML)
├── pyproject.toml                # Project packaging and pytest configuration
│
├── configs/                      # Application Configurations
│   ├── config.yaml               # Pipeline, camera, blending, and lighting parameters
│   ├── models.yaml               # ONNX model paths and provider configurations
│   └── targets.yaml              # Identity categories and target database mappings
│
├── models/                       # Local Pretrained ONNX Models
│   ├── face_analysis/            # ArcFace / InsightFace 512-D identity embedding model
│   ├── face_detection/           # SCRFD / MediaPipe FaceMesh detectors
│   ├── face_swap/                # Replaceable ONNX face swap model (model.onnx)
│   └── enhancement/              # GFPGAN super-resolution model (face_enhancer.onnx)
│
├── training/                     # Custom Hero Model Training Pipeline
│   ├── train_hero_model.py       # PyTorch Shared-Encoder + Hero-Decoder training script
│   └── export_to_onnx.py         # Production ONNX exporter with dynamic batch axes
│
├── notebooks/                    # Jupyter & Google Colab Notebooks
│   ├── 01_camera_testing.ipynb   # Camera capture validation
│   ├── 02_face_detection.ipynb   # Face detector benchmarks
│   ├── 05_face_swap_testing.ipynb# Swap pipeline validation
│   └── 07_train_hero_model_colab.ipynb # Free Google Colab T4 GPU training workflow
│
├── faces/                        # Active Target Celebrity Database
│   ├── actors/                   # Bollywood & Indian cinema actors
│   ├── actresses/                # Cinema actresses
│   ├── telugu_heroes/            # Tollywood stars (Prabhas, Chiranjeevi, Mahesh Babu, etc.)
│   └── custom/                   # User-imported dynamic identities
│
├── src/                          # Application Source Code
│   ├── core/                     # Device detection, config loading, governor
│   ├── camera/                   # DirectShow MJPEG capture engine & camera manager
│   ├── detection/                # MediaPipe FaceMesh & 5-point landmark alignment
│   ├── tracking/                 # High-speed face tracker with EMA smoothing
│   ├── models/                   # Pluggable ONNX swapper & enhancement adapters
│   ├── processing/               # Masks, color correction, relighting, and blending
│   ├── pipeline/                 # Real-time multi-threaded frame orchestrator
│   └── ui/                       # PyQt6 desktop user interface
│
├── scripts/                      # Operational Scripts & Utilities
│   ├── run_app.py                # Main desktop application launcher
│   ├── package_training_dataset.py # 1-click dataset packager for Google Colab
│   ├── process_video.py          # Offline video and batch image face swapper
│   ├── add_target.py             # CLI for importing custom celebrity portraits
│   └── test_pipeline.py          # End-to-end integration test runner
│
└── outputs/                      # Generated Captures and Recordings
    ├── captures/                 # Timestamped high-resolution screenshots
    ├── recordings/               # Recorded MP4 videos and metadata JSON
    └── benchmarks/               # Performance throughput reports
```

---

## 💻 Installation

### Prerequisites
* **Python**: 3.11 or 3.12 (64-bit)
* **Operating System**: Windows 10/11, macOS, or Linux
* **Hardware**: Dual-core CPU minimum (NVIDIA GPU with CUDA 11.8+ / 12.x recommended for maximum performance)

### 1. Clone the Repository & Initialize Environment
```powershell
git clone https://github.com/vasudev1876961/Real-Time-AI-Face-Swap.git
cd Real-Time-AI-Face-Swap

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Linux/macOS: source .venv/bin/activate
```

### 2. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. (Optional) GPU Acceleration (NVIDIA CUDA)
To enable GPU acceleration via ONNX Runtime:
```powershell
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
```
> The application automatically queries available execution providers (`CUDAExecutionProvider`, `DmlExecutionProvider`, `CPUExecutionProvider`) and selects the fastest available engine.

---

## 🚀 Quickstart & Running the App

### Launch the Desktop GUI
```powershell
python scripts/run_app.py
```

### Desktop App Keyboard Shortcuts

| Key | Action | Description |
| :---: | :--- | :--- |
| `T` | **Next Target** | Cycles to the next target identity in the selected category |
| `1` | **Multiband Blend** | 3-Level Laplacian pyramid blending (seamless transitions) |
| `2` | **Vectorized Alpha** | Ultra-fast zero-allocation alpha blending (lowest latency) |
| `3` | **Poisson Clone** | Normal gradient seamless cloning |
| `4` | **Mixed Clone** | Preserves background high-frequency skin textures |
| `S` | **Save Capture** | Captures full-resolution JPEG snapshot to `outputs/captures/` |
| `R` | **Record Video** | Toggles live MP4 video recording to `outputs/recordings/` |
| `Q` / `Esc` | **Exit** | Safely releases camera hardware and closes application |

### Command-Line Arguments
```powershell
# Select specific camera index
python scripts/run_app.py --camera 1

# Force CPU execution
python scripts/run_app.py --device cpu

# Run automated headless smoke test
python scripts/run_app.py --headless-check
```

---

## 🎨 512×512 HD Neural Face Swapping & Masking

This project resolves the common low-resolution artifacts found in real-time deepfake tools:

1. **Resolution Harmonization**:
   Standard face swap models produce 128×128 pixel crops. When coupled with neural super-resolution models (GFPGAN 512×512), standard pipelines suffer from coordinate misalignment. Our pipeline dynamically harmonizes resolution:
   $$\text{Scale} = \frac{W_{\text{enhanced}}}{128.0}$$
   Landmarks, masks, and inverse affine transformation matrices are scaled consistently to prevent boundary distortion or shrinking.

2. **Anatomical Curvature-Adaptive Masking**:
   - **Mandibular Jawline**: Tight signed distance field (SDF) feathering (3.0–4.5px) with steep Hermite falloff prevents discoloration spilling onto collars or necks.
   - **Forehead & Temples**: Gentle feathering (8.0–12.0px) blends smoothly into natural hairline gradients.
   - **Active Edge Boundary Snapper**: Evaluates bilateral Sobel gradient magnitude and YCrCb skin probability, snapping the mask directly to the subject's true jawline.

3. **Color Balance & Lighting Harmonization**:
   - **Reinhard Lab Color Correction**: Matches statistical mean and standard deviation across $L$, $a$, and $b$ color channels to synchronize skin tone without losing identity characteristics.
   - **Directional Volumetric Relighting**: Calculates the 3D key-light vector $[L_x, L_y, L_z]$ from facial luminance moments, casting natural directional shadows.
   - **Subsurface Scattering (SSS) Epidermal Warmth**: Injects warm chrominance ($Cr$ channel) along shadow terminators, eliminating cold gray skin tones.

---

## 🏋️ Custom Hero Model Training (Cloud GPU & Local)

You can train a specialized model for any actor, celebrity, or individual with the upgraded **Shared-Encoder + Hero-Decoder** pipeline.

### Model Architecture
* **Shared Encoder**: Compresses face crops into a 512-dimensional identity-agnostic latent representation capturing pose, expression, and gaze.
* **Hero Decoder**: Features residual blocks, Instance Normalization, and sub-pixel PixelShuffle upscalers to reconstruct sharp facial details, clear irises, and lifelike skin textures.
* **Multi-Loss Optimization**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_1 + 0.5 \cdot \mathcal{L}_{\text{SSIM}} + 0.3 \cdot \mathcal{L}_{\text{Sobel}} + 0.2 \cdot \mathcal{L}_{\text{VGG19}}$$

---

### Option A: 1-Click Free Google Colab GPU Training (Recommended)

Train 200–300 epochs in **5–8 minutes** using a free Google Colab NVIDIA T4 GPU:

1. **Package your target face crops** into a zip archive with one command:
   ```powershell
   python scripts/package_training_dataset.py --input faces/actors/prabhas --output outputs/hero_dataset.zip
   ```
2. Open [`notebooks/07_train_hero_model_colab.ipynb`](notebooks/07_train_hero_model_colab.ipynb) in [Google Colab](https://colab.research.google.com).
3. Set runtime to GPU (*Runtime* -> *Change runtime type* -> *T4 GPU*).
4. Run the notebook cells: upload `hero_dataset.zip`, monitor the visual preview progression, and download the exported `model.onnx`.
5. Place the downloaded `model.onnx` into `models/face_swap/model.onnx` on your local PC.

---

### Option B: Local Training via PyTorch

```powershell
# Train custom model on local dataset (uses SSIM + Sobel Gradient Loss)
python training/train_hero_model.py --dataset faces/actors/prabhas --epochs 100 --batch_size 16

# If you have an NVIDIA CUDA GPU, enable deep VGG19 perceptual loss:
python training/train_hero_model.py --dataset faces/actors/prabhas --epochs 200 --batch_size 16 --use_vgg --device cuda
```

* **Live Visual Progress**: The training script automatically saves side-by-side progression grids `[Target | Model Output]` to `checkpoints/preview_epoch_XXX.jpg` every 5 epochs.

### Export Trained Checkpoint to ONNX
```powershell
python training/export_to_onnx.py --checkpoint checkpoints/hero_model_best.pt --output models/face_swap/model.onnx
```

---

## 🎬 Offline Video & Image Processing

Process pre-recorded videos and photo directories with full audio preservation:

```powershell
# Swap face in a recorded video
python scripts/process_video.py --input inputs/interview.mp4 --output outputs/recordings/swapped.mp4 --target prabhas --enhance 0.80

# Swap face in a single portrait image
python scripts/process_video.py --image inputs/photo.jpg --output outputs/captures/swapped.jpg --target chiranjeevi

# Batch swap an entire folder of photos
python scripts/process_video.py --dir inputs/cast_photos --dir-out outputs/batch_swapped --target rekha
```

---

## 🌐 Virtual Camera & Broadcast Integration

Broadcast swapped video streams directly into third-party software:

### 1. DirectShow Virtual Camera (OBS Studio, Zoom, Discord, Google Meet)
Set `virtual_camera.enabled: true` in [`configs/config.yaml`](configs/config.yaml). The pipeline automatically creates an OS-level DirectShow webcam stream visible in any video conferencing or streaming tool.

### 2. Zero-Config HTTP MJPEG Network Stream
Access the live feed from any web browser or network device:
```
http://localhost:8080/stream
```
* **OBS Studio Integration**: Add a **Browser Source** pointing to `http://localhost:8080/stream`.
* **Mobile / LAN Viewing**: Open the host machine's local IP address on any phone or tablet connected to the same WiFi network.

---

## ⚙️ Configuration Reference (`configs/config.yaml`)

| Setting | Default | Description |
| :--- | :---: | :--- |
| `camera.width` / `height` | `1280x720` | Native camera resolution (supports 720p, 1080p, 4K) |
| `camera.backend` | `"AUTO"` | Camera backend: `"AUTO"`, `"DSHOW"`, `"MSMF"`, `"V4L2"` |
| `performance.mode` | `"quality"` | Performance profile: `"performance"`, `"quality"`, `"debug"` |
| `processing.blending_method` | `"multiband"` | Blending mode: `"multiband"`, `"alpha"`, `"seamless_clone"` |
| `processing.color_correction`| `"reinhard"` | Color matching: `"reinhard"`, `"gain_matching"`, `"histogram"` |
| `processing.color_blend_ratio`| `0.70` | Skin tone adaptation balance ($0.0 = \text{target}, 1.0 = \text{source}$) |
| `processing.enhancement_mode`| `"onnx"` | Face enhancement: `"onnx"` (GFPGAN), `"adaptive"`, `"off"` |
| `processing.enhancement_strength`| `0.85` | Neural face restoration intensity [0.0–1.0] |
| `processing.postprocess_sharpen`| `0.35` | Unsharp mask high-frequency facial clarity [0.0–1.0] |
| `processing.enable_dense_mesh_mask`| `true` | Sub-pixel 468-point anatomical MediaPipe contour fitting |
| `processing.enable_edge_snapping`| `true` | Snaps mask boundary to subject's physical jawline |
| `processing.enable_volumetric_relighting`| `true` | 3D directional cast shadow and SSS warmth transfer |
| `virtual_camera.enabled` | `false` | Automatic DirectShow virtual webcam broadcast on launch |

---

## 🧪 Automated Testing & Verification

Run the comprehensive automated pytest suite covering unit, integration, and regression tests:

```powershell
# Run full test suite
python -m pytest tests/ -v

# Run pipeline integration test
python scripts/test_pipeline.py
```

---

## 📋 Comprehensive Feature Index (Phases 1–17)

<details>
<summary><b>Click to expand full architecture roadmap details</b></summary>

* **Phase 1**: Foundations — Modular camera backend, configuration loader, and logging framework.
* **Phase 2**: Geometric Pipeline — 5-point ArcFace landmark alignment with Umeyama affine transforms.
* **Phase 3**: Target Identity Database — ArcFace 512-D identity embeddings with cosine distance matching.
* **Phase 4**: ONNX Swapper Adapter — Abstract `BaseFaceSwapper` interface with dynamic execution provider detection.
* **Phase 5**: Compositing & Blending — 3-level Laplacian multi-band pyramids and Reinhard Lab color transfer.
* **Phase 6**: Occlusion & Anti-Jitter — Foreground occlusion carving and EMA affine motion stabilization.
* **Phase 7**: Hardware Governor — Persistent async inference worker and zero-allocation frame pools.
* **Phase 8**: Studio Color Grading — 1D LUT engine with cinematic presets (Golden Hour, Film Noir, Warm Studio).
* **Phase 9**: Eye Realism & 3D Pose — Blink synchronization, corneal catchlights, and 3D head-pose clamping.
* **Phase 10**: Interactive Web Studio — Browser-based streaming dashboard with real-time parameter controls.
* **Phase 11**: Speech & Expression Transfer — Inter-labial mouth aperture tracking and speech phoneme preservation.
* **Phase 12**: Spectacles & Hand Carving — Nasal bridge ridge filtering and skin-on-skin finger occlusion carving.
* **Phase 13**: Sub-Pixel Dense Mesh Fitting — 468-point MediaPipe contour projection and Chaikin spline smoothing.
* **Phase 14**: Anti-Stutter DirectShow Engine — Native hardware MJPEG FourCC streaming and frame deduplication.
* **Phase 15**: Volumetric Relighting & SLERP Morphing — 3D key-light estimation, SSS warmth, and hypersphere morphing.
* **Phase 16**: Depth-of-Field Bokeh & Micro-Expressions — Monocular depth field estimation and cutaneous strain wrinkles.
* **Phase 17**: Multi-Identity Matrix — Multi-subject spatial binding, 3-point studio rim lighting, and latent temporal coherence.

</details>

---

## 📄 License

This project is open-source software licensed under the [MIT License](LICENSE).