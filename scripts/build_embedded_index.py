"""
Builds the standalone, fully embedded index.html.
Embeds the entire CSS directly into <style>...</style> in <head> so that
Brave Shields, ad-blockers, Windows mime-types, and cache issues can NEVER
cause unstyled or distorted layouts again.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(ROOT, "src", "web", "static", "index.html")
CSS_PATH = os.path.join(ROOT, "src", "web", "static", "css", "studio.css")

with open(CSS_PATH, "r", encoding="utf-8") as f:
    css_content = f.read()

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FaceMorph Studio | Real-Time AI Neural Face Transformation</title>
    <meta name="description" content="Next-generation real-time AI face swapping studio with live webcam streaming, photo swap studio, cinematic portrait lighting, and photorealistic identity transfer.">
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    
    <!-- External Stylesheet with Cache Buster -->
    <link rel="stylesheet" href="/static/css/studio.css?v=2.5">
    
    <!-- Guaranteed Direct Inlined Styles (Immune to AdBlock, Brave Shields, or MIME type glitches) -->
    <style>
{css_content}
    </style>
</head>
<body class="studio-dark">
    <!-- Top Modern Navigation Header -->
    <header class="studio-navbar">
        <div class="nav-left">
            <div class="brand-container">
                <div class="brand-gem">
                    <span class="brand-spark">⚡</span>
                </div>
                <div class="brand-titles">
                    <h1 class="brand-title">FaceMorph <span class="gradient-text">Studio</span></h1>
                    <span class="brand-subtitle">REAL-TIME NEURAL CINEMA ENGINE</span>
                </div>
            </div>

            <!-- Mode Quick Switcher -->
            <div class="nav-modes">
                <button class="mode-tab-btn active" id="mode-btn-live">📹 Live Camera</button>
                <button class="mode-tab-btn" id="mode-btn-photo">📸 Photo Studio</button>
                <button class="mode-tab-btn" id="mode-btn-outputs">🖼️ Outputs & Tests</button>
            </div>
        </div>

        <div class="nav-center">
            <!-- Live Broadcast Pill -->
            <div class="status-pill live-pill" id="live-indicator">
                <span class="pulsing-beacon"></span>
                <span class="pill-text">LIVE FEED</span>
            </div>
            
            <!-- FPS Telemetry Pill -->
            <div class="status-pill fps-pill" id="fps-badge" title="Live Pipeline Framerate">
                <span class="pill-glyph">⚡</span>
                <span id="header-fps-text" class="mono-text">30.0 FPS</span>
            </div>

            <!-- Active Target Pill -->
            <div class="status-pill target-pill" id="active-target-pill" title="Current Selected Actor">
                <span class="pill-glyph">🎭</span>
                <span id="active-target-badge">Amitabh Bachchan</span>
            </div>

            <!-- Hardware Execution Provider Pill -->
            <div class="status-pill hw-pill" id="hw-badge" title="ONNX Runtime Provider">
                <span class="pill-glyph">⚙️</span>
                <span id="header-hw-text">CPU ACCELERATED</span>
            </div>

            <span id="header-gov-text" style="display:none;">OPTIMAL</span>
        </div>

        <div class="nav-right">
            <!-- Source Selector -->
            <button class="nav-btn nav-btn-glass" id="btn-toggle-source" title="Toggle between physical camera and synthetic benchmark stream">
                <span class="btn-glyph">🔄</span>
                <span id="source-toggle-text">Webcam</span>
            </button>

            <!-- Video Recording -->
            <button class="nav-btn nav-btn-danger" id="btn-record-toggle" title="Record live stream to high-definition MP4">
                <span class="btn-glyph" id="record-icon">⏺️</span>
                <span id="record-btn-text">Record</span>
                <span class="record-clock" id="record-timer" style="display: none;">00:00</span>
            </button>

            <!-- Instant Snapshot -->
            <button class="nav-btn nav-btn-primary" id="btn-quick-capture" title="Take a high-resolution snapshot">
                <span class="btn-glyph">📸</span>
                <span>Snapshot</span>
            </button>
        </div>
    </header>

    <!-- Quick User Guide Banner -->
    <div style="padding: 16px 28px 0 28px; max-width: 1920px; margin: 0 auto; width: 100%;">
        <div class="studio-guide-banner">
            <div class="guide-steps">
                <div class="guide-step">
                    <span class="step-num">1</span>
                    <span class="step-text"><strong>Pick Target Actor:</strong> Click any of the 123 actors in the carousel below</span>
                </div>
                <div class="guide-arrow">➔</div>
                <div class="guide-step">
                    <span class="step-num">2</span>
                    <span class="step-text"><strong>Face the Camera:</strong> Your face is replaced in real time at 30 FPS (or use <em>Photo Studio</em> tab)</span>
                </div>
                <div class="guide-arrow">➔</div>
                <div class="guide-step">
                    <span class="step-num">3</span>
                    <span class="step-text"><strong>Test & Compare:</strong> Click <em>Split Compare</em> to slide original vs swapped, or view <em>Outputs</em></span>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Workspace Stage -->
    <main class="studio-workspace">
        <!-- Left Hero Column: Viewport & Actor Shelf -->
        <section class="viewport-stage">
            <!-- Viewport Card -->
            <div class="viewport-cinema-card" id="viewport-card">
                <!-- Viewport Top Header -->
                <div class="viewport-bar">
                    <div class="viewport-status-info">
                        <span class="status-ring emerald"></span>
                        <span class="viewport-heading">Live Neural Transformation Viewport</span>
                        <span class="resolution-badge">1280 × 720 • 30 FPS</span>
                    </div>
                    <div class="viewport-quick-tools">
                        <button class="tool-btn" id="btn-split-toggle" title="Toggle Before/After Split Comparison View">
                            <span class="tool-icon">🌓</span>
                            <span class="tool-text">Split Compare</span>
                        </button>
                        <button class="tool-btn" id="btn-toggle-swap" title="Toggle Neural Face Swapping On/Off">
                            <span class="tool-icon">✨</span>
                            <span id="btn-swap-text">Swap: ON</span>
                        </button>
                        <button class="tool-btn" id="btn-fullscreen" title="Enter Fullscreen Theater Mode">
                            <span class="tool-icon">⛶</span>
                        </button>
                    </div>
                </div>

                <!-- Video Viewport Frame -->
                <div class="video-container" id="viewport-wrapper">
                    <!-- Shutter Flash Effect for Snapshots -->
                    <div class="shutter-flash" id="shutter-flash"></div>

                    <!-- Live Main Stream -->
                    <img id="live-video-feed" class="video-canvas" src="/stream" alt="Live Face Swap Video Stream">

                    <!-- Interactive Split View Mode (Hidden by default) -->
                    <div class="split-comparison-deck" id="split-container" style="display: none;">
                        <img id="split-original" class="split-stream original-stream" src="/stream/original" alt="Original Camera Stream">
                        <img id="split-swapped" class="split-stream swapped-stream" src="/stream" alt="Swapped Stream">
                        <div class="split-divider-bar" id="split-divider">
                            <div class="divider-grabber">↔</div>
                        </div>
                        <span class="split-tag tag-before">ORIGINAL</span>
                        <span class="split-tag tag-after">SWAPPED</span>
                    </div>

                    <!-- Subtle Floating Telemetry Pill (corner, non-obscuring) -->
                    <div class="stream-telemetry-badge" id="viewport-hud">
                        <div class="telem-item"><span class="telem-dot"></span><span id="hud-fps">30.0</span> FPS</div>
                        <div class="telem-sep">•</div>
                        <div class="telem-item"><span id="hud-latency">215</span> ms</div>
                        <div class="telem-sep">•</div>
                        <div class="telem-item"><span id="status-message-text">Face active</span></div>
                        <span id="hud-viewers" style="display:none;">1</span>
                    </div>
                </div>

                <!-- Floating Glass Control Strip under Video -->
                <div class="viewport-dock">
                    <div class="dock-left">
                        <span class="dock-caption">Active Target:</span>
                        <span class="dock-target-name" id="dock-target-label">Amitabh Bachchan</span>
                    </div>
                    <div class="dock-center">
                        <button class="dock-action-btn" id="btn-capture-main" title="Capture high-definition image">
                            <span class="dock-glyph">📸</span>
                            <span>Capture Frame</span>
                        </button>
                        <button class="dock-action-btn" id="btn-dock-photo-swap" title="Open Photo Swap Studio">
                            <span class="dock-glyph">🖼️</span>
                            <span>Swap Any Photo</span>
                        </button>
                    </div>
                    <div class="dock-right">
                        <span class="status-indicator-tag" id="dock-pipeline-status">Face Active</span>
                    </div>
                </div>
            </div>

            <!-- Target Actors Carousel Shelf -->
            <div class="actors-shelf-card">
                <div class="shelf-top-row">
                    <div class="shelf-headline">
                        <h2 class="shelf-title">🎭 Target Identities <span class="actor-count" id="actor-total-count">(123 Actors)</span></h2>
                        <p class="shelf-subtitle">Click any actor card below to immediately swap into their identity</p>
                    </div>

                    <!-- Search Filter Input -->
                    <div class="actor-search-box">
                        <span class="search-icon">🔍</span>
                        <input type="text" id="actor-search-input" placeholder="Search actor name... (e.g. Prabhas, Salman)" autocomplete="off">
                        <button class="clear-search-btn" id="btn-clear-search" style="display:none;">✕</button>
                    </div>

                    <!-- Category Pills -->
                    <div class="category-tabs" id="category-tabs">
                        <button class="category-pill active" data-category="all">All</button>
                        <button class="category-pill" data-category="actors">Bollywood</button>
                        <button class="category-pill" data-category="telugu_heroes">Telugu Heroes</button>
                        <button class="category-pill" data-category="actresses">Actresses</button>
                        <button class="category-pill" data-category="custom">Custom</button>
                    </div>
                </div>

                <!-- Carousel Cards Deck with Navigation Arrows -->
                <div class="carousel-nav-wrapper">
                    <button class="carousel-scroll-btn btn-scroll-left" id="btn-carousel-left" title="Scroll actors left">‹</button>
                    <div class="target-carousel" id="target-carousel">
                        <!-- Upload Custom Target Card -->
                        <div class="target-card upload-custom-card" id="card-upload-trigger" title="Upload a photo of yourself or any custom face">
                            <div class="upload-icon-wrapper">
                                <span class="upload-plus">+</span>
                            </div>
                            <span class="upload-card-label">Add Custom</span>
                            <span class="upload-card-sub">Upload photo</span>
                        </div>
                        <!-- Dynamically populated actor cards inserted here -->
                    </div>
                    <button class="carousel-scroll-btn btn-scroll-right" id="btn-carousel-right" title="Scroll actors right">›</button>
                </div>
            </div>
        </section>

        <!-- Right Studio Controls Dock (Clean 5-Tab Interface) -->
        <aside class="studio-dock">
            <!-- Dock Tab Header -->
            <div class="dock-tabs-nav">
                <button class="dock-tab-btn active" data-tab="tab-outputs" id="btn-tab-outputs">
                    <span class="tab-glyph">🖼️</span>
                    <span class="tab-label">Outputs</span>
                </button>
                <button class="dock-tab-btn" data-tab="tab-photo-swap" id="btn-tab-photo">
                    <span class="tab-glyph">📸</span>
                    <span class="tab-label">Photo Studio</span>
                </button>
                <button class="dock-tab-btn" data-tab="tab-fx" id="btn-tab-fx">
                    <span class="tab-glyph">💡</span>
                    <span class="tab-label">Lighting & FX</span>
                </button>
                <button class="dock-tab-btn" data-tab="tab-engine" id="btn-tab-engine">
                    <span class="tab-glyph">⚙️</span>
                    <span class="tab-label">Tuning</span>
                </button>
                <button class="dock-tab-btn" data-tab="tab-telemetry" id="btn-tab-telemetry">
                    <span class="tab-glyph">📊</span>
                    <span class="tab-label">Telemetry</span>
                </button>
            </div>

            <!-- Tab 1: Outputs & Showcase Gallery -->
            <div class="dock-tab-content active" id="tab-outputs">
                <div class="dock-card">
                    <div class="card-heading-row">
                        <h3 class="card-title">🖼️ Generated Output Gallery</h3>
                        <span class="card-pill emerald" id="outputs-count-badge">Loading...</span>
                    </div>
                    <p class="card-desc">Review high-definition neural swap outputs, 512×512 HD model tests, and live pipeline benchmarks. Click any card to open in full screen.</p>

                    <!-- Filter Pills -->
                    <div class="outputs-filter-pills" id="outputs-filter-pills">
                        <button class="output-filter-btn active" data-filter="all">All</button>
                        <button class="output-filter-btn" data-filter="hd_model">512p HD</button>
                        <button class="output-filter-btn" data-filter="showcase">Cinematic</button>
                        <button class="output-filter-btn" data-filter="comparison">Split & Grid</button>
                        <button class="output-filter-btn" data-filter="capture">Captures</button>
                    </div>

                    <!-- Outputs Grid -->
                    <div class="outputs-gallery-grid" id="outputs-gallery-grid">
                        <!-- Populated dynamically via JS -->
                    </div>
                </div>
            </div>

            <!-- Tab 2: Instant Photo Swap Studio -->
            <div class="dock-tab-content" id="tab-photo-swap">
                <div class="dock-card">
                    <div class="card-heading-row">
                        <h3 class="card-title">Instant Photo Swap Studio</h3>
                        <span class="card-pill purple">High-Res</span>
                    </div>
                    <p class="card-desc">Upload any selfie or portrait to instantly swap it with the selected actor in full photographic resolution.</p>

                    <!-- Dropzone -->
                    <div class="photo-dropzone" id="photo-dropzone">
                        <input type="file" id="photo-file-input" accept="image/jpeg,image/png,image/webp" style="display: none;">
                        <div class="dropzone-content" id="dropzone-prompt">
                            <span class="dropzone-icon">📁</span>
                            <span class="dropzone-title">Drop your selfie / portrait here</span>
                            <span class="dropzone-subtitle">or click to browse files (JPEG, PNG, WebP)</span>
                        </div>
                        <!-- Preview Container -->
                        <div class="photo-preview-box" id="photo-preview-box" style="display: none;">
                            <img id="uploaded-photo-preview" src="" alt="Uploaded Face Preview">
                            <button class="btn-change-photo" id="btn-change-photo" title="Choose another photo">Change Photo</button>
                        </div>
                    </div>

                    <!-- Quick Demo Photos -->
                    <div class="quick-demos-row">
                        <span class="quick-demo-label">Or test sample portrait:</span>
                        <div class="quick-demo-pills">
                            <button class="demo-pill-btn" data-src="/faces/actors/akshay_kumar/reference.jpg" data-name="Akshay Kumar">Akshay Kumar</button>
                            <button class="demo-pill-btn" data-src="/faces/actors/ajay_devgn/reference.jpg" data-name="Ajay Devgn">Ajay Devgn</button>
                            <button class="demo-pill-btn" data-src="/faces/telugu_heroes/prabhas/reference.jpg" data-name="Prabhas">Prabhas</button>
                        </div>
                    </div>

                    <!-- Swap Action Button -->
                    <button class="btn-gradient-action" id="btn-execute-photo-swap" disabled>
                        <span class="action-glyph">✨</span>
                        <span id="btn-photo-swap-text">Select a Photo to Swap</span>
                    </button>

                    <!-- Swapped Result Container -->
                    <div class="photo-result-card" id="photo-result-card" style="display: none;">
                        <div class="result-header">
                            <span class="result-title">Transformed Result</span>
                            <a id="btn-download-result" class="btn-download" download="face_swap_result.jpg" href="#">⬇ Download High-Res</a>
                        </div>
                        <div class="result-image-box">
                            <img id="photo-result-img" src="" alt="Swapped Result Output">
                        </div>
                    </div>
                </div>
            </div>

            <!-- Tab 3: Studio Lighting & Cinematic FX -->
            <div class="dock-tab-content" id="tab-fx">
                <!-- Virtual Studio Rim Lighting (Phase 17) -->
                <div class="dock-card">
                    <div class="card-heading-row">
                        <h3 class="card-title">💡 Virtual Studio Rim Lighting</h3>
                        <label class="toggle-switch">
                            <input type="checkbox" id="toggle-studio-lighting">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <p class="card-desc">Adds a 3-point photographic rim light contour around the subject and softly dims ambient background.</p>

                    <div class="fx-control-group">
                        <label class="fx-label-row">
                            <span>Rim Glow Color</span>
                        </label>
                        <div class="color-swatch-pills" id="rim-color-pills">
                            <button class="swatch-btn active" data-color="warm_gold"><span class="swatch-dot gold"></span>Gold</button>
                            <button class="swatch-btn" data-color="cyber_cyan"><span class="swatch-dot cyan"></span>Cyan</button>
                            <button class="swatch-btn" data-color="neon_magenta"><span class="swatch-dot magenta"></span>Magenta</button>
                            <button class="swatch-btn" data-color="studio_white"><span class="swatch-dot white"></span>White</button>
                        </div>
                        <select id="select-studio-rim-color" style="display:none;">
                            <option value="warm_gold" selected>Warm Gold</option>
                            <option value="cyber_cyan">Cyber Cyan</option>
                            <option value="neon_magenta">Neon Magenta</option>
                            <option value="studio_white">Studio White</option>
                        </select>
                    </div>

                    <div class="fx-control-group">
                        <div class="fx-label-row">
                            <span>Rim Light Intensity</span>
                            <span class="fx-val" id="val-studio-rim-intensity">50%</span>
                        </div>
                        <input type="range" id="slider-studio-rim-intensity" min="0" max="100" value="50" class="styled-slider">
                    </div>

                    <div class="fx-control-group">
                        <div class="fx-label-row">
                            <span>Background Dimming</span>
                            <span class="fx-val" id="val-studio-backdrop-dim">15%</span>
                        </div>
                        <input type="range" id="slider-studio-backdrop-dim" min="0" max="60" value="15" class="styled-slider">
                        <input type="range" id="slider-studio-rim-thickness" min="1" max="10" value="3" style="display:none;">
                    </div>
                </div>

                <!-- Cinematic DSLR Portrait Bokeh (Phase 16) -->
                <div class="dock-card">
                    <div class="card-heading-row">
                        <h3 class="card-title">📷 DSLR Portrait Bokeh Blur</h3>
                        <label class="toggle-switch">
                            <input type="checkbox" id="toggle-bokeh">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                    <p class="card-desc">Simulates a shallow depth-of-field lens (f/1.4 aperture) blurring the room behind the subject.</p>

                    <div class="fx-control-group">
                        <div class="fx-label-row">
                            <span>Aperture Blur Radius</span>
                            <span class="fx-val" id="val-bokeh-radius">12px</span>
                        </div>
                        <input type="range" id="slider-bokeh-radius" min="0" max="30" value="12" class="styled-slider">
                        <input type="range" id="slider-bokeh-depth" min="5" max="95" value="35" style="display:none;">
                    </div>
                </div>

                <!-- Studio Color Grading Presets -->
                <div class="dock-card">
                    <div class="card-heading-row">
                        <h3 class="card-title">🎨 Color Grading Atmosphere</h3>
                    </div>
                    <div class="preset-pill-grid" id="preset-pill-group">
                        <button class="preset-pill active" data-preset="neutral">Neutral Clean</button>
                        <button class="preset-pill" data-preset="warm_studio">Warm Studio</button>
                        <button class="preset-pill" data-preset="golden_hour">Golden Hour</button>
                        <button class="preset-pill" data-preset="cool_daylight">Cool Daylight</button>
                        <button class="preset-pill" data-preset="film_noir">Film Noir</button>
                        <button class="preset-pill" data-preset="cinematic_vibrant">Vibrant Cinema</button>
                    </div>
                </div>
            </div>

            <!-- Tab 4: Neural Engine Tuning -->
            <div class="dock-tab-content" id="tab-engine">
                <div class="dock-card">
                    <div class="card-heading-row">
                        <h3 class="card-title">⚡ Identity & Appearance Tuning</h3>
                        <button class="btn-micro-reset" id="btn-reset-params">Reset Defaults</button>
                    </div>

                    <!-- Face Enhancement Super-Resolution -->
                    <div class="fx-control-group">
                        <div class="fx-label-row">
                            <span>GFPGAN Face Super-Resolution</span>
                            <span class="fx-val" id="val-enhancement">50%</span>
                        </div>
                        <input type="range" id="slider-enhancement" min="0" max="100" value="50" class="styled-slider">
                    </div>

                    <!-- Identity Blend / Color Blend Ratio -->
                    <div class="fx-control-group">
                        <div class="fx-label-row">
                            <span>Skin Tone Harmonization (Reinhard)</span>
                            <span class="fx-val" id="val-color-blend">45%</span>
                        </div>
                        <input type="range" id="slider-lighting" min="0" max="100" value="45" class="styled-slider">
                    </div>

                    <!-- Spectacles Preservation -->
                    <div class="toggle-row">
                        <div class="toggle-info">
                            <span class="toggle-title">Spectacles & Eyewear Glare</span>
                            <span class="toggle-sub">Preserves eyeglasses and frames over the swap</span>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="slider-spectacles-toggle" checked>
                            <span class="toggle-slider"></span>
                        </label>
                        <input type="range" id="slider-spectacles" min="0" max="100" value="60" style="display:none;">
                    </div>

                    <!-- Dense Mesh 468p Fitting -->
                    <div class="toggle-row">
                        <div class="toggle-info">
                            <span class="toggle-title">Dense 468-Point Mesh Fitting</span>
                            <span class="toggle-sub">Sub-pixel MediaPipe face fitting and edge adherence</span>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="toggle-dense-mesh" checked>
                            <span class="toggle-slider"></span>
                        </label>
                    </div>

                    <!-- Temporal Anti-Jitter Stabilization -->
                    <div class="toggle-row">
                        <div class="toggle-info">
                            <span class="toggle-title">Temporal Micro-Jitter Suppression</span>
                            <span class="toggle-sub">Smooths geodesic pose vectors across consecutive frames</span>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="toggle-temporal-coherence" checked>
                            <span class="toggle-slider"></span>
                        </label>
                        <input type="range" id="slider-coherence-factor" min="0" max="98" value="85" style="display:none;">
                        <input type="range" id="slider-jitter-suppression" min="0" max="100" value="40" style="display:none;">
                    </div>

                    <!-- Hidden legacy slider references for JS compatibility -->
                    <div style="display: none;">
                        <input type="range" id="slider-eye-gaze" min="0" max="100" value="30">
                        <span id="val-eye-gaze">30%</span>
                        <input type="range" id="slider-mouth" min="0" max="100" value="20">
                        <span id="val-mouth">20%</span>
                        <input type="range" id="slider-expression" min="0" max="100" value="35">
                        <span id="val-expression">35%</span>
                        <input type="range" id="slider-occlusion" min="0" max="100" value="30">
                        <span id="val-occlusion">30%</span>
                        <input type="range" id="slider-hand-carving" min="0" max="100" value="40">
                        <span id="val-hand-carving">40%</span>
                        <span id="val-spectacles">60%</span>
                        <input type="checkbox" id="toggle-occlusion-hud">
                        <input type="checkbox" id="toggle-edge-snapping" checked>
                        <input type="range" id="slider-edge-snapping" min="0" max="100" value="65">
                        <span id="val-edge-snapping">65%</span>
                        <input type="checkbox" id="toggle-hairline-carving" checked>
                        <input type="checkbox" id="toggle-curvature-feathering" checked>
                        <input type="checkbox" id="toggle-mask-hud">
                        <input type="checkbox" id="toggle-fps-hud">
                        <select id="select-color-correction"><option value="reinhard" selected>reinhard</option></select>
                        <input type="checkbox" id="toggle-volumetric-relighting" checked>
                        <input type="range" id="slider-volumetric-shadow" min="0" max="100" value="50">
                        <input type="range" id="slider-sss-warmth" min="0" max="100" value="45">
                        <select id="select-secondary-target"><option value="">None</option></select>
                        <input type="range" id="slider-fusion-ratio" min="0" max="100" value="0">
                        <input type="range" id="slider-morph-duration" min="1" max="30" value="5">
                        <select id="select-split-mode"><option value="off">off</option></select>
                        <input type="range" id="slider-split-pos" min="5" max="95" value="50">
                        <input type="checkbox" id="toggle-micro-expressions" checked>
                        <input type="range" id="slider-micro-expressions" min="0" max="100" value="60">
                        <input type="checkbox" id="toggle-motion-blur" checked>
                        <input type="range" id="slider-motion-blur" min="5" max="100" value="50">
                        <select id="select-multi-id-mode"><option value="primary" selected>Primary</option></select>
                        <div id="bar-fps"></div>
                        <span id="telemetry-fps-stat"></span>
                    </div>
                </div>
            </div>

            <!-- Tab 5: Live Telemetry & Hardware Health -->
            <div class="dock-tab-content" id="tab-telemetry">
                <div class="dock-card">
                    <div class="card-heading-row">
                        <h3 class="card-title">📊 Real-Time Engine Telemetry</h3>
                    </div>

                    <div class="telemetry-grid">
                        <div class="telem-metric-card">
                            <span class="metric-label">Framerate</span>
                            <span class="metric-big-val" id="telem-fps-val">30.0</span>
                            <span class="metric-unit">FPS</span>
                        </div>
                        <div class="telem-metric-card">
                            <span class="metric-label">Swap Latency</span>
                            <span class="metric-big-val" id="time-swap">0.0</span>
                            <span class="metric-unit">ms</span>
                        </div>
                        <div class="telem-metric-card">
                            <span class="metric-label">Face Tracking</span>
                            <span class="metric-big-val" id="time-tracking">0.0</span>
                            <span class="metric-unit">ms</span>
                        </div>
                        <div class="telem-metric-card">
                            <span class="metric-label">Alpha Blending</span>
                            <span class="metric-big-val" id="time-blend">0.0</span>
                            <span class="metric-unit">ms</span>
                        </div>
                    </div>

                    <div class="telem-details-list">
                        <div class="telem-row">
                            <span>Active Provider</span>
                            <strong id="hw-device-val">CPUExecutionProvider</strong>
                        </div>
                        <div class="telem-row">
                            <span>Governor State</span>
                            <strong id="hw-gov-val">Optimal</strong>
                        </div>
                        <div class="telem-row">
                            <span>Alignment Latency</span>
                            <span id="time-align">0.0 ms</span>
                        </div>
                        <div class="telem-row">
                            <span>Enhancement Latency</span>
                            <span id="time-enh">0.0 ms</span>
                        </div>
                        <div class="telem-row">
                            <span>Color Harmonization</span>
                            <span id="time-color">0.0 ms</span>
                        </div>
                    </div>
                </div>
            </div>
        </aside>
    </main>

    <!-- Upload Custom Face Modal -->
    <div class="modal-backdrop" id="modal-upload" style="display: none;">
        <div class="modal-card">
            <div class="modal-header">
                <h3>🎭 Upload Custom Identity</h3>
                <button class="modal-close-btn" id="btn-modal-close">✕</button>
            </div>
            <form id="form-upload-target" class="modal-body">
                <div class="form-field">
                    <label for="input-target-name">Identity / Actor Name</label>
                    <input type="text" id="input-target-name" name="name" placeholder="e.g. Tony Stark" required>
                </div>
                <div class="form-field">
                    <label for="select-target-category">Category</label>
                    <select id="select-target-category" name="category">
                        <option value="custom" selected>Custom</option>
                        <option value="actors">Actors</option>
                        <option value="actresses">Actresses</option>
                        <option value="telugu_heroes">Telugu Heroes</option>
                    </select>
                </div>
                <div class="form-field">
                    <label for="input-target-file">Reference Portrait Photo (Clear front-facing)</label>
                    <input type="file" id="input-target-file" name="file" accept="image/jpeg,image/png,image/webp" required>
                </div>
                <div class="modal-actions">
                    <button type="button" class="btn-cancel" id="btn-modal-cancel">Cancel</button>
                    <button type="submit" class="btn-submit" id="btn-modal-submit">Upload & Select</button>
                </div>
            </form>
        </div>
    </div>

    <!-- Output Lightbox Modal -->
    <div class="modal-backdrop" id="modal-lightbox" style="display: none;">
        <div class="modal-card modal-lightbox-card">
            <div class="modal-header">
                <h3 id="lightbox-title">Output Preview</h3>
                <button class="modal-close-btn" id="btn-lightbox-close">✕</button>
            </div>
            <div class="lightbox-body">
                <img id="lightbox-img" src="" alt="Full Output Preview">
                <div class="lightbox-meta">
                    <span id="lightbox-desc"></span>
                    <a id="btn-lightbox-download" class="btn-download" href="#" target="_blank" download>⬇ Open Full Resolution</a>
                </div>
            </div>
        </div>
    </div>

    <!-- Toast Notification Container -->
    <div class="toast-stack" id="toast-stack"></div>

    <!-- Client Script -->
    <script src="/static/js/studio.js?v=2.5"></script>
</body>
</html>
"""

with open(INDEX_PATH, "w", encoding="utf-8") as f:
    f.write(html_content)

print("Built fully embedded index.html successfully.")
