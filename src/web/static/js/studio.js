/**
 * FaceMorph Studio — Interactive Client Logic (Vanilla JS)
 * Ultra-Responsive Real-Time AI Face Swap Studio
 */

document.addEventListener("DOMContentLoaded", () => {
    // Application State
    const state = {
        activeCategory: "all",
        searchQuery: "",
        targets: [],
        selectedTargetId: null,
        isSplitView: false,
        isSwappingEnabled: true,
        isSynthetic: false,
        selectedPhotoFile: null,
        isPhotoSwapping: false,
        telemetryTimer: null,
        outputs: [],
        activeOutputFilter: "all",
    };

    // DOM References
    const elements = {
        // Headers & Badges
        headerFps: document.getElementById("header-fps-text"),
        headerHw: document.getElementById("header-hw-text"),
        headerGov: document.getElementById("header-gov-text"),
        hudFps: document.getElementById("hud-fps"),
        hudLatency: document.getElementById("hud-latency"),
        hudViewers: document.getElementById("hud-viewers"),
        activeTargetBadge: document.getElementById("active-target-badge"),
        dockTargetLabel: document.getElementById("dock-target-label"),
        statusMessageText: document.getElementById("status-message-text"),
        dockPipelineStatus: document.getElementById("dock-pipeline-status"),
        actorTotalCount: document.getElementById("actor-total-count"),
        shutterFlash: document.getElementById("shutter-flash"),

        // Video Viewport & Split Deck
        liveVideoFeed: document.getElementById("live-video-feed"),
        splitContainer: document.getElementById("split-container"),
        splitSwapped: document.getElementById("split-swapped"),
        splitOriginal: document.getElementById("split-original"),
        splitDivider: document.getElementById("split-divider"),
        btnSplitToggle: document.getElementById("btn-split-toggle"),
        btnFullscreen: document.getElementById("btn-fullscreen"),
        viewportCard: document.getElementById("viewport-card"),

        // Nav & Action Buttons
        btnToggleSource: document.getElementById("btn-toggle-source"),
        sourceToggleText: document.getElementById("source-toggle-text"),
        btnRecordToggle: document.getElementById("btn-record-toggle"),
        recordIcon: document.getElementById("record-icon"),
        recordBtnText: document.getElementById("record-btn-text"),
        recordTimer: document.getElementById("record-timer"),
        btnQuickCapture: document.getElementById("btn-quick-capture"),
        btnCaptureMain: document.getElementById("btn-capture-main"),
        btnToggleSwap: document.getElementById("btn-toggle-swap"),
        btnSwapText: document.getElementById("btn-swap-text"),
        btnDockPhotoSwap: document.getElementById("btn-dock-photo-swap"),
        btnResetParams: document.getElementById("btn-reset-params"),

        // Carousel, Search & Category Tabs
        categoryTabs: document.getElementById("category-tabs"),
        targetCarousel: document.getElementById("target-carousel"),
        cardUploadTrigger: document.getElementById("card-upload-trigger"),
        actorSearchInput: document.getElementById("actor-search-input"),
        btnClearSearch: document.getElementById("btn-clear-search"),

        // Right Dock Tabs
        dockTabs: document.querySelectorAll(".dock-tab-btn"),
        dockTabContents: document.querySelectorAll(".dock-tab-content"),

        // Photo Swap Studio
        photoDropzone: document.getElementById("photo-dropzone"),
        photoFileInput: document.getElementById("photo-file-input"),
        dropzonePrompt: document.getElementById("dropzone-prompt"),
        photoPreviewBox: document.getElementById("photo-preview-box"),
        uploadedPhotoPreview: document.getElementById("uploaded-photo-preview"),
        btnChangePhoto: document.getElementById("btn-change-photo"),
        btnExecutePhotoSwap: document.getElementById("btn-execute-photo-swap"),
        btnPhotoSwapText: document.getElementById("btn-photo-swap-text"),
        photoResultCard: document.getElementById("photo-result-card"),
        photoResultImg: document.getElementById("photo-result-img"),
        btnDownloadResult: document.getElementById("btn-download-result"),
        demoPillBtns: document.querySelectorAll(".demo-pill-btn"),

        // FX & Tuning Sliders
        toggleStudioLighting: document.getElementById("toggle-studio-lighting"),
        rimColorPills: document.getElementById("rim-color-pills"),
        selectStudioRimColor: document.getElementById("select-studio-rim-color"),
        sliderStudioRimIntensity: document.getElementById("slider-studio-rim-intensity"),
        valStudioRimIntensity: document.getElementById("val-studio-rim-intensity"),
        sliderStudioBackdropDim: document.getElementById("slider-studio-backdrop-dim"),
        valStudioBackdropDim: document.getElementById("val-studio-backdrop-dim"),
        toggleBokeh: document.getElementById("toggle-bokeh"),
        sliderBokehRadius: document.getElementById("slider-bokeh-radius"),
        valBokehRadius: document.getElementById("val-bokeh-radius"),
        presetPills: document.querySelectorAll(".preset-pill"),

        sliderEnhancement: document.getElementById("slider-enhancement"),
        valEnhancement: document.getElementById("val-enhancement"),
        sliderLighting: document.getElementById("slider-lighting"),
        valColorBlend: document.getElementById("val-color-blend"),
        toggleDenseMesh: document.getElementById("toggle-dense-mesh"),
        toggleTemporalCoherence: document.getElementById("toggle-temporal-coherence"),
        sliderSpectaclesToggle: document.getElementById("slider-spectacles-toggle"),

        // Telemetry Tab
        telemFpsVal: document.getElementById("telem-fps-val"),
        timeSwap: document.getElementById("time-swap"),
        timeTracking: document.getElementById("time-tracking"),
        timeBlend: document.getElementById("time-blend"),
        timeAlign: document.getElementById("time-align"),
        timeEnh: document.getElementById("time-enh"),
        timeColor: document.getElementById("time-color"),
        hwDeviceVal: document.getElementById("hw-device-val"),
        hwGovVal: document.getElementById("hw-gov-val"),

        // Upload Modal
        modalUpload: document.getElementById("modal-upload"),
        btnModalClose: document.getElementById("btn-modal-close"),
        btnModalCancel: document.getElementById("btn-modal-cancel"),
        formUploadTarget: document.getElementById("form-upload-target"),

        // Toast Container
        toastStack: document.getElementById("toast-stack"),

        // Outputs Gallery & Lightbox
        btnTabOutputs: document.getElementById("btn-tab-outputs"),
        outputsCountBadge: document.getElementById("outputs-count-badge"),
        outputsFilterPills: document.getElementById("outputs-filter-pills"),
        outputsGalleryGrid: document.getElementById("outputs-gallery-grid"),
        modalLightbox: document.getElementById("modal-lightbox"),
        lightboxTitle: document.getElementById("lightbox-title"),
        lightboxImg: document.getElementById("lightbox-img"),
        lightboxDesc: document.getElementById("lightbox-desc"),
        btnLightboxDownload: document.getElementById("btn-lightbox-download"),
        btnLightboxClose: document.getElementById("btn-lightbox-close"),

        // Carousel Scroll & Mode Buttons
        btnCarouselLeft: document.getElementById("btn-carousel-left"),
        btnCarouselRight: document.getElementById("btn-carousel-right"),
        modeBtnLive: document.getElementById("mode-btn-live"),
        modeBtnPhoto: document.getElementById("mode-btn-photo"),
        modeBtnOutputs: document.getElementById("mode-btn-outputs"),
    };

    // =========================================================================
    // Toast Notification System
    // =========================================================================
    function showToast(message, type = "info") {
        if (!elements.toastStack) return;
        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        const icons = { success: "✓", info: "ℹ", error: "✕" };
        toast.innerHTML = `<span>${icons[type] || "•"}</span><span>${message}</span>`;
        elements.toastStack.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateX(40px)";
            toast.style.transition = "all 0.25s ease-out";
            setTimeout(() => toast.remove(), 250);
        }, 3200);
    }

    // =========================================================================
    // Tab & Mode Switching Navigation
    // =========================================================================
    function switchDockTab(tabId) {
        elements.dockTabs.forEach(b => {
            if (b.dataset.tab === tabId) b.classList.add("active");
            else b.classList.remove("active");
        });
        elements.dockTabContents.forEach(c => {
            if (c.id === tabId) c.classList.add("active");
            else c.classList.remove("active");
        });
    }

    elements.dockTabs.forEach(btn => {
        btn.addEventListener("click", () => {
            switchDockTab(btn.dataset.tab);
        });
    });

    if (elements.modeBtnLive) {
        elements.modeBtnLive.addEventListener("click", () => {
            document.querySelectorAll(".mode-tab-btn").forEach(b => b.classList.remove("active"));
            elements.modeBtnLive.classList.add("active");
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    }

    if (elements.modeBtnPhoto) {
        elements.modeBtnPhoto.addEventListener("click", () => {
            document.querySelectorAll(".mode-tab-btn").forEach(b => b.classList.remove("active"));
            elements.modeBtnPhoto.classList.add("active");
            switchDockTab("tab-photo-swap");
            const dropzone = document.getElementById("photo-dropzone");
            if (dropzone) dropzone.scrollIntoView({ behavior: "smooth", block: "center" });
        });
    }

    if (elements.modeBtnOutputs) {
        elements.modeBtnOutputs.addEventListener("click", () => {
            document.querySelectorAll(".mode-tab-btn").forEach(b => b.classList.remove("active"));
            elements.modeBtnOutputs.classList.add("active");
            switchDockTab("tab-outputs");
            fetchOutputs();
            const gallery = document.getElementById("outputs-gallery-grid");
            if (gallery) gallery.scrollIntoView({ behavior: "smooth", block: "center" });
        });
    }

    if (elements.btnCarouselLeft && elements.targetCarousel) {
        elements.btnCarouselLeft.addEventListener("click", () => {
            elements.targetCarousel.scrollBy({ left: -320, behavior: "smooth" });
        });
    }

    if (elements.btnCarouselRight && elements.targetCarousel) {
        elements.btnCarouselRight.addEventListener("click", () => {
            elements.targetCarousel.scrollBy({ left: 320, behavior: "smooth" });
        });
    }

    if (elements.btnDockPhotoSwap) {
        elements.btnDockPhotoSwap.addEventListener("click", () => {
            switchDockTab("tab-photo-swap");
            const dropzone = document.getElementById("photo-dropzone");
            if (dropzone) dropzone.scrollIntoView({ behavior: "smooth", block: "center" });
        });
    }

    // =========================================================================
    // API & Target Identity Management
    // =========================================================================
    async function fetchTargets() {
        try {
            const res = await fetch("/api/targets");
            const data = await res.json();
            state.targets = data.targets || [];
            state.selectedTargetId = data.selected_id || (state.targets[0] ? state.targets[0].id : null);
            if (elements.actorTotalCount) {
                elements.actorTotalCount.textContent = `(${state.targets.length} Actors)`;
            }
            renderTargetCarousel();
            updateTargetLabels();
        } catch (e) {
            console.error("Failed to fetch targets:", e);
        }
    }

    async function selectTarget(targetId) {
        try {
            const res = await fetch("/api/target/select", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ target_id: targetId }),
            });
            const data = await res.json();
            if (data.success) {
                state.selectedTargetId = targetId;
                renderTargetCarousel();
                updateTargetLabels();
                const sel = state.targets.find(t => t.id === targetId);
                showToast(`Switched active identity to: ${sel ? sel.name : targetId}`, "success");
            }
        } catch (e) {
            showToast("Failed to switch target identity", "error");
        }
    }

    function updateTargetLabels() {
        const sel = state.targets.find(t => t.id === state.selectedTargetId);
        const name = sel ? sel.name : "None";
        if (elements.activeTargetBadge) elements.activeTargetBadge.textContent = name;
        if (elements.dockTargetLabel) elements.dockTargetLabel.textContent = name;
    }

    function renderTargetCarousel() {
        if (!elements.targetCarousel) return;
        const cards = elements.targetCarousel.querySelectorAll(".target-card:not(.upload-custom-card)");
        cards.forEach(c => c.remove());

        const query = state.searchQuery.toLowerCase().trim();
        const filtered = state.targets.filter(t => {
            const matchesCat = (state.activeCategory === "all" || t.category === state.activeCategory);
            const matchesSearch = (!query || t.name.toLowerCase().includes(query) || t.id.toLowerCase().includes(query));
            return matchesCat && matchesSearch;
        });

        filtered.forEach(t => {
            const card = document.createElement("div");
            const isSelected = (t.id === state.selectedTargetId);
            card.className = `target-card ${isSelected ? "selected" : ""}`;
            card.dataset.id = t.id;

            card.innerHTML = `
                <div class="card-avatar-box" style="width: 58px; height: 58px; min-width: 58px; min-height: 58px; max-width: 58px; max-height: 58px; border-radius: 50%; overflow: hidden; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
                    <img class="card-avatar-img" src="${t.image_url}" alt="${t.name}" width="58" height="58" style="width: 58px; height: 58px; object-fit: cover; border-radius: 50%; display: block;" loading="lazy" onerror="this.src='/static/icons/default_avatar.png'">
                </div>
                <span class="card-actor-name" title="${t.name}" style="font-size: 11px; font-weight: 600; text-align: center; max-width: 90px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: block;">${t.name}</span>
                <span class="card-category-tag" style="font-size: 9px; font-weight: 700; text-transform: uppercase;">${t.category}</span>
            `;

            card.addEventListener("click", () => {
                selectTarget(t.id);
            });

            elements.targetCarousel.appendChild(card);
        });
    }

    // Search filter input listener
    if (elements.actorSearchInput) {
        elements.actorSearchInput.addEventListener("input", (e) => {
            state.searchQuery = e.target.value;
            if (elements.btnClearSearch) {
                elements.btnClearSearch.style.display = state.searchQuery ? "block" : "none";
            }
            renderTargetCarousel();
        });
    }

    if (elements.btnClearSearch) {
        elements.btnClearSearch.addEventListener("click", () => {
            elements.actorSearchInput.value = "";
            state.searchQuery = "";
            elements.btnClearSearch.style.display = "none";
            renderTargetCarousel();
        });
    }

    // Category filter pills
    if (elements.categoryTabs) {
        elements.categoryTabs.querySelectorAll(".category-pill").forEach(pill => {
            pill.addEventListener("click", () => {
                elements.categoryTabs.querySelectorAll(".category-pill").forEach(p => p.classList.remove("active"));
                pill.classList.add("active");
                state.activeCategory = pill.dataset.category;
                renderTargetCarousel();
            });
        });
    }

    // =========================================================================
    // Photo Swap Studio (Instant High-Res Image Swapping)
    // =========================================================================
    if (elements.photoDropzone && elements.photoFileInput) {
        elements.photoDropzone.addEventListener("click", () => {
            elements.photoFileInput.click();
        });

        elements.photoDropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            elements.photoDropzone.classList.add("drag-over");
        });

        elements.photoDropzone.addEventListener("dragleave", () => {
            elements.photoDropzone.classList.remove("drag-over");
        });

        elements.photoDropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            elements.photoDropzone.classList.remove("drag-over");
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handlePhotoSelection(e.dataTransfer.files[0]);
            }
        });

        elements.photoFileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                handlePhotoSelection(e.target.files[0]);
            }
        });
    }

    function handlePhotoSelection(file) {
        state.selectedPhotoFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            elements.uploadedPhotoPreview.src = e.target.result;
            elements.dropzonePrompt.style.display = "none";
            elements.photoPreviewBox.style.display = "flex";
            elements.btnExecutePhotoSwap.disabled = false;
            const sel = state.targets.find(t => t.id === state.selectedTargetId);
            const targetName = sel ? sel.name : "Selected Actor";
            elements.btnPhotoSwapText.textContent = `Swap Photo with ${targetName}`;
            showToast("Photo loaded! Click 'Swap Photo' to transform.", "info");
        };
        reader.readAsDataURL(file);
    }

    if (elements.btnChangePhoto) {
        elements.btnChangePhoto.addEventListener("click", (e) => {
            e.stopPropagation();
            elements.photoFileInput.click();
        });
    }

    // Demo portrait buttons
    elements.demoPillBtns.forEach(btn => {
        btn.addEventListener("click", async () => {
            const src = btn.dataset.src;
            try {
                const response = await fetch(src);
                const blob = await response.blob();
                const file = new File([blob], `${btn.dataset.name.toLowerCase().replace(" ", "_")}.jpg`, { type: "image/jpeg" });
                handlePhotoSelection(file);
            } catch (err) {
                showToast("Could not load sample portrait", "error");
            }
        });
    });

    // Execute photo swap action
    if (elements.btnExecutePhotoSwap) {
        elements.btnExecutePhotoSwap.addEventListener("click", async () => {
            if (!state.selectedPhotoFile || state.isPhotoSwapping) return;

            state.isPhotoSwapping = true;
            elements.btnExecutePhotoSwap.disabled = true;
            elements.btnPhotoSwapText.textContent = "Neural Swapping in Progress...";

            try {
                const formData = new FormData();
                formData.append("file", state.selectedPhotoFile);
                if (state.selectedTargetId) {
                    formData.append("target_id", state.selectedTargetId);
                }

                const res = await fetch("/api/photo/swap", {
                    method: "POST",
                    body: formData,
                });

                const data = await res.json();
                if (data.success && data.image_base64) {
                    elements.photoResultImg.src = data.image_base64;
                    elements.btnDownloadResult.href = data.image_base64;
                    elements.btnDownloadResult.download = data.filename || "face_swap_result.jpg";
                    elements.photoResultCard.style.display = "flex";
                    showToast(`Photo transformed to ${data.target_name}!`, "success");
                } else {
                    showToast(data.detail || "Face swap failed on uploaded photo.", "error");
                }
            } catch (err) {
                showToast("Network error during photo swap.", "error");
            } finally {
                state.isPhotoSwapping = false;
                elements.btnExecutePhotoSwap.disabled = false;
                const sel = state.targets.find(t => t.id === state.selectedTargetId);
                elements.btnPhotoSwapText.textContent = `Swap Photo with ${sel ? sel.name : "Selected Actor"}`;
            }
        });
    }

    // =========================================================================
    // Config Update Helpers & Debounce
    // =========================================================================
    let configDebounceTimer = null;
    function sendConfigUpdate(params) {
        clearTimeout(configDebounceTimer);
        configDebounceTimer = setTimeout(async () => {
            try {
                await fetch("/api/pipeline/config", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(params),
                });
            } catch (e) {
                console.error("Config update error:", e);
            }
        }, 120);
    }

    // Sliders bindings
    if (elements.sliderEnhancement) {
        elements.sliderEnhancement.addEventListener("input", (e) => {
            const val = parseInt(e.target.value, 10);
            if (elements.valEnhancement) elements.valEnhancement.textContent = `${val}%`;
            sendConfigUpdate({ enhancement_strength: val / 100.0 });
        });
    }

    if (elements.sliderLighting) {
        elements.sliderLighting.addEventListener("input", (e) => {
            const val = parseInt(e.target.value, 10);
            if (elements.valColorBlend) elements.valColorBlend.textContent = `${val}%`;
            sendConfigUpdate({ color_blend_ratio: val / 100.0 });
        });
    }

    // Studio Rim Lighting
    if (elements.toggleStudioLighting) {
        elements.toggleStudioLighting.addEventListener("change", (e) => {
            sendConfigUpdate({ enable_studio_lighting: e.target.checked });
        });
    }

    if (elements.sliderStudioRimIntensity) {
        elements.sliderStudioRimIntensity.addEventListener("input", (e) => {
            const val = parseInt(e.target.value, 10);
            if (elements.valStudioRimIntensity) elements.valStudioRimIntensity.textContent = `${val}%`;
            sendConfigUpdate({ studio_rim_intensity: val / 100.0 });
        });
    }

    if (elements.sliderStudioBackdropDim) {
        elements.sliderStudioBackdropDim.addEventListener("input", (e) => {
            const val = parseInt(e.target.value, 10);
            if (elements.valStudioBackdropDim) elements.valStudioBackdropDim.textContent = `${val}%`;
            sendConfigUpdate({ studio_backdrop_dim: val / 100.0 });
        });
    }

    // Rim Color Swatches
    if (elements.rimColorPills) {
        elements.rimColorPills.querySelectorAll(".swatch-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                elements.rimColorPills.querySelectorAll(".swatch-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const color = btn.dataset.color;
                if (elements.selectStudioRimColor) elements.selectStudioRimColor.value = color;
                sendConfigUpdate({ studio_rim_color: color });
            });
        });
    }

    // Bokeh Blur
    if (elements.toggleBokeh) {
        elements.toggleBokeh.addEventListener("change", (e) => {
            sendConfigUpdate({ enable_cinematic_bokeh: e.target.checked });
        });
    }

    if (elements.sliderBokehRadius) {
        elements.sliderBokehRadius.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            if (elements.valBokehRadius) elements.valBokehRadius.textContent = `${val.toFixed(0)}px`;
            sendConfigUpdate({ bokeh_blur_radius: val });
        });
    }

    // Color Grading Presets
    elements.presetPills.forEach(pill => {
        pill.addEventListener("click", () => {
            elements.presetPills.forEach(p => p.classList.remove("active"));
            pill.classList.add("active");
            sendConfigUpdate({ color_grading_preset: pill.dataset.preset });
        });
    });

    // Dense Mesh & Coherence switches
    if (elements.toggleDenseMesh) {
        elements.toggleDenseMesh.addEventListener("change", (e) => {
            sendConfigUpdate({ enable_dense_mesh_mask: e.target.checked });
        });
    }

    if (elements.toggleTemporalCoherence) {
        elements.toggleTemporalCoherence.addEventListener("change", (e) => {
            sendConfigUpdate({ enable_temporal_coherence: e.target.checked });
        });
    }

    // Reset Defaults
    if (elements.btnResetParams) {
        elements.btnResetParams.addEventListener("click", () => {
            if (elements.sliderEnhancement) elements.sliderEnhancement.value = 50;
            if (elements.valEnhancement) elements.valEnhancement.textContent = "50%";
            if (elements.sliderLighting) elements.sliderLighting.value = 45;
            if (elements.valColorBlend) elements.valColorBlend.textContent = "45%";
            if (elements.toggleStudioLighting) elements.toggleStudioLighting.checked = false;
            if (elements.toggleBokeh) elements.toggleBokeh.checked = false;
            sendConfigUpdate({
                enhancement_strength: 0.50,
                color_blend_ratio: 0.45,
                enable_studio_lighting: false,
                enable_cinematic_bokeh: false,
                enable_occlusion: false,
                enable_mouth_preservation: false,
                enable_eye_realism: false,
            });
            showToast("Reset to optimal cinematic defaults.", "info");
        });
    }

    // =========================================================================
    // Snapshot Flash & Capture Trigger
    // =========================================================================
    async function triggerCapture() {
        if (elements.shutterFlash) {
            elements.shutterFlash.classList.add("active");
            setTimeout(() => elements.shutterFlash.classList.remove("active"), 120);
        }
        try {
            const res = await fetch("/api/capture", { method: "POST" });
            const data = await res.json();
            if (data.success) {
                showToast(`Snapshot saved: ${data.filename}`, "success");
            }
        } catch (e) {
            showToast("Capture failed", "error");
        }
    }

    if (elements.btnQuickCapture) elements.btnQuickCapture.addEventListener("click", triggerCapture);
    if (elements.btnCaptureMain) elements.btnCaptureMain.addEventListener("click", triggerCapture);

    // Toggle swap on/off
    if (elements.btnToggleSwap) {
        elements.btnToggleSwap.addEventListener("click", () => {
            state.isSwappingEnabled = !state.isSwappingEnabled;
            sendConfigUpdate({ enable_swapping: state.isSwappingEnabled });
            elements.btnSwapText.textContent = state.isSwappingEnabled ? "Swap: ON" : "Swap: OFF";
            showToast(state.isSwappingEnabled ? "Face swapping enabled" : "Face swapping paused (preview only)", "info");
        });
    }

    // Toggle source
    if (elements.btnToggleSource) {
        elements.btnToggleSource.addEventListener("click", async () => {
            try {
                const res = await fetch("/api/source/toggle", { method: "POST" });
                const data = await res.json();
                if (data.success) {
                    state.isSynthetic = data.is_synthetic;
                    elements.sourceToggleText.textContent = data.is_synthetic ? "Synthetic" : "Webcam";
                    showToast(`Switched input to ${data.is_synthetic ? "Synthetic Benchmark Loop" : "Physical Webcam"}`, "info");
                }
            } catch (e) {
                showToast("Could not toggle camera source", "error");
            }
        });
    }

    // Split Comparison View
    if (elements.btnSplitToggle && elements.splitContainer) {
        elements.btnSplitToggle.addEventListener("click", () => {
            state.isSplitView = !state.isSplitView;
            elements.splitContainer.style.display = state.isSplitView ? "block" : "none";
            elements.liveVideoFeed.style.display = state.isSplitView ? "none" : "block";
        });
    }

    // Split Divider Dragging
    let isDraggingSplit = false;
    if (elements.splitDivider && elements.splitContainer) {
        elements.splitDivider.addEventListener("mousedown", (e) => {
            isDraggingSplit = true;
            e.preventDefault();
        });

        window.addEventListener("mousemove", (e) => {
            if (!isDraggingSplit) return;
            const rect = elements.splitContainer.getBoundingClientRect();
            const posX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
            const pct = (posX / rect.width) * 100;
            elements.splitDivider.style.left = `${pct}%`;
            elements.splitSwapped.style.clipPath = `inset(0 0 0 ${pct}%)`;
        });

        window.addEventListener("mouseup", () => {
            isDraggingSplit = false;
        });
    }

    // Fullscreen Mode
    if (elements.btnFullscreen && elements.viewportCard) {
        elements.btnFullscreen.addEventListener("click", () => {
            if (!document.fullscreenElement) {
                elements.viewportCard.requestFullscreen().catch(() => {});
            } else {
                document.exitFullscreen().catch(() => {});
            }
        });
    }

    // =========================================================================
    // Upload Custom Identity Modal
    // =========================================================================
    if (elements.cardUploadTrigger && elements.modalUpload) {
        elements.cardUploadTrigger.addEventListener("click", () => {
            elements.modalUpload.style.display = "flex";
        });
    }

    if (elements.btnModalClose && elements.modalUpload) {
        elements.btnModalClose.addEventListener("click", () => {
            elements.modalUpload.style.display = "none";
        });
    }

    if (elements.btnModalCancel && elements.modalUpload) {
        elements.btnModalCancel.addEventListener("click", () => {
            elements.modalUpload.style.display = "none";
        });
    }

    if (elements.formUploadTarget) {
        elements.formUploadTarget.addEventListener("submit", async (e) => {
            e.preventDefault();
            const formData = new FormData(elements.formUploadTarget);
            try {
                const res = await fetch("/api/target/upload", {
                    method: "POST",
                    body: formData,
                });
                const data = await res.json();
                if (data.success) {
                    elements.modalUpload.style.display = "none";
                    elements.formUploadTarget.reset();
                    showToast(`Custom face '${data.name}' uploaded and selected!`, "success");
                    await fetchTargets();
                    if (data.target_id) selectTarget(data.target_id);
                } else {
                    showToast("Failed to upload face", "error");
                }
            } catch (err) {
                showToast("Network error uploading target", "error");
            }
        });
    }

    // =========================================================================
    // Live Telemetry Polling / WebSocket
    // =========================================================================
    function startTelemetry() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
        let ws;

        try {
            ws = new WebSocket(wsUrl);
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateTelemetryUI(data);
            };
            ws.onerror = () => {
                fallbackPolling();
            };
            ws.onclose = () => {
                fallbackPolling();
            };
        } catch (e) {
            fallbackPolling();
        }
    }

    function fallbackPolling() {
        if (state.telemetryTimer) return;
        state.telemetryTimer = setInterval(async () => {
            try {
                const res = await fetch("/api/telemetry");
                const data = await res.json();
                updateTelemetryUI(data);
            } catch (e) {
                // Ignore transient polling failure
            }
        }, 1000);
    }

    function updateTelemetryUI(data) {
        const fps = (data.fps || 30.0).toFixed(1);
        if (elements.headerFps) elements.headerFps.textContent = `${fps} FPS`;
        if (elements.hudFps) elements.hudFps.textContent = fps;
        if (elements.telemFpsVal) elements.telemFpsVal.textContent = fps;

        const timings = data.stage_timings || {};
        const totalMs = (timings.total_ms || 215.0).toFixed(1);
        if (elements.hudLatency) elements.hudLatency.textContent = totalMs;

        if (elements.timeTracking) elements.timeTracking.textContent = (timings.tracking_ms || 8.0).toFixed(1);
        if (elements.timeSwap) elements.timeSwap.textContent = (timings.swap_ms || 190.0).toFixed(1);
        if (elements.timeBlend) elements.timeBlend.textContent = (timings.blend_ms || 6.0).toFixed(1);
        if (elements.timeAlign) elements.timeAlign.textContent = `${(timings.alignment_ms || 5.0).toFixed(1)} ms`;
        if (elements.timeEnh) elements.timeEnh.textContent = `${(timings.enhancement_ms || 0.0).toFixed(1)} ms`;
        if (elements.timeColor) elements.timeColor.textContent = `${(timings.color_ms || 4.0).toFixed(1)} ms`;

        if (elements.statusMessageText) elements.statusMessageText.textContent = data.message || "Face Active";
        if (elements.dockPipelineStatus) {
            elements.dockPipelineStatus.textContent = data.is_swapped ? "Identity Swapped" : "Face Active";
        }

        const hw = data.hardware || {};
        const dev = hw.device || "CPU";
        if (elements.headerHw) elements.headerHw.textContent = `${dev.toUpperCase()} ACCELERATED`;
        if (elements.hwDeviceVal) elements.hwDeviceVal.textContent = dev;
        if (elements.hwGovVal) elements.hwGovVal.textContent = (data.governor_state || "optimal").toUpperCase();
    }

    // =========================================================================
    // Outputs Gallery & Lightbox Viewer
    // =========================================================================
    async function fetchOutputs() {
        try {
            const res = await fetch("/api/outputs");
            const data = await res.json();
            state.outputs = data.outputs || [];
            if (elements.outputsCountBadge) {
                elements.outputsCountBadge.textContent = `${state.outputs.length} Items`;
            }
            renderOutputsGallery();
        } catch (e) {
            console.error("Failed to load outputs:", e);
        }
    }

    function renderOutputsGallery() {
        if (!elements.outputsGalleryGrid) return;
        elements.outputsGalleryGrid.innerHTML = "";

        const filtered = state.outputs.filter(item => {
            if (state.activeOutputFilter === "all") return true;
            return item.category === state.activeOutputFilter;
        });

        if (filtered.length === 0) {
            elements.outputsGalleryGrid.innerHTML = `
                <div style="padding: 30px 10px; text-align: center; color: var(--text-muted); font-size: 13px;">
                    No outputs in this category yet.
                </div>
            `;
            return;
        }

        filtered.forEach(item => {
            const card = document.createElement("div");
            card.className = "output-gallery-card";
            card.innerHTML = `
                <div class="output-card-thumb-box">
                    <img class="output-card-thumb-img" src="${item.url}" alt="${item.title}" loading="lazy">
                    <span class="output-badge-tag ${item.category === 'hd_model' ? 'emerald' : (item.category === 'showcase' ? 'purple' : '')}">${item.badge || 'OUTPUT'}</span>
                </div>
                <div class="output-card-content">
                    <div class="output-card-title">${item.title}</div>
                    <div class="output-card-desc">${item.description}</div>
                    <div class="output-card-footer">
                        <span>${item.size_kb ? item.size_kb + ' KB' : ''} ${item.modified_at ? '• ' + item.modified_at : ''}</span>
                        <a href="${item.url}" class="output-open-link" target="_blank" onclick="event.stopPropagation()">Open ↗</a>
                    </div>
                </div>
            `;

            card.addEventListener("click", () => {
                openLightbox(item);
            });

            elements.outputsGalleryGrid.appendChild(card);
        });
    }

    function openLightbox(item) {
        if (!elements.modalLightbox) return;
        if (elements.lightboxTitle) elements.lightboxTitle.textContent = item.title;
        if (elements.lightboxImg) elements.lightboxImg.src = item.url;
        if (elements.lightboxDesc) elements.lightboxDesc.textContent = item.description;
        if (elements.btnLightboxDownload) {
            elements.btnLightboxDownload.href = item.url;
            elements.btnLightboxDownload.setAttribute("download", item.filename || "output.jpg");
        }
        elements.modalLightbox.style.display = "flex";
    }

    if (elements.btnLightboxClose && elements.modalLightbox) {
        elements.btnLightboxClose.addEventListener("click", () => {
            elements.modalLightbox.style.display = "none";
        });
    }

    if (elements.modalLightbox) {
        elements.modalLightbox.addEventListener("click", (e) => {
            if (e.target === elements.modalLightbox) {
                elements.modalLightbox.style.display = "none";
            }
        });
    }

    if (elements.outputsFilterPills) {
        elements.outputsFilterPills.querySelectorAll(".output-filter-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                elements.outputsFilterPills.querySelectorAll(".output-filter-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                state.activeOutputFilter = btn.dataset.filter;
                renderOutputsGallery();
            });
        });
    }

    // Refresh outputs when snapshot or photo swap happens
    const origTriggerCapture = triggerCapture;
    triggerCapture = async function() {
        await origTriggerCapture();
        fetchOutputs();
    };

    // Initialize
    fetchTargets();
    fetchOutputs();
    startTelemetry();
});
