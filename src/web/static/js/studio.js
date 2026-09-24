/**
 * Real-Time AI Face Swap Web Studio — Client Logic (Vanilla JS)
 */

document.addEventListener("DOMContentLoaded", () => {
    // State
    const state = {
        activeCategory: "all",
        targets: [],
        selectedTargetId: null,
        isSplitView: false,
        isSwappingEnabled: true,
        isSynthetic: false,
        telemetryTimer: null,
    };

    // DOM Elements
    const elements = {
        headerFps: document.getElementById("header-fps-text"),
        headerHw: document.getElementById("header-hw-text"),
        headerGov: document.getElementById("header-gov-text"),
        hudFps: document.getElementById("hud-fps"),
        hudLatency: document.getElementById("hud-latency"),
        hudViewers: document.getElementById("hud-viewers"),
        activeTargetBadge: document.getElementById("active-target-badge"),
        statusMessageText: document.getElementById("status-message-text"),
        barFps: document.getElementById("bar-fps"),
        telemetryFpsStat: document.getElementById("telemetry-fps-stat"),

        // Timings
        timeTracking: document.getElementById("time-tracking"),
        timeAlign: document.getElementById("time-align"),
        timeSwap: document.getElementById("time-swap"),
        timeEnh: document.getElementById("time-enh"),
        timeColor: document.getElementById("time-color"),
        timeBlend: document.getElementById("time-blend"),

        // Hardware
        hwDeviceVal: document.getElementById("hw-device-val"),
        hwGovVal: document.getElementById("hw-gov-val"),

        // Video & Viewport
        liveVideoFeed: document.getElementById("live-video-feed"),
        splitContainer: document.getElementById("split-container"),
        splitSwapped: document.getElementById("split-swapped"),
        splitDivider: document.getElementById("split-divider"),
        btnSplitToggle: document.getElementById("btn-split-toggle"),
        btnFullscreen: document.getElementById("btn-fullscreen"),
        viewportCard: document.getElementById("viewport-card"),

        // Buttons
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
        btnResetParams: document.getElementById("btn-reset-params"),

        // Carousel & Tabs
        categoryTabs: document.getElementById("category-tabs"),
        targetCarousel: document.getElementById("target-carousel"),
        cardUploadTrigger: document.getElementById("card-upload-trigger"),

        // Sliders & Inputs
        sliderEnhancement: document.getElementById("slider-enhancement"),
        valEnhancement: document.getElementById("val-enhancement"),
        sliderEyeGaze: document.getElementById("slider-eye-gaze"),
        valEyeGaze: document.getElementById("val-eye-gaze"),
        sliderMouth: document.getElementById("slider-mouth"),
        valMouth: document.getElementById("val-mouth"),
        sliderExpression: document.getElementById("slider-expression"),
        valExpression: document.getElementById("val-expression"),
        sliderLighting: document.getElementById("slider-lighting"),
        valLighting: document.getElementById("val-lighting"),
        sliderOcclusion: document.getElementById("slider-occlusion"),
        valOcclusion: document.getElementById("val-occlusion"),
        selectColorCorrection: document.getElementById("select-color-correction"),
        presetPillGroup: document.getElementById("preset-pill-group"),

        // Captures & Recordings Drawer
        tabDrawCaptures: document.getElementById("tab-draw-captures"),
        tabDrawRecordings: document.getElementById("tab-draw-recordings"),
        capturesDrawer: document.getElementById("captures-drawer"),
        emptyCapturesText: document.getElementById("empty-captures-text"),
        recordingsDrawer: document.getElementById("recordings-drawer"),
        emptyRecordingsText: document.getElementById("empty-recordings-text"),

        // Modal
        uploadModal: document.getElementById("upload-modal"),
        btnModalClose: document.getElementById("btn-modal-close"),
        btnModalCancel: document.getElementById("btn-modal-cancel"),
        uploadForm: document.getElementById("upload-form"),
        inputTargetName: document.getElementById("input-target-name"),
        inputTargetCategory: document.getElementById("input-target-category"),
        inputTargetFile: document.getElementById("input-target-file"),
        filePreviewBox: document.getElementById("file-preview-box"),
        filePreviewImg: document.getElementById("file-preview-img"),

        // Toast Container
        toastContainer: document.getElementById("toast-container"),
    };

    // =========================================================================
    // Toast Notifications
    // =========================================================================
    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        toast.innerHTML = `<span>${type === 'success' ? '✓' : 'ℹ'}</span><span>${message}</span>`;
        elements.toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateY(12px)";
            setTimeout(() => toast.remove(), 300);
        }, 3200);
    }

    // =========================================================================
    // API Services
    // =========================================================================
    async function fetchTelemetry() {
        try {
            const res = await fetch("/api/telemetry");
            if (!res.ok) return;
            const data = await res.json();
            updateTelemetryUI(data);
        } catch (e) {
            console.warn("Telemetry fetch error:", e);
        }
    }

    async function fetchTargets() {
        try {
            const res = await fetch("/api/targets");
            if (!res.ok) return;
            const data = await res.json();
            state.targets = data.targets || [];
            state.selectedTargetId = data.selected_id;
            renderTargetCarousel();
        } catch (e) {
            console.error("Targets fetch error:", e);
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
                showToast(`Switched active face to: ${targetId || "None"}`, "success");
            }
        } catch (e) {
            showToast("Failed to switch target identity", "info");
        }
    }

    // Debounced config updater
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
        }, 150);
    }

    async function triggerCapture() {
        try {
            const res = await fetch("/api/capture", { method: "POST" });
            const data = await res.json();
            if (data.success) {
                showToast(`Snapshot captured: ${data.filename}`, "success");
                addCaptureToDrawer(data.url, data.filename);
            }
        } catch (e) {
            showToast("Capture failed", "info");
        }
    }

    async function toggleSource() {
        try {
            const res = await fetch("/api/source/toggle", { method: "POST" });
            const data = await res.json();
            if (data.success) {
                state.isSynthetic = data.is_synthetic;
                elements.sourceToggleText.textContent = `Source: ${data.is_synthetic ? "Synthetic" : "Webcam"}`;
                showToast(`Source set to ${data.mode}`, "info");
            }
        } catch (e) {
            showToast("Could not toggle source", "info");
        }
    }

    // =========================================================================
    // UI Update Functions
    // =========================================================================
    function updateTelemetryUI(data) {
        const fps = (data.fps || 0.0).toFixed(1);
        const targetFps = data.target_fps || 30;
        elements.headerFps.textContent = `${fps} FPS`;
        elements.hudFps.textContent = fps;

        const fpsPercent = Math.min(100, Math.round((data.fps / targetFps) * 100));
        elements.barFps.style.width = `${fpsPercent}%`;
        elements.telemetryFpsStat.textContent = `${fps} / ${targetFps} FPS`;

        // Latency
        const timings = data.stage_timings || {};
        const totalMs = (timings.total_ms || 0.0).toFixed(1);
        elements.hudLatency.textContent = `${totalMs} ms`;
        elements.hudViewers.textContent = data.active_clients || 1;

        elements.timeTracking.textContent = `${(timings.tracking_ms || 0.0).toFixed(1)} ms`;
        elements.timeAlign.textContent = `${(timings.alignment_ms || 0.0).toFixed(1)} ms`;
        elements.timeSwap.textContent = `${(timings.swap_ms || 0.0).toFixed(1)} ms`;
        elements.timeEnh.textContent = `${(timings.enhancement_ms || 0.0).toFixed(1)} ms`;
        elements.timeColor.textContent = `${(timings.color_ms || 0.0).toFixed(1)} ms`;
        elements.timeBlend.textContent = `${(timings.blend_ms || 0.0).toFixed(1)} ms`;

        // Hardware & Governor
        const hw = data.hardware || {};
        const devName = hw.device || "CPU";
        elements.headerHw.textContent = devName.toUpperCase();
        elements.hwDeviceVal.textContent = devName;

        const gov = (data.governor_state || "optimal").toUpperCase();
        elements.headerGov.textContent = gov;
        elements.hwGovVal.textContent = gov;

        // Target badge
        const target = data.active_target;
        if (target && target.name && target.name !== "None") {
            elements.activeTargetBadge.textContent = `Target: ${target.name}`;
        } else {
            elements.activeTargetBadge.textContent = "Target: None";
        }

        // Status message
        elements.statusMessageText.textContent = data.message || "Active";
    }

    function renderTargetCarousel() {
        // Clear all except upload card
        const cards = elements.targetCarousel.querySelectorAll(".target-card:not(.upload-card)");
        cards.forEach(c => c.remove());

        const filtered = state.targets.filter(t => {
            if (state.activeCategory === "all") return true;
            return t.category === state.activeCategory;
        });

        filtered.forEach(t => {
            const card = document.createElement("div");
            const isSelected = (t.id === state.selectedTargetId);
            card.className = `target-card ${isSelected ? "selected" : ""}`;
            card.dataset.id = t.id;

            card.innerHTML = `
                <div class="target-avatar-wrapper">
                    <img class="target-avatar" src="${t.image_url}" alt="${t.name}" loading="lazy" onerror="this.src='/static/icons/default_avatar.png'">
                </div>
                <span class="target-name">${t.name}</span>
                <span class="target-category-tag">${t.category}</span>
            `;

            card.addEventListener("click", () => {
                selectTarget(t.id);
            });

            elements.targetCarousel.appendChild(card);
        });
    }

    function addCaptureToDrawer(url, filename) {
        if (elements.emptyCapturesText) {
            elements.emptyCapturesText.style.display = "none";
        }
        const item = document.createElement("div");
        item.className = "capture-item";
        item.title = `Click to download ${filename}`;
        item.innerHTML = `<img src="${url}" alt="${filename}">`;
        item.addEventListener("click", () => {
            const a = document.createElement("a");
            a.href = url;
            a.download = filename;
            a.click();
        });
        elements.capturesDrawer.prepend(item);
    }

    // =========================================================================
    // Split View Slider Dragging
    // =========================================================================
    let isDraggingSplit = false;

    elements.btnSplitToggle.addEventListener("click", () => {
        state.isSplitView = !state.isSplitView;
        if (state.isSplitView) {
            elements.liveVideoFeed.style.display = "none";
            elements.splitContainer.style.display = "block";
            elements.btnSplitToggle.classList.add("active");
            showToast("Split comparison active — Drag middle slider", "info");
        } else {
            elements.liveVideoFeed.style.display = "block";
            elements.splitContainer.style.display = "none";
            elements.btnSplitToggle.classList.remove("active");
        }
    });

    elements.splitDivider.addEventListener("mousedown", (e) => {
        isDraggingSplit = true;
        e.preventDefault();
    });

    window.addEventListener("mouseup", () => {
        isDraggingSplit = false;
    });

    window.addEventListener("mousemove", (e) => {
        if (!isDraggingSplit || !state.isSplitView) return;
        const rect = elements.splitContainer.getBoundingClientRect();
        const offsetX = e.clientX - rect.left;
        let percent = (offsetX / rect.width) * 100;
        percent = Math.max(5, Math.min(95, percent));

        elements.splitDivider.style.left = `${percent}%`;
        elements.splitSwapped.style.clipPath = `inset(0 0 0 ${percent}%)`;
    });

    // Touch support for split slider
    elements.splitDivider.addEventListener("touchstart", (e) => {
        isDraggingSplit = true;
    });
    window.addEventListener("touchend", () => {
        isDraggingSplit = false;
    });
    window.addEventListener("touchmove", (e) => {
        if (!isDraggingSplit || !state.isSplitView) return;
        const touch = e.touches[0];
        const rect = elements.splitContainer.getBoundingClientRect();
        const offsetX = touch.clientX - rect.left;
        let percent = (offsetX / rect.width) * 100;
        percent = Math.max(5, Math.min(95, percent));

        elements.splitDivider.style.left = `${percent}%`;
        elements.splitSwapped.style.clipPath = `inset(0 0 0 ${percent}%)`;
    });

    // Fullscreen toggle
    elements.btnFullscreen.addEventListener("click", () => {
        const target = elements.viewportCard;
        if (!document.fullscreenElement) {
            target.requestFullscreen().catch(err => alert(err.message));
        } else {
            document.exitFullscreen();
        }
    });

    // =========================================================================
    // Category Tabs Filtering
    // =========================================================================
    elements.categoryTabs.addEventListener("click", (e) => {
        const btn = e.target.closest(".tab-btn");
        if (!btn) return;

        elements.categoryTabs.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.activeCategory = btn.dataset.category || "all";
        renderTargetCarousel();
    });

    // =========================================================================
    // Slider & Input Bindings
    // =========================================================================
    elements.sliderEnhancement.addEventListener("input", (e) => {
        elements.valEnhancement.textContent = `${e.target.value}%`;
        sendConfigUpdate({ enhancement_strength: parseFloat(e.target.value) / 100.0 });
    });

    elements.sliderEyeGaze.addEventListener("input", (e) => {
        elements.valEyeGaze.textContent = `${e.target.value}%`;
        sendConfigUpdate({ eye_realism_strength: parseFloat(e.target.value) / 100.0 });
    });

    elements.sliderMouth.addEventListener("input", (e) => {
        elements.valMouth.textContent = `${e.target.value}%`;
        sendConfigUpdate({ mouth_preservation_strength: parseFloat(e.target.value) / 100.0 });
    });

    if (elements.sliderExpression) {
        elements.sliderExpression.addEventListener("input", (e) => {
            elements.valExpression.textContent = `${e.target.value}%`;
            sendConfigUpdate({ expression_transfer_strength: parseFloat(e.target.value) / 100.0 });
        });
    }

    elements.sliderLighting.addEventListener("input", (e) => {
        elements.valLighting.textContent = `${e.target.value}%`;
        sendConfigUpdate({ specular_lighting_strength: parseFloat(e.target.value) / 100.0 });
    });

    elements.sliderOcclusion.addEventListener("input", (e) => {
        elements.valOcclusion.textContent = `${e.target.value}%`;
        sendConfigUpdate({ occlusion_sensitivity: parseFloat(e.target.value) / 100.0 });
    });

    elements.selectColorCorrection.addEventListener("change", (e) => {
        sendConfigUpdate({ color_correction: e.target.value });
    });

    // Preset Pills
    elements.presetPillGroup.addEventListener("click", (e) => {
        const btn = e.target.closest(".preset-btn");
        if (!btn) return;
        elements.presetPillGroup.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        sendConfigUpdate({ color_grading_preset: btn.dataset.preset });
    });

    // Reset parameters button
    elements.btnResetParams.addEventListener("click", () => {
        elements.sliderEnhancement.value = 40; elements.valEnhancement.textContent = "40%";
        elements.sliderEyeGaze.value = 70; elements.valEyeGaze.textContent = "70%";
        elements.sliderMouth.value = 65; elements.valMouth.textContent = "65%";
        if (elements.sliderExpression) {
            elements.sliderExpression.value = 65; elements.valExpression.textContent = "65%";
        }
        elements.sliderLighting.value = 50; elements.valLighting.textContent = "50%";
        elements.sliderOcclusion.value = 50; elements.valOcclusion.textContent = "50%";
        elements.selectColorCorrection.value = "reinhard";
        
        elements.presetPillGroup.querySelectorAll(".preset-btn").forEach(b => {
            b.classList.toggle("active", b.dataset.preset === "neutral");
        });

        sendConfigUpdate({
            enhancement_strength: 0.40,
            eye_realism_strength: 0.70,
            mouth_preservation_strength: 0.65,
            expression_transfer_strength: 0.65,
            specular_lighting_strength: 0.50,
            occlusion_sensitivity: 0.50,
            color_correction: "reinhard",
            color_grading_preset: "neutral",
        });
        showToast("Parameters reset to optimal defaults", "info");
    });

    // Toggle Swapping
    elements.btnToggleSwap.addEventListener("click", () => {
        state.isSwappingEnabled = !state.isSwappingEnabled;
        elements.btnSwapText.textContent = state.isSwappingEnabled ? "Disable Swap" : "Enable Swap";
        sendConfigUpdate({ enable_swapping: state.isSwappingEnabled });
        showToast(state.isSwappingEnabled ? "Neural Face Swapping Enabled" : "Swapping Bypassed (Preview Only)", "info");
    });

    // Capture Buttons
    elements.btnQuickCapture.addEventListener("click", triggerCapture);
    elements.btnCaptureMain.addEventListener("click", triggerCapture);

    // Source Toggle Button
    elements.btnToggleSource.addEventListener("click", toggleSource);

    // =========================================================================
    // Live MP4 Video Recording Controls
    // =========================================================================
    let recordTimerInterval = null;
    let recordStartSeconds = 0;

    async function toggleRecording() {
        if (!state.isRecording) {
            try {
                const res = await fetch("/api/recording/start", { method: "POST" });
                const data = await res.json();
                if (data.success) {
                    state.isRecording = true;
                    elements.btnRecordToggle.classList.add("is-recording");
                    elements.recordBtnText.textContent = "Stop Recording";
                    elements.recordTimer.style.display = "inline-block";
                    recordStartSeconds = Date.now();
                    elements.recordTimer.textContent = "00:00";
                    recordTimerInterval = setInterval(() => {
                        const elapsed = Math.floor((Date.now() - recordStartSeconds) / 1000);
                        const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
                        const secs = String(elapsed % 60).padStart(2, "0");
                        elements.recordTimer.textContent = `${mins}:${secs}`;
                    }, 500);
                    showToast("🔴 MP4 Recording started", "info");
                } else {
                    showToast(data.message || "Could not start recording", "info");
                }
            } catch (err) {
                showToast(`Recording error: ${err.message}`, "info");
            }
        } else {
            try {
                const res = await fetch("/api/recording/stop", { method: "POST" });
                const data = await res.json();
                state.isRecording = false;
                elements.btnRecordToggle.classList.remove("is-recording");
                elements.recordBtnText.textContent = "Record Video";
                elements.recordTimer.style.display = "none";
                if (recordTimerInterval) {
                    clearInterval(recordTimerInterval);
                    recordTimerInterval = null;
                }
                if (data.success && data.url) {
                    showToast(`✅ Video saved: ${data.filename}`, "success");
                    await fetchRecordings();
                } else {
                    showToast("Recording finalized", "info");
                }
            } catch (err) {
                showToast(`Error stopping recording: ${err.message}`, "info");
            }
        }
    }

    if (elements.btnRecordToggle) {
        elements.btnRecordToggle.addEventListener("click", toggleRecording);
    }

    // Drawer Tabs switching
    if (elements.tabDrawCaptures && elements.tabDrawRecordings) {
        elements.tabDrawCaptures.addEventListener("click", () => {
            elements.tabDrawCaptures.classList.add("active");
            elements.tabDrawRecordings.classList.remove("active");
            elements.capturesDrawer.style.display = "grid";
            elements.recordingsDrawer.style.display = "none";
        });
        elements.tabDrawRecordings.addEventListener("click", () => {
            elements.tabDrawRecordings.classList.add("active");
            elements.tabDrawCaptures.classList.remove("active");
            elements.capturesDrawer.style.display = "none";
            elements.recordingsDrawer.style.display = "block";
            fetchRecordings();
        });
    }

    async function fetchRecordings() {
        try {
            const res = await fetch("/api/recordings");
            const data = await res.json();
            const list = data.recordings || [];
            if (!elements.recordingsDrawer) return;

            if (list.length === 0) {
                elements.recordingsDrawer.innerHTML = `
                    <div class="empty-captures">No MP4 videos recorded yet. Click ⏺️ Record Video to record live streams.</div>
                `;
                return;
            }

            elements.recordingsDrawer.innerHTML = "";
            list.forEach(item => {
                const card = document.createElement("div");
                card.className = "recording-card-item";
                card.innerHTML = `
                    <div class="recording-info">
                        <span class="recording-name">${item.filename}</span>
                        <span class="recording-meta">${item.size_mb} MB • ${item.modified_at ? item.modified_at.slice(0, 16).replace("T", " ") : ""}</span>
                    </div>
                    <a href="${item.url}" download="${item.filename}" class="recording-download-btn">⬇️ Download</a>
                `;
                elements.recordingsDrawer.appendChild(card);
            });
        } catch (err) {
            console.error("Could not fetch recordings:", err);
        }
    }

    // =========================================================================
    // Upload Modal Handling
    // =========================================================================
    elements.cardUploadTrigger.addEventListener("click", () => {
        elements.uploadModal.style.display = "flex";
    });

    function closeModal() {
        elements.uploadModal.style.display = "none";
        elements.uploadForm.reset();
        elements.filePreviewBox.style.display = "none";
    }

    elements.btnModalClose.addEventListener("click", closeModal);
    elements.btnModalCancel.addEventListener("click", closeModal);

    elements.inputTargetFile.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (evt) => {
                elements.filePreviewImg.src = evt.target.result;
                elements.filePreviewBox.style.display = "block";
            };
            reader.readAsDataURL(file);
        }
    });

    elements.uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const file = elements.inputTargetFile.files[0];
        const name = elements.inputTargetName.value.trim();
        const category = elements.inputTargetCategory.value;

        if (!file || !name) {
            alert("Please provide both name and portrait photo.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);
        formData.append("name", name);
        formData.append("category", category);

        try {
            showToast("Processing and extracting face embeddings...", "info");
            const res = await fetch("/api/target/upload", {
                method: "POST",
                body: formData,
            });
            const data = await res.json();
            if (data.success) {
                closeModal();
                showToast(`Target '${data.name}' added successfully!`, "success");
                await fetchTargets();
            } else {
                alert("Upload failed.");
            }
        } catch (err) {
            alert(`Error uploading target: ${err.message}`);
        }
    });

    // =========================================================================
    // Initialization & Heartbeat Polling / WebSocket Telemetry
    // =========================================================================
    function connectWebSocket() {
        try {
            const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
            const ws = new WebSocket(wsUrl);

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    updateTelemetryUI(data);
                } catch (e) {}
            };

            ws.onerror = () => {
                ws.close();
            };

            ws.onclose = () => {
                if (!state.telemetryTimer) {
                    state.telemetryTimer = setInterval(fetchTelemetry, 800);
                }
            };
        } catch (e) {
            state.telemetryTimer = setInterval(fetchTelemetry, 800);
        }
    }

    fetchTargets();
    fetchTelemetry();
    connectWebSocket();
});
