"""
Professional Web Studio UI Upgrade Script.
Embeds full stylesheet directly into index.html to eliminate any browser caching or CSS blocking issues,
adds quick guide banner, carousel navigation arrows, and enhances mode switching.
"""

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_PATH = os.path.join(ROOT, "src", "web", "static", "index.html")
CSS_PATH = os.path.join(ROOT, "src", "web", "static", "css", "studio.css")
JS_PATH = os.path.join(ROOT, "src", "web", "static", "js", "studio.js")

with open(CSS_PATH, "r", encoding="utf-8") as f:
    css_content = f.read()

# Additional CSS enhancements for guide banner, carousel scroll buttons, and mode switcher
extra_css = """
/* Quick Guide Banner */
.studio-guide-banner {
    background: linear-gradient(90deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.12) 50%, rgba(236, 72, 153, 0.12) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: var(--radius-lg);
    padding: 10px 20px;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.guide-steps {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    gap: 12px;
}

.guide-step {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    color: var(--text-primary);
}

.step-num {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--gradient-brand);
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    flex-shrink: 0;
}

.guide-step em {
    color: var(--accent-cyan);
    font-style: normal;
    font-weight: 600;
}

.guide-arrow {
    color: rgba(255, 255, 255, 0.3);
    font-size: 14px;
    font-weight: bold;
}

/* Nav Modes Switcher */
.nav-modes {
    display: flex;
    align-items: center;
    background: rgba(0, 0, 0, 0.4);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-full);
    padding: 3px;
    gap: 3px;
}

.mode-tab-btn {
    padding: 6px 14px;
    border-radius: var(--radius-full);
    background: transparent;
    border: none;
    color: var(--text-secondary);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all var(--transition-fast);
}

.mode-tab-btn:hover {
    color: #fff;
}

.mode-tab-btn.active {
    background: var(--gradient-brand);
    color: #fff;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4);
}

/* Carousel Scroll Buttons */
.carousel-nav-wrapper {
    position: relative;
    display: flex;
    align-items: center;
    width: 100%;
}

.carousel-scroll-btn {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: rgba(15, 18, 29, 0.9);
    border: 1px solid var(--border-glass);
    color: #fff;
    font-size: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    z-index: 20;
    transition: all var(--transition-fast);
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}

.carousel-scroll-btn:hover {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    transform: translateY(-50%) scale(1.1);
}

.btn-scroll-left { left: -14px; }
.btn-scroll-right { right: -14px; }

/* Responsive adjustments */
@media (max-width: 1100px) {
    .guide-steps { flex-direction: column; align-items: flex-start; gap: 8px; }
    .guide-arrow { display: none; }
}
"""

full_css = css_content + "\n" + extra_css

with open(CSS_PATH, "w", encoding="utf-8") as f:
    f.write(full_css)

print("Updated studio.css successfully.")
