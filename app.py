"""
Multi-Digit Handwritten Number Recognition AI - NiceGUI Web Studio
A state-of-the-art deep learning studio for recognizing multi-digit handwritten numbers
(e.g., 123, 2026, 0075, 45678) using a CRNN (CNN + BiLSTM + CTC) sequence recognition engine.
"""

import os
import sys
import json
import base64
import hashlib
import inspect
from typing import Optional, Dict, Any, List
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from nicegui import ui, app

from src.inference import predict_number, compute_image_sha256
from src.preprocessing import pil_to_base64_data_uri
from src.model import MODEL_PATH, METRICS_PATH

# Configuration & Color Palette
THEME = {
    "navy": "#172B4D",
    "teal": "#0F9488",
    "bg_soft": "#F7F9FC",
    "card_bg": "#FFFFFF",
    "text_dark": "#202B38",
    "text_muted": "#64748B",
    "border": "#E2E8F0",
    "border_light": "#EDF2F7",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
}


def load_metrics_data() -> Optional[Dict[str, Any]]:
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# Custom CSS Injection
ui.add_head_html(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    body {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: {THEME['bg_soft']};
        color: {THEME['text_dark']};
        margin: 0;
        padding: 0;
    }}
    
    .font-mono {{
        font-family: 'JetBrains Mono', monospace;
    }}
    
    .nav-btn {{
        border-radius: 10px;
        transition: all 0.2s ease-in-out;
        font-weight: 600;
        justify-content: flex-start;
        padding: 10px 16px;
        text-transform: none;
        letter-spacing: 0;
    }}
    
    .nav-btn:hover {{
        background-color: rgba(15, 148, 136, 0.12);
        color: #ffffff;
    }}
    
    .nav-btn-active {{
        background-color: {THEME['teal']} !important;
        color: #ffffff !important;
    }}
    
    .studio-card {{
        background-color: {THEME['card_bg']};
        border: 1px solid {THEME['border']};
        border-radius: 14px;
        box-shadow: 0 4px 20px -2px rgba(23, 43, 77, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    
    .studio-card:hover {{
        box-shadow: 0 8px 30px -4px rgba(23, 43, 77, 0.08);
    }}
    
    .stat-badge {{
        background: rgba(15, 148, 136, 0.1);
        color: {THEME['teal']};
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 0.85rem;
        font-weight: 700;
    }}
    
    .q-tab-panel {{
        padding: 0 !important;
        background: transparent !important;
    }}
</style>
<script>
    window.addEventListener('load', function() {{
        const params = new URLSearchParams(window.location.search);
        const tab = params.get('tab');
        if (tab) {{
            setTimeout(() => {{
                const btn = document.querySelector(`[data-nav="${{tab}}"]`);
                if (btn) btn.click();
                const sample = params.get('sample');
                if (sample) {{
                    setTimeout(() => {{
                        const sampleBtn = document.querySelector(`[data-sample="${{sample}}"]`);
                        if (sampleBtn) sampleBtn.click();
                    }}, 600);
                }}
            }}, 600);
        }}
    }});
</script>
""")

# Build Application Layout
with ui.tabs().classes('hidden') as tabs:
    tab_overview = ui.tab('Overview')
    tab_recognizer = ui.tab('Number Recognizer')
    tab_insights = ui.tab('Model Insights')
    tab_about = ui.tab('About Project')

tab_map = {
    "Overview": tab_overview,
    "Number Recognizer": tab_recognizer,
    "Model Insights": tab_insights,
    "About Project": tab_about,
}

nav_buttons = {}

def switch_to(target_name: str):
    panels.value = tab_map[target_name]
    for name, btn in nav_buttons.items():
        if name == target_name:
            btn.classes(add='nav-btn-active', remove='text-slate-300')
        else:
            btn.classes(add='text-slate-300', remove='nav-btn-active')

# Sidebar Navigation
with ui.left_drawer(value=True).classes('bg-[#172B4D] text-white p-4 w-72 flex flex-col justify-between shadow-xl'):
    with ui.column().classes('w-full gap-2'):
        # Brand Header
        with ui.row().classes('items-center gap-3 px-2 py-4 mb-2'):
            with ui.element('div').classes('w-10 h-10 rounded-xl bg-teal-500 flex items-center justify-center shadow-lg'):
                ui.icon('pin', size='24px').classes('text-white')
            with ui.column().classes('gap-0'):
                ui.label('DigitAI Studio').classes('text-lg font-bold text-white tracking-tight')
                ui.label('Multi-Digit CRNN 2.0').classes('text-xs text-teal-300 font-medium')

        ui.separator().classes('bg-slate-700/60 my-2')

        # Navigation Buttons
        items = [
            ("Overview", "dashboard"),
            ("Number Recognizer", "pin"),
            ("Model Insights", "insights"),
            ("About Project", "info"),
        ]

        for name, icon_name in items:
            is_active = (name == "Overview")
            btn = ui.button(
                name,
                icon=icon_name,
                on_click=lambda n=name: switch_to(n)
            ).classes(
                f'w-full nav-btn {"nav-btn-active" if is_active else "text-slate-300"}'
            ).props(f'data-nav="{name}"')
            nav_buttons[name] = btn

    # Footer Status
    with ui.column().classes('w-full p-3 rounded-xl bg-slate-800/80 border border-slate-700 gap-1'):
        with ui.row().classes('items-center justify-between w-full'):
            ui.label('Model Status').classes('text-xs text-slate-400 font-medium')
            model_exists = os.path.exists(MODEL_PATH)
            color = "bg-emerald-500" if model_exists else "bg-amber-500"
            status_text = "Active" if model_exists else "Ready"
            with ui.row().classes('items-center gap-1.5'):
                ui.element('span').classes(f'w-2 h-2 rounded-full {color}')
                ui.label(status_text).classes('text-xs font-semibold text-white')
        ui.label('Architecture: CRNN + CTC').classes('text-[11px] text-teal-300 font-mono')
        ui.label('Engine: TensorFlow 2.21 CPU').classes('text-[11px] text-slate-400')


# Main Content Area driven by Tab Panels
with ui.column().classes('w-full max-w-6xl mx-auto p-6 md:p-8 gap-8'):
    with ui.tab_panels(tabs, value=tab_overview).classes('w-full bg-transparent') as panels:

        # ==============================================================
        # 1. OVERVIEW PANEL
        # ==============================================================
        with ui.tab_panel(tab_overview):
            with ui.column().classes('w-full gap-8'):
                # Welcome Hero Banner
                with ui.card().classes('studio-card w-full p-8 relative overflow-hidden bg-gradient-to-r from-[#172B4D] to-[#1E3A8A] text-white border-0'):
                    with ui.column().classes('gap-3 max-w-2xl z-10'):
                        ui.label('Deep Learning Sequence Recognition').classes('stat-badge bg-teal-500/20 text-teal-300 w-fit')
                        ui.label('Multi-Digit Number Recognition AI').classes('text-3xl md:text-4xl font-extrabold tracking-tight text-white')
                        ui.label(
                            'Recognize complete handwritten numbers (e.g. 123, 2026, 0075, 45678) from single uploaded images '
                            'using an end-to-end CRNN (CNN + Bidirectional LSTM + CTC) deep learning architecture.'
                        ).classes('text-slate-200 text-sm md:text-base leading-relaxed')

                        with ui.row().classes('gap-4 mt-2'):
                            ui.button('Start Number Recognition', icon='auto_awesome', on_click=lambda: switch_to('Number Recognizer')).classes(
                                'bg-[#0F9488] hover:bg-[#0D8276] text-white font-bold px-6 py-3 rounded-xl shadow-lg shadow-teal-900/30 text-sm'
                            )
                            ui.button('Explore Model Insights', icon='insights', on_click=lambda: switch_to('Model Insights')).classes(
                                'bg-white/10 hover:bg-white/20 text-white font-semibold px-5 py-3 rounded-xl border border-white/20 text-sm'
                            )

                # Key Statistics Grid
                with ui.grid().classes('grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 w-full'):
                    stats = [
                        ("1 to 5+", "Sequence Length", "pin", "Variable Length Number Strings"),
                        ("0075", "Leading Zeros", "exposure_zero", "Full String Preservation"),
                        ("32 × 160", "Input Resolution", "aspect_ratio", "CRNN Normalized Canvas"),
                        ("CTC Loss", "Sequence Decoding", "account_tree", "Connectionist Temporal Classif."),
                    ]
                    for val, title, icon_name, desc in stats:
                        with ui.card().classes('studio-card p-5'):
                            with ui.row().classes('items-center justify-between w-full'):
                                ui.label(title).classes('text-xs font-bold text-slate-500 uppercase tracking-wider')
                                ui.icon(icon_name, size='20px').classes('text-teal-600')
                            ui.label(val).classes('text-2xl font-extrabold text-[#172B4D] mt-2')
                            ui.label(desc).classes('text-xs text-slate-500 mt-1')

                # Architecture & Features Section
                with ui.grid().classes('grid-cols-1 md:grid-cols-2 gap-6 w-full'):
                    with ui.card().classes('studio-card p-6 gap-3'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('layers', size='22px').classes('text-teal-600')
                            ui.label('CRNN Architecture').classes('text-lg font-bold text-[#172B4D]')
                        ui.label(
                            'Convolutional Recurrent Neural Network with 4-stage Conv2D feature extractors, '
                            'dual Bidirectional LSTMs for horizontal context, and CTC sequence decoding.'
                        ).classes('text-sm text-slate-600 leading-relaxed')

                        with ui.column().classes('gap-2 mt-2'):
                            for item in [
                                "4 Conv2D + BatchNorm blocks extracting visual features",
                                "Horizontal temporal compression to 40 sequence time-slices",
                                "Dual Bidirectional LSTM (128 units) sequence modeling",
                                "CTC Greedy Decoding handling variable-length numbers",
                            ]:
                                with ui.row().classes('items-center gap-2'):
                                    ui.icon('check_circle', size='16px').classes('text-teal-600')
                                    ui.label(item).classes('text-xs text-slate-700 font-medium')

                    with ui.card().classes('studio-card p-6 gap-3'):
                        with ui.row().classes('items-center gap-2'):
                            ui.icon('tune', size='22px').classes('text-teal-600')
                            ui.label('Data Provenance & Robustness').classes('text-lg font-bold text-[#172B4D]')
                        ui.label(
                            'Strict single-source-of-truth image handling, SHA-256 upload tracking, '
                            'and calibrated uncertainty rejection for degraded or ambiguous handwriting.'
                        ).classes('text-sm text-slate-600 leading-relaxed')

                        with ui.column().classes('gap-2 mt-2'):
                            for item in [
                                "Cryptographic SHA-256 provenance tracking per upload",
                                "Automatic polarity inversion for paper scans vs dark mode",
                                "Calibrated confidence score with uncertainty alerts",
                                "Full preservation of leading zeros (e.g. 0075)",
                            ]:
                                with ui.row().classes('items-center gap-2'):
                                    ui.icon('check_circle', size='16px').classes('text-teal-600')
                                    ui.label(item).classes('text-xs text-slate-700 font-medium')

        # ==============================================================
        # 2. NUMBER RECOGNIZER PANEL
        # ==============================================================
        with ui.tab_panel(tab_recognizer):
            with ui.column().classes('w-full gap-6'):
                ui.label('Multi-Digit Number Recognizer Studio').classes('text-2xl font-extrabold text-[#172B4D]')
                ui.label('Upload an image containing any handwritten number (e.g. 123, 2026, 0075, 45678).').classes('text-sm text-slate-500 -mt-4')

                # Recognizer State
                rec_state = {
                    "bytes": None,
                    "uri": None,
                    "sha256": None,
                }

                # Grid
                with ui.grid().classes('grid-cols-1 lg:grid-cols-12 gap-6 w-full'):
                    # Left Controls (6 cols)
                    with ui.column().classes('lg:col-span-6 gap-5'):
                        # Upload Card
                        with ui.card().classes('studio-card p-6 w-full gap-4'):
                            with ui.row().classes('items-center justify-between w-full'):
                                ui.label('1. Upload Handwritten Number').classes('font-bold text-base text-[#172B4D]')
                                ui.label('PNG / JPG / JPEG').classes('text-xs bg-slate-100 text-slate-600 px-2.5 py-1 rounded-md font-semibold')

                            async def on_upload(e):
                                try:
                                    raw_bytes = None
                                    if hasattr(e, 'file'):
                                        res = e.file.read()
                                        raw_bytes = await res if inspect.isawaitable(res) else res
                                    elif hasattr(e, 'content'):
                                        res = e.content.read()
                                        raw_bytes = await res if inspect.isawaitable(res) else res
                                    else:
                                        raise ValueError(f"Unknown upload structure: {e}")

                                    if not raw_bytes:
                                        ui.notify("Empty file uploaded.", type="warning")
                                        return

                                    rec_state["bytes"] = raw_bytes
                                    rec_state["sha256"] = hashlib.sha256(raw_bytes).hexdigest()

                                    b64 = base64.b64encode(raw_bytes).decode('utf-8')
                                    rec_state["uri"] = f"data:image/png;base64,{b64}"
                                    uploaded_img_view.set_source(rec_state["uri"])
                                    uploaded_img_view.classes(remove='hidden')
                                    uploaded_placeholder.classes(add='hidden')

                                    # Update SHA256 badge
                                    sha_label.set_text(f"SHA-256: {rec_state['sha256'][:16]}...")
                                    sha_badge_row.classes(remove='hidden')

                                    # Hide processed preview until inference completes
                                    processed_img_view.classes(add='hidden')
                                    processed_placeholder.classes(remove='hidden')

                                    # Run inference automatically on the uploaded image
                                    do_predict()
                                except Exception as exc:
                                    print(f"[ERROR in on_upload]: {exc}", flush=True)
                                    ui.notify(f"Upload error: {exc}", type="negative")

                            upload_widget = ui.upload(
                                label="Drop handwritten number image here or click to browse",
                                auto_upload=True,
                                on_upload=on_upload,
                                max_file_size=5_000_000
                            ).classes('w-full').props('accept=".png,.jpg,.jpeg"')

                            # Benchmark Multi-Digit Test Samples (Loads genuine test bytes)
                            sample_dir = os.path.join(os.path.dirname(__file__), "assets", "sample_numbers")
                            sample_files = [
                                ("7", "seq_7.png"),
                                ("42", "seq_42.png"),
                                ("123", "seq_123.png"),
                                ("2026", "seq_2026.png"),
                                ("0075", "seq_0075.png"),
                                ("45678", "seq_45678.png"),
                            ]

                            def load_benchmark_sample(filename: str):
                                path = os.path.join(sample_dir, filename)
                                if os.path.exists(path):
                                    with open(path, "rb") as f:
                                        raw_bytes = f.read()
                                    rec_state["bytes"] = raw_bytes
                                    rec_state["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
                                    b64 = base64.b64encode(raw_bytes).decode('utf-8')
                                    rec_state["uri"] = f"data:image/png;base64,{b64}"
                                    uploaded_img_view.set_source(rec_state["uri"])
                                    uploaded_img_view.classes(remove='hidden')
                                    uploaded_placeholder.classes(add='hidden')
                                    sha_label.set_text(f"SHA-256: {rec_state['sha256'][:16]}...")
                                    sha_badge_row.classes(remove='hidden')
                                    processed_img_view.classes(add='hidden')
                                    processed_placeholder.classes(remove='hidden')
                                    do_predict()

                            with ui.column().classes('w-full gap-1.5 mt-2'):
                                ui.label('Or pick a test sample number:').classes('text-xs font-semibold text-slate-500')
                                with ui.row().classes('w-full gap-2 flex-wrap'):
                                    for label, fname in sample_files:
                                        ui.button(
                                            label,
                                            on_click=lambda fn=fname: load_benchmark_sample(fn)
                                        ).classes(
                                            'bg-slate-100 hover:bg-teal-50 hover:text-teal-700 text-slate-700 text-xs font-mono font-bold px-3 py-1 rounded-lg border border-slate-200 transition-colors'
                                        ).props(f'data-sample="{label}"')

                            # Action Buttons Row
                            with ui.row().classes('w-full gap-3 mt-1'):
                                def do_clear():
                                    rec_state["bytes"] = None
                                    rec_state["uri"] = None
                                    rec_state["sha256"] = None

                                    # Reset previews
                                    uploaded_img_view.classes(add='hidden')
                                    uploaded_placeholder.classes(remove='hidden')
                                    processed_img_view.classes(add='hidden')
                                    processed_placeholder.classes(remove='hidden')
                                    sha_badge_row.classes(add='hidden')

                                    # Reset results to idle
                                    result_idle_box.classes(remove='hidden')
                                    result_active_box.classes(add='hidden')
                                    result_error_box.classes(add='hidden')
                                    uncertain_banner.classes(add='hidden')

                                    ui.notify("Image and results cleared.", type="info")

                                ui.button('Recognize Number', icon='psychology', on_click=lambda: do_predict()).classes(
                                    'flex-1 bg-[#0F9488] hover:bg-[#0D8276] text-white font-bold py-3 rounded-xl shadow-md text-sm'
                                )
                                ui.button('Clear', icon='delete_sweep', on_click=do_clear).classes(
                                    'bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold px-4 py-3 rounded-xl border border-slate-200 text-sm'
                                )

                        # Image Previews Card
                        with ui.card().classes('studio-card p-5 w-full gap-3'):
                            ui.label('Image Previews & Data Provenance').classes('font-bold text-sm text-[#172B4D]')
                            with ui.row().classes('w-full justify-around items-center gap-4'):
                                # Uploaded Preview
                                with ui.column().classes('items-center gap-1'):
                                    ui.label('Uploaded Image').classes('text-xs text-slate-500 font-semibold')
                                    uploaded_img_view = ui.image('').classes('w-44 h-24 object-contain rounded-lg border border-slate-200 bg-white shadow-inner hidden')
                                    with ui.element('div').classes('w-44 h-24 rounded-lg border-2 border-dashed border-slate-200 flex items-center justify-center bg-slate-50 text-slate-400 text-xs text-center p-2') as uploaded_placeholder:
                                        ui.label('No image selected')

                                # Processed Sequence (32x160) Preview
                                with ui.column().classes('items-center gap-1'):
                                    ui.label('Processed (32×160 Sequence)').classes('text-xs text-slate-500 font-semibold')
                                    processed_img_view = ui.image('').classes('w-44 h-24 object-contain rounded-lg border border-slate-700 bg-black shadow-inner hidden')
                                    with ui.element('div').classes('w-44 h-24 rounded-lg border-2 border-dashed border-slate-200 flex items-center justify-center bg-slate-50 text-slate-400 text-xs text-center p-2') as processed_placeholder:
                                        ui.label('Awaiting prediction')

                            # Provenance SHA256 badge
                            with ui.row().classes('items-center justify-center w-full gap-2 text-xs text-slate-500 font-mono hidden') as sha_badge_row:
                                ui.icon('fingerprint', size='16px').classes('text-teal-600')
                                sha_label = ui.label('')

                    # Right Side: Results (6 cols)
                    with ui.column().classes('lg:col-span-6 gap-6'):
                        with ui.card().classes('studio-card p-6 w-full gap-5'):
                            ui.label('Recognition Results').classes('font-bold text-base text-[#172B4D]')

                            # Idle state
                            with ui.column().classes('items-center justify-center py-16 gap-2 text-slate-400 w-full') as result_idle_box:
                                ui.icon('pin', size='48px').classes('text-slate-300')
                                ui.label('Upload a handwritten number to see predictions').classes('text-sm font-medium')

                            # Error state
                            with ui.column().classes('p-4 rounded-xl bg-rose-50 border border-rose-200 gap-2 w-full hidden') as result_error_box:
                                with ui.row().classes('items-center gap-2 text-rose-700 font-bold'):
                                    ui.icon('error', size='22px')
                                    ui.label('Recognition Error')
                                error_msg_label = ui.label('').classes('text-sm text-rose-600')

                            # Active Prediction state
                            with ui.column().classes('w-full gap-4 hidden') as result_active_box:
                                # Uncertainty banner
                                with ui.column().classes('p-3 rounded-xl bg-amber-50 border border-amber-200 gap-1 w-full hidden') as uncertain_banner:
                                    with ui.row().classes('items-center gap-2 text-amber-800 font-bold text-xs'):
                                        ui.icon('warning', size='18px').classes('text-amber-600')
                                        ui.label('Uncertainty Detected')
                                    uncertain_msg_label = ui.label('').classes('text-xs text-amber-700')

                                # Main Result Card
                                with ui.row().classes('items-center justify-between w-full p-4 rounded-xl bg-teal-50 border border-teal-200'):
                                    with ui.column().classes('gap-0'):
                                        ui.label('Recognized Number').classes('text-xs font-bold text-teal-800 uppercase tracking-wider')
                                        pred_badge = ui.label('').classes('text-5xl font-extrabold text-[#0F9488] font-mono tracking-wider')
                                    with ui.column().classes('items-end gap-0'):
                                        ui.label('Sequence Confidence').classes('text-xs font-bold text-teal-800 uppercase tracking-wider')
                                        conf_badge = ui.label('').classes('text-2xl font-black text-slate-800')

                                # Per-Digit Breakdown Section
                                ui.label('Character Breakdown & Timing:').classes('text-xs font-bold text-slate-500 uppercase tracking-wider mt-2')
                                breakdown_container = ui.column().classes('w-full gap-2')

                    def do_predict():
                        if rec_state["bytes"] is None:
                            ui.notify('Please upload an image first.', type='warning')
                            return

                        res = predict_number(rec_state["bytes"])

                        # Update Previews
                        if res.get("processed_image_uri"):
                            processed_img_view.set_source(res["processed_image_uri"])
                            processed_img_view.classes(remove='hidden')
                            processed_placeholder.classes(add='hidden')

                        if not res["success"] and not res.get("uncertain"):
                            # Hard failure (e.g. blank image, unreadable)
                            result_idle_box.classes(add='hidden')
                            result_active_box.classes(add='hidden')
                            result_error_box.classes(remove='hidden')
                            error_msg_label.set_text(res["message"])
                            ui.notify(res["message"], type='negative')
                            return

                        result_idle_box.classes(add='hidden')
                        result_error_box.classes(add='hidden')
                        result_active_box.classes(remove='hidden')

                        # Update Prediction Badge
                        pred_str = res.get("predicted_number") or "?"
                        pred_badge.set_text(pred_str)

                        # Update Calibrated Confidence
                        conf_val = res.get("confidence", 0.0)
                        conf_badge.set_text(f"{conf_val * 100:.1f}%")

                        # Uncertainty warning banner
                        if res.get("uncertain"):
                            uncertain_banner.classes(remove='hidden')
                            uncertain_msg_label.set_text(res["message"])
                            pred_badge.classes(add='text-amber-600', remove='text-[#0F9488]')
                        else:
                            uncertain_banner.classes(add='hidden')
                            pred_badge.classes(add='text-[#0F9488]', remove='text-amber-600')

                        # Rebuild Per-Digit Breakdown
                        breakdown_container.clear()
                        with breakdown_container:
                            details = res.get("per_digit_details", [])
                            if not details:
                                ui.label('No digit character breakdown available.').classes('text-xs text-slate-400')
                            else:
                                for idx, d_info in enumerate(details):
                                    with ui.row().classes('items-center w-full gap-3 p-2 rounded-lg bg-slate-50 border border-slate-200 text-xs'):
                                        ui.label(f"Position {idx + 1}").classes('w-20 text-slate-500 font-semibold')
                                        with ui.element('div').classes('w-7 h-7 rounded-md bg-teal-500 text-white font-bold flex items-center justify-center font-mono text-sm'):
                                            ui.label(str(d_info['digit']))
                                        with ui.element('div').classes('flex-1 h-2 rounded-full bg-slate-200 overflow-hidden'):
                                            ui.element('div').classes('h-full bg-teal-600 rounded-full').style(f"width: {d_info['confidence']*100:.1f}%;")
                                        ui.label(f"{d_info['confidence']*100:.1f}%").classes('w-12 text-right font-mono font-bold text-slate-700')
                                        time_slice = d_info.get('time_slice', [])
                                        if time_slice:
                                            ui.label(f"t={time_slice[0]}-{time_slice[1]}").classes('text-[10px] text-slate-400 font-mono')

                        status_type = 'warning' if res.get('uncertain') else 'positive'
                        ui.notify(res["message"], type=status_type)

        # ==============================================================
        # 3. MODEL INSIGHTS PANEL
        # ==============================================================
        with ui.tab_panel(tab_insights):
            with ui.column().classes('w-full gap-6'):
                ui.label('Model Insights & Benchmark Metrics').classes('text-2xl font-extrabold text-[#172B4D]')
                ui.label('Actual test performance, character error rate (CER), and sequence accuracy evaluated on unseen MNIST test sequences.').classes('text-sm text-slate-500 -mt-4')

                metrics = load_metrics_data()
                if not metrics:
                    with ui.card().classes('studio-card p-8 w-full items-center justify-center gap-3'):
                        ui.icon('hourglass_empty', size='40px').classes('text-amber-500')
                        ui.label('No Model Metrics Available').classes('text-lg font-bold text-slate-800')
                        ui.label('Please run training first using "python train.py".').classes('text-sm text-slate-500')
                else:
                    # Metric summary cards
                    with ui.grid().classes('grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 w-full'):
                        cards = [
                            (f"{metrics.get('exact_sequence_accuracy', 0.0) * 100:.2f}%", "Exact Sequence Acc.", "verified", "Full Multi-Digit Sequence Match"),
                            (f"{metrics.get('character_level_accuracy', 0.0) * 100:.2f}%", "Character Accuracy", "pin", "Per-Digit Recognition Accuracy"),
                            (f"{metrics.get('character_error_rate_cer', 0.0) * 100:.2f}%", "Character Error Rate", "trending_down", "Levenshtein CER on Test Set"),
                            (f"{metrics.get('epochs', 5)} Epochs", "Training Duration", "history", "Graph-Compiled Fast Convergence"),
                        ]
                        for val, title, icon_name, desc in cards:
                            with ui.card().classes('studio-card p-5'):
                                with ui.row().classes('items-center justify-between w-full'):
                                    ui.label(title).classes('text-xs font-bold text-slate-500 uppercase tracking-wider')
                                    ui.icon(icon_name, size='20px').classes('text-teal-600')
                                ui.label(val).classes('text-2xl font-extrabold text-[#172B4D] mt-2')
                                ui.label(desc).classes('text-xs text-slate-500 mt-1')

                    # Per-Length Performance Breakdown Table
                    if "per_length_performance" in metrics:
                        with ui.card().classes('studio-card p-6 w-full gap-4'):
                            with ui.row().classes('items-center justify-between w-full'):
                                ui.label('Accuracy by Sequence Length (1 to 5 Digits)').classes('font-bold text-base text-[#172B4D]')
                                ui.label('Evaluated on 1,500 unseen multi-digit test sequences').classes('text-xs text-slate-500')

                            length_rows = [
                                "<tr style='background:#F1F5F9; color:#334155; font-size:12px; font-weight:700;'><th style='padding:8px; border:1px solid #E2E8F0;'>Sequence Length</th><th style='padding:8px; border:1px solid #E2E8F0;'>Test Samples</th><th style='padding:8px; border:1px solid #E2E8F0;'>Exact Match Accuracy</th><th style='padding:8px; border:1px solid #E2E8F0;'>Character Accuracy</th></tr>"
                            ]
                            for l_key, l_data in metrics["per_length_performance"].items():
                                clean_name = l_key.replace("_", " ").title()
                                length_rows.append(
                                    f"<tr style='font-size:12px;'>"
                                    f"<td style='padding:8px; font-weight:700; border:1px solid #E2E8F0; color:#172B4D;'>{clean_name}</td>"
                                    f"<td style='padding:8px; font-family:monospace; border:1px solid #E2E8F0;'>{l_data.get('total_sequences', 0)}</td>"
                                    f"<td style='padding:8px; font-family:monospace; font-weight:700; color:#0F9488; border:1px solid #E2E8F0;'>{l_data.get('exact_accuracy', 0.0) * 100:.2f}%</td>"
                                    f"<td style='padding:8px; font-family:monospace; font-weight:700; color:#1E3A8A; border:1px solid #E2E8F0;'>{l_data.get('char_accuracy', 0.0) * 100:.2f}%</td>"
                                    f"</tr>"
                                )
                            l_html = f"<div style='overflow-x:auto; width:100%;'><table style='width:100%; text-align:left; border-collapse:collapse;'>{''.join(length_rows)}</table></div>"
                            ui.html(l_html).classes('w-full')

                    # Training CTC Loss Curve
                    if "history_ctc_loss" in metrics:
                        losses = metrics["history_ctc_loss"]
                        epochs_range = list(range(1, len(losses) + 1))

                        with ui.card().classes('studio-card p-6 w-full gap-4'):
                            ui.label('CTC Loss Convergence Curve').classes('font-bold text-base text-[#172B4D]')

                            chart_options = {
                                'tooltip': {'trigger': 'axis'},
                                'legend': {'data': ['CTC Training Loss']},
                                'grid': {'left': '3%', 'right': '4%', 'bottom': '3%', 'containLabel': True},
                                'xAxis': {'type': 'category', 'boundaryGap': False, 'data': [f"Epoch {e}" for e in epochs_range]},
                                'yAxis': {'type': 'value', 'name': 'CTC Loss'},
                                'series': [
                                    {'name': 'CTC Training Loss', 'type': 'line', 'data': [round(x, 4) for x in losses], 'color': '#0F9488', 'smooth': True, 'areaStyle': {'opacity': 0.1, 'color': '#0F9488'}}
                                ]
                            }
                            ui.echart(chart_options).classes('w-full h-72')

        # ==============================================================
        # 4. ABOUT PROJECT PANEL
        # ==============================================================
        with ui.tab_panel(tab_about):
            with ui.column().classes('w-full gap-6'):
                ui.label('About Multi-Digit Number Recognition AI').classes('text-2xl font-extrabold text-[#172B4D]')
                ui.label('Technical specifications, CRNN architecture, and CTC loss principles.').classes('text-sm text-slate-500 -mt-4')

                with ui.card().classes('studio-card p-6 w-full gap-4'):
                    ui.label('Sequence Recognition Architecture (CRNN + CTC)').classes('font-bold text-base text-[#172B4D]')
                    ui.label(
                        'Rather than relying on brittle segmentation heuristics (which fail on touching or overlapping numbers), '
                        'this system adopts a modern sequence-to-sequence optical character recognition approach based on CRNN (Convolutional Recurrent Neural Network).'
                    ).classes('text-sm text-slate-600 leading-relaxed')

                    with ui.column().classes('p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs font-mono text-slate-700 gap-1'):
                        ui.label('1. Feature Extractor: 4 Conv2D + BatchNorm blocks (32, 64, 128, 128 filters)')
                        ui.label('2. Sequence Folding: Reshape layer converting 2D spatial maps into 40 horizontal time slices')
                        ui.label('3. Sequence Encoder: Dual Bidirectional LSTM (64 units per direction, 128 total)')
                        ui.label('4. Connectionist Temporal Classification (CTC): Alignment-free loss & greedy collapse decoding')

                with ui.card().classes('studio-card p-6 w-full gap-4'):
                    ui.label('Guaranteed Data Provenance & Safety').classes('font-bold text-base text-[#172B4D]')
                    steps = [
                        ("Exact Uploaded Bytes", "Pre-processing, canvas preview, and model tensors are derived strictly from the same byte stream."),
                        ("SHA-256 Provenance", "Every upload is fingerprinted cryptographically to eliminate stale prediction cache bugs."),
                        ("Calibrated Uncertainty", "Ambiguous or degraded handwriting is rejected safely instead of fabricating artificial 99.9% confidence."),
                        ("Leading Zero Support", "Sequence decoding treats zeros as distinct tokens, accurately capturing sequences like 0075 and 05."),
                    ]
                    with ui.column().classes('gap-3'):
                        for title, desc in steps:
                            with ui.row().classes('items-start gap-3'):
                                ui.icon('check_circle', size='18px').classes('text-teal-600 mt-0.5')
                                with ui.column().classes('gap-0.5'):
                                    ui.label(title).classes('text-xs font-bold text-slate-800')
                                    ui.label(desc).classes('text-xs text-slate-600')


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Multi-Digit Handwritten Number Recognition Studio",
        port=8080,
        reload=False,
        show=False
    )
