import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import DRModel
from utils import generate_gradcam, process_image_for_gradcam

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="RetinaScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GLOBAL CSS — Dark clinical theme with teal/cyan accents
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ---- Reset & Base ---- */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main, .block-container {
    background-color: #080d14 !important;
    color: #e2eaf4 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1520 0%, #091018 100%) !important;
    border-right: 1px solid #1a2a3a !important;
}
[data-testid="stSidebar"] * {
    color: #c8d8e8 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ---- Typography ---- */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}
h1 { color: #ffffff !important; font-weight: 800 !important; }
h2 { color: #7dd3fc !important; font-weight: 700 !important; }
h3 { color: #a5c8e8 !important; font-weight: 600 !important; }
p, li, span { color: #b8cfe4 !important; line-height: 1.75 !important; }

/* ---- Streamlit default text elements ---- */
[data-testid="stMarkdownContainer"] p { color: #b8cfe4 !important; }
[data-testid="stText"] { color: #b8cfe4 !important; }
label { color: #b8cfe4 !important; }

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 20px rgba(14, 165, 233, 0.35) !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 28px rgba(14, 165, 233, 0.55) !important;
    transform: translateY(-2px) !important;
}

/* ---- File Uploader ---- */
[data-testid="stFileUploader"] {
    background: rgba(14, 165, 233, 0.05) !important;
    border: 2px dashed #0ea5e9 !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploader"] * { color: #7dd3fc !important; }

/* ---- Dataframe ---- */
[data-testid="stDataFrame"] {
    background: #0d1a27 !important;
    border: 1px solid #1a2e42 !important;
    border-radius: 10px !important;
}
[data-testid="stDataFrame"] * { color: #c8d8e8 !important; }

/* ---- Metrics ---- */
[data-testid="metric-container"] {
    background: #0d1a27 !important;
    border: 1px solid #1a2e42 !important;
    border-radius: 12px !important;
    padding: 1.2rem !important;
}
[data-testid="metric-container"] * { color: #e2eaf4 !important; }

/* ---- Alerts ---- */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ---- Divider ---- */
hr {
    border-color: #1a2e42 !important;
    margin: 2rem 0 !important;
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1520; }
::-webkit-scrollbar-thumb { background: #1e3a52; border-radius: 3px; }

/* ---- Custom cards ---- */
.rs-card {
    background: linear-gradient(145deg, #0d1a27, #0a1520);
    border: 1px solid #1a2e42;
    border-radius: 14px;
    padding: 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.3s;
}
.rs-card:hover { border-color: #0ea5e9; }

.rs-stat {
    background: linear-gradient(145deg, #0d1a27, #0a1520);
    border: 1px solid #1a2e42;
    border-radius: 14px;
    padding: 1.6rem 1.2rem;
    text-align: center;
}

.rs-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.5px;
}

.rs-hero {
    background: linear-gradient(135deg, #061020 0%, #0d2035 50%, #061528 100%);
    border: 1px solid #1a3050;
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.rs-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(14,165,233,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.rs-hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 40px;
    width: 300px; height: 200px;
    background: radial-gradient(ellipse, rgba(6,182,212,0.07) 0%, transparent 70%);
    pointer-events: none;
}

.rs-section-header {
    background: linear-gradient(90deg, rgba(14,165,233,0.15) 0%, transparent 100%);
    border-left: 4px solid #0ea5e9;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.5rem;
    margin-bottom: 1.5rem;
}

.rs-heatmap-legend {
    display: flex;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    background: #0d1a27;
    border-radius: 8px;
    border: 1px solid #1a2e42;
    align-items: center;
    flex-wrap: wrap;
}

.rs-grade-pill {
    padding: 0.2rem 0.6rem;
    border-radius: 5px;
    font-size: 0.82rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CONSTANTS
# ============================================================================
DR_CLASSES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

DR_DESCRIPTIONS = {
    0: "No diabetic retinopathy detected. Continue routine annual screening.",
    1: "Mild non-proliferative DR. Microaneurysms present. Follow-up in 12 months.",
    2: "Moderate NPDR. Extensive lesions beyond microaneurysms. Ophthalmologist referral recommended within 3–6 months.",
    3: "Severe NPDR. Numerous hemorrhages and venous beading. Urgent ophthalmologist referral required.",
    4: "Proliferative DR. Neovascularization or vitreous hemorrhage detected. Immediate specialist intervention needed."
}

DR_SEVERITY = ["None", "Low", "Moderate", "High", "Critical"]

GRADE_COLORS = {
    0: ("#22c55e", "#052e16"),   # green
    1: ("#eab308", "#1c1500"),   # yellow
    2: ("#f97316", "#1c0a00"),   # orange
    3: ("#ef4444", "#1c0000"),   # red
    4: ("#dc2626", "#200000"),   # deep red
}

GRADE_ACCENT = ["#22c55e", "#eab308", "#f97316", "#ef4444", "#dc2626"]


# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model():
    model_path = "EfficientNet.pth"
    try:
        model = DRModel()
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(
            f"❌ **Model file not found:** `{model_path}`\n\n"
            "Place `EfficientNet.pth` in the same directory as `app.py`."
        )
        st.stop()
    except RuntimeError as e:
        st.error(f"❌ **Model architecture mismatch:** {str(e)[:300]}")
        st.stop()
    except Exception as e:
        st.error(f"❌ **Unexpected error:** {str(e)}")
        st.stop()


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 0.5rem 0;'>
        <p style='font-family: Syne, sans-serif; font-size: 1.4rem; font-weight: 800;
                  color: #7dd3fc !important; margin: 0; letter-spacing: -0.3px;'>
            🔬 RetinaScan AI
        </p>
        <p style='font-size: 0.78rem; color: #4a7a9a !important; margin: 2px 0 0 0;
                  letter-spacing: 1px; text-transform: uppercase;'>
             EfficientNetV2-S + CBAM
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div class='rs-card'>
        <p style='font-size:0.82rem; color:#4a9ab8 !important; text-transform:uppercase;
                  letter-spacing:1px; font-weight:700; margin:0 0 10px 0;'>Model Architecture</p>
        <p style='margin:0; font-size:0.9rem; line-height:1.8; color:#c8d8e8 !important;'>
            🧠 <strong style='color:#7dd3fc !important;'>Backbone:</strong> EfficientNetV2-S<br>
            👁️ <strong style='color:#7dd3fc !important;'>Attention:</strong> CBAM (ch. attention)<br>
            🎯 <strong style='color:#7dd3fc !important;'>Task:</strong> 5-class classification<br>
            🔍 <strong style='color:#7dd3fc !important;'>XAI:</strong> Grad-CAM
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='font-family:Syne,sans-serif; font-size:1rem; font-weight:700;
              color:#7dd3fc !important; margin: 1rem 0 0.75rem 0;'>DR Grading Scale</p>
    """, unsafe_allow_html=True)

    grade_info = [
        ("0", "No DR",           "#22c55e", "No abnormalities"),
        ("1", "Mild NPDR",       "#eab308", "Microaneurysms only"),
        ("2", "Moderate NPDR",   "#f97316", "Extensive lesions"),
        ("3", "Severe NPDR",     "#ef4444", "Numerous hemorrhages"),
        ("4", "Proliferative",   "#dc2626", "Neovascularization"),
    ]

    for grade, label, color, desc in grade_info:
        st.markdown(f"""
        <div style='display:flex; align-items:flex-start; gap:10px; margin-bottom:8px;
                    padding:8px 10px; background:#0d1a27; border-radius:8px;
                    border-left:3px solid {color};'>
            <span style='font-family:Syne,sans-serif; font-weight:800; font-size:1rem;
                         color:{color} !important; min-width:18px;'>{grade}</span>
            <div>
                <p style='margin:0; font-weight:600; font-size:0.88rem;
                          color:#e2eaf4 !important;'>{label}</p>
                <p style='margin:0; font-size:0.78rem; color:#5a8aaa !important;'>{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style='background:rgba(234,179,8,0.08); border:1px solid rgba(234,179,8,0.25);
                border-radius:10px; padding:1rem;'>
        <p style='margin:0; font-size:0.85rem; color:#a8940a !important; line-height:1.7;'>
            <strong style='color:#eab308 !important;'>⚠️ Clinical Disclaimer</strong><br>
            This tool is designed to <em>assist</em> healthcare professionals — not replace them.
            Always confirm findings with a qualified ophthalmologist.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# HERO HEADER
# ============================================================================
st.markdown("""
<div class='rs-hero'>
    <p style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800; margin:0;
              color:#ffffff !important; letter-spacing:-0.8px; line-height:1.2;'>
        🔬 RetinaScan <span style='color:#0ea5e9;'>AI</span>
    </p>
    <p style='font-size:1rem; color:#5a9fcc !important; margin:8px 0 16px 0;'>
        Diabetic Retinopathy Detection · EfficientNetV2-S + CBAM Attention
    </p>
    <p style='font-size:0.9rem; color:#4a7a9a !important; margin:0; max-width:680px; line-height:1.7;'>
        State-of-the-art deep learning model with channel attention mechanism for precise 5-grade 
        DR classification. Every prediction includes Grad-CAM visual explanations to support 
        clinical decision-making.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# UPLOAD SECTION
# ============================================================================
st.markdown("""
<div class='rs-section-header'>
    <p style='font-family:Syne,sans-serif; font-weight:700; font-size:1.15rem;
              color:#7dd3fc !important; margin:0;'>📤 Upload Fundus Image</p>
    <p style='margin:4px 0 0 0; font-size:0.85rem; color:#4a7a9a !important;'>
        Accepts JPG · PNG · JPEG — high-resolution fundus photographs recommended
    </p>
</div>
""", unsafe_allow_html=True)

up_col1, up_col2 = st.columns([3, 1])

with up_col1:
    uploaded_file = st.file_uploader(
        "Select fundus image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )

    # ✅ Paste HERE
    if uploaded_file is not None:
        st.image(uploaded_file, use_container_width=True)
        st.markdown(
            f"<p style='text-align:center; font-size:0.85rem; color:#7dd3fc;'>{uploaded_file.name}</p>",
            unsafe_allow_html=True
        )

with up_col2:
    st.markdown("""
    <div class='rs-card' style='height:100%; padding:1.2rem;'>
        <p style='font-size:0.78rem; text-transform:uppercase; letter-spacing:1px;
                  color:#0ea5e9 !important; font-weight:700; margin:0 0 8px 0;'>Requirements</p>
        <p style='font-size:0.82rem; color:#6a9ab8 !important; margin:0; line-height:1.8;'>
            Format: JPG / PNG<br>
            Focus: Sharp, clear<br>
            Lighting: Even, no glare<br>
            Output: 380×380 px
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# ANALYSIS
# ============================================================================
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        # The rest of the transform stays the same:
        transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    # Preprocessing — EfficientNetV2-S uses 380×380
    transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image)

    # ---------- Load Model & Predict ----------
    model = load_model()

    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_class].item() * 100)

    accent, dark_bg = GRADE_COLORS[predicted_class]
    severity = DR_SEVERITY[predicted_class]

    st.markdown("---")

    # ---- RESULTS HEADER ----
    st.markdown("""
    <div class='rs-section-header'>
        <p style='font-family:Syne,sans-serif; font-weight:700; font-size:1.15rem;
                  color:#7dd3fc !important; margin:0;'>🩺 Analysis Results</p>
        <p style='margin:4px 0 0 0; font-size:0.85rem; color:#4a7a9a !important;'>
            Real-time AI inference from EfficientNetV2-S + CBAM
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- 3-STAT ROW ----
    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(f"""
        <div class='rs-stat' style='border-top: 3px solid {accent};'>
            <p style='font-size:0.75rem; text-transform:uppercase; letter-spacing:1.2px;
                      color:#3a6a8a !important; margin:0 0 8px 0; font-weight:700;'>Predicted Grade</p>
            <p style='font-family:Syne,sans-serif; font-size:3rem; font-weight:800; margin:0;
                      color:{accent} !important; line-height:1;'>{predicted_class}</p>
            <p style='font-size:1rem; font-weight:600; margin:6px 0 0 0;
                      color:#e2eaf4 !important;'>{DR_CLASSES[predicted_class]}</p>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        # confidence bar width
        bar_w = int(confidence)
        st.markdown(f"""
        <div class='rs-stat' style='border-top: 3px solid #0ea5e9;'>
            <p style='font-size:0.75rem; text-transform:uppercase; letter-spacing:1.2px;
                      color:#3a6a8a !important; margin:0 0 8px 0; font-weight:700;'>Confidence</p>
            <p style='font-family:Syne,sans-serif; font-size:3rem; font-weight:800; margin:0;
                      color:#0ea5e9 !important; line-height:1;'>{confidence:.1f}<span style='font-size:1.5rem;'>%</span></p>
            <div style='margin-top:10px; height:5px; background:#1a2e42; border-radius:5px; overflow:hidden;'>
                <div style='height:100%; width:{bar_w}%; background:linear-gradient(90deg,#0ea5e9,#06b6d4);
                            border-radius:5px; transition:width 0.8s ease;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown(f"""
        <div class='rs-stat' style='border-top: 3px solid {accent};'>
            <p style='font-size:0.75rem; text-transform:uppercase; letter-spacing:1.2px;
                      color:#3a6a8a !important; margin:0 0 8px 0; font-weight:700;'>Severity Level</p>
            <p style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800; margin:0;
                      color:{accent} !important; line-height:1.1;'>{severity}</p>
            <p style='font-size:0.82rem; margin:8px 0 0 0; color:#5a8aaa !important;'>
                Risk classification
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ---- CLINICAL RECOMMENDATION ----
    st.markdown(f"""
    <div style='background:{dark_bg}; border:1px solid {accent}40;
                border-left:5px solid {accent}; border-radius:12px;
                padding:1.2rem 1.5rem; margin: 1.5rem 0;'>
        <p style='font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;
                  color:{accent} !important; font-weight:700; margin:0 0 6px 0;'>
            💊 Clinical Recommendation
        </p>
        <p style='font-size:0.95rem; color:#d0e4f4 !important; margin:0; line-height:1.7;'>
            {DR_DESCRIPTIONS[predicted_class]}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- PROBABILITY TABLE + CHART ----
    st.markdown("""
    <div class='rs-section-header'>
        <p style='font-family:Syne,sans-serif; font-weight:700; font-size:1.05rem;
                  color:#7dd3fc !important; margin:0;'>📊 Classification Probabilities</p>
    </div>
    """, unsafe_allow_html=True)

    tc, gc = st.columns([1, 2])

    with tc:
        prob_data = {
            "Grade": [f"Grade {i}" for i in range(5)],
            "Class": [DR_CLASSES[i] for i in range(5)],
            "Probability": [f"{float(probabilities[i].item()*100):.2f}%" for i in range(5)]
        }
        prob_df = pd.DataFrame(prob_data)
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

    with gc:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("#0d1a27")
        ax.set_facecolor("#0d1a27")

        labels = [f"G{i}\n{DR_CLASSES[i].replace(' ', chr(10))}" for i in range(5)]
        probs_list = [float(probabilities[i].item() * 100) for i in range(5)]
        bar_colors = [GRADE_ACCENT[i] if i == predicted_class else "#1a3050" for i in range(5)]
        edge_colors = [GRADE_ACCENT[i] for i in range(5)]

        bars = ax.bar(labels, probs_list, color=bar_colors,
                      edgecolor=edge_colors, linewidth=1.5, width=0.55)

        for bar, prob in zip(bars, probs_list):
            if prob > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8,
                        f'{prob:.1f}%', ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='#b0ccee')

        ax.set_ylim(0, 105)
        ax.set_ylabel("Probability (%)", fontsize=10, color='#4a7a9a', labelpad=10)
        ax.tick_params(colors='#4a7a9a', labelsize=8)
        ax.spines['bottom'].set_color('#1a2e42')
        ax.spines['left'].set_color('#1a2e42')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, color='#1a2e42', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ---- GRAD-CAM ----
    st.markdown("---")
    st.markdown("""
    <div class='rs-section-header'>
        <p style='font-family:Syne,sans-serif; font-weight:700; font-size:1.15rem;
                  color:#7dd3fc !important; margin:0;'>🧬 Explainable AI · Grad-CAM</p>
        <p style='margin:4px 0 0 0; font-size:0.85rem; color:#4a7a9a !important;'>
            Gradient-weighted Class Activation Maps — visualising model attention regions
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='rs-card'>
        <p style='font-size:0.88rem; color:#6a9ab8 !important; margin:0; line-height:1.75;'>
            <strong style='color:#7dd3fc !important;'>What you're seeing:</strong>
            Grad-CAM computes gradient-weighted feature activations from the final convolutional layer 
            of EfficientNetV2-S to highlight retinal regions that most influenced the prediction — 
            such as microaneurysms, exudates, and neovascularization.
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        # EfficientNetV2-S target layer
        target_layer = model.backbone.conv_head

        grayscale_cam = generate_gradcam(
            model, image_tensor, target_layer, predicted_class
        )

        original_img = process_image_for_gradcam(image.resize((380, 380)))

        visualization = show_cam_on_image(
            original_img,
            grayscale_cam,
            use_rgb=True,
            image_weight=0.55
        )

        v1, v2 = st.columns(2)

        with v1:
            st.markdown("""
            <p style='font-family:Syne,sans-serif; font-weight:700; font-size:0.95rem;
                      color:#a5c8e8 !important; margin:0 0 8px 0;'>Original Fundus Image</p>
            """, unsafe_allow_html=True)
            st.image(original_img, use_container_width=True, clamp=True)

        with v2:
            st.markdown("""
            <p style='font-family:Syne,sans-serif; font-weight:700; font-size:0.95rem;
                      color:#a5c8e8 !important; margin:0 0 8px 0;'>Grad-CAM Attention Map</p>
            """, unsafe_allow_html=True)
            st.image(visualization, use_container_width=True, clamp=True)

        # Legend
        st.markdown("""
        <div style='margin-top:1rem;'>
            <p style='font-size:0.78rem; text-transform:uppercase; letter-spacing:1px;
                      color:#3a6a8a !important; font-weight:700; margin:0 0 8px 0;'>
                Heatmap Colour Guide
            </p>
            <div style='display:flex; gap:12px; flex-wrap:wrap;'>
                <div style='background:#0d1a27; border:1px solid #1a2e42; border-radius:8px;
                            padding:0.5rem 1rem; text-align:center;'>
                    <div style='width:28px; height:8px; background:#ff2200; border-radius:4px;
                                margin: 0 auto 4px auto;'></div>
                    <p style='font-size:0.78rem; color:#8ab0cc !important; margin:0;'>
                        <strong>Red</strong><br>Highest influence
                    </p>
                </div>
                <div style='background:#0d1a27; border:1px solid #1a2e42; border-radius:8px;
                            padding:0.5rem 1rem; text-align:center;'>
                    <div style='width:28px; height:8px; background:#ffaa00; border-radius:4px;
                                margin: 0 auto 4px auto;'></div>
                    <p style='font-size:0.78rem; color:#8ab0cc !important; margin:0;'>
                        <strong>Yellow</strong><br>Moderate
                    </p>
                </div>
                <div style='background:#0d1a27; border:1px solid #1a2e42; border-radius:8px;
                            padding:0.5rem 1rem; text-align:center;'>
                    <div style='width:28px; height:8px; background:#22cc44; border-radius:4px;
                                margin: 0 auto 4px auto;'></div>
                    <p style='font-size:0.78rem; color:#8ab0cc !important; margin:0;'>
                        <strong>Green</strong><br>Low
                    </p>
                </div>
                <div style='background:#0d1a27; border:1px solid #1a2e42; border-radius:8px;
                            padding:0.5rem 1rem; text-align:center;'>
                    <div style='width:28px; height:8px; background:#0044ff; border-radius:4px;
                                margin: 0 auto 4px auto;'></div>
                    <p style='font-size:0.78rem; color:#8ab0cc !important; margin:0;'>
                        <strong>Blue</strong><br>Minimal
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='margin-top:1rem; padding:0.75rem 1rem; background:#0d1a27;
                    border-radius:8px; border:1px solid #1a2e42;'>
            <p style='font-size:0.85rem; color:#6a9ab8 !important; margin:0; line-height:1.7;'>
                <strong style='color:#7dd3fc !important;'>Common highlighted lesions:</strong>
                Microaneurysms · Dot and blot hemorrhages · Hard / soft exudates ·
                Venous beading · Intraretinal microvascular abnormalities (IRMA) · Neovascularization
            </p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Grad-CAM generation error: {e}")

    # ---- SUMMARY ----
    st.markdown("---")
    st.markdown("""
    <div class='rs-section-header'>
        <p style='font-family:Syne,sans-serif; font-weight:700; font-size:1.1rem;
                  color:#7dd3fc !important; margin:0;'>📄 Analysis Summary</p>
    </div>
    """, unsafe_allow_html=True)

    sm1, sm2 = st.columns(2)

    probs_np = probabilities.cpu().numpy()
    sorted_idx = np.argsort(probs_np)[::-1]

    with sm1:
        st.markdown(f"""
        <div class='rs-card'>
            <p style='font-size:0.78rem; text-transform:uppercase; letter-spacing:1px;
                      color:#0ea5e9 !important; font-weight:700; margin:0 0 12px 0;'>📌 Key Findings</p>
            <p style='font-size:0.9rem; color:#b8cfe4 !important; line-height:2; margin:0;'>
                <strong style='color:#7dd3fc !important;'>Grade:</strong> {predicted_class} — {DR_CLASSES[predicted_class]}<br>
                <strong style='color:#7dd3fc !important;'>Confidence:</strong> {confidence:.2f}%<br>
                <strong style='color:#7dd3fc !important;'>Severity:</strong> {severity}<br>
                <strong style='color:#7dd3fc !important;'>Backbone:</strong> EfficientNetV2-S<br>
                <strong style='color:#7dd3fc !important;'>Attention:</strong> CBAM (Channel)<br>
                <strong style='color:#7dd3fc !important;'>Runner-up:</strong> {DR_CLASSES[sorted_idx[1]]} ({float(probs_np[sorted_idx[1]])*100:.2f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)

    with sm2:
        st.warning(
            "**⚠️ Disclaimer:** This system is designed to assist healthcare professionals "
            "in diabetic retinopathy screening. It must not be used as a standalone "
            "diagnostic tool. All findings should be verified by a qualified ophthalmologist."
        )

# ============================================================================
# EMPTY STATE
# ============================================================================
else:
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding:3rem 1rem;'>
        <p style='font-size:3.5rem; margin:0;'>🔬</p>
        <p style='font-family:Syne,sans-serif; font-size:1.5rem; font-weight:700;
                  color:#5a9fcc !important; margin:12px 0 8px 0;'>Ready to Analyse</p>
        <p style='color:#3a6a8a !important; font-size:0.95rem;'>
            Upload a fundus image above to begin automated DR detection
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    steps = [
        ("1️⃣", "Upload Image", "Select a high-quality fundus photograph"),
        ("2️⃣", "Auto Analysis", "EfficientNetV2-S + CBAM processes the image"),
        ("3️⃣", "Review Results", "View grade, confidence, and Grad-CAM maps"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], steps):
        with col:
            col.markdown(f"""
            <div class='rs-card' style='text-align:center;'>
                <p style='font-size:2rem; margin:0 0 8px 0;'>{icon}</p>
                <p style='font-family:Syne,sans-serif; font-weight:700; font-size:1rem;
                          color:#7dd3fc !important; margin:0 0 6px 0;'>{title}</p>
                <p style='font-size:0.85rem; color:#4a7a9a !important; margin:0;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
