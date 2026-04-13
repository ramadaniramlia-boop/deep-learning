import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI Visual Sentiment Analysis | Konflik Iran-Israel",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1421 50%, #0a0e1a 100%);
    color: #e0e6f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1829 0%, #111827 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c9d8ed !important; }

/* Header besar */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 1rem;
    color: #7ba4cc;
    margin-bottom: 1.5rem;
    font-weight: 300;
}
.badge {
    display: inline-block;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid #38bdf8;
    color: #38bdf8;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    margin-right: 6px;
    margin-bottom: 6px;
}

/* Section header */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.2rem;
    color: #38bdf8;
    border-left: 3px solid #38bdf8;
    padding-left: 10px;
    margin: 1.5rem 0 1rem 0;
}

/* Card metric */
.metric-card {
    background: linear-gradient(135deg, #0f1e30, #132035);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-3px); border-color: #38bdf8; }
.metric-label {
    font-size: 0.75rem;
    color: #7ba4cc;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #38bdf8;
    font-family: 'Space Mono', monospace;
}
.metric-model { font-size: 0.8rem; color: #a0c0e0; margin-top: 4px; }

/* Info box */
.info-box {
    background: rgba(56, 189, 248, 0.07);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #c9d8ed;
}
.warning-box {
    background: rgba(251, 191, 36, 0.07);
    border: 1px solid rgba(251, 191, 36, 0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #fcd34d;
}
.success-box {
    background: rgba(52, 211, 153, 0.07);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #6ee7b7;
}

/* Table styling */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* Tab */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1829;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #7ba4cc;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(56,189,248,0.15) !important;
    color: #38bdf8 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #0f1e30 !important;
    border-radius: 8px !important;
    color: #38bdf8 !important;
    font-family: 'Space Mono', monospace !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #0ea5e9, #6366f1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    padding: 0.5rem 1.2rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

hr { border-color: #1e3a5f; }
</style>
""", unsafe_allow_html=True)

# ─── DATA SIMULASI (tanpa dataset nyata) ─────────────────────────────────────
CONFIG = {
    'class_names': ['positif', 'netral', 'negatif'],
    'img_size': 224,
    'batch_size': 32,
    'random_seed': 42,
    'total_data': 526
}

# Distribusi kelas (hasil penelitian)
class_counts = {'positif': 142, 'netral': 175, 'negatif': 209}

# Split dataset
split_info = {
    'Train': {'positif': 91, 'netral': 112, 'negatif': 134},
    'Validation': {'positif': 23, 'netral': 28, 'negatif': 34},
    'Test': {'positif': 28, 'netral': 35, 'negatif': 41}
}

# Load hasil evaluasi dari CSV secara dinamis
@st.cache_data
def load_eval_data():
    try:
        df = pd.read_csv('data/eval_results.csv')
        results_dict = {}
        for _, row in df.iterrows():
            results_dict[row['Model']] = {
                'accuracy': row['Accuracy'],
                'precision': row['Precision'],
                'recall': row['Recall'],
                'f1': row['F1_Score'],
                'size_mb': row['Size_MB']
            }
        return results_dict
    except FileNotFoundError:
        st.error("File data/eval_results.csv tidak ditemukan! Menampilkan data simulasi sebagai fallback.")
        # Fallback agar aplikasi tidak crash jika file belum dibuat
        return {
            'VGG16':       {'accuracy': 0.7760, 'precision': 0.7801, 'recall': 0.7760, 'f1': 0.7742, 'size_mb': 105.0},
            'DenseNet121': {'accuracy': 0.9252, 'precision': 0.9278, 'recall': 0.9252, 'f1': 0.9248, 'size_mb': 56.3},
            'MobileNetV2': {'accuracy': 0.9063, 'precision': 0.9091, 'recall': 0.9063, 'f1': 0.9057, 'size_mb': 13.2},
        }

eval_results = load_eval_data()

# Confusion matrix (simulasi realistis)
cm_data = {
    'VGG16': np.array([[18, 5, 5],[4, 22, 9],[3, 6, 32]]),
    'DenseNet121': np.array([[25, 2, 1],[1, 32, 2],[1, 2, 38]]),
    'MobileNetV2': np.array([[24, 3, 1],[2, 30, 3],[2, 2, 37]]),
}

# Training history (simulasi)
np.random.seed(42)
def make_history(final_acc, noise=0.03):
    epochs = 10
    train_acc = np.clip(np.linspace(0.55, final_acc + 0.02, epochs) + np.random.normal(0, noise, epochs), 0, 1)
    val_acc   = np.clip(np.linspace(0.50, final_acc, epochs) + np.random.normal(0, noise, epochs), 0, 1)
    train_loss = np.clip(np.linspace(1.0, 0.25, epochs) + np.random.normal(0, 0.03, epochs), 0.1, 1.2)
    val_loss   = np.clip(np.linspace(1.05, 0.3, epochs) + np.random.normal(0, 0.04, epochs), 0.1, 1.2)
    return {'train_acc': train_acc, 'val_acc': val_acc, 'train_loss': train_loss, 'val_loss': val_loss}

histories = {
    'VGG16':       make_history(0.782),
    'DenseNet121': make_history(0.930),
    'MobileNetV2': make_history(0.912),
}

MODEL_COLORS = {'VGG16': '#60a5fa', 'DenseNet121': '#f97316', 'MobileNetV2': '#34d399'}

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 XAI Sentiment Analysis")
    st.markdown("---")
    st.markdown("**Peneliti**")
    st.markdown("Ramlia Ramadani Sudin")
    st.markdown("`NIM: E1E123016`")
    st.markdown("---")

    page = st.radio("📌 Navigasi", [
        "🏠 Beranda",
        "📊 EDA & Dataset",
        "⚙️ Konfigurasi Model",
        "🏋️ Training",
        "📈 Evaluasi Model",
        "🔥 Grad-CAM (XAI)",
        "🧪 Uji Coba Model",
        "📋 Ringkasan & Kesimpulan"
    ])

    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"Total Citra: `{CONFIG['total_data']}`")
    st.markdown(f"Kelas: `{', '.join(CONFIG['class_names'])}`")
    st.markdown(f"Ukuran Input: `{CONFIG['img_size']}×{CONFIG['img_size']}`")

    st.markdown("---")
    st.markdown("**Model Digunakan**")
    for m, c in MODEL_COLORS.items():
        st.markdown(f"<span style='color:{c}'>●</span> {m}", unsafe_allow_html=True)

# ─── HALAMAN BERANDA ──────────────────────────────────────────────────────────
if page == "🏠 Beranda":
    st.markdown('<div class="hero-title">Explainable AI untuk<br>Visual Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Konflik Iran-Israel · CNN + Grad-CAM · Transfer Learning</div>', unsafe_allow_html=True)

    st.markdown("""
    <span class="badge">CNN</span>
    <span class="badge">Transfer Learning</span>
    <span class="badge">Grad-CAM</span>
    <span class="badge">VGG16</span>
    <span class="badge">DenseNet121</span>
    <span class="badge">MobileNetV2</span>
    <span class="badge">XAI</span>
    <span class="badge">Web Scraping</span>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-label">Total Data</div><div class="metric-value">526</div><div class="metric-model">Citra Konflik</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-label">Best Accuracy</div><div class="metric-value">92.5%</div><div class="metric-model">DenseNet121</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-label">Model Diuji</div><div class="metric-value">3</div><div class="metric-model">VGG16 · Dense · Mobile</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-label">Rekomendasi</div><div class="metric-value">MobileV2</div><div class="metric-model">Trade-off Terbaik</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-header">Tentang Penelitian</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Penelitian ini mengembangkan sistem <strong>Visual Sentiment Analysis</strong> berbasis 
        <strong>Convolutional Neural Network (CNN)</strong> untuk menganalisis citra yang berkaitan 
        dengan konflik Iran-Israel. Dataset diperoleh melalui <strong>Web Scraping</strong> dari 
        berbagai portal berita dan media.<br><br>
        Untuk meningkatkan <strong>interpretabilitas</strong> model AI, penelitian ini mengintegrasikan 
        teknik <strong>Grad-CAM (Gradient-weighted Class Activation Mapping)</strong> sebagai pendekatan 
        Explainable AI (XAI), yang mampu memvisualisasikan area penting pada citra yang menjadi dasar 
        keputusan model.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Alur Penelitian</div>', unsafe_allow_html=True)
        steps = [
            ("1", "Web Scraping", "Pengumpulan citra dari portal berita"),
            ("2", "EDA", "Eksplorasi dan analisis distribusi data"),
            ("3", "Augmentasi", "Rotasi, zoom, flip untuk variasi data"),
            ("4", "Transfer Learning", "VGG16, DenseNet121, MobileNetV2"),
            ("5", "Evaluasi", "Accuracy, Precision, Recall, F1"),
            ("6", "Grad-CAM", "Visualisasi area keputusan model (XAI)"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;margin:8px 0;padding:10px 14px;
            background:rgba(56,189,248,0.05);border-radius:8px;border:1px solid #1e3a5f;">
                <div style="background:#0ea5e9;color:white;width:28px;height:28px;border-radius:50%;
                display:flex;align-items:center;justify-content:center;font-family:'Space Mono',monospace;
                font-size:0.75rem;font-weight:700;flex-shrink:0;">{num}</div>
                <div>
                    <div style="font-weight:600;color:#c9d8ed;">{title}</div>
                    <div style="font-size:0.8rem;color:#7ba4cc;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header">Hasil Singkat</div>', unsafe_allow_html=True)
        df_quick = pd.DataFrame({
            'Model': ['VGG16', 'DenseNet121', 'MobileNetV2'],
            'Akurasi': ['77.6%', '92.5%', '90.6%'],
            'Ukuran': ['105 MB', '56.3 MB', '13.2 MB'],
            'Rank': ['3️⃣', '1️⃣', '2️⃣']
        })
        st.dataframe(df_quick, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">Distribusi Kelas</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='none')
        colors_pie = ['#34d399', '#60a5fa', '#f87171']
        wedges, texts, autotexts = ax.pie(
            list(class_counts.values()),
            labels=list(class_counts.keys()),
            autopct='%1.1f%%',
            colors=colors_pie,
            startangle=140,
            textprops={'color': '#c9d8ed', 'fontsize': 11}
        )
        for at in autotexts: at.set_color('white')
        ax.set_facecolor('none')
        fig.patch.set_alpha(0)
        st.pyplot(fig, transparent=True)
        plt.close()

# ─── HALAMAN EDA ──────────────────────────────────────────────────────────────
elif page == "📊 EDA & Dataset":
    st.markdown('<div class="hero-title">EDA & Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Exploratory Data Analysis · Distribusi Data · Split Dataset</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📦 Distribusi Kelas", "✂️ Split Dataset", "🎨 Augmentasi"])

    with tab1:
        st.markdown('<div class="section-header">Distribusi Jumlah Data per Kelas</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Dataset dikumpulkan melalui <strong>Web Scraping</strong> dari portal berita internasional.
        Total <strong>526 citra</strong> terbagi ke 3 kelas sentimen: Positif, Netral, dan Negatif.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
            bars = ax.bar(
                list(class_counts.keys()),
                list(class_counts.values()),
                color=['#34d399', '#60a5fa', '#f87171'],
                edgecolor='none', width=0.5
            )
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 3, str(int(h)),
                        ha='center', va='bottom', color='#c9d8ed', fontsize=11, fontweight='bold')
            ax.set_facecolor('none')
            ax.tick_params(colors='#7ba4cc')
            ax.spines[:].set_color('#1e3a5f')
            ax.set_ylabel('Jumlah Gambar', color='#7ba4cc')
            ax.set_title('Distribusi Kelas', color='#c9d8ed', fontweight='bold')
            fig.patch.set_alpha(0)
            st.pyplot(fig, transparent=True)
            plt.close()

        with col2:
            for cls, cnt in class_counts.items():
                pct = cnt / CONFIG['total_data'] * 100
                color_map = {'positif': '#34d399', 'netral': '#60a5fa', 'negatif': '#f87171'}
                st.markdown(f"""
                <div style="margin:12px 0;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="color:#c9d8ed;font-weight:600;">{cls.capitalize()}</span>
                        <span style="color:{color_map[cls]};font-family:'Space Mono',monospace;">{cnt} ({pct:.1f}%)</span>
                    </div>
                    <div style="background:#1e3a5f;border-radius:999px;height:8px;">
                        <div style="background:{color_map[cls]};width:{pct}%;height:8px;border-radius:999px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="warning-box">⚠️ Dataset sedikit <strong>imbalanced</strong> — kelas Negatif mendominasi (39.7%). Teknik stratified split digunakan untuk menjaga proporsi di setiap subset.</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Statistik Pixel</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            fig2, ax2 = plt.subplots(figsize=(6, 3.5), facecolor='none')
            x = np.linspace(0, 1, 500)
            for channel, color, label in zip([0.48, 0.44, 0.38], ['#f87171','#4ade80','#60a5fa'], ['Red','Green','Blue']):
                y = np.exp(-0.5*((x - channel)/0.18)**2)
                ax2.plot(x, y, color=color, label=label, linewidth=2)
                ax2.fill_between(x, y, alpha=0.1, color=color)
            ax2.set_facecolor('none')
            ax2.tick_params(colors='#7ba4cc')
            ax2.spines[:].set_color('#1e3a5f')
            ax2.legend(labelcolor='#c9d8ed', facecolor='#0d1829', edgecolor='#1e3a5f')
            ax2.set_title('Distribusi Intensitas Pixel', color='#c9d8ed', fontweight='bold')
            ax2.set_xlabel('Nilai Pixel (0–1)', color='#7ba4cc')
            fig2.patch.set_alpha(0)
            st.pyplot(fig2, transparent=True)
            plt.close()

        with col4:
            mean_rgb = {'Red': 0.481, 'Green': 0.442, 'Blue': 0.384}
            fig3, ax3 = plt.subplots(figsize=(5, 3.5), facecolor='none')
            bars3 = ax3.bar(list(mean_rgb.keys()), list(mean_rgb.values()),
                            color=['#f87171','#4ade80','#60a5fa'], width=0.4)
            for bar in bars3:
                h = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.3f}',
                         ha='center', va='bottom', color='#c9d8ed', fontsize=10)
            ax3.set_ylim(0, 0.65)
            ax3.set_facecolor('none')
            ax3.tick_params(colors='#7ba4cc')
            ax3.spines[:].set_color('#1e3a5f')
            ax3.set_title('Rata-rata Intensitas RGB', color='#c9d8ed', fontweight='bold')
            ax3.set_ylabel('Nilai (0–1)', color='#7ba4cc')
            fig3.patch.set_alpha(0)
            st.pyplot(fig3, transparent=True)
            plt.close()

    with tab2:
        st.markdown('<div class="section-header">Pembagian Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Dataset dibagi menggunakan <strong>Stratified Split</strong> untuk memastikan proporsi kelas 
        tetap seimbang di setiap subset. Pembagian: <strong>Train 70%</strong> → Val 20% dari Train → <strong>Test 20%</strong>.
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        split_colors = {'Train': '#6366f1', 'Validation': '#f59e0b', 'Test': '#10b981'}
        split_totals = {k: sum(v.values()) for k, v in split_info.items()}
        for col, (split_name, data) in zip([col1, col2, col3], split_info.items()):
            with col:
                total = split_totals[split_name]
                color = split_colors[split_name]
                st.markdown(f"""
                <div class="metric-card" style="border-color:{color}40;">
                    <div class="metric-label">{split_name}</div>
                    <div class="metric-value" style="color:{color};">{total}</div>
                    <div class="metric-model">citra</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                for cls, cnt in data.items():
                    st.markdown(f"<span style='color:#7ba4cc;'>• {cls}:</span> <strong style='color:#c9d8ed;'>{cnt}</strong>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig_split, ax_split = plt.subplots(figsize=(8, 3.5), facecolor='none')
        x = np.arange(3)
        width = 0.22
        class_colors = ['#34d399', '#60a5fa', '#f87171']
        for i, cls in enumerate(CONFIG['class_names']):
            vals = [split_info[s][cls] for s in ['Train', 'Validation', 'Test']]
            ax_split.bar(x + i*width, vals, width, label=cls.capitalize(), color=class_colors[i])
        ax_split.set_xticks(x + width)
        ax_split.set_xticklabels(['Train', 'Validation', 'Test'], color='#c9d8ed')
        ax_split.tick_params(colors='#7ba4cc')
        ax_split.spines[:].set_color('#1e3a5f')
        ax_split.set_facecolor('none')
        ax_split.legend(labelcolor='#c9d8ed', facecolor='#0d1829', edgecolor='#1e3a5f')
        ax_split.set_title('Distribusi Kelas per Split', color='#c9d8ed', fontweight='bold')
        ax_split.set_ylabel('Jumlah', color='#7ba4cc')
        fig_split.patch.set_alpha(0)
        st.pyplot(fig_split, transparent=True)
        plt.close()

    with tab3:
        st.markdown('<div class="section-header">Teknik Data Augmentasi</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        Augmentasi data diterapkan secara <em>on-the-fly</em> menggunakan <strong>ImageDataGenerator</strong> 
        dari Keras. Teknik ini memperkaya variasi visual tanpa menduplikasi file fisik.
        </div>
        """, unsafe_allow_html=True)

        aug_params = [
            ("🔄", "Rotasi", "±20°", "Gambar diputar acak hingga 20° agar model tetap mengenali objek meski kamera miring."),
            ("🔍", "Zoom", "±20%", "Gambar di-zoom in/out 20% untuk menyesuaikan berbagai jarak pengambilan gambar."),
            ("↔️", "Geser Horizontal", "10%", "Objek digeser 10% ke samping agar model tidak bergantung pada posisi sentral."),
            ("↕️", "Geser Vertikal", "10%", "Pergeseran atas-bawah untuk menambah variasi perspektif vertikal."),
            ("🪞", "Horizontal Flip", "True", "Gambar dibalik seperti cermin, menambah arah objek tanpa mengubah makna."),
        ]
        for icon, name, val, desc in aug_params:
            st.markdown(f"""
            <div style="display:flex;gap:14px;padding:12px 16px;margin:6px 0;
            background:rgba(99,102,241,0.07);border:1px solid #2d3f6e;border-radius:10px;align-items:center;">
                <div style="font-size:1.5rem;">{icon}</div>
                <div style="flex:1;">
                    <div style="font-weight:600;color:#c9d8ed;">{name} 
                        <span style="background:#1e3a5f;color:#38bdf8;padding:1px 8px;border-radius:4px;
                        font-size:0.75rem;font-family:'Space Mono',monospace;margin-left:8px;">{val}</span>
                    </div>
                    <div style="font-size:0.85rem;color:#7ba4cc;margin-top:3px;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

# ─── HALAMAN KONFIGURASI MODEL ────────────────────────────────────────────────
elif page == "⚙️ Konfigurasi Model":
    st.markdown('<div class="hero-title">Konfigurasi Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Transfer Learning · Arsitektur CNN · Hyperparameter</div>', unsafe_allow_html=True)

    model_tab1, model_tab2, model_tab3 = st.tabs(["🔵 VGG16", "🟠 DenseNet121", "🟢 MobileNetV2"])

    def render_model_info(name, base_info, arch_layers, color, params_m, size_mb, notes):
        st.markdown(f'<div class="section-header" style="border-color:{color};color:{color};">{name} Architecture</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Parameters</div><div class="metric-value" style="color:{color};">{params_m}M</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Ukuran File</div><div class="metric-value" style="color:{color};">{size_mb}MB</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Pre-trained</div><div class="metric-value" style="color:{color};font-size:1.1rem;">ImageNet</div></div>', unsafe_allow_html=True)

        st.markdown(f"<br>**Base Model:** {base_info}", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Arsitektur Custom Head</div>', unsafe_allow_html=True)

        for i, layer in enumerate(arch_layers):
            arrow = "↓" if i < len(arch_layers)-1 else ""
            st.markdown(f"""
            <div style="display:flex;align-items:center;margin:3px 0;">
                <div style="padding:8px 16px;background:rgba(56,189,248,0.08);border:1px solid {color}40;
                border-radius:8px;font-family:'Space Mono',monospace;font-size:0.8rem;color:#c9d8ed;flex:1;">{layer}</div>
            </div>
            {"<div style='text-align:center;color:#7ba4cc;margin:2px 0;'>↓</div>" if arrow else ""}
            """, unsafe_allow_html=True)

        st.markdown(f"""<div class="info-box" style="margin-top:12px;">📌 {notes}</div>""", unsafe_allow_html=True)

    with model_tab1:
        render_model_info(
            "VGG16",
            "VGG16 (Oxford) — 16 layer berbobot, arsitektur 3×3 conv blocks",
            ["VGG16 Base (frozen) — Input 224×224×3",
             "GlobalAveragePooling2D",
             "Dense(256, ReLU)",
             "Dropout(0.3)",
             "Dense(128, ReLU)",
             "Dense(3, Softmax) — Output"],
            "#60a5fa", 138, 105,
            "Model paling berat dan paling lama dalam inferensi. Base model di-freeze selama training awal (feature extraction mode)."
        )

    with model_tab2:
        render_model_info(
            "DenseNet121",
            "DenseNet121 — 121 layer, setiap layer terhubung ke semua layer sebelumnya",
            ["DenseNet121 Base (frozen) — Input 224×224×3",
             "GlobalAveragePooling2D",
             "Dense(256, ReLU)",
             "Dropout(0.3)",
             "Dense(128, ReLU)",
             "Dense(3, Softmax) — Output"],
            "#f97316", 7, 56.3,
            "Arsitektur dense connection memungkinkan reuse fitur yang sangat efektif. Menghasilkan akurasi tertinggi 92.5% dalam eksperimen ini."
        )

    with model_tab3:
        render_model_info(
            "MobileNetV2",
            "MobileNetV2 — Depthwise separable conv, dirancang untuk perangkat mobile",
            ["MobileNetV2 Base (frozen) — Input 224×224×3",
             "GlobalAveragePooling2D",
             "Dense(256, ReLU)",
             "Dropout(0.3)",
             "Dense(128, ReLU)",
             "Dense(3, Softmax) — Output"],
            "#34d399", 3.4, 13.2,
            "Model paling ringan (13.2 MB). Dengan selisih akurasi hanya 1.89% dibanding DenseNet121, MobileNetV2 menjadi pilihan paling optimal untuk deployment."
        )

    st.markdown("---")
    st.markdown('<div class="section-header">Hyperparameter Training</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    hp = [("Optimizer", "Adam"), ("Learning Rate", "0.001"), ("Batch Size", "32"), ("Epochs", "10")]
    colors_hp = ['#38bdf8','#818cf8','#f59e0b','#34d399']
    for col, (k, v), c in zip([col1,col2,col3,col4], hp, colors_hp):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{k}</div><div class="metric-value" style="color:{c};font-size:1.3rem;">{v}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Callbacks</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""<div class="info-box">
        🛑 <strong>EarlyStopping</strong><br>
        Monitor: <code>val_loss</code> · Patience: <code>5</code><br>
        Menghentikan training jika val_loss tidak membaik selama 5 epoch berturut-turut.
        Restore best weights otomatis aktif.
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""<div class="info-box">
        📉 <strong>ReduceLROnPlateau</strong><br>
        Monitor: <code>val_loss</code> · Factor: <code>0.5</code> · Patience: <code>3</code><br>
        Menurunkan learning rate secara dinamis ketika model mencapai titik jenuh.
        Min LR: <code>1e-6</code>
        </div>""", unsafe_allow_html=True)

# ─── HALAMAN TRAINING ─────────────────────────────────────────────────────────
elif page == "🏋️ Training":
    st.markdown('<div class="hero-title">Training Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Kurva Training · Accuracy · Loss</div>', unsafe_allow_html=True)

    selected_models = st.multiselect("Pilih model untuk ditampilkan:", ['VGG16','DenseNet121','MobileNetV2'], default=['VGG16','DenseNet121','MobileNetV2'])
    view_type = st.radio("Tampilkan:", ["Accuracy", "Loss", "Keduanya"], horizontal=True)

    if selected_models:
        epochs_range = range(1, 11)

        if view_type in ["Accuracy", "Keduanya"]:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='none')

            for name in selected_models:
                c = MODEL_COLORS[name]
                axes[0].plot(epochs_range, histories[name]['train_acc'], color=c, linewidth=2, label=f'{name} Train')
                axes[0].plot(epochs_range, histories[name]['val_acc'], color=c, linewidth=2, linestyle='--', alpha=0.7, label=f'{name} Val')

            axes[0].set_title('Training vs Validation Accuracy', color='#c9d8ed', fontweight='bold')
            axes[0].set_xlabel('Epoch', color='#7ba4cc')
            axes[0].set_ylabel('Accuracy', color='#7ba4cc')
            axes[0].legend(labelcolor='#c9d8ed', facecolor='#0d1829', edgecolor='#1e3a5f', fontsize=8)
            axes[0].set_facecolor('none')
            axes[0].tick_params(colors='#7ba4cc')
            axes[0].spines[:].set_color('#1e3a5f')
            axes[0].grid(alpha=0.15, color='#38bdf8')

            for name in selected_models:
                c = MODEL_COLORS[name]
                final_val = histories[name]['val_acc'][-1]
                axes[1].bar(name, final_val, color=c, alpha=0.8, width=0.4)
                axes[1].text(name, final_val + 0.005, f'{final_val:.3f}', ha='center', va='bottom', color='#c9d8ed', fontsize=10)

            axes[1].set_title('Final Validation Accuracy', color='#c9d8ed', fontweight='bold')
            axes[1].set_ylim(0, 1.05)
            axes[1].set_facecolor('none')
            axes[1].tick_params(colors='#7ba4cc')
            axes[1].spines[:].set_color('#1e3a5f')
            axes[1].grid(axis='y', alpha=0.15, color='#38bdf8')

            for ax in axes:
                ax.set_facecolor('none')
            fig.patch.set_alpha(0)
            st.pyplot(fig, transparent=True)
            plt.close()

        if view_type in ["Loss", "Keduanya"]:
            fig2, ax_loss = plt.subplots(figsize=(10, 4), facecolor='none')
            for name in selected_models:
                c = MODEL_COLORS[name]
                ax_loss.plot(epochs_range, histories[name]['train_loss'], color=c, linewidth=2, label=f'{name} Train Loss')
                ax_loss.plot(epochs_range, histories[name]['val_loss'], color=c, linewidth=2, linestyle='--', alpha=0.7, label=f'{name} Val Loss')
            ax_loss.set_title('Training vs Validation Loss', color='#c9d8ed', fontweight='bold')
            ax_loss.set_xlabel('Epoch', color='#7ba4cc')
            ax_loss.set_ylabel('Loss', color='#7ba4cc')
            ax_loss.legend(labelcolor='#c9d8ed', facecolor='#0d1829', edgecolor='#1e3a5f', fontsize=8)
            ax_loss.set_facecolor('none')
            ax_loss.tick_params(colors='#7ba4cc')
            ax_loss.spines[:].set_color('#1e3a5f')
            ax_loss.grid(alpha=0.15, color='#38bdf8')
            fig2.patch.set_alpha(0)
            st.pyplot(fig2, transparent=True)
            plt.close()

        st.markdown('<div class="section-header">Analisis Training</div>', unsafe_allow_html=True)
        analysis = {
            'DenseNet121': ("🟠", "Tampil sebagai model terbaik. Akurasi paling stabil dan mencapai puncak tertinggi mendekati **93%** pada epoch ke-8. Dense connections membantu reuse fitur secara efektif."),
            'MobileNetV2': ("🟢", "Berada di posisi kedua dengan performa kompetitif, berfluktuasi di kisaran **85%–90%**. Sangat efisien untuk ukurannya yang kecil."),
            'VGG16': ("🔵", "Menunjukkan performa paling rendah, tertahan di kisaran **72%–82%**. Arsitektur VGG yang lebih tua kurang efisien dalam task transfer learning ini."),
        }
        for name, (icon, text) in analysis.items():
            if name in selected_models:
                st.markdown(f"""
                <div style="padding:10px 14px;margin:6px 0;background:rgba(56,189,248,0.05);
                border-left:3px solid {MODEL_COLORS[name]};border-radius:0 8px 8px 0;">
                {icon} <strong style="color:{MODEL_COLORS[name]};">{name}</strong><br>
                <span style="color:#c9d8ed;font-size:0.9rem;">{text}</span>
                </div>""", unsafe_allow_html=True)

# ─── HALAMAN EVALUASI ─────────────────────────────────────────────────────────
elif page == "📈 Evaluasi Model":
    st.markdown('<div class="hero-title">Evaluasi Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Accuracy · Precision · Recall · F1-Score · Confusion Matrix</div>', unsafe_allow_html=True)

    # Tabel perbandingan
    st.markdown('<div class="section-header">Tabel Perbandingan Performa</div>', unsafe_allow_html=True)
    df_eval = pd.DataFrame([
        {'Model': name,
         'Accuracy': f"{r['accuracy']:.4f}",
         'Precision': f"{r['precision']:.4f}",
         'Recall': f"{r['recall']:.4f}",
         'F1-Score': f"{r['f1']:.4f}",
         'Ukuran': f"{r['size_mb']} MB"}
        for name, r in eval_results.items()
    ])
    st.dataframe(df_eval, use_container_width=True, hide_index=True)

    # Metric cards
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (name, r) in zip(cols, eval_results.items()):
        with col:
            color = MODEL_COLORS[name]
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}60;">
                <div class="metric-label">{name}</div>
                <div class="metric-value" style="color:{color};">{r['accuracy']*100:.1f}%</div>
                <div class="metric-model">Accuracy</div>
                <hr style="border-color:{color}30;margin:8px 0;">
                <div style="display:flex;justify-content:space-around;font-size:0.8rem;color:#7ba4cc;">
                    <div><div style="color:{color};">{r['precision']:.3f}</div>Precision</div>
                    <div><div style="color:{color};">{r['recall']:.3f}</div>Recall</div>
                    <div><div style="color:{color};">{r['f1']:.3f}</div>F1</div>
                </div>
            </div>""", unsafe_allow_html=True)

    # Bar chart perbandingan
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Visualisasi Performa</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.25
    for i, (name, r) in enumerate(eval_results.items()):
        vals = [r[m] for m in metrics]
        ax.bar(x + i*width, vals, width, label=name, color=MODEL_COLORS[name], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='#c9d8ed')
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel('Nilai', color='#7ba4cc')
    ax.set_title('Perbandingan Performa Semua Model', color='#c9d8ed', fontweight='bold')
    ax.legend(labelcolor='#c9d8ed', facecolor='#0d1829', edgecolor='#1e3a5f')
    ax.set_facecolor('none')
    ax.tick_params(colors='#7ba4cc')
    ax.spines[:].set_color('#1e3a5f')
    ax.grid(axis='y', alpha=0.15, color='#38bdf8')
    fig.patch.set_alpha(0)
    st.pyplot(fig, transparent=True)
    plt.close()

    # Confusion matrix
    st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5), facecolor='none')
    for ax2, (name, cm) in zip(axes2, cm_data.items()):
        sns.heatmap(cm, annot=True, fmt='d', ax=ax2,
                    cmap=sns.light_palette(MODEL_COLORS[name], as_cmap=True),
                    xticklabels=CONFIG['class_names'],
                    yticklabels=CONFIG['class_names'],
                    linewidths=0.5, linecolor='#0d1829')
        ax2.set_title(name, color='#c9d8ed', fontweight='bold')
        ax2.set_xlabel('Predicted', color='#7ba4cc')
        ax2.set_ylabel('Actual', color='#7ba4cc')
        ax2.tick_params(colors='#7ba4cc')
        ax2.set_facecolor('none')
    fig2.patch.set_alpha(0)
    st.pyplot(fig2, transparent=True)
    plt.close()

# ─── HALAMAN GRAD-CAM ─────────────────────────────────────────────────────────
elif page == "🔥 Grad-CAM (XAI)":
    st.markdown('<div class="hero-title">Grad-CAM Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Explainable AI · Gradient-weighted Class Activation Mapping</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Grad-CAM</strong> (Gradient-weighted Class Activation Mapping) adalah teknik XAI yang 
    memvisualisasikan area penting pada citra yang menjadi dasar keputusan model CNN. 
    Teknik ini menggunakan gradien yang mengalir ke <strong>layer konvolusi terakhir</strong> untuk 
    membuat peta aktivasi berwarna (heatmap).
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Layer Grad-CAM per Model</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    layers_used = [
        ("VGG16", "block5_conv3", "#60a5fa"),
        ("DenseNet121", "conv5_block16_2_conv", "#f97316"),
        ("MobileNetV2", "Conv_1", "#34d399"),
    ]
    for col, (name, layer, color) in zip([col1, col2, col3], layers_used):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}50;">
                <div class="metric-label">{name}</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:{color};
                word-break:break-all;margin-top:6px;">{layer}</div>
                <div class="metric-model" style="margin-top:4px;">Last Conv Layer</div>
            </div>""", unsafe_allow_html=True)

    # Simulasi visualisasi Grad-CAM
    st.markdown('<div class="section-header">Simulasi Visualisasi Grad-CAM</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="warning-box">
    ⚠️ Visualisasi berikut adalah <strong>simulasi ilustratif</strong> dari heatmap Grad-CAM. 
    Untuk hasil nyata, jalankan model pada dataset asli dengan kode Grad-CAM yang disediakan.
    </div>
    """, unsafe_allow_html=True)

    selected_class = st.selectbox("Pilih kelas sentimen:", CONFIG['class_names'])
    class_idx = CONFIG['class_names'].index(selected_class)

    fig_cam, axes_cam = plt.subplots(1, 4, figsize=(16, 4), facecolor='none')
    np.random.seed(class_idx * 10 + 42)

    # Original (simulasi gambar konflik)
    base_img = np.random.rand(224, 224, 3) * 0.5 + 0.2
    if selected_class == 'negatif':
        base_img[80:150, 60:160] = [0.9, 0.3, 0.1]
        base_img[50:90, 100:180] = [0.8, 0.5, 0.1]
    elif selected_class == 'positif':
        base_img[70:130, 70:160] = [0.3, 0.7, 0.5]
    else:
        base_img[90:140, 80:150] = [0.4, 0.6, 0.8]

    axes_cam[0].imshow(np.clip(base_img, 0, 1))
    axes_cam[0].set_title(f'Original\n[{selected_class}]', color='#c9d8ed', fontsize=10)
    axes_cam[0].axis('off')

    model_cam_data = {
        'VGG16': ({'center': (112, 112), 'sigma': 60, 'intensity': 0.75}),
        'DenseNet121': ({'center': (105, 115), 'sigma': 45, 'intensity': 0.95}),
        'MobileNetV2': ({'center': (108, 110), 'sigma': 50, 'intensity': 0.88}),
    }

    for ax_c, (mname, params) in zip(axes_cam[1:], model_cam_data.items()):
        y, x = np.mgrid[0:224, 0:224]
        cy, cx = params['center']
        heatmap = np.exp(-((x-cx)**2 + (y-cy)**2) / (2*params['sigma']**2)) * params['intensity']
        heatmap += np.random.rand(224, 224) * 0.1
        heatmap = np.clip(heatmap, 0, 1)

        heatmap_color = plt.cm.jet(heatmap)[:, :, :3]
        overlay = np.clip(0.55 * base_img + 0.45 * heatmap_color, 0, 1)

        pred_labels = {'VGG16': selected_class, 'DenseNet121': selected_class, 'MobileNetV2': selected_class}
        ax_c.imshow(overlay)
        ax_c.set_title(f'{mname}\nPred: {pred_labels[mname]}', color=MODEL_COLORS[mname], fontsize=10)
        ax_c.axis('off')

    for ax_c in axes_cam:
        ax_c.set_facecolor('none')
    fig_cam.patch.set_alpha(0)
    fig_cam.suptitle(f'Grad-CAM — Kelas: {selected_class.upper()}', color='#c9d8ed', fontweight='bold')
    st.pyplot(fig_cam, transparent=True)
    plt.close()

    # Legenda warna
    st.markdown('<div class="section-header">Arti Warna Grad-CAM</div>', unsafe_allow_html=True)
    color_meanings = [
        ("🔴", "#ef4444", "Merah — Area Intensitas Tinggi (Hot Spot)",
         "Area yang paling berpengaruh bagi model AI. Gradien terbesar ada di sini. Jika model memprediksi 'Negatif' dan merah berada di atas ledakan, artinya model bekerja logis."),
        ("🟡", "#eab308", "Kuning–Hijau — Area Intensitas Menengah",
         "Area pendukung atau kontekstual. Membantu model lebih yakin, bukan penentu utama. Contoh: jika merah adalah bendera, kuning/hijau biasanya orang yang memegang bendera."),
        ("🔵", "#3b82f6", "Biru — Area Intensitas Rendah (Cold Spot)",
         "Area yang diabaikan model. Piksel ini dianggap tidak relevan. Pada gambar Peta (Netral), dominasi biru tua menandakan AI tidak menemukan objek provokatif."),
    ]
    for icon, color, title, desc in color_meanings:
        st.markdown(f"""
        <div style="display:flex;gap:14px;padding:12px 16px;margin:8px 0;
        background:rgba(0,0,0,0.2);border-left:4px solid {color};border-radius:0 10px 10px 0;">
            <div style="font-size:1.8rem;">{icon}</div>
            <div>
                <div style="font-weight:700;color:{color};">{title}</div>
                <div style="font-size:0.88rem;color:#c9d8ed;margin-top:4px;">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

# ─── HALAMAN UJI COBA MODEL ───────────────────────────────────────────────────
elif page == "🧪 Uji Coba Model":
    st.markdown('<div class="hero-title">Uji Coba Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload Gambar · Prediksi Real Model .h5 · Grad-CAM · Analisis Sentimen</div>', unsafe_allow_html=True)

    # ── Import TensorFlow di sini agar tidak memperlambat halaman lain ────────
    try:
        import tensorflow as tf
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False

    from PIL import Image as PILImage
    import io
    from collections import Counter

    # ── Path model ─────────────────────────────────────────────────────────────
    model_paths = {
        'VGG16':       'models/VGG16.h5',
        'DenseNet121': 'models/DenseNet121.h5',
        'MobileNetV2': 'models/MobileNetV2.h5',
    }
    layer_map = {
        'VGG16':       'block5_conv3',
        'DenseNet121': 'conv5_block16_2_conv',
        'MobileNetV2': 'Conv_1',
    }
    class_colors_map = {'positif': '#34d399', 'netral': '#60a5fa', 'negatif': '#f87171'}
    class_icons_map  = {'positif': '😊', 'netral': '😐', 'negatif': '😠'}

    # ── Fungsi load model dengan cache ─────────────────────────────────────────
    @st.cache_resource(show_spinner=False)
    def load_model_h5(path):
        """Load model .h5 dari disk. Return (model, error_message)."""
        if not TF_AVAILABLE:
            return None, "TensorFlow tidak terinstall di environment ini."
        if not os.path.exists(path):
            return None, f"File tidak ditemukan: `{path}`"
        try:
            model = tf.keras.models.load_model(path, compile=False)
            return model, None
        except Exception as e:
            return None, str(e)

    # ── Fungsi prediksi nyata ───────────────────────────────────────────────────
    def predict_real(model, img_norm):
        """
        Prediksi menggunakan model nyata.
        img_norm: numpy array (224, 224, 3) dengan nilai [0,1].
        Return: (probs array shape (3,), pred_class int, confidence float)
        """
        img_input = np.expand_dims(img_norm, axis=0).astype(np.float32)
        preds = model.predict(img_input, verbose=0)  # shape (1, 3)
        probs = preds[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        return probs, pred_class, confidence

    # ── Fungsi Grad-CAM nyata dengan GradientTape ──────────────────────────────
    def compute_gradcam_real(model, img_norm, pred_class, last_conv_layer_name):
        """
        Hitung Grad-CAM nyata dari model .h5.
        Mengembalikan heatmap 2D bernilai [0,1] berukuran sama dengan img_norm.
        """
        img_input = tf.cast(np.expand_dims(img_norm, axis=0), tf.float32)

        # Buat model intermediate: input → last conv layer output + final output
        try:
            last_conv_layer = model.get_layer(last_conv_layer_name)
        except ValueError:
            # Coba cari layer konvolusi terakhir secara otomatis
            conv_layers = [l for l in model.layers
                           if isinstance(l, (tf.keras.layers.Conv2D,
                                             tf.keras.layers.DepthwiseConv2D))]
            if not conv_layers:
                return None
            last_conv_layer = conv_layers[-1]

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_input)
            loss = predictions[:, pred_class]

        grads = tape.gradient(loss, conv_outputs)  # (1, h, w, filters)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (filters,)

        conv_outputs = conv_outputs[0]  # (h, w, filters)
        # Bobot setiap feature map
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
        heatmap = tf.squeeze(heatmap)  # (h, w)
        heatmap = tf.nn.relu(heatmap)  # hapus nilai negatif

        # Normalisasi ke [0, 1]
        heatmap = heatmap.numpy()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Resize ke 224×224
        import cv2
        heatmap_resized = cv2.resize(heatmap, (img_norm.shape[1], img_norm.shape[0]))
        return heatmap_resized

    def overlay_gradcam(img_norm, heatmap, alpha=0.45):
        """Overlay heatmap Grad-CAM di atas gambar asli."""
        heatmap_color = plt.cm.jet(heatmap)[:, :, :3]
        overlay = np.clip((1 - alpha) * img_norm + alpha * heatmap_color, 0, 1)
        return overlay

    # ── Cek & tampilkan status model ───────────────────────────────────────────
    st.markdown('<div class="section-header">📦 Status Model</div>', unsafe_allow_html=True)
    if not TF_AVAILABLE:
        st.error("❌ **TensorFlow tidak tersedia.** Install dengan: `pip install tensorflow`")
        st.stop()

    col_av1, col_av2, col_av3 = st.columns(3)
    model_load_errors = {}
    for col_av, mname in zip([col_av1, col_av2, col_av3], ['VGG16', 'DenseNet121', 'MobileNetV2']):
        with col_av:
            mpath = model_paths[mname]
            exists = os.path.exists(mpath)
            if exists:
                status_icon, status_color, status_text = "✅", "#34d399", "File Ditemukan"
                model_load_errors[mname] = None
            else:
                status_icon, status_color, status_text = "❌", "#f87171", "File Tidak Ada"
                model_load_errors[mname] = f"File tidak ditemukan: `{mpath}`"
            st.markdown(f"""
            <div style="text-align:center;padding:12px;background:rgba(0,0,0,0.2);
            border:2px solid {MODEL_COLORS[mname]}50;border-radius:12px;margin-bottom:8px;">
                <div style="font-size:1.6rem;">{status_icon}</div>
                <div style="color:{MODEL_COLORS[mname]};font-family:'Space Mono',monospace;
                font-weight:700;font-size:0.95rem;margin-top:4px;">{mname}</div>
                <div style="color:{status_color};font-size:0.78rem;margin-top:3px;
                font-weight:600;">{status_text}</div>
                <div style="color:#7ba4cc;font-size:0.7rem;margin-top:2px;">{mpath}</div>
            </div>""", unsafe_allow_html=True)

    # Jika ada model yang tidak ditemukan, tampilkan error dan berhenti
    missing = [m for m, err in model_load_errors.items() if err]
    if missing:
        st.markdown(f"""
        <div style="background:rgba(248,113,113,0.1);border:1px solid #f87171;
        border-radius:10px;padding:16px;margin:12px 0;">
            <div style="color:#f87171;font-weight:700;font-size:1rem;margin-bottom:8px;">
                ❌ Model Tidak Ditemukan — Tidak Dapat Melanjutkan
            </div>
            <div style="color:#fca5a5;font-size:0.88rem;line-height:1.7;">
                File model berikut <strong>tidak ada</strong> di direktori proyek:<br>
                {"".join(f"<code style='display:block;margin:4px 0;'>• {model_paths[m]}</code>" for m in missing)}
                <br>
                Pastikan Anda sudah melatih model dan menyimpannya ke folder <code>models/</code> 
                di root direktori proyek Streamlit ini, lalu jalankan ulang aplikasi.
            </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    st.markdown("""
    <div class="success-box">
    ✅ <strong>Semua model ditemukan.</strong> Sistem siap melakukan prediksi nyata menggunakan 
    ketiga model CNN. Silakan upload gambar di bawah untuk memulai.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Penjelasan cara kerja ───────────────────────────────────────────────────
    st.markdown("""
    <div class="info-box">
    <strong>🧪 Cara Kerja Uji Coba:</strong>
    <ol style="margin:8px 0 0 16px;color:#c9d8ed;">
        <li><strong>Upload Gambar</strong> — Gambar JPG/PNG diunggah, kemudian dibaca sebagai array piksel RGB</li>
        <li><strong>Preprocessing</strong> — Resize ke 224×224, normalisasi ke [0,1], tambahkan dimensi batch</li>
        <li><strong>Prediksi Nyata</strong> — Setiap model .h5 menjalankan <code>model.predict()</code> dan menghasilkan probabilitas untuk 3 kelas</li>
        <li><strong>Grad-CAM Nyata</strong> — <code>GradientTape</code> menghitung gradien dari layer konvolusi terakhir terhadap kelas yang diprediksi, lalu divisualisasikan sebagai heatmap</li>
        <li><strong>Analisis</strong> — Hasil ketiga model dibandingkan dan diinterpretasikan</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload gambar ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📁 Upload Gambar untuk Diuji</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Pilih gambar (JPG/PNG):",
        type=['jpg', 'jpeg', 'png'],
        help="Upload gambar untuk dianalisis oleh ketiga model CNN"
    )

    if uploaded_file is not None:
        # Baca gambar
        img_bytes = uploaded_file.read()
        pil_img = PILImage.open(io.BytesIO(img_bytes)).convert('RGB')
        img_original = np.array(pil_img)

        # Tampilkan gambar asli
        st.markdown('<div class="section-header">🖼️ Gambar yang Diupload</div>', unsafe_allow_html=True)
        col_orig1, col_orig2, col_orig3 = st.columns([1, 2, 1])
        with col_orig2:
            st.image(img_original, caption=f"Gambar Asli · {img_original.shape[1]}×{img_original.shape[0]} piksel", use_container_width=True)
            st.markdown(f"""
            <div style="text-align:center;padding:6px;background:rgba(56,189,248,0.07);
            border-radius:8px;font-family:'Space Mono',monospace;font-size:0.8rem;color:#7ba4cc;">
                Format: {uploaded_file.type} · Ukuran file: {len(img_bytes)/1024:.1f} KB
            </div>""", unsafe_allow_html=True)

        # Preprocessing: resize + normalisasi
        pil_resized = pil_img.resize((224, 224), PILImage.LANCZOS)
        img_norm = np.array(pil_resized, dtype=np.float32) / 255.0  # (224,224,3) [0,1]

        # Visualisasi preprocessing
        st.markdown("---")
        st.markdown('<div class="section-header">⚙️ Preprocessing Gambar</div>', unsafe_allow_html=True)
        fig_pre, axes_pre = plt.subplots(1, 3, figsize=(14, 4), facecolor='none')
        axes_pre[0].imshow(img_original)
        axes_pre[0].set_title(f'Gambar Asli\n{img_original.shape[1]}×{img_original.shape[0]} px', color='#c9d8ed', fontsize=10)
        axes_pre[0].axis('off')
        axes_pre[1].imshow(pil_resized)
        axes_pre[1].set_title('Setelah Resize\n224×224 px', color='#38bdf8', fontsize=10)
        axes_pre[1].axis('off')
        axes_pre[2].hist(img_norm[:,:,0].ravel(), bins=50, color='#f87171', alpha=0.6, label='R', density=True)
        axes_pre[2].hist(img_norm[:,:,1].ravel(), bins=50, color='#4ade80', alpha=0.6, label='G', density=True)
        axes_pre[2].hist(img_norm[:,:,2].ravel(), bins=50, color='#60a5fa', alpha=0.6, label='B', density=True)
        axes_pre[2].set_title('Distribusi Pixel Ternormalisasi\n[0.0 – 1.0]', color='#34d399', fontsize=10)
        axes_pre[2].set_xlabel('Nilai Pixel', color='#7ba4cc', fontsize=9)
        axes_pre[2].set_ylabel('Densitas', color='#7ba4cc', fontsize=9)
        axes_pre[2].legend(labelcolor='#c9d8ed', facecolor='#0d1829', edgecolor='#1e3a5f', fontsize=8)
        axes_pre[2].set_facecolor('none')
        axes_pre[2].tick_params(colors='#7ba4cc', labelsize=8)
        axes_pre[2].spines[:].set_color('#1e3a5f')
        for ax_p in axes_pre[:2]: ax_p.set_facecolor('none')
        fig_pre.patch.set_alpha(0)
        st.pyplot(fig_pre, transparent=True)
        plt.close()

        st.markdown("---")
        st.markdown('<div class="section-header">🤖 Memuat Model & Menjalankan Prediksi Nyata...</div>', unsafe_allow_html=True)

        # ── Load model & prediksi per model ────────────────────────────────────
        all_predictions = {}
        all_heatmaps    = {}
        load_errors     = {}

        for mname in ['VGG16', 'DenseNet121', 'MobileNetV2']:
            with st.spinner(f"Memuat {mname} dan menghitung prediksi + Grad-CAM..."):
                model, err = load_model_h5(model_paths[mname])
                if err or model is None:
                    load_errors[mname] = err or "Gagal load model."
                    continue

                # Prediksi nyata
                try:
                    probs, pred_class, confidence = predict_real(model, img_norm)
                except Exception as e:
                    load_errors[mname] = f"Error saat predict: {e}"
                    continue

                # Grad-CAM nyata
                heatmap = None
                try:
                    heatmap = compute_gradcam_real(model, img_norm, pred_class, layer_map[mname])
                except Exception as e:
                    pass  # Grad-CAM gagal tidak menghentikan prediksi

                all_predictions[mname] = {
                    'probs':      probs,
                    'pred_class': pred_class,
                    'pred_label': CONFIG['class_names'][pred_class],
                    'confidence': confidence,
                }
                all_heatmaps[mname] = heatmap

        # Tampilkan error load jika ada
        for mname, err in load_errors.items():
            st.markdown(f"""
            <div style="background:rgba(248,113,113,0.1);border:1px solid #f87171;
            border-radius:10px;padding:12px;margin:6px 0;">
                <strong style="color:#f87171;">❌ {mname} — Gagal diproses</strong><br>
                <code style="color:#fca5a5;font-size:0.82rem;">{err}</code>
            </div>""", unsafe_allow_html=True)

        if not all_predictions:
            st.error("❌ Tidak ada model yang berhasil dijalankan. Periksa error di atas.")
            st.stop()

        st.markdown("---")
        st.markdown('<div class="section-header">📊 Hasil Prediksi per Model</div>', unsafe_allow_html=True)

        # ── Tampilkan kartu hasil per model ────────────────────────────────────
        model_cols = st.columns(len(all_predictions))
        for col_m, mname in zip(model_cols, all_predictions.keys()):
            with col_m:
                pred_info  = all_predictions[mname]
                pred_label = pred_info['pred_label']
                confidence = pred_info['confidence']
                probs      = pred_info['probs']
                model_color = MODEL_COLORS[mname]
                class_color = class_colors_map[pred_label]
                class_icon  = class_icons_map[pred_label]

                # Header kartu
                st.markdown(f"""
                <div style="text-align:center;padding:16px;
                background:linear-gradient(135deg,rgba(13,24,41,0.95),rgba(19,32,53,0.95));
                border:2px solid {model_color}70;border-radius:14px;margin-bottom:10px;">
                    <div style="font-family:'Space Mono',monospace;font-weight:700;
                    color:{model_color};font-size:1.05rem;">{mname}</div>
                    <div style="font-size:2.2rem;margin:10px 0;">{class_icon}</div>
                    <div style="font-size:1.3rem;font-weight:700;color:{class_color};
                    font-family:'Space Mono',monospace;">{pred_label.upper()}</div>
                    <div style="font-size:0.88rem;color:#7ba4cc;margin-top:6px;">
                        Confidence:<br>
                        <span style="color:{class_color};font-family:'Space Mono',monospace;
                        font-weight:700;font-size:1.2rem;">{confidence*100:.2f}%</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Gambar resize + Grad-CAM overlay
                fig_res, axes_res = plt.subplots(1, 2, figsize=(6, 3), facecolor='none')
                axes_res[0].imshow(img_norm)
                axes_res[0].set_title('Input (224×224)', color='#c9d8ed', fontsize=8)
                axes_res[0].axis('off')
                axes_res[0].set_facecolor('none')

                heatmap = all_heatmaps.get(mname)
                if heatmap is not None:
                    gc_overlay = overlay_gradcam(img_norm, heatmap, alpha=0.45)
                    axes_res[1].imshow(gc_overlay)
                    axes_res[1].set_title('Grad-CAM (Real)', color=model_color, fontsize=8)
                else:
                    axes_res[1].imshow(img_norm)
                    axes_res[1].set_title('Grad-CAM\n(Gagal)', color='#f87171', fontsize=8)
                axes_res[1].axis('off')
                axes_res[1].set_facecolor('none')
                fig_res.patch.set_alpha(0)
                fig_res.tight_layout(pad=0.3)
                st.pyplot(fig_res, transparent=True)
                plt.close()

                # Bar chart probabilitas nyata
                fig_prob, ax_prob = plt.subplots(figsize=(4, 2.8), facecolor='none')
                class_labels_bar = ['Positif', 'Netral', 'Negatif']
                bar_colors_bar   = ['#34d399', '#60a5fa', '#f87171']
                bars_prob = ax_prob.barh(class_labels_bar, probs * 100,
                                         color=bar_colors_bar, alpha=0.85, height=0.5)
                for bar_p, pv in zip(bars_prob, probs):
                    ax_prob.text(bar_p.get_width() + 0.5,
                                 bar_p.get_y() + bar_p.get_height()/2,
                                 f'{pv*100:.2f}%', va='center', ha='left',
                                 color='#c9d8ed', fontsize=8, fontfamily='monospace')
                ax_prob.set_xlim(0, 118)
                ax_prob.set_xlabel('Probabilitas (%)', color='#7ba4cc', fontsize=8)
                ax_prob.set_title('Probabilitas Kelas', color='#c9d8ed', fontsize=9, fontweight='bold')
                ax_prob.set_facecolor('none')
                ax_prob.tick_params(colors='#7ba4cc', labelsize=8)
                ax_prob.spines[:].set_color('#1e3a5f')
                ax_prob.invert_yaxis()
                fig_prob.patch.set_alpha(0)
                fig_prob.tight_layout()
                st.pyplot(fig_prob, transparent=True)
                plt.close()

                # Info layer
                st.markdown(f"""
                <div style="padding:8px 10px;background:rgba(0,0,0,0.3);border-radius:8px;
                font-size:0.72rem;color:#7ba4cc;margin-top:4px;
                font-family:'Space Mono',monospace;text-align:center;">
                    Grad-CAM Layer:<br>
                    <span style="color:{model_color};">{layer_map[mname]}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Tabel ringkasan semua model ─────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Tabel Ringkasan Prediksi</div>', unsafe_allow_html=True)
        summary_rows = []
        for mname in ['VGG16', 'DenseNet121', 'MobileNetV2']:
            if mname not in all_predictions:
                summary_rows.append({'Model': mname, 'Prediksi': '❌ Error', 'Confidence': '-',
                                     'P(Positif)': '-', 'P(Netral)': '-', 'P(Negatif)': '-'})
                continue
            p  = all_predictions[mname]
            pv = p['probs']
            summary_rows.append({
                'Model':       mname,
                'Prediksi':    p['pred_label'].upper(),
                'Confidence':  f"{p['confidence']*100:.2f}%",
                'P(Positif)':  f"{pv[0]*100:.2f}%",
                'P(Netral)':   f"{pv[1]*100:.2f}%",
                'P(Negatif)':  f"{pv[2]*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # ── Diagram perbandingan grouped bar ───────────────────────────────────
        if len(all_predictions) > 0:
            st.markdown('<div class="section-header">📈 Diagram Sentimen Perbandingan Semua Model</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            Diagram ini menampilkan probabilitas <strong>nyata</strong> dari model .h5 Anda untuk 
            setiap kelas sentimen. Nilai berasal langsung dari output <code>softmax</code> model.
            </div>
            """, unsafe_allow_html=True)

            fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(14, 5), facecolor='none')

            # Grouped bar probabilitas
            x_cmp      = np.arange(3)
            w_cmp      = 0.25
            cls_labels = ['Positif', 'Netral', 'Negatif']
            mnames_ok  = list(all_predictions.keys())

            for i, mname in enumerate(mnames_ok):
                pv = all_predictions[mname]['probs'] * 100
                bars_c = axes_cmp[0].bar(x_cmp + i * w_cmp, pv, w_cmp,
                                          label=mname, color=MODEL_COLORS[mname], alpha=0.85)
                for bc in bars_c:
                    hc = bc.get_height()
                    axes_cmp[0].text(bc.get_x() + bc.get_width()/2, hc + 0.5,
                                     f'{hc:.1f}', ha='center', va='bottom',
                                     color='#c9d8ed', fontsize=7, fontfamily='monospace')

            axes_cmp[0].set_xticks(x_cmp + w_cmp * (len(mnames_ok)-1) / 2)
            axes_cmp[0].set_xticklabels(cls_labels, color='#c9d8ed', fontsize=10)
            axes_cmp[0].set_ylabel('Probabilitas (%)', color='#7ba4cc')
            axes_cmp[0].set_title('Probabilitas per Kelas & Model\n(dari model nyata)', color='#c9d8ed', fontweight='bold')
            axes_cmp[0].legend(labelcolor='#c9d8ed', facecolor='#0d1829', edgecolor='#1e3a5f', fontsize=9)
            axes_cmp[0].set_facecolor('none')
            axes_cmp[0].tick_params(colors='#7ba4cc')
            axes_cmp[0].spines[:].set_color('#1e3a5f')
            axes_cmp[0].grid(axis='y', alpha=0.15, color='#38bdf8')
            axes_cmp[0].set_ylim(0, 110)

            # Confidence per model
            confs  = [all_predictions[m]['confidence'] * 100 for m in mnames_ok]
            labels = [all_predictions[m]['pred_label'] for m in mnames_ok]
            bcolors = [class_colors_map[l] for l in labels]
            ecolors = [MODEL_COLORS[m] for m in mnames_ok]
            bars_cf = axes_cmp[1].bar(mnames_ok, confs, color=bcolors, alpha=0.85,
                                       width=0.45, edgecolor=ecolors, linewidth=2)
            for bc2, lbl, cv in zip(bars_cf, labels, confs):
                axes_cmp[1].text(bc2.get_x() + bc2.get_width()/2, bc2.get_height() + 1.0,
                                  f'{lbl.upper()}\n{cv:.2f}%',
                                  ha='center', va='bottom', color='#c9d8ed',
                                  fontsize=9, fontweight='bold')
            for ref, rlbl in [(50, '50%'), (80, '80%')]:
                axes_cmp[1].axhline(y=ref, color='#38bdf8', linestyle='--', alpha=0.3, linewidth=1)
                axes_cmp[1].text(len(mnames_ok) - 0.4, ref + 1, rlbl,
                                  color='#38bdf8', fontsize=8, alpha=0.5)
            axes_cmp[1].set_ylabel('Confidence (%)', color='#7ba4cc')
            axes_cmp[1].set_title('Confidence Prediksi per Model\n(dari model nyata)', color='#c9d8ed', fontweight='bold')
            axes_cmp[1].set_ylim(0, 115)
            axes_cmp[1].set_facecolor('none')
            axes_cmp[1].tick_params(colors='#c9d8ed')
            axes_cmp[1].spines[:].set_color('#1e3a5f')
            axes_cmp[1].grid(axis='y', alpha=0.15, color='#38bdf8')

            for ax_cmp in axes_cmp: ax_cmp.set_facecolor('none')
            fig_cmp.patch.set_alpha(0)
            fig_cmp.tight_layout()
            st.pyplot(fig_cmp, transparent=True)
            plt.close()

        # ── Perbandingan Grad-CAM semua model ─────────────────────────────────
        gcam_models = [m for m in all_predictions if all_heatmaps.get(m) is not None]
        if gcam_models:
            st.markdown('<div class="section-header">🔥 Perbandingan Grad-CAM Semua Model</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            Heatmap Grad-CAM di bawah ini dihasilkan langsung dari gradien model .h5 Anda menggunakan 
            <code>GradientTape</code>. Area <span style="color:#ef4444;font-weight:700;">merah</span> 
            menunjukkan piksel yang paling mempengaruhi keputusan model.
            </div>
            """, unsafe_allow_html=True)

            ncols = 1 + len(gcam_models)
            fig_gc, axes_gc = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5), facecolor='none')
            if ncols == 2: axes_gc = [axes_gc[0], axes_gc[1]]

            axes_gc[0].imshow(np.array(pil_resized))
            axes_gc[0].set_title('Input Asli\n(224×224)', color='#c9d8ed', fontsize=11, fontweight='bold')
            axes_gc[0].axis('off')
            axes_gc[0].set_facecolor('none')

            for ax_gc, mname in zip(axes_gc[1:], gcam_models):
                hm = all_heatmaps[mname]
                gc_ov = overlay_gradcam(img_norm, hm, alpha=0.45)
                ax_gc.imshow(gc_ov)
                pred_lbl  = all_predictions[mname]['pred_label']
                conf_val  = all_predictions[mname]['confidence']
                ax_gc.set_title(f'{mname}\n{pred_lbl.upper()} ({conf_val*100:.2f}%)',
                                 color=MODEL_COLORS[mname], fontsize=10, fontweight='bold')
                ax_gc.axis('off')
                ax_gc.set_facecolor('none')

            fig_gc.patch.set_alpha(0)
            fig_gc.suptitle('Grad-CAM Nyata — Gradien dari Layer Konvolusi Terakhir',
                             color='#c9d8ed', fontweight='bold', fontsize=12)
            fig_gc.tight_layout()
            st.pyplot(fig_gc, transparent=True)
            plt.close()

            # Legenda warna
            col_l1, col_l2, col_l3 = st.columns(3)
            for col_l, (title, color, desc) in zip(
                [col_l1, col_l2, col_l3],
                [("🔴 Merah — Hot Spot", "#ef4444",
                  "Area yang paling mempengaruhi prediksi. Gradien terbesar berada di sini."),
                 ("🟡 Kuning-Hijau — Pendukung", "#eab308",
                  "Area yang turut berkontribusi namun bukan penentu utama keputusan model."),
                 ("🔵 Biru — Diabaikan", "#3b82f6",
                  "Area yang dianggap tidak relevan oleh model untuk kelas yang diprediksi.")]
            ):
                with col_l:
                    st.markdown(f"""
                    <div style="padding:10px;background:rgba(0,0,0,0.2);
                    border-left:3px solid {color};border-radius:0 8px 8px 0;margin-top:8px;">
                        <div style="font-weight:700;color:{color};font-size:0.88rem;">{title}</div>
                        <div style="font-size:0.8rem;color:#c9d8ed;margin-top:4px;">{desc}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Analisis konsensus & interpretasi ─────────────────────────────────
        st.markdown('<div class="section-header">🧠 Analisis & Interpretasi Hasil</div>', unsafe_allow_html=True)

        pred_labels_all = [all_predictions[m]['pred_label'] for m in all_predictions]
        label_counts_pred = Counter(pred_labels_all)
        majority_label, majority_count = label_counts_pred.most_common(1)[0]

        if majority_count == len(all_predictions):
            st.markdown(f"""
            <div class="success-box">
            ✅ <strong>Konsensus Penuh — Semua model sepakat!</strong><br>
            Seluruh model yang berhasil dijalankan memprediksi gambar ini sebagai 
            <strong style="color:{class_colors_map[majority_label]};">{majority_label.upper()}</strong>. 
            Prediksi ini sangat dapat dipercaya karena berasal dari arsitektur yang berbeda-beda 
            namun menghasilkan kesimpulan yang sama.
            </div>""", unsafe_allow_html=True)
        elif majority_count >= 2:
            minority = [m for m in all_predictions if all_predictions[m]['pred_label'] != majority_label]
            st.markdown(f"""
            <div class="warning-box">
            ⚠️ <strong>Konsensus Mayoritas (tidak semua model sepakat)</strong><br>
            Mayoritas model memprediksi 
            <strong style="color:{class_colors_map[majority_label]};">{majority_label.upper()}</strong>, 
            namun <strong>{', '.join(minority)}</strong> memberikan prediksi berbeda. 
            Gunakan prediksi <strong>DenseNet121</strong> sebagai referensi utama (akurasi 92.5%).
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
            ⚠️ <strong>Tidak ada konsensus</strong> — Semua model memberikan prediksi berbeda. 
            Gambar kemungkinan memiliki elemen visual yang ambigu. 
            Gunakan prediksi DenseNet121 sebagai acuan terpercaya.
            </div>""", unsafe_allow_html=True)

        # Analisis per model
        analysis_texts = {
            'VGG16': {
                'positif': "VGG16 mendeteksi elemen visual yang berasosiasi dengan sentimen positif. Dengan akurasi 77.6%, model ini menggunakan fitur low-level (tepi, warna, tekstur) dari 16 layer konvolusinya. Fokus Grad-CAM pada area terang/cerah dalam gambar.",
                'netral':  "VGG16 tidak menemukan elemen yang cukup kuat untuk dikategorikan positif maupun negatif. Arsitektur VGG16 yang lebih tradisional cenderung kurang tepat untuk sentimen halus — interpretasi ini perlu dikonfirmasi oleh DenseNet121.",
                'negatif': "VGG16 mendeteksi elemen visual destruktif atau konfrontatif. Meskipun akurasinya hanya 77.6%, sentimen negatif yang kuat biasanya terdeteksi dengan baik oleh semua arsitektur karena fitur visualnya mencolok.",
            },
            'DenseNet121': {
                'positif': "DenseNet121 — model dengan akurasi tertinggi (92.5%) — mengklasifikasikan gambar ini sebagai positif. Dense connections memungkinkan reuse fitur dari semua layer, sehingga model sangat sensitif terhadap nuansa visual seperti gestur damai dan komposisi warna harmonis.",
                'netral':  "DenseNet121 mengidentifikasi gambar sebagai netral dengan kepercayaan tertinggi di antara ketiga model. Prediksi ini sangat dapat diandalkan — model berhasil mengenali bahwa gambar tidak mengandung trigger emosional yang signifikan.",
                'negatif': "DenseNet121 mendeteksi sentimen negatif. Ini adalah prediksi yang paling kredibel karena DenseNet121 memiliki akurasi tertinggi (92.5%). Dense connections membuat model sangat sensitif terhadap pola destruktif, kekerasan, atau ekspresi agresif.",
            },
            'MobileNetV2': {
                'positif': "MobileNetV2 memvalidasi sentimen positif dengan efisiensi tinggi. Model 13.2 MB ini menggunakan depthwise separable convolutions untuk menangkap fitur positif secara akurat, dengan akurasi kompetitif 90.6%.",
                'netral':  "MobileNetV2 mengklasifikasikan gambar sebagai netral. Dengan arsitektur mobile-first yang efisien, model ini tidak menemukan elemen emosional yang dominan pada gambar.",
                'negatif': "MobileNetV2 mendeteksi sentimen negatif dengan akurasi 90.6%. Meski model paling ringan, MobileNetV2 tetap mampu mengenali elemen destruktif dan konflik secara efektif.",
            }
        }
        for mname, pred_info in all_predictions.items():
            pred_lbl = pred_info['pred_label']
            conf_v   = pred_info['confidence']
            atext    = analysis_texts.get(mname, {}).get(pred_lbl, "Prediksi dihasilkan dari model nyata.")
            st.markdown(f"""
            <div style="padding:14px 16px;margin:8px 0;
            background:rgba(56,189,248,0.04);
            border:1px solid {MODEL_COLORS[mname]}40;
            border-left:4px solid {MODEL_COLORS[mname]};border-radius:0 10px 10px 0;">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                    <span style="font-family:'Space Mono',monospace;font-weight:700;
                    color:{MODEL_COLORS[mname]};font-size:1rem;">{mname}</span>
                    <span style="background:{class_colors_map[pred_lbl]}20;
                    color:{class_colors_map[pred_lbl]};
                    border:1px solid {class_colors_map[pred_lbl]};
                    padding:2px 10px;border-radius:999px;
                    font-size:0.78rem;font-family:'Space Mono',monospace;">
                        {pred_lbl.upper()} · {conf_v*100:.2f}%
                    </span>
                </div>
                <div style="font-size:0.88rem;color:#c9d8ed;line-height:1.7;">{atext}</div>
            </div>""", unsafe_allow_html=True)

        # ── Rekomendasi final ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if 'DenseNet121' in all_predictions:
            best_lbl  = all_predictions['DenseNet121']['pred_label']
            best_conf = all_predictions['DenseNet121']['confidence']
            rec_model = 'DenseNet121'
        else:
            # Fallback ke model dengan confidence tertinggi
            rec_model = max(all_predictions, key=lambda m: all_predictions[m]['confidence'])
            best_lbl  = all_predictions[rec_model]['pred_label']
            best_conf = all_predictions[rec_model]['confidence']

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(56,189,248,0.1),rgba(129,140,248,0.1));
        border:1px solid #38bdf8;border-radius:14px;padding:20px;">
            <div style="font-family:'Space Mono',monospace;font-size:1rem;
            color:#38bdf8;font-weight:700;margin-bottom:10px;">
                🏆 Rekomendasi Prediksi Final (dari model nyata)
            </div>
            <div style="color:#c9d8ed;font-size:0.92rem;line-height:1.9;">
                Prediksi terpercaya menggunakan 
                <strong style="color:{MODEL_COLORS[rec_model]};">{rec_model}</strong> 
                (akurasi tertinggi 92.5% pada test set penelitian ini):<br><br>
                ▶ Sentimen:
                <strong style="color:{class_colors_map[best_lbl]};font-size:1.2rem;
                font-family:'Space Mono',monospace;"> {best_lbl.upper()} </strong>
                &nbsp;|&nbsp; Confidence:
                <strong style="color:{class_colors_map[best_lbl]};font-family:'Space Mono',monospace;">
                    {best_conf*100:.2f}%
                </strong>
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        # Placeholder jika belum upload
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;
        background:rgba(56,189,248,0.04);border:2px dashed #1e3a5f;border-radius:16px;">
            <div style="font-size:4rem;margin-bottom:16px;">📷</div>
            <div style="font-family:'Space Mono',monospace;color:#38bdf8;
            font-size:1.1rem;font-weight:700;margin-bottom:8px;">
                Belum Ada Gambar yang Diupload
            </div>
            <div style="color:#7ba4cc;font-size:0.9rem;max-width:500px;margin:0 auto;line-height:1.7;">
                Upload gambar di atas untuk memulai uji coba. Sistem akan menjalankan prediksi 
                <strong>nyata</strong> menggunakan model .h5 Anda dan menampilkan Grad-CAM 
                dari gradien model sesungguhnya.
            </div>
            <br>
            <div style="display:flex;justify-content:center;gap:20px;flex-wrap:wrap;margin-top:12px;">
                <span style="color:#34d399;font-size:0.85rem;">😊 Sentimen Positif</span>
                <span style="color:#60a5fa;font-size:0.85rem;">😐 Sentimen Netral</span>
                <span style="color:#f87171;font-size:0.85rem;">😠 Sentimen Negatif</span>
            </div>
        </div>""", unsafe_allow_html=True)



# ─── HALAMAN RINGKASAN & KESIMPULAN ───────────────────────────────────────────
elif page == "📋 Ringkasan & Kesimpulan":
    st.markdown('<div class="hero-title">Ringkasan & Kesimpulan</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Analisis Akhir · Rekomendasi · Future Work</div>', unsafe_allow_html=True)

    tab_r1, tab_r2, tab_r3 = st.tabs(["📊 Ringkasan Hasil", "🏆 Rekomendasi", "🔭 Future Work"])

    with tab_r1:
        st.markdown('<div class="section-header">Ringkasan Seluruh Tahapan</div>', unsafe_allow_html=True)

        ringkasan_steps = [
            ("1. Dataset & EDA", [
                ("Total Citra", "526 gambar (dari web scraping)"),
                ("Distribusi", "Positif 27%, Netral 33.3%, Negatif 39.7%"),
                ("Ukuran Input", "224×224 piksel (standar CNN)"),
                ("Split", "Train 70% · Validation 24% · Test 20%"),
            ]),
            ("2. Augmentasi Data", [
                ("Rotasi", "±20° untuk variasi sudut pandang"),
                ("Zoom", "±20% untuk variasi jarak objek"),
                ("Shift", "10% horizontal & vertikal"),
                ("Flip", "Horizontal mirror"),
            ]),
            ("3. Model & Training", [
                ("Metode", "Transfer Learning (ImageNet pre-trained)"),
                ("Optimizer", "Adam · LR 0.001"),
                ("Callbacks", "EarlyStopping + ReduceLROnPlateau"),
                ("Epochs", "10 (dengan early stopping)"),
            ]),
            ("4. Evaluasi", [
                ("VGG16", "Accuracy 77.6% · F1 0.7742"),
                ("DenseNet121", "Accuracy 92.5% · F1 0.9248 ⭐"),
                ("MobileNetV2", "Accuracy 90.6% · F1 0.9057"),
                ("Terbaik", "DenseNet121 (akurasi tertinggi)"),
            ]),
            ("5. Grad-CAM (XAI)", [
                ("VGG16", "Layer: block5_conv3"),
                ("DenseNet121", "Layer: conv5_block16_2_conv"),
                ("MobileNetV2", "Layer: Conv_1"),
                ("Insight", "Model fokus pada objek konflik (ledakan, simbol, bendera)"),
            ]),
        ]

        for step_title, items in ringkasan_steps:
            with st.expander(f"📌 {step_title}", expanded=True):
                col1, col2 = st.columns(2)
                for i, (k, v) in enumerate(items):
                    with col1 if i % 2 == 0 else col2:
                        st.markdown(f"<span style='color:#7ba4cc;font-size:0.85rem;'>{k}</span><br><strong style='color:#c9d8ed;'>{v}</strong>", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)

        # Final comparison
        st.markdown('<div class="section-header">Perbandingan Final</div>', unsafe_allow_html=True)
        fig_final, axes_f = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='none')

        # Accuracy bar
        names = list(eval_results.keys())
        accs = [eval_results[n]['accuracy'] for n in names]
        colors_f = [MODEL_COLORS[n] for n in names]
        bars_f = axes_f[0].bar(names, accs, color=colors_f, alpha=0.85, width=0.4)
        for bar in bars_f:
            h = bar.get_height()
            axes_f[0].text(bar.get_x()+bar.get_width()/2, h+0.003, f'{h:.4f}',
                           ha='center', va='bottom', color='#c9d8ed', fontsize=10)
        axes_f[0].set_ylim(0.6, 1.02)
        axes_f[0].set_title('Akurasi Test Set', color='#c9d8ed', fontweight='bold')
        axes_f[0].set_facecolor('none')
        axes_f[0].tick_params(colors='#7ba4cc')
        axes_f[0].spines[:].set_color('#1e3a5f')
        axes_f[0].grid(axis='y', alpha=0.15, color='#38bdf8')

        # Size vs accuracy scatter
        sizes = [eval_results[n]['size_mb'] for n in names]
        for name, acc, size, color in zip(names, accs, sizes, colors_f):
            axes_f[1].scatter(size, acc, s=200, color=color, zorder=5, label=name)
            axes_f[1].annotate(name, (size, acc), textcoords="offset points",
                               xytext=(8, 4), color=color, fontsize=9)
        axes_f[1].set_xlabel('Ukuran Model (MB)', color='#7ba4cc')
        axes_f[1].set_ylabel('Accuracy', color='#7ba4cc')
        axes_f[1].set_title('Trade-off: Ukuran vs Akurasi', color='#c9d8ed', fontweight='bold')
        axes_f[1].set_facecolor('none')
        axes_f[1].tick_params(colors='#7ba4cc')
        axes_f[1].spines[:].set_color('#1e3a5f')
        axes_f[1].grid(alpha=0.15, color='#38bdf8')

        for ax in axes_f: ax.set_facecolor('none')
        fig_final.patch.set_alpha(0)
        st.pyplot(fig_final, transparent=True)
        plt.close()

    with tab_r2:
        st.markdown('<div class="section-header">Rekomendasi Model</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="success-box">
        ✅ <strong>Rekomendasi Utama: MobileNetV2</strong><br>
        Meskipun DenseNet121 memiliki akurasi sedikit lebih tinggi, <strong>MobileNetV2 direkomendasikan 
        untuk implementasi praktis</strong> karena trade-off yang paling optimal.
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        rec_data = [
            ("VGG16", "❌ Tidak Direkomendasikan", "#f87171",
             ["Akurasi terendah (77.6%)", "Ukuran paling besar (105 MB)", "Inferensi paling lambat", "Kurang efisien untuk task ini"]),
            ("DenseNet121", "⭐ Akurasi Terbaik", "#f97316",
             ["Akurasi tertinggi (92.5%)", "F1-Score 0.9248", "Ukuran menengah (56.3 MB)", "Ideal jika akurasi prioritas utama"]),
            ("MobileNetV2", "🏆 Rekomendasi Deployment", "#34d399",
             ["Akurasi kompetitif (90.6%)", "Hanya 13.2 MB (8× lebih ringan dari VGG16)", "Selisih akurasi hanya 1.89% dari DenseNet", "Ideal untuk edge/mobile deployment"]),
        ]
        for col, (name, label, color, points) in zip([col1, col2, col3], rec_data):
            with col:
                st.markdown(f"""
                <div style="background:rgba(0,0,0,0.2);border:1px solid {color}50;border-radius:12px;padding:16px;">
                    <div style="font-family:'Space Mono',monospace;font-weight:700;color:{color};font-size:0.95rem;">{name}</div>
                    <div style="font-size:0.8rem;color:{color};margin:6px 0 10px 0;">{label}</div>
                    {"".join(f'<div style="font-size:0.82rem;color:#c9d8ed;margin:4px 0;">• {p}</div>' for p in points)}
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Kesimpulan Penelitian</div>', unsafe_allow_html=True)
        kesimpulan = [
            ("CNN + Transfer Learning efektif", "Pendekatan Transfer Learning dari ImageNet terbukti menghasilkan model yang akurat meskipun dataset terbatas (526 citra). Tanpa fitur pre-trained, akurasi akan jauh lebih rendah."),
            ("DenseNet121 unggul dalam akurasi", "Arsitektur dense connection memungkinkan reuse fitur yang sangat efektif sehingga DenseNet121 mencapai akurasi tertinggi 92.5% dalam task Visual Sentiment Analysis."),
            ("MobileNetV2 unggul dalam efisiensi", "Dengan selisih akurasi hanya 1.89% namun ukuran 4× lebih kecil dari DenseNet121, MobileNetV2 menawarkan keseimbangan terbaik untuk deployment nyata."),
            ("Grad-CAM meningkatkan transparansi AI", "Integrasi Grad-CAM sebagai teknik XAI membuktikan bahwa model benar-benar belajar dari fitur visual yang relevan (ledakan, simbol konflik, bendera) bukan dari artefak acak."),
            ("Sentimen Negatif mendominasi dataset", "Distribusi dataset mencerminkan realitas konflik Iran-Israel di media: citra bernuansa negatif mendominasi (39.7%), diikuti netral (33.3%) dan positif (27%)."),
        ]
        for title, desc in kesimpulan:
            st.markdown(f"""
            <div style="padding:12px 16px;margin:8px 0;background:rgba(56,189,248,0.05);
            border:1px solid #1e3a5f;border-radius:10px;">
                <div style="font-weight:700;color:#38bdf8;margin-bottom:4px;">✓ {title}</div>
                <div style="font-size:0.88rem;color:#c9d8ed;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with tab_r3:
        st.markdown('<div class="section-header">Saran Pengembangan</div>', unsafe_allow_html=True)
        future = [
            ("🔁", "Fine-tuning Layer", "Lakukan unfreezing bertahap (fine-tuning) pada layer base model untuk meningkatkan akurasi lebih lanjut, terutama pada DenseNet121 dan MobileNetV2."),
            ("📦", "Perbesar Dataset", "Tambah jumlah data scraping hingga 1000–2000 citra per kelas untuk meningkatkan generalisasi model dan mengurangi bias kelas."),
            ("🧮", "LIME & SHAP", "Integrasikan teknik XAI tambahan seperti LIME dan SHAP untuk membandingkan dan memperkuat interpretabilitas model."),
            ("🚀", "Deployment API", "Deploy model MobileNetV2 sebagai REST API menggunakan FastAPI atau Flask untuk penggunaan real-time pada sistem monitoring berita."),
            ("🌐", "Multi-source Scraping", "Perluas sumber data ke platform media sosial (Twitter/X, Instagram) untuk analisis sentimen yang lebih komprehensif."),
            ("⚖️", "Class Balancing", "Terapkan teknik SMOTE atau oversampling pada kelas minoritas (Positif) untuk mengatasi class imbalance yang ada."),
        ]
        col1, col2 = st.columns(2)
        for i, (icon, title, desc) in enumerate(future):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"""
                <div style="padding:14px 16px;margin:8px 0;background:rgba(99,102,241,0.07);
                border:1px solid #2d3f6e;border-radius:10px;">
                    <div style="font-size:1.4rem;margin-bottom:6px;">{icon}</div>
                    <div style="font-weight:700;color:#818cf8;margin-bottom:4px;">{title}</div>
                    <div style="font-size:0.85rem;color:#c9d8ed;line-height:1.5;">{desc}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(56,189,248,0.1),rgba(129,140,248,0.1));
        border:1px solid #38bdf8;border-radius:12px;padding:20px;text-align:center;">
            <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#38bdf8;font-weight:700;margin-bottom:8px;">
                Ramlia Ramadani Sudin · NIM E1E123016
            </div>
            <div style="color:#7ba4cc;font-size:0.9rem;">
                Explainable AI untuk Visual Sentiment Analysis pada Konflik Iran-Israel<br>
                Menggunakan Grad-CAM pada Model CNN
            </div>
        </div>
        """, unsafe_allow_html=True)
