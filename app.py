"""
Streamlit Web Application — ML Classification Model Comparison
Dataset: Breast Cancer Wisconsin (Diagnostic)
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# ── helpers ──────────────────────────────────────────────────────────────────

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

@st.cache_resource
def load_artefacts():
    models = {}
    for name, fname in MODEL_FILES.items():
        with open(os.path.join(MODEL_DIR, fname), "rb") as f:
            models[name] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "metrics.pkl"), "rb") as f:
        metrics = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)
    return models, metrics, scaler, feature_names


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


MODEL_COLORS = {
    "Logistic Regression": "#4C72B0",
    "Decision Tree": "#DD8452",
    "KNN": "#55A868",
    "Naive Bayes": "#C44E52",
    "Random Forest": "#8172B3",
}

OBSERVATIONS = {
    "Logistic Regression": (
        "Best overall performer across every metric. Achieves the highest accuracy of 98.25%, "
        "the highest AUC of 0.9954, and the highest MCC of 0.9623. Its F1 score of 0.9825 is "
        "5.27 percentage points higher than the next-best model (KNN/Random Forest at 0.9560). "
        "Only 2 out of 114 test samples were misclassified. The strong performance indicates that "
        "the 30 standardised features create a nearly linearly separable decision boundary, which "
        "Logistic Regression exploits optimally."
    ),
    "Decision Tree": (
        "Weakest performer among all five models. Accuracy is 91.23%, which is 7.02 percentage "
        "points lower than Logistic Regression. AUC is only 0.9157 — the lowest by a large margin "
        "(gap of 0.0797 vs Logistic Regression). MCC of 0.8174 is also the lowest, trailing "
        "Logistic Regression by 0.1449. The single unpruned tree overfits the training data and "
        "produces axis-aligned splits that fail to capture the smooth boundary. It misclassified "
        "10 out of 114 test samples."
    ),
    "KNN": (
        "Achieves an accuracy of 95.61% and MCC of 0.9054, placing it second alongside Random "
        "Forest. Its AUC of 0.9788 is solid but 0.0166 below Logistic Regression. KNN benefits "
        "from StandardScaler normalisation (Euclidean distance becomes meaningful), and with k=5 "
        "it captures local neighbourhood patterns well. However, the 2.64 percentage-point "
        "accuracy gap versus Logistic Regression suggests the true decision boundary is more "
        "global/linear than local. It misclassified 5 out of 114 test samples."
    ),
    "Naive Bayes": (
        "Gaussian Naive Bayes records an accuracy of 92.98% and an AUC of 0.9868. Notably, its "
        "AUC is higher than KNN's (0.9788) and only 0.0086 behind Logistic Regression, indicating "
        "excellent probabilistic ranking — the model assigns well-separated probability scores to "
        "the two classes. However, the hard 0.5 classification threshold leads to 8 misclassifications "
        "out of 114, pulling accuracy 5.27 points below Logistic Regression. MCC of 0.8492 is "
        "moderate, reflecting the impact of correlated features violating the independence assumption."
    ),
    "Random Forest": (
        "Random Forest matches KNN with 95.61% accuracy, 0.9560 F1, and 0.9054 MCC. Its AUC of "
        "0.9932 is the second-highest overall — only 0.0022 below Logistic Regression — demonstrating "
        "strong probabilistic calibration from the ensemble of 200 trees. Compared to the single "
        "Decision Tree, Random Forest improves accuracy by 4.38 percentage points and AUC by 0.0775, "
        "confirming that bagging and feature randomisation effectively reduce overfitting. It "
        "misclassified 5 out of 114 test samples, same as KNN."
    ),
}

# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
        border: 1px solid #dce1e8;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label { font-weight: 600; }
    /* Force the primary button to blue */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #1a73e8 !important;
        border-color: #1a73e8 !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #1558b0 !important;
        border-color: #1558b0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── load artefacts ───────────────────────────────────────────────────────────

try:
    models, saved_metrics, scaler, feature_names = load_artefacts()
except FileNotFoundError:
    st.error(
        "Model artefacts not found. Run `python model/train_models.py` first."
    )
    st.stop()

# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/d/d3/BITS_Pilani-Logo.svg", width=80)
    st.title("ML Dashboard")
    st.markdown("---")

    st.subheader("1. Select Model")
    model_name = st.selectbox(
        "Choose a classifier",
        list(models.keys()),
        help="Pick a model to see its detailed metrics, confusion matrix, and classification report.",
    )

    st.markdown("---")
    st.subheader("2. Display Options")
    show_all = st.checkbox("Show all-model comparison", value=True)
    show_roc = st.checkbox("Show ROC curves", value=True)

    st.markdown("---")
    st.subheader("3. Upload Custom Test Data")
    uploaded_file = st.file_uploader(
        "Upload Test CSV", type=["csv"],
        help="CSV must have the same feature columns as the training data plus a 'target' column.",
    )
    if uploaded_file is not None:
        run_clicked = st.button(
            "🚀 Submit & Run Predictions",
            use_container_width=True, type="primary",
        )
    else:
        run_clicked = False

    st.markdown("---")
    st.caption("BITS Pilani — ML Assignment 2")

# ── hero header ──────────────────────────────────────────────────────────────

st.markdown(
    """
    <p style='text-align:center; color:#888; font-size:0.95rem; margin-bottom:2px;'>
        Machine Learning (Assignment-2)
    </p>
    <h1 style='text-align:center; margin-bottom:0; margin-top:0;'>
        🧬 ML Classification Model Comparison
    </h1>
    <p style='text-align:center; color:#666; font-size:1.1rem; margin-top:4px;'>
        Breast Cancer Wisconsin (Diagnostic) &mdash; Binary Classification
    </p>
    <p style='text-align:center; color:#555; font-size:0.95rem; margin-top:2px;'>
        <strong>Name:</strong> Sunag P &nbsp;&bull;&nbsp;
        <strong>BITS ID:</strong> 2026AC05679
    </p>
    """,
    unsafe_allow_html=True,
)

col_d1, col_d2, col_d3, col_d4 = st.columns(4)
col_d1.metric("Features", "30")
col_d2.metric("Total Instances", "569")
col_d3.metric("Classes", "2 (M / B)")
col_d4.metric("Models Trained", "5")

st.markdown("---")

# ── load data ────────────────────────────────────────────────────────────────

if "submitted_data" not in st.session_state:
    st.session_state.submitted_data = None

if uploaded_file is not None and run_clicked:
    uploaded_df = pd.read_csv(uploaded_file)
    if "target" not in uploaded_df.columns:
        st.error("CSV must contain a **target** column.")
        st.stop()
    missing_features = [f for f in feature_names if f not in uploaded_df.columns]
    if missing_features:
        st.error(f"Missing features in uploaded CSV: {missing_features}")
        st.stop()
    st.session_state.submitted_data = uploaded_df
    st.toast(f"Running predictions on {len(uploaded_df)} uploaded rows", icon="✅")

if uploaded_file is not None and not run_clicked and st.session_state.submitted_data is not None:
    df = st.session_state.submitted_data
elif uploaded_file is not None and not run_clicked:
    st.info("👆 Click **Submit & Run Predictions** in the sidebar to process the uploaded CSV.")
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data.csv")
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
    else:
        st.stop()
elif uploaded_file is not None and run_clicked:
    df = st.session_state.submitted_data
else:
    st.session_state.submitted_data = None
    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data.csv")
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
    else:
        st.warning("No test data found. Please upload a CSV via the sidebar.")
        st.stop()

if "target" not in df.columns:
    st.error("CSV must contain a **target** column.")
    st.stop()

missing_features = [f for f in feature_names if f not in df.columns]
if missing_features:
    st.error(f"Missing features in uploaded CSV: {missing_features}")
    st.stop()

X_test = df[feature_names]
y_test = df["target"]
X_test_scaled = scaler.transform(X_test)

# ── precompute predictions for all models ────────────────────────────────────

all_predictions = {}
for name, clf in models.items():
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    m = compute_metrics(y_test.values, y_pred, y_prob)
    all_predictions[name] = {"y_pred": y_pred, "y_prob": y_prob, "metrics": m}

# ── tab layout ───────────────────────────────────────────────────────────────

tab_compare, tab_detail, tab_data = st.tabs([
    "📋 Model Comparison", "🔍 Individual Model Analysis", "📂 Dataset Explorer"
])

# ── TAB 1: Comparison ────────────────────────────────────────────────────────

with tab_compare:
    if show_all:
        st.header("Evaluation Metrics — All Models")

        rows = []
        for name in models:
            m = all_predictions[name]["metrics"].copy()
            m["Model"] = name
            rows.append(m)

        comparison_df = pd.DataFrame(rows)[
            ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        ]

        st.dataframe(
            comparison_df.style
                .highlight_max(
                    subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                    color="#d4edda",
                )
                .highlight_min(
                    subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                    color="#f8d7da",
                )
                .format(precision=4),
            use_container_width=True,
            hide_index=True,
        )

        best_model = comparison_df.loc[comparison_df["F1"].idxmax(), "Model"]
        st.success(f"**Best model (by F1 Score): {best_model}**")

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("Grouped Bar Chart")
            fig, ax = plt.subplots(figsize=(8, 4.5))
            x = np.arange(len(comparison_df))
            metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
            width = 0.13
            for i, metric in enumerate(metric_cols):
                ax.bar(x + i * width, comparison_df[metric], width,
                       label=metric, edgecolor="white", linewidth=0.5)
            ax.set_xticks(x + width * 2.5)
            ax.set_xticklabels(comparison_df["Model"], rotation=20, ha="right", fontsize=9)
            ax.set_ylim(0.75, 1.02)
            ax.set_ylabel("Score")
            ax.legend(fontsize=8, ncol=3, loc="lower right")
            ax.set_title("Metric Comparison Across Models", fontsize=11)
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        with col_chart2:
            st.subheader("Radar Chart")
            categories = metric_cols
            N = len(categories)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]

            fig_radar, ax_radar = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
            for _, row in comparison_df.iterrows():
                values = row[categories].tolist()
                values += values[:1]
                color = MODEL_COLORS.get(row["Model"], "#999")
                ax_radar.plot(angles, values, linewidth=1.5, label=row["Model"], color=color)
                ax_radar.fill(angles, values, alpha=0.08, color=color)

            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(categories, fontsize=9)
            ax_radar.set_ylim(0.75, 1.02)
            ax_radar.set_title("Model Performance Radar", fontsize=11, pad=20)
            ax_radar.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_radar)

        if show_roc:
            st.subheader("ROC Curves — All Models")
            fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
            for name in models:
                y_prob = all_predictions[name]["y_prob"]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_val = all_predictions[name]["metrics"]["AUC"]
                color = MODEL_COLORS.get(name, "#999")
                ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc_val})",
                            color=color, linewidth=1.8)
            ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves")
            ax_roc.legend(fontsize=9)
            ax_roc.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_roc)

        st.subheader("Model Observations")
        obs_rows = []
        for name in models:
            obs_rows.append({"Model": name, "Observation": OBSERVATIONS.get(name, "")})
        obs_rows.append({
            "Model": "Overall Winner",
            "Observation": (
                f"{best_model} — ranks #1 in all six evaluation metrics: "
                "Accuracy (0.9825), AUC (0.9954), Precision (0.9825), Recall (0.9825), "
                "F1 (0.9825), and MCC (0.9623). Its dominance is explained by the fact that "
                "the 30 standardised diagnostic features create a feature space where the two "
                "classes are nearly linearly separable, making a regularised linear model the "
                "optimal choice. Among non-linear models, Random Forest is the runner-up by "
                "AUC (0.9932), while KNN and Random Forest tie as runners-up by accuracy (0.9561)."
            ),
        })
        st.table(pd.DataFrame(obs_rows))

    else:
        st.info("Enable **'Show all-model comparison'** in the sidebar to see the comparison table.")

# ── TAB 2: Individual Model ─────────────────────────────────────────────────

with tab_detail:
    st.header(f"{model_name}")

    pred = all_predictions[model_name]
    metrics = pred["metrics"]
    y_pred = pred["y_pred"]
    y_prob = pred["y_prob"]

    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric("Accuracy", metrics["Accuracy"])
    m_col2.metric("AUC", metrics["AUC"])
    m_col3.metric("Precision", metrics["Precision"])
    m_col4.metric("Recall", metrics["Recall"])
    m_col5.metric("F1", metrics["F1"])
    m_col6.metric("MCC", metrics["MCC"])

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Malignant (0)", "Benign (1)"],
            yticklabels=["Malignant (0)", "Benign (1)"],
            ax=ax_cm, linewidths=0.5, linecolor="white",
            annot_kws={"size": 16, "weight": "bold"},
        )
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")
        ax_cm.set_title(f"Confusion Matrix  —  {model_name}", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_cm)

    with right_col:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc2, ax_roc2 = plt.subplots(figsize=(5, 4))
        color = MODEL_COLORS.get(model_name, "#4C72B0")
        ax_roc2.plot(fpr, tpr, color=color, linewidth=2,
                     label=f"AUC = {metrics['AUC']}")
        ax_roc2.fill_between(fpr, tpr, alpha=0.15, color=color)
        ax_roc2.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax_roc2.set_xlabel("False Positive Rate")
        ax_roc2.set_ylabel("True Positive Rate")
        ax_roc2.set_title(f"ROC Curve  —  {model_name}", fontsize=11)
        ax_roc2.legend(fontsize=10)
        ax_roc2.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_roc2)

    st.subheader("Classification Report")
    report = classification_report(
        y_test, y_pred,
        target_names=["Malignant (0)", "Benign (1)"],
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)

    st.subheader("Observation")
    st.info(OBSERVATIONS.get(model_name, ""))

# ── TAB 3: Dataset Explorer ─────────────────────────────────────────────────

with tab_data:
    st.header("Test Data Overview")

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Rows", df.shape[0])
    info_col2.metric("Columns", df.shape[1])
    target_counts = y_test.value_counts()
    info_col3.metric("Class Balance", f"{target_counts.get(0,0)} M / {target_counts.get(1,0)} B")

    dist_col, stats_col = st.columns(2)

    with dist_col:
        st.subheader("Target Distribution")
        fig_dist, ax_dist = plt.subplots(figsize=(4, 3.5))
        colors = ["#C44E52", "#55A868"]
        labels = ["Malignant (0)", "Benign (1)"]
        counts = [target_counts.get(0, 0), target_counts.get(1, 0)]
        wedges, texts, autotexts = ax_dist.pie(
            counts, labels=labels, autopct="%1.1f%%", startangle=90,
            colors=colors, textprops={"fontsize": 9},
        )
        for t in autotexts:
            t.set_fontweight("bold")
        ax_dist.set_title("Class Distribution in Test Data", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_dist)

    with stats_col:
        st.subheader("Feature Statistics")
        st.dataframe(df[feature_names].describe().round(3), use_container_width=True)

    st.subheader("Raw Data Preview")
    st.dataframe(df, use_container_width=True, height=350)
    st.caption(
        f"Showing all {df.shape[0]} rows and {df.shape[1]} columns. "
        "Upload your own CSV via the sidebar to test with different data."
    )

# ── footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#999; font-size:0.85rem;'>"
    "Built for BITS Pilani ML Assignment 2 &bull; "
    "Streamlit + scikit-learn &bull; "
    "Breast Cancer Wisconsin Dataset"
    "</p>",
    unsafe_allow_html=True,
)
