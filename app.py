
"""
TP2 Deep Learning
====================================================
Sujet B : Réseaux convolutifs (CNN) et vision : architectures modernes

Par: Tchuenteu Guetchueng David – 20U2891 – Info4
Et: Nafiesatou Hamadicko 20U2891
Sous la supervision de Stéphane C. K. TÉKOUABOU (PhD & Ing.) - UY1 2025-2026
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, roc_curve,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import time

# ============================================================
# CONFIGURATION PAGE & THÈME
# ============================================================
st.set_page_config(page_title="TP Deep Learning", layout="wide")

# CSS amélioré
st.markdown("""
<style>
    .stApp {
        background-color: #F8F9FA;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1F4E79 !important;
    }
    .stButton > button {
        background-color: #F39C12;
        color: white;
        border: none;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #E67E22;
        color: white;
    }
    .stMetric .label {
        color: #1F4E79;
        font-weight: bold;
    }
    .stMetric .value {
        color: #E67E22;
        font-size: 1.8rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #1F4E79;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        border-bottom-color: #F39C12;
    }
    hr {
        border-color: #F39C12;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stExpander"] details summary p {
        font-weight: bold;
        color: #1F4E79;
    }
</style>
""", unsafe_allow_html=True)

st.title("TP Deep Learning – Réseaux convolutifs (CNN) et vision : architectures modernes")
st.markdown("### Tchuenteu Guetchueng David – matricule 20U2891 – Info4")
st.markdown("### Nafiesatou Hamadicko - matricule 21L545 – Info4")
st.markdown("#### Sous la supervision de Stéphane C. K. TÉKOUABOU (PhD & Ing.) - UY1 2025-20264")
st.markdown("---")