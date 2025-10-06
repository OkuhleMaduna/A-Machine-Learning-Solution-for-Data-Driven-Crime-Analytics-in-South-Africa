import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from prophet import Prophet
import io

# --------------------------------------------
# DASHBOARD CONFIG
# --------------------------------------------
st.set_page_config(page_title="Crime Analytics in South Africa", layout="wide")

TOPIC = "A Machine Learning Solution for Data-Driven Crime Analytics in South Africa"

# --------------------------------------------
# FILE UPLOAD (shared for all tabs)
# --------------------------------------------
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Dataset uploaded successfully!")
else:
    st.sidebar.warning("⚠️ Please upload your dataset first.")

# --------------------------------------------
# TABS
# --------------------------------------------
tabs = st.tabs(["🏠 Home", "📊 EDA & Filtering", "🤖 Classification", "🔮 Forecasting", "⚙️ Settings"])

# --------------------------------------------
# HOME TAB
# --------------------------------------------
with tabs[0]:
    st.title(TOPIC)
    st.header("📘 Project Presentation")

    st.markdown("""
    This Streamlit dashboard visualizes and models crime trends in South Africa.
    It provides EDA tools, classification of crime hotspots, and time-series forecasting.
    """)

    st.subheader("🧩 Key Tasks")
    st.markdown("""
    1. Data Acquisition & Justification  
    2. Data Understanding & Preprocessing  
    3. Exploratory Data Analysis (EDA)  
    4. Crime Hotspot Classification  
    5. Crime Trend Forecasting  
    6. Model Evaluation & Insights
    """)

    st.subheader("🎯 Purpose")
    st.markdown("""
    The purpose of this dashboard is to convert raw crime data into insights that help
    policymakers and law enforcement agencies identify and predict crime patterns.
    """)

    st.markdown("---")
    st.markdown("**Developed by OKUHLE | 22410169 Maduna O.A.U**")

# --------------------------------------------
# EDA TAB
# --------------------------------------------
with tabs[1]:
    st.title("📊 Exploratory Data Analysis")

    if df is not None:
        st.dataframe(df.head())

        # Auto-detect columns
        year_cols = [c for c in df.columns if "-" in c]
        provs = ["All"] + sorted(df["Province"].unique().tolist()) if "Province" in df.columns else ["All"]
        cats = ["All"] + sorted(df["Category"].unique().tolist()) if "Category" in df.columns else ["All"]

        col1, col2, col3 = st.columns(3)
        prov = col1.selectbox("Province", provs)
        cat = col2.selectbox("Crime Category", cats)
        yr = col3.selectbox("Year", year_cols)

        filt = df.copy()
        if prov != "All":
            filt = filt[filt["Province"] == prov]
        if cat != "All":
            filt = filt[filt["Category"] == cat]

        st.write(f"Filtered rows: {len(filt)}")

        if yr in filt.columns:
            plotdata = filt.groupby("Station")[yr].sum().sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(8, 4))
            plotdata.plot(kind="bar", ax=ax)
            ax.set_ylabel("Incident Count")
            ax.set_title(f"{cat if cat!='All' else 'All Crimes'} in {prov if prov!='All' else 'All Provinces'} ({yr})")
            st.pyplot(fig)
    else:
        st.info("Upload a dataset in the sidebar first.")

# --------------------------------------------
# CLASSIFICATION TAB
# --------------------------------------------
with tabs[2]:
    st.title("🤖 Crime Hotspot Classification")

    if df is not None:
        # Prepare features
        year_cols = [c for c in df.columns if "-" in c]
        df["Total"] = df[year_cols].sum(axis=1)
        df["is_hotspot"] = (df["Total"] > df["Total"].quantile(0.75)).astype(int)

        X = pd.get_dummies(df[["Province"]], drop_first=True)
        y = df["is_hotspot"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.metric("✅ Model Accuracy", f"{acc * 100:.2f}%")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=["Not Hotspot", "Hotspot"]).plot(ax=ax)
        st.pyplot(fig)

        st.markdown("The model successfully classifies areas as hotspots or not, aiding resource allocation.")
    else:
        st.info("Upload a dataset in the sidebar first.")

# --------------------------------------------
# FORECASTING TAB
# --------------------------------------------
with tabs[3]:
    st.title("🔮 Crime Forecasting (Prophet Model)")

    if df is not None:
        year_cols = [c for c in df.columns if "-" in c]
        yearly = df[year_cols].sum().reset_index()
        yearly.columns = ["ds", "y"]
        yearly["ds"] = yearly["ds"].str.split("-").str[0]
        yearly["ds"] = pd.to_datetime(yearly["ds"])

        model = Prophet()
        model.fit(yearly)
        future = model.make_future_dataframe(periods=24, freq="M")
        forecast = model.predict(future)

        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        st.markdown("This forecast predicts future crime trends with 24-month confidence intervals.")
    else:
        st.info("Upload a dataset in the sidebar first.")

# --------------------------------------------
# SETTINGS TAB
# --------------------------------------------
with tabs[4]:
    st.title("⚙️ Settings")
    theme_choice = st.radio("Choose Theme", ["Light", "Dark"])

    if st.button("Apply Theme"):
        if theme_choice == "Dark":
            st.markdown(
                "<style>body{background-color:#0e1117;color:white;}</style>",
                unsafe_allow_html=True,
            )
            st.success("Dark mode applied!")
        else:
            st.markdown(
                "<style>body{background-color:white;color:black;}</style>",
                unsafe_allow_html=True,
            )
            st.success("Light mode applied!")
