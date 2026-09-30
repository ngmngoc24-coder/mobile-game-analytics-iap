
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Page setup
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "game_data_clean.csv"

st.set_page_config(
    page_title="Mobile Game Analytics",
    page_icon="🎮",
    layout="wide",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.65rem;
    }
    .small-note {
        color: #666;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Data loading
# ============================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. "
            "Run 02_data_cleaning.ipynb first so that "
            "data/game_data_clean.csv exists."
        )

    df = pd.read_csv(DATA_PATH)
    df["LastPurchaseDate"] = pd.to_datetime(
        df["LastPurchaseDate"], errors="coerce"
    )
    return df


df = load_data()


# ============================================================
# Reusable analytical transformations
# ============================================================

@st.cache_data
def create_behavior_data(df: pd.DataFrame) -> pd.DataFrame:
    behavior_df = df[df["InAppPurchaseAmount"].notna()].copy()

    behavior_df["LogRevenue"] = np.log1p(
        behavior_df["InAppPurchaseAmount"]
    )

    behavior_df["SessionCount_Z"] = (
        behavior_df["SessionCount"] - behavior_df["SessionCount"].mean()
    ) / behavior_df["SessionCount"].std()

    behavior_df["SessionLength_Z"] = (
        behavior_df["AverageSessionLength"]
        - behavior_df["AverageSessionLength"].mean()
    ) / behavior_df["AverageSessionLength"].std()

    behavior_df["EngagementScore"] = (
        behavior_df["SessionCount_Z"]
        + behavior_df["SessionLength_Z"]
    ) / 2

    engagement_threshold = behavior_df["EngagementScore"].median()
    monetization_threshold = behavior_df["LogRevenue"].median()

    behavior_df["BehavioralSegment"] = np.select(
        [
            (behavior_df["EngagementScore"] >= engagement_threshold)
            & (behavior_df["LogRevenue"] >= monetization_threshold),

            (behavior_df["EngagementScore"] >= engagement_threshold)
            & (behavior_df["LogRevenue"] < monetization_threshold),

            (behavior_df["EngagementScore"] < engagement_threshold)
            & (behavior_df["LogRevenue"] >= monetization_threshold),

            (behavior_df["EngagementScore"] < engagement_threshold)
            & (behavior_df["LogRevenue"] < monetization_threshold),
        ],
        [
            "High Engagement - High Monetization",
            "High Engagement - Low Monetization",
            "Low Engagement - High Monetization",
            "Low Engagement - Low Monetization",
        ],
        default="Unknown",
    )

    behavior_df["FirstPurchaseTiming"] = pd.cut(
        behavior_df["FirstPurchaseDaysAfterInstall"],
        bins=[-1, 3, 7, 14, 30],
        labels=["0–3 days", "4–7 days", "8–14 days", "15–30 days"],
    )

    reference_date = behavior_df["LastPurchaseDate"].max()
    behavior_df["PurchaseRecencyDays"] = (
        reference_date - behavior_df["LastPurchaseDate"]
    ).dt.days

    behavior_df["PurchaseRecencyGroup"] = pd.cut(
        behavior_df["PurchaseRecencyDays"],
        bins=[-1, 7, 30, 90, np.inf],
        labels=["0–7 days", "8–30 days", "31–90 days", "90+ days"],
    )

    return behavior_df


@st.cache_data
def create_hvp_data(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    modeling_df = df[df["InAppPurchaseAmount"].notna()].copy()

    hvp_percentile = 0.90
    hvp_threshold = modeling_df["InAppPurchaseAmount"].quantile(
        hvp_percentile
    )

    modeling_df["HVP"] = (
        modeling_df["InAppPurchaseAmount"] >= hvp_threshold
    ).astype(int)

    reference_date = modeling_df["LastPurchaseDate"].max()
    modeling_df["PurchaseRecencyDays"] = (
        reference_date - modeling_df["LastPurchaseDate"]
    ).dt.days

    return modeling_df, hvp_threshold


@st.cache_resource
def train_logistic_model(df: pd.DataFrame):
    modeling_df, hvp_threshold = create_hvp_data(df)

    feature_columns = [
        "Age",
        "Gender",
        "Country",
        "Device",
        "GameGenre",
        "SessionCount",
        "AverageSessionLength",
        "FirstPurchaseDaysAfterInstall",
        "PurchaseRecencyDays",
        "PaymentMethod",
    ]

    X = modeling_df[feature_columns].copy()
    y = modeling_df["HVP"].copy()

    numeric_features = [
        "Age",
        "SessionCount",
        "AverageSessionLength",
        "FirstPurchaseDaysAfterInstall",
        "PurchaseRecencyDays",
    ]

    categorical_features = [
        "Gender",
        "Country",
        "Device",
        "GameGenre",
        "PaymentMethod",
    ]

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob),
    }

    # Out-of-fold probabilities for the probability-distribution view.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_prob = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    probability_df = pd.DataFrame(
        {
            "Actual HVP": y.values,
            "Predicted Probability": cv_prob,
        }
    )

    # Coefficients.
    fitted_preprocessor = model.named_steps["preprocessor"]
    fitted_classifier = model.named_steps["classifier"]

    feature_names = fitted_preprocessor.get_feature_names_out()
    coefficients = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": fitted_classifier.coef_[0],
        }
    )
    coefficients["AbsCoefficient"] = coefficients["Coefficient"].abs()
    coefficients = coefficients.sort_values(
        "AbsCoefficient",
        ascending=False,
    )

    return {
        "modeling_df": modeling_df,
        "threshold": hvp_threshold,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "probability_df": probability_df,
        "coefficients": coefficients,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


behavior_df = create_behavior_data(df)
model_results = train_logistic_model(df)


# ============================================================
# Sidebar filters
# ============================================================

with st.sidebar:
    st.header("Filters")

    st.caption(
        "Filters affect the descriptive analysis tabs. "
        "The predictive model is trained on the full modeling population."
    )

    country_options = sorted(
        df["Country"].dropna().unique().tolist()
    )
    genre_options = sorted(
        df["GameGenre"].dropna().unique().tolist()
    )
    device_options = sorted(
        df["Device"].dropna().unique().tolist()
    )
    spending_options = sorted(
        df["SpendingSegment"].dropna().unique().tolist()
    )

    selected_countries = st.multiselect(
        "Country",
        country_options,
        default=country_options,
    )

    selected_genres = st.multiselect(
        "Game Genre",
        genre_options,
        default=genre_options,
    )

    selected_devices = st.multiselect(
        "Device",
        device_options,
        default=device_options,
    )

    selected_spending = st.multiselect(
        "Spending Segment",
        spending_options,
        default=spending_options,
    )


filtered_df = df[
    df["Country"].isin(selected_countries)
    & df["GameGenre"].isin(selected_genres)
    & df["Device"].isin(selected_devices)
    & df["SpendingSegment"].isin(selected_spending)
].copy()

filtered_behavior = behavior_df[
    behavior_df["Country"].isin(selected_countries)
    & behavior_df["GameGenre"].isin(selected_genres)
    & behavior_df["Device"].isin(selected_devices)
    & behavior_df["SpendingSegment"].isin(selected_spending)
].copy()


if filtered_df.empty:
    st.warning("No players match the selected filters.")
    st.stop()


# ============================================================
# Header
# ============================================================

st.title("🎮 Mobile Game Analytics")
st.caption(
    "Mobile Game Monetization & Player Behavior Analytics · "
    "Player-level observational dataset"
)

st.markdown(
    """
    **Business question:** What player behaviors and characteristics are
    associated with higher monetization value, and how can players be
    segmented to support game monetization and engagement decisions?
    """
)


# ============================================================
# KPI row
# ============================================================

observed_revenue = filtered_behavior["InAppPurchaseAmount"].sum()
median_revenue = filtered_behavior["InAppPurchaseAmount"].median()
revenue_per_player = (
    observed_revenue / len(filtered_behavior)
    if len(filtered_behavior)
    else 0
)

whale_share = (
    filtered_behavior.loc[
        filtered_behavior["SpendingSegment"].eq("Whale"),
        "InAppPurchaseAmount",
    ].sum()
    / observed_revenue
    if observed_revenue
    else 0
)

hvp_share = (
    model_results["modeling_df"]["HVP"].mean()
)

kpis = st.columns(5)
kpis[0].metric("Players", f"{len(filtered_df):,}")
kpis[1].metric("Observed Revenue", f"${observed_revenue:,.0f}")
kpis[2].metric("Revenue / Player", f"${revenue_per_player:,.2f}")
kpis[3].metric("Median Revenue", f"${median_revenue:,.2f}")
kpis[4].metric("HVP Target Share", f"{hvp_share:.1%}")


# ============================================================
# Tabs
# ============================================================

(
    overview_tab,
    monetization_tab,
    segmentation_tab,
    behavior_tab,
    modeling_tab,
    methodology_tab,
) = st.tabs(
    [
        "Overview",
        "Monetization",
        "Player Segmentation",
        "Behavior",
        "Predictive Modeling",
        "Methodology",
    ]
)


# ============================================================
# Overview
# ============================================================

with overview_tab:
    st.subheader("Executive Overview")

    st.info(
        "Revenue is highly concentrated: a small share of players "
        "contributes a disproportionately large share of observed revenue."
    )

    col1, col2 = st.columns(2)

    with col1:
        segment_summary = (
            filtered_behavior.groupby("SpendingSegment")
            .agg(
                Players=("UserID", "count"),
                Revenue=("InAppPurchaseAmount", "sum"),
                AvgRevenue=("InAppPurchaseAmount", "mean"),
            )
            .reset_index()
        )

        segment_summary["PlayerShare"] = (
            segment_summary["Players"]
            / segment_summary["Players"].sum()
        )

        segment_summary["RevenueShare"] = (
            segment_summary["Revenue"]
            / segment_summary["Revenue"].sum()
        )

        fig = px.bar(
            segment_summary,
            x="SpendingSegment",
            y="RevenueShare",
            text=segment_summary["RevenueShare"].map(
                lambda x: f"{x:.1%}"
            ),
            title="Observed Revenue Share by Spending Segment",
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        revenue_distribution = filtered_behavior[
            "InAppPurchaseAmount"
        ].copy()

        fig = px.histogram(
            revenue_distribution,
            nbins=50,
            x="InAppPurchaseAmount",
            title="Observed Revenue Distribution",
        )
        fig.update_xaxes(title="Observed In-App Purchase Amount")
        fig.update_yaxes(title="Players")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Revenue Concentration")

    ranked = filtered_behavior.sort_values(
        "InAppPurchaseAmount",
        ascending=False,
    )

    total_revenue = ranked["InAppPurchaseAmount"].sum()

    concentration_rows = []
    for pct in [1, 5, 10, 25, 50]:
        n = max(1, int(len(ranked) * pct / 100))
        share = (
            ranked.head(n)["InAppPurchaseAmount"].sum()
            / total_revenue
            if total_revenue
            else 0
        )
        concentration_rows.append(
            {
                "Top Player Share": f"Top {pct}%",
                "Players": n,
                "Revenue Share": share,
            }
        )

    concentration = pd.DataFrame(concentration_rows)
    concentration["Revenue Share"] = concentration[
        "Revenue Share"
    ].map(lambda x: f"{x:.1%}")

    st.dataframe(
        concentration,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        """
        **Interpretation:** Revenue concentration means aggregate revenue
        should be interpreted alongside player distribution. A small number
        of high-spending players can materially affect the mean.
        """
    )


# ============================================================
# Monetization
# ============================================================

with monetization_tab:
    st.subheader("Monetization Analysis")

    col1, col2 = st.columns(2)

    with col1:
        gender_summary = (
            filtered_behavior.groupby("Gender")
            .agg(
                Players=("UserID", "count"),
                Revenue=("InAppPurchaseAmount", "sum"),
            )
            .reset_index()
        )
        gender_summary["RevenuePerPlayer"] = (
            gender_summary["Revenue"]
            / gender_summary["Players"]
        )

        fig = px.bar(
            gender_summary.sort_values(
                "RevenuePerPlayer",
                ascending=False,
            ),
            x="Gender",
            y="RevenuePerPlayer",
            title="Observed Revenue per Player by Gender",
        )
        fig.update_yaxes(title="Revenue per player (USD)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        device_summary = (
            filtered_behavior.groupby("Device")
            .agg(
                Players=("UserID", "count"),
                Revenue=("InAppPurchaseAmount", "sum"),
            )
            .reset_index()
        )
        device_summary["RevenuePerPlayer"] = (
            device_summary["Revenue"]
            / device_summary["Players"]
        )

        fig = px.bar(
            device_summary,
            x="Device",
            y="RevenuePerPlayer",
            title="Observed Revenue per Player by Device",
        )
        fig.update_yaxes(title="Revenue per player (USD)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Revenue by Country")

    country_summary = (
        filtered_behavior.groupby("Country")
        .agg(
            Players=("UserID", "count"),
            Revenue=("InAppPurchaseAmount", "sum"),
            AvgRevenue=("InAppPurchaseAmount", "mean"),
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )

    country_summary["RevenueShare"] = (
        country_summary["Revenue"]
        / country_summary["Revenue"].sum()
    )

    fig = px.bar(
        country_summary.head(10),
        x="Country",
        y="Revenue",
        title="Top Countries by Observed Revenue",
        hover_data=["Players", "AvgRevenue"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        country_summary.assign(
            Revenue=lambda x: x["Revenue"].map(
                lambda v: f"${v:,.2f}"
            ),
            AvgRevenue=lambda x: x["AvgRevenue"].map(
                lambda v: f"${v:,.2f}"
            ),
            RevenueShare=lambda x: x["RevenueShare"].map(
                lambda v: f"{v:.1%}"
            ),
        ),
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# Player Segmentation
# ============================================================

with segmentation_tab:
    st.subheader("Behavioral Player Segmentation")

    st.caption(
        "Four rule-based profiles are created using median engagement "
        "and log-transformed observed revenue. This is an analytical "
        "segmentation, not a clustering model."
    )

    segment_profile = (
        filtered_behavior.groupby("BehavioralSegment")
        .agg(
            Players=("UserID", "count"),
            Revenue=("InAppPurchaseAmount", "sum"),
            MedianRevenue=("InAppPurchaseAmount", "median"),
            AvgSessions=("SessionCount", "mean"),
            AvgSessionLength=("AverageSessionLength", "mean"),
        )
        .reset_index()
    )

    segment_profile["PlayerShare"] = (
        segment_profile["Players"]
        / segment_profile["Players"].sum()
    )

    segment_profile["RevenueShare"] = (
        segment_profile["Revenue"]
        / segment_profile["Revenue"].sum()
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            filtered_behavior,
            x="EngagementScore",
            y="LogRevenue",
            color="BehavioralSegment",
            hover_data=[
                "UserID",
                "GameGenre",
                "SpendingSegment",
            ],
            title="Engagement × Monetization",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            segment_profile.sort_values("Players"),
            x="Players",
            y="BehavioralSegment",
            orientation="h",
            title="Players by Behavioral Segment",
        )
        st.plotly_chart(fig, use_container_width=True)

    display_profile = segment_profile.copy()
    display_profile["PlayerShare"] = display_profile[
        "PlayerShare"
    ].map(lambda x: f"{x:.1%}")
    display_profile["RevenueShare"] = display_profile[
        "RevenueShare"
    ].map(lambda x: f"{x:.1%}")
    display_profile["Revenue"] = display_profile[
        "Revenue"
    ].map(lambda x: f"${x:,.2f}")
    display_profile["MedianRevenue"] = display_profile[
        "MedianRevenue"
    ].map(lambda x: f"${x:,.2f}")
    display_profile["AvgSessions"] = display_profile[
        "AvgSessions"
    ].map(lambda x: f"{x:.1f}")
    display_profile["AvgSessionLength"] = display_profile[
        "AvgSessionLength"
    ].map(lambda x: f"{x:.1f}")

    st.dataframe(
        display_profile,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Behavioral Segment × Game Genre")

    genre_mix = pd.crosstab(
        filtered_behavior["GameGenre"],
        filtered_behavior["BehavioralSegment"],
        normalize="index",
    ) * 100

    fig = px.imshow(
        genre_mix,
        text_auto=".0f",
        aspect="auto",
        labels={
            "x": "Behavioral Segment",
            "y": "Game Genre",
            "color": "Player Share (%)",
        },
        title="Behavioral Segment Composition Within Each Genre",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Behavioral Analysis
# ============================================================

with behavior_tab:
    st.subheader("Behavioral Patterns")

    col1, col2 = st.columns(2)

    with col1:
        timing_summary = (
            filtered_behavior.groupby(
                "FirstPurchaseTiming",
                observed=True,
            )
            .agg(
                Players=("UserID", "count"),
                MeanRevenue=("InAppPurchaseAmount", "mean"),
                MedianRevenue=("InAppPurchaseAmount", "median"),
            )
            .reset_index()
        )

        fig = px.bar(
            timing_summary,
            x="FirstPurchaseTiming",
            y="MeanRevenue",
            title="Mean Observed Revenue by First Purchase Timing",
        )
        fig.update_yaxes(title="Mean observed revenue (USD)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        timing_engagement = (
            filtered_behavior.groupby(
                "FirstPurchaseTiming",
                observed=True,
            )
            .agg(
                AvgSessionCount=("SessionCount", "mean"),
                AvgSessionLength=(
                    "AverageSessionLength",
                    "mean",
                ),
            )
            .reset_index()
        )

        fig = px.bar(
            timing_engagement,
            x="FirstPurchaseTiming",
            y=[
                "AvgSessionCount",
                "AvgSessionLength",
            ],
            barmode="group",
            title="Engagement by First Purchase Timing",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Purchase Recency")

    recency_distribution = (
        filtered_behavior["PurchaseRecencyGroup"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).reset_index()

    recency_distribution.columns = [
        "PurchaseRecencyGroup",
        "PlayerShare",
    ]

    fig = px.bar(
        recency_distribution,
        x="PurchaseRecencyGroup",
        y="PlayerShare",
        text=recency_distribution["PlayerShare"].map(
            lambda x: f"{x:.1f}%"
        ),
        title="Purchase Recency Distribution",
    )
    fig.update_yaxes(title="Players (%)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Behavioral Contrasts")

    low_value_high_engagement = filtered_behavior[
        (
            filtered_behavior["EngagementScore"]
            >= filtered_behavior["EngagementScore"].quantile(0.75)
        )
        & (
            filtered_behavior["InAppPurchaseAmount"]
            <= filtered_behavior["InAppPurchaseAmount"].quantile(0.25)
        )
    ]

    high_value_low_engagement = filtered_behavior[
        (
            filtered_behavior["EngagementScore"]
            <= filtered_behavior["EngagementScore"].quantile(0.25)
        )
        & (
            filtered_behavior["InAppPurchaseAmount"]
            >= filtered_behavior["InAppPurchaseAmount"].quantile(0.75)
        )
    ]

    contrast = pd.DataFrame(
        {
            "Group": [
                "High Engagement / Low Revenue",
                "Low Engagement / High Revenue",
            ],
            "Players": [
                len(low_value_high_engagement),
                len(high_value_low_engagement),
            ],
            "Avg First Purchase Days": [
                low_value_high_engagement[
                    "FirstPurchaseDaysAfterInstall"
                ].mean(),
                high_value_low_engagement[
                    "FirstPurchaseDaysAfterInstall"
                ].mean(),
            ],
            "Avg Purchase Recency Days": [
                low_value_high_engagement[
                    "PurchaseRecencyDays"
                ].mean(),
                high_value_low_engagement[
                    "PurchaseRecencyDays"
                ].mean(),
            ],
            "Avg Sessions": [
                low_value_high_engagement["SessionCount"].mean(),
                high_value_low_engagement["SessionCount"].mean(),
            ],
            "Avg Session Length": [
                low_value_high_engagement[
                    "AverageSessionLength"
                ].mean(),
                high_value_low_engagement[
                    "AverageSessionLength"
                ].mean(),
            ],
        }
    )

    st.dataframe(
        contrast.round(2),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "These are contrasting behavioral profiles rather than formal "
        "statistical outliers."
    )


# ============================================================
# Predictive Modeling
# ============================================================

with modeling_tab:
    st.subheader("Predictive Modeling — High-Value Player (HVP)")

    st.warning(
        "HVP is an analytical label created specifically for this modeling "
        "exercise. It is not the same as the High Monetization behavioral "
        "segments used elsewhere in the project."
    )

    modeling_df = model_results["modeling_df"]
    hvp_threshold = model_results["threshold"]
    metrics = model_results["metrics"]

    st.markdown(
        f"""
        **HVP definition:** players in the top 10% of observed
        In-App Purchase amount.

        **Observed revenue threshold:** **${hvp_threshold:,.2f}**

        The target is defined from observed player-level revenue; players
        with missing purchase amounts are excluded from the modeling population.
        """
    )

    metric_cols = st.columns(6)
    metric_cols[0].metric("HVP Players", f"{modeling_df['HVP'].sum():,}")
    metric_cols[1].metric(
        "HVP Share",
        f"{modeling_df['HVP'].mean():.1%}",
    )
    metric_cols[2].metric(
        "Accuracy",
        f"{metrics['Accuracy']:.3f}",
    )
    metric_cols[3].metric(
        "Precision",
        f"{metrics['Precision']:.3f}",
    )
    metric_cols[4].metric(
        "Recall",
        f"{metrics['Recall']:.3f}",
    )
    metric_cols[5].metric(
        "ROC-AUC",
        f"{metrics['ROC-AUC']:.3f}",
    )

    st.markdown("### Model Performance")

    perf_df = pd.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values()),
        }
    )

    fig = px.bar(
        perf_df,
        x="Metric",
        y="Value",
        title="Logistic Regression Performance",
        range_y=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)

    cm = model_results["confusion_matrix"]

    col1, col2 = st.columns(2)

    with col1:
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Non-HVP", "Actual HVP"],
            columns=["Predicted Non-HVP", "Predicted HVP"],
        )

        fig = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        probability_df = model_results["probability_df"]

        probability_summary = (
            probability_df.groupby("Actual HVP")[
                "Predicted Probability"
            ]
            .agg(["count", "mean", "median", "min", "max"])
            .reset_index()
        )

        st.markdown("#### Out-of-fold probability distribution")
        st.dataframe(
            probability_summary.round(3),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "The predicted probabilities for HVP and Non-HVP players "
            "are very similar, consistent with weak discrimination."
        )

    st.markdown("### ROC Curve")

    # Reconstruct an ROC-style curve from the held-out test set.
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(
        model_results["y_test"],
        model_results["y_prob"],
    )

    roc_fig = go.Figure()
    roc_fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Logistic Regression (AUC = {metrics['ROC-AUC']:.3f})",
        )
    )
    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(dash="dash"),
        )
    )
    roc_fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    st.plotly_chart(roc_fig, use_container_width=True)

    st.markdown("### Model Interpretation")

    coefficients = model_results["coefficients"].head(15).copy()

    fig = px.bar(
        coefficients.sort_values("Coefficient"),
        x="Coefficient",
        y="Feature",
        orientation="h",
        title="Largest Logistic Regression Coefficients",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Coefficient direction indicates association within the fitted "
        "logistic model; it should not be interpreted as causation."
    )

    st.markdown(
        """
        ### Modeling conclusion

        The Logistic Regression model did **not** identify meaningful
        predictive signal for HVP membership with the available player-level
        features. The ROC-AUC is close to 0.50, and the model predicts the
        majority class at the default classification threshold.

        This is a valid analytical result: the available variables and this
        modeling approach were not sufficient to reliably distinguish HVPs.
        It should not be interpreted as evidence that the underlying player
        characteristics have no causal effect on spending.
        """
    )


# ============================================================
# Methodology
# ============================================================

with methodology_tab:
    st.subheader("Methodology & Limitations")

    st.markdown(
        """
        ### Dataset

        - 3,024 player-level records
        - One row per player
        - Demographic, platform, genre, engagement, monetization,
          purchase timing, payment method, and purchase-date variables

        ### Analytical population

        Purchase-related analyses use players with observed
        `InAppPurchaseAmount`. Missing purchase information is not assumed
        to represent zero revenue.

        ### Behavioral segmentation

        Players are classified using:

        - standardized session count
        - standardized average session length
        - log-transformed observed revenue
        - median thresholds

        This produces four rule-based behavioral profiles.

        ### Predictive modeling

        HVP is defined as the top 10% of players by observed
        In-App Purchase amount, using the 90th percentile threshold.

        Logistic Regression uses demographic, platform, genre, engagement,
        purchase-timing, recency, and payment-method features.

        `InAppPurchaseAmount`, `LogRevenue`, and `BehavioralSegment` are
        excluded from the predictive feature set to avoid target leakage.

        ### Important limitations

        - The dataset is observational, so associations are not causal.
        - Revenue is highly right-skewed and concentrated among a small group.
        - The data is aggregated at player level rather than event level.
        - Purchase timing and recency are observed behavioral signals and
          should not be interpreted as valid pre-purchase prediction inputs
          for new players.
        - HVP is an analytical target created for this project, not an
          industry-standard definition.
        - The Logistic Regression result should not be generalized to other
          games or populations.
        """
    )

    st.markdown("### Data completeness")

    missingness = (
        df.isna()
        .sum()
        .reset_index()
    )
    missingness.columns = ["Column", "Missing Values"]
    missingness["Missing Share"] = (
        missingness["Missing Values"] / len(df)
    )

    st.dataframe(
        missingness.assign(
            MissingShare=lambda x: x["Missing Share"].map(
                lambda v: f"{v:.1%}"
            )
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "The master dataset retains the full player population rather than "
        "dropping every row with any missing value."
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Mobile Game Monetization & Player Behavior Analytics · "
    "Portfolio project"
)
