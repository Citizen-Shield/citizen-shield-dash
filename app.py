import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.colors as mcolors
import streamlit as st
from utils import run_catboost_var

# CONFIG
st.set_page_config(
    page_title="Citizen Shield Multi Method Dashboard",
    page_icon=":sparkles:",
    layout="wide",
)

# HIDE STREAMLIT STYLE
hide_streamlit_style = """
                        <style>
                        #MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}
                        header {visibility: hidden;}
                        .st-dn {background-color: #743de0;}
                        .st-ep {
                            background-color: #743de0;
                        }
                        .css-w770g5:hover {
                            border-top-color: #743de0;
                            border-right-color: #743de0;
                            border-bottom-color: #743de0;
                            border-left-color: #743de0;
                            color: #743de0;
                        }
                        .css-w770g5:active {
                            color: #743de0;
                            border-top-color: #743de0;
                            border-right-color: #743de0;
                            border-bottom-color: #743de0;
                            border-left-color: #743de0;
                            background-color: #743de0;
                        }
                        .css-10y5sf6 {
                            color: white;
                        }
                        .css-1vzeuhh {
                            background-color: white;
                        }
                        </style>
                        """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# CREATE CACHE DATA FUNCTION
@st.cache_data(ttl=3600)
def get_data(file_name="data.csv"):
    df = pd.read_csv(file_name, index_col=[0]).reset_index(drop=True)
    return df


# CREATE ANALYSIS CACHE FUNCTION
@st.cache_data(ttl=3600)
def get_analysis_output(df, outcome, feature_list, amount_splits, amount_repeats):
    shap_df = run_catboost_var(
        data=df,
        outcome=outcome,
        feature_list=feature_list,
        amount_splits=amount_splits,
        amount_repeats=amount_repeats,
    )
    return shap_df


# SIDEBAR - TITLE AND LOGO
st.sidebar.markdown(
    """
    <div style="text-align: center; padding-right: 10px;">
        <img alt="logo" src="https://citizenshieldproject.files.wordpress.com/2021/01/citizenshield-logo-colour.png?w=580&h=360" width="200">
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    """
    <div style="text-align: center; color: #E8C003; margin-top: 40px; margin-bottom: 40px;">
        <a href="https://citizenshield.fi" style="color: #E8C003;">Part of Citizen Shield</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# SIDEBAR - TITLE AND DATA SOURCE
# SIDEBAR - OUTCOME AND FEATURE LIST
chosen_file = st.sidebar.file_uploader("Upload your own data", type="csv")

if chosen_file is not None:
    # READ DATA
    df = get_data(file_name=chosen_file)

    outcome = st.sidebar.selectbox(
        "Select an outcome measure of interest:", options=df.columns.tolist()
    )

    amount_splits = st.sidebar.slider(
        "Select the amount of splits for the cross validation:",
        min_value=2,
        max_value=10,
        value=2,
    )

    amount_repeats = st.sidebar.slider(
        "Select the amount of repeats for the cross validation:",
        min_value=2,
        max_value=10,
        value=2,
    )

    if df[outcome].dtype == "object":
        st.warning("Please select a numeric outcome measure.")

    else:
        feature_list = st.sidebar.multiselect(
            "Select features to include in the analysis:",
            options=df.drop(columns=[outcome]).columns.tolist(),
            default=df.drop(columns=[outcome]).columns.tolist(),
        )

        # SIDEBAR - LOGO AND CREDITS
        st.sidebar.markdown("---")
        # st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
        st.sidebar.markdown(
            """
            <div style="text-align: center; padding-right: 10px;">
                <img alt="logo" src="https://services.jms.rocks/img/logo.png" width="100">
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.sidebar.markdown(
            """
            <div style="text-align: center; color: #E8C003; margin-top: 40px; margin-bottom: 40px;">
                <a href="https://services.jms.rocks" style="color: #E8C003;">Created by James Twose</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # RUN MAIN ANALYSIS
        shap_df = get_analysis_output(
            df=df,
            outcome=outcome,
            feature_list=feature_list,
            amount_splits=amount_splits,
            amount_repeats=amount_repeats,
        )

        # GET BEST FEATURE
        shap_feature_importance_sorted = (
            shap_df.drop(columns=["fold_number"])
            .median()
            .sort_values(ascending=True)
            .index.tolist()
        )
        shap_best_feature = shap_feature_importance_sorted[-1]

        # MAINPAGE
        st.markdown(
            "<h1>Citizen Shield Multi Method Dashboard</h1>", unsafe_allow_html=True
        )
        left_column, middle_column, right_column = st.columns(3)
        with left_column:
            st.subheader("Chosen Outcome")
            st.subheader(outcome)
        with middle_column:
            st.subheader("Mean of chosen outcome")
            st.subheader(f"{df[outcome].mean():,.3f}")
        with right_column:
            st.subheader("Variance of chosen outcome")
            st.subheader(f"{df[outcome].var():,.3f}")
        st.markdown("---")

        st.header("Main Report")
        st.markdown(
            f"""Based on the variance in SHAP values, the following feature
            is the most important: :green[{shap_best_feature}]"""
        )
        st.markdown("---")

        st.header("Selected Dataframe")
        st.dataframe(df)
        st.markdown("---")
        st.header("Descriptive Statistics")
        st.dataframe(df.describe())
        st.markdown("---")

        st.header("Pearson Correlations between all columns")
        plot_df = df.select_dtypes("number").corr().round(3)
        # correlation plot
        mycmap = mcolors.LinearSegmentedColormap.from_list(
            "mycmap", ["#E8C003", "white", "#743de0"], N=256
        )
        color_list = [mcolors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]
        corr_heat = px.imshow(
            plot_df,
            text_auto=True,
            color_continuous_scale=color_list,
        )

        st.plotly_chart(corr_heat)
        st.markdown("---")
        
        # SHOW CIBER
        st.header("CIBER Calculation")
        CIBER_fig = px.strip(
            df[feature_list].select_dtypes("number")
            .melt()
            .assign(**{"value": lambda x: x["value"].add(np.random.rand(x["value"].shape[0])*0.75)}),
            x="value",
            y="variable",
            # height=1000,
            stripmode="group",
            color_discrete_sequence=["rgba(184, 184, 184, 0.25)"],
        )
        st.plotly_chart(CIBER_fig)

        # SHOW MAIN ANALYSIS OUTPUT
        st.header("CatBoost and SHAP (variance) Calculation")

        # SHOW OUTCOME DISTRIBUTION
        outcome_fig = px.box(
            df[[outcome]].melt(),
            x="value",
            y="variable",
            points="all",
            title=f"Outcome Distribution - {outcome}",
            color_discrete_sequence=["#E8C003"],
        )
        st.plotly_chart(outcome_fig)

        # SHAP PLOT
        st.markdown(
            """
            <div>
                <p>For more information on SHAP Values, please see the following: 
                    <a href="https://christophm.github.io/interpretable-ml-book/shap.html"
                    style="color: #E8C003;">SHAP Values Explanation</a>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        shap_fig = px.box(
            shap_df[shap_feature_importance_sorted].melt(),
            x="value",
            y="variable",
            points="all",
            color_discrete_sequence=["#E8C003"],
            title="Feature Importance Based on Variance in SHAP Values",
            height=800,
        )

        shap_fig.update_layout(showlegend=False, coloraxis_showscale=True)
        st.plotly_chart(shap_fig)

else:
    st.warning("Please upload a CSV file.")
