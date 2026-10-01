import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import re
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="South Africa Crime Analytics",
    page_icon="🇿🇦",
    layout="wide"
)


# ============================================================
# FILE NAMES
# ============================================================

CRIME_FILE = "2023-2024 _Annual_Stats.xlsx"
STATION_FILE = "police_stations.csv"
MODEL_FILE = "crime_hotspot_random_forest_model.pkl"
MODEL_INFO_FILE = "crime_hotspot_model_info.pkl"


# ============================================================
# HELPER FUNCTION - CLEAN STATION NAMES
# ============================================================

def clean_station_name(name):
    """
    Standardises station names so that the crime dataset can be
    matched with the police station contextual dataset.
    """

    name = str(name).lower().strip()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


# ============================================================
# STATION NAME MAPPING
# ============================================================

station_name_mapping = {
    "bityi": "bhityi",
    "grahamstown": "makhanda",
    "jhb central": "johannesburg central",
    "king william s town": "qonce",
    "int airp king shaka": "king shaka intl airport",
    "or tambo intern airp": "or tambo intl airport",
    "queenstown": "komani",
    "uitenhage": "kariega",
    "hilton kzn": "hilton",
    "mayville kzn": "mayville",
    "maluti": "maloti",
    "middelburg mpumalang": "middelburg mp",
    "morgenzon transvaal": "morgenzon",
    "mount ayliff": "maxesibeni",
    "mount fletcher": "tlokoeng",
    "mount frere": "kwabhaca",
    "somerset east": "kwanojoli",

    # Remaining manual mappings
    "balfour": "balfour ec",
    "balfour tvl": "balfour mp",
    "heidelberg c": "heidelberg wc",
    "richmond c": "richmond nc",
    "berlin": "ntabozuko",
    "cradock": "nxuba",
    "elliot": "khowa",
    "fort beaufort": "kwamaqoma",
    "jamestown": "james calata",
    "katkop": "ngqayi",
    "komga": "qumrha",
    "lady frere": "cacadu",
    "maclear": "nqanqarhu"
}


# ============================================================
# LOAD SAVED MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_FILE)
    model_info = joblib.load(MODEL_INFO_FILE)

    return model, model_info


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

@st.cache_data
def load_data():

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    crime_df = pd.read_excel(
        CRIME_FILE,
        sheet_name="RAW Data",
        header=2
    )

    stations_df = pd.read_csv(STATION_FILE)


    # --------------------------------------------------------
    # Basic crime-data cleaning
    # --------------------------------------------------------

    crime_clean = crime_df.dropna(
        subset=[
            "Station",
            "District",
            "Province",
            "Crime_Category"
        ]
    ).copy()


    # Remove unnecessary unnamed Excel columns
    unnamed_columns = [
        col for col in crime_clean.columns
        if str(col).startswith("Unnamed")
    ]

    crime_clean = crime_clean.drop(
        columns=unnamed_columns,
        errors="ignore"
    )


    # --------------------------------------------------------
    # Keep station-level observations only
    # --------------------------------------------------------

    station_crime = crime_clean[
        crime_clean["Comp level"] == "Station"
    ].copy()


    # --------------------------------------------------------
    # Standardise station names
    # --------------------------------------------------------

    station_crime["station_clean"] = (
        station_crime["Station"]
        .apply(clean_station_name)
    )

    stations_df["station_clean"] = (
        stations_df["name"]
        .apply(clean_station_name)
    )


    # Apply known historical / spelling mappings
    station_crime["station_match"] = (
        station_crime["station_clean"]
        .replace(station_name_mapping)
    )


    # --------------------------------------------------------
    # Select contextual station information
    # --------------------------------------------------------

    station_info = stations_df[
        [
            "station_clean",
            "code",
            "muni_code",
            "dc_code",
            "prov_code",
            "population",
            "area_km2"
        ]
    ].copy()


    # --------------------------------------------------------
    # Merge datasets
    # --------------------------------------------------------

    merged_df = station_crime.merge(
        station_info,
        left_on="station_match",
        right_on="station_clean",
        how="left",
        suffixes=("", "_context")
    )


    # --------------------------------------------------------
    # Convert annual columns from wide to long format
    # --------------------------------------------------------

    year_columns = [
        "2013-2014",
        "2014-2015",
        "2015-2016",
        "2016-2017",
        "2017-2018",
        "2018-2019",
        "2019-2020",
        "2020-2021",
        "2021-2022",
        "2022-2023"
    ]


    crime_long = merged_df.melt(

        id_vars=[
            "Station",
            "District",
            "Province",
            "Crime_Category",
            "Code",
            "population",
            "area_km2",
            "muni_code",
            "dc_code",
            "prov_code"
        ],

        value_vars=year_columns,

        var_name="Year",

        value_name="Crime_Count"
    )


    # Make sure Crime_Count is numeric
    crime_long["Crime_Count"] = pd.to_numeric(
        crime_long["Crime_Count"],
        errors="coerce"
    )


    # Remove missing and invalid negative crime counts
    crime_long = crime_long.dropna(
        subset=["Crime_Count"]
    )

    crime_long = crime_long[
        crime_long["Crime_Count"] >= 0
    ].copy()


    # Year number used by our classification model
    crime_long["Year_Number"] = (
        crime_long["Year"]
        .str[:4]
        .astype(int)
    )


    return crime_long, stations_df


# ============================================================
# TRY TO LOAD EVERYTHING
# ============================================================

try:

    model, model_info = load_model()
    crime_long, stations_df = load_data()

except Exception as e:

    st.error("The application could not load the required project files.")

    st.write("Error details:")

    st.code(str(e))

    st.info(
        "Make sure the Excel dataset, police_stations.csv, "
        "and both .pkl model files are in the same folder as app.py."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🇿🇦 Crime Analytics")

st.sidebar.write(
    "Machine Learning Solution for "
    "Data-driven Crime Analytics in South Africa"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Crime Analysis",
        "🔥 Hotspot Prediction",
        "🤖 Model Performance",
        "ℹ️ About"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Technical Programming 2 Project"
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title("🇿🇦 South Africa Crime Analytics")

    st.subheader(
        "A Machine Learning Solution for Data-driven "
        "Crime Analytics in South Africa"
    )

    st.write(
        """
        This application analyses historical South African crime data
        and uses machine learning to classify police-station areas
        according to hotspot status.

        The dashboard combines historical crime statistics with
        contextual police-station information such as population and
        geographical area.
        """
    )


    st.divider()


    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Police Stations",
        f"{crime_long['Station'].nunique():,}"
    )

    col2.metric(
        "Provinces",
        crime_long["Province"].nunique()
    )

    col3.metric(
        "Crime Categories",
        crime_long["Crime_Category"].nunique()
    )

    col4.metric(
        "Years",
        crime_long["Year"].nunique()
    )


    st.divider()


    st.subheader("📌 Application Features")

    st.markdown(
        """
        - Explore historical crime statistics.
        - Filter crime data by province, station, category and year.
        - Visualise crime patterns and trends.
        - Predict whether an area is classified as a crime hotspot.
        - View the performance of the final Random Forest model.
        """
    )


    st.info(
        "Hotspot classification is based on the "
        "'17 Community reported serious crime' category."
    )


# ============================================================
# CRIME ANALYSIS PAGE
# ============================================================

elif page == "📊 Crime Analysis":

    st.title("📊 Crime Analysis")

    st.write(
        "Use the filters below to explore historical crime patterns."
    )


    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    filter_col1, filter_col2 = st.columns(2)


    provinces = sorted(
        crime_long["Province"]
        .dropna()
        .unique()
    )

    selected_province = filter_col1.selectbox(
        "Province",
        ["All Provinces"] + provinces
    )


    if selected_province == "All Provinces":

        province_data = crime_long.copy()

    else:

        province_data = crime_long[
            crime_long["Province"] == selected_province
        ].copy()


    categories = sorted(
        province_data["Crime_Category"]
        .dropna()
        .unique()
    )

    selected_category = filter_col2.selectbox(
        "Crime Category",
        categories
    )


    filtered_data = province_data[
        province_data["Crime_Category"] == selected_category
    ].copy()


    # --------------------------------------------------------
    # Station and year filters
    # --------------------------------------------------------

    filter_col3, filter_col4 = st.columns(2)


    stations = sorted(
        filtered_data["Station"]
        .dropna()
        .unique()
    )

    selected_station = filter_col3.selectbox(
        "Police Station",
        ["All Stations"] + stations
    )


    years = sorted(
        filtered_data["Year"]
        .dropna()
        .unique()
    )

    selected_year = filter_col4.selectbox(
        "Year",
        ["All Years"] + years
    )


    if selected_station != "All Stations":

        filtered_data = filtered_data[
            filtered_data["Station"] == selected_station
        ]


    if selected_year != "All Years":

        filtered_data = filtered_data[
            filtered_data["Year"] == selected_year
        ]


    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    st.divider()

    total_crime = filtered_data["Crime_Count"].sum()

    average_crime = filtered_data["Crime_Count"].mean()

    number_stations = filtered_data["Station"].nunique()


    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Total Recorded Incidents",
        f"{total_crime:,.0f}"
    )

    metric2.metric(
        "Average Incidents",
        f"{average_crime:,.2f}"
        if not np.isnan(average_crime)
        else "0"
    )

    metric3.metric(
        "Stations",
        number_stations
    )


    # --------------------------------------------------------
    # CRIME TREND
    # --------------------------------------------------------

    st.subheader("📈 Crime Trend Over Time")


    trend_data = (
        filtered_data
        .groupby("Year", as_index=False)["Crime_Count"]
        .sum()
    )


    year_order = [
        "2013-2014",
        "2014-2015",
        "2015-2016",
        "2016-2017",
        "2017-2018",
        "2018-2019",
        "2019-2020",
        "2020-2021",
        "2021-2022",
        "2022-2023"
    ]


    trend_data["Year"] = pd.Categorical(
        trend_data["Year"],
        categories=year_order,
        ordered=True
    )

    trend_data = trend_data.sort_values("Year")


    if len(trend_data) > 0:

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            trend_data["Year"].astype(str),
            trend_data["Crime_Count"],
            marker="o"
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("Crime Count")

        ax.set_title(
            f"Trend: {selected_category}"
        )

        plt.xticks(rotation=45)

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.warning(
            "No records are available for the selected filters."
        )


    # --------------------------------------------------------
    # TOP STATIONS
    # --------------------------------------------------------

    st.subheader("🏢 Police Stations with Highest Recorded Counts")


    station_totals = (
        filtered_data
        .groupby("Station", as_index=False)["Crime_Count"]
        .sum()
        .sort_values(
            "Crime_Count",
            ascending=False
        )
        .head(10)
    )


    if len(station_totals) > 0:

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(
            station_totals["Station"][::-1],
            station_totals["Crime_Count"][::-1]
        )

        ax.set_xlabel("Crime Count")

        ax.set_ylabel("Police Station")

        ax.set_title(
            "Top 10 Police Stations"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


    # --------------------------------------------------------
    # PROVINCE COMPARISON
    # --------------------------------------------------------

    st.subheader("🗺️ Crime by Province")


    province_totals = (
        filtered_data
        .groupby("Province", as_index=False)["Crime_Count"]
        .sum()
        .sort_values(
            "Crime_Count",
            ascending=False
        )
    )


    if len(province_totals) > 0:

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            province_totals["Province"],
            province_totals["Crime_Count"]
        )

        ax.set_xlabel("Province")

        ax.set_ylabel("Crime Count")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


    st.caption(
        "Note: These are raw incident counts. They should not "
        "automatically be interpreted as crime rates because "
        "population sizes differ between locations."
    )


    # --------------------------------------------------------
    # DATA TABLE
    # --------------------------------------------------------

    with st.expander("🔎 View Filtered Data"):

        st.dataframe(
            filtered_data[
                [
                    "Station",
                    "District",
                    "Province",
                    "Crime_Category",
                    "Year",
                    "Crime_Count"
                ]
            ],
            use_container_width=True
        )


# ============================================================
# HOTSPOT PREDICTION PAGE
# ============================================================

elif page == "🔥 Hotspot Prediction":

    st.title("🔥 Crime Hotspot Prediction")

    st.write(
        """
        The trained Random Forest model predicts whether a
        police-station area is classified as a crime hotspot.
        """
    )

    st.info(
        "The classification model was developed using the "
        "'17 Community reported serious crime' category."
    )


    # --------------------------------------------------------
    # Get station context data
    # --------------------------------------------------------

    prediction_data = crime_long[
        crime_long["Crime_Category"]
        == "17 Community reported serious crime"
    ].copy()


    prediction_data = prediction_data.dropna(
        subset=[
            "population",
            "area_km2"
        ]
    )


    prediction_provinces = sorted(
        prediction_data["Province"]
        .dropna()
        .unique()
    )


    selected_prediction_province = st.selectbox(
        "Select Province",
        prediction_provinces
    )


    province_stations = prediction_data[
        prediction_data["Province"]
        == selected_prediction_province
    ]


    station_options = sorted(
        province_stations["Station"]
        .dropna()
        .unique()
    )


    selected_prediction_station = st.selectbox(
        "Select Police Station",
        station_options
    )


    station_record = province_stations[
        province_stations["Station"]
        == selected_prediction_station
    ].iloc[0]


    population = float(
        station_record["population"]
    )

    area = float(
        station_record["area_km2"]
    )


    # --------------------------------------------------------
    # Show contextual information
    # --------------------------------------------------------

    info1, info2 = st.columns(2)

    info1.metric(
        "Population",
        f"{population:,.0f}"
    )

    info2.metric(
        "Area",
        f"{area:,.2f} km²"
    )


    # --------------------------------------------------------
    # Year selection
    # --------------------------------------------------------

    available_prediction_years = [
        2013,
        2014,
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
        2022
    ]


    selected_year_number = st.selectbox(
        "Select Start Year",
        available_prediction_years,
        index=len(available_prediction_years) - 1
    )


    st.caption(
        "For example, 2022 represents the 2022-2023 reporting year."
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Hotspot Status",
        type="primary",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            {
                "Province": [
                    selected_prediction_province
                ],

                "Year_Number": [
                    selected_year_number
                ],

                "population": [
                    population
                ],

                "area_km2": [
                    area
                ]
            }
        )


        try:

            prediction = model.predict(
                input_data
            )[0]


            # Probability if available
            probability = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                probability = probabilities[
                    int(prediction)
                ]


            if prediction == 1:

                st.error(
                    "🔥 HOTSPOT CLASSIFICATION"
                )

                st.write(
                    f"**{selected_prediction_station}** "
                    "is classified by the model as a "
                    "**crime hotspot** for the selected inputs."
                )

            else:

                st.success(
                    "✅ NON-HOTSPOT CLASSIFICATION"
                )

                st.write(
                    f"**{selected_prediction_station}** "
                    "is classified by the model as a "
                    "**non-hotspot** for the selected inputs."
                )


            if probability is not None:

                st.metric(
                    "Model Confidence",
                    f"{probability * 100:.2f}%"
                )


            st.divider()


            st.subheader("Prediction Inputs")

            display_input = pd.DataFrame(
                {
                    "Variable": [
                        "Province",
                        "Police Station",
                        "Year",
                        "Population",
                        "Area (km²)"
                    ],

                    "Value": [
                        selected_prediction_province,
                        selected_prediction_station,
                        selected_year_number,
                        f"{population:,.0f}",
                        f"{area:,.2f}"
                    ]
                }
            )

            st.dataframe(
                display_input,
                use_container_width=True,
                hide_index=True
            )


            st.warning(
                "This classification is produced by a machine "
                "learning model and should not be interpreted as "
                "a guarantee of future crime."
            )


        except Exception as e:

            st.error(
                "The prediction could not be completed."
            )

            st.code(str(e))


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    st.write(
        """
        Multiple classification algorithms were evaluated during
        model development. The final Random Forest model was selected
        after chronological testing on later reporting years.
        """
    )


    st.subheader("🏆 Final Model")

    st.success(
        "Random Forest Classifier"
    )


    # --------------------------------------------------------
    # Performance metrics
    # --------------------------------------------------------

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Accuracy",
        "96.76%"
    )

    metric2.metric(
        "Precision",
        "98.98%"
    )

    metric3.metric(
        "Recall",
        "87.43%"
    )

    metric4.metric(
        "F1-Score",
        "92.85%"
    )


    st.divider()


    st.subheader("📅 Chronological Evaluation")

    st.write(
        """
        To provide a more realistic evaluation, earlier reporting
        years were used for model training and later years were
        reserved for testing.
        """
    )


    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Training Period")

        st.write(
            "2013-2014 to 2020-2021"
        )

        st.metric(
            "Training Observations",
            "9,272"
        )


    with col2:

        st.markdown("#### Testing Period")

        st.write(
            "2021-2022 and 2022-2023"
        )

        st.metric(
            "Testing Observations",
            "2,318"
        )


    st.divider()


    # --------------------------------------------------------
    # Reproduce test evaluation
    # --------------------------------------------------------

    st.subheader("📊 Confusion Matrix")


    hotspot_category = crime_long[
        crime_long["Crime_Category"]
        == "17 Community reported serious crime"
    ].copy()


    hotspot_threshold = float(
        model_info.get(
            "hotspot_threshold",
            1896.25
        )
    )


    hotspot_category["Hotspot"] = (
        hotspot_category["Crime_Count"]
        >= hotspot_threshold
    ).astype(int)


    evaluation_df = hotspot_category.dropna(
        subset=[
            "population",
            "area_km2"
        ]
    ).copy()


    test_years = [
        "2021-2022",
        "2022-2023"
    ]


    evaluation_test = evaluation_df[
        evaluation_df["Year"].isin(
            test_years
        )
    ].copy()


    X_evaluation = evaluation_test[
        [
            "Province",
            "Year_Number",
            "population",
            "area_km2"
        ]
    ]


    y_evaluation = evaluation_test[
        "Hotspot"
    ]


    try:

        y_pred_evaluation = model.predict(
            X_evaluation
        )


        cm = confusion_matrix(
            y_evaluation,
            y_pred_evaluation
        )


        fig, ax = plt.subplots(
            figsize=(6, 5)
        )


        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Non-Hotspot",
                "Hotspot"
            ]
        )


        display.plot(
            ax=ax,
            values_format="d"
        )


        ax.set_title(
            "Random Forest Confusion Matrix"
        )


        st.pyplot(fig)

        plt.close(fig)


        # ----------------------------------------------------
        # Recalculated performance
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_evaluation,
            y_pred_evaluation
        )

        precision = precision_score(
            y_evaluation,
            y_pred_evaluation,
            zero_division=0
        )

        recall = recall_score(
            y_evaluation,
            y_pred_evaluation,
            zero_division=0
        )

        f1 = f1_score(
            y_evaluation,
            y_pred_evaluation,
            zero_division=0
        )


        st.subheader(
            "Recalculated Test Metrics"
        )


        eval1, eval2, eval3, eval4 = st.columns(4)

        eval1.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        eval2.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

        eval3.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

        eval4.metric(
            "F1-Score",
            f"{f1 * 100:.2f}%"
        )


        st.subheader(
            "Classification Report"
        )


        report = classification_report(
            y_evaluation,
            y_pred_evaluation,
            target_names=[
                "Non-Hotspot",
                "Hotspot"
            ],
            output_dict=True,
            zero_division=0
        )


        report_df = (
            pd.DataFrame(report)
            .transpose()
        )


        st.dataframe(
            report_df.round(3),
            use_container_width=True
        )


    except Exception as e:

        st.warning(
            "The evaluation results could not be "
            "recalculated."
        )

        st.code(str(e))


    st.divider()


    st.subheader("🎯 Hotspot Definition")

    st.write(
        f"""
        A station-year observation was labelled as a hotspot when
        its number of incidents for **17 Community reported serious
        crime** was at or above the 75th percentile threshold.

        **Hotspot threshold: {hotspot_threshold:.2f} incidents**

        Since crime counts are whole numbers, this corresponds
        approximately to **1,897 or more incidents**.
        """
    )


    st.subheader("Why was Crime_Count not an input?")

    st.write(
        """
        Crime_Count was used to create the hotspot target.
        Therefore, using Crime_Count as an input feature would reveal
        the answer to the model and cause target leakage.

        The final model instead uses Province, reporting year,
        population and geographical area.
        """
    )


# ============================================================
# ABOUT PAGE
# ============================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About the Project")

    st.subheader(
        "Machine Learning Solution for Data-driven "
        "Crime Analytics in South Africa"
    )


    st.write(
        """
        This project demonstrates how data analytics and machine
        learning can be applied to historical crime information.

        The system combines crime statistics with contextual
        information about South African police stations and provides
        interactive exploratory analysis and hotspot classification.
        """
    )


    st.divider()


    st.subheader("📂 Data Used")

    st.write(
        """
        **Crime Dataset**

        Historical South African crime statistics containing police
        station, district, province, crime category and annual crime
        counts.

        **Police Station Dataset**

        Contextual information about police stations including
        population, geographical area, municipality, district and
        province codes.
        """
    )


    st.divider()


    st.subheader("🤖 Machine Learning")

    st.write(
        """
        Several classification algorithms were investigated,
        including:

        - Logistic Regression
        - Decision Tree
        - Random Forest
        - K-Nearest Neighbours
        - Gradient Boosting

        Random Forest was selected as the final classification model
        after chronological evaluation on later reporting years.
        """
    )


    st.divider()


    st.subheader("⚠️ Important Limitations")

    st.write(
        """
        Crime counts represent recorded incidents and are not
        automatically equivalent to crime rates.

        Population and geographical differences between locations
        should be considered when interpreting the results.

        The hotspot classifier is a machine-learning analytical tool.
        Its predictions should not be interpreted as guarantees of
        future crime or as a replacement for official crime analysis.
        """
    )


    st.divider()


    st.caption(
        "South Africa Crime Analytics • Technical Programming 2"
    )