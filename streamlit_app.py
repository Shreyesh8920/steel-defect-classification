import streamlit as st
from services.prediction import predict_defect

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Steel Surface Defect Classification",
    page_icon="🏭",
    layout="wide"
)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("Project Information")

st.sidebar.write("**Project:** Steel Surface Defect Classification")
st.sidebar.write("**Model:** XGBoost Classifier")
st.sidebar.write("**Input Features:** 11")
st.sidebar.write("**Output:** Defect Category")

with st.sidebar.expander("About"):
    st.write("""
    This application predicts steel surface defects
    using a machine learning model trained on
    manufacturing process data.
    """)

# ---------------- HEADER ---------------- #

st.title("🏭 Steel Surface Defect Classification")

st.markdown("""
Predict surface defects in steel sheets using a trained
machine learning model.
""")

st.divider()

# ---------------- INPUT FORM ---------------- #

with st.form("prediction_form"):

    st.subheader("Input Parameters")

    col1, col2, col3 = st.columns(3)

with col1:

    pixels_areas = st.number_input(
        "Pixels Areas",
        min_value=6,
        max_value=152655,
        value=500
    )

    x_perimeter = st.number_input(
        "X Perimeter",
        min_value=2,
        max_value=7553,
        value=100
    )

    y_perimeter = st.number_input(
        "Y Perimeter",
        min_value=2,
        max_value=903,
        value=50
    )

    steel_plate_thickness = st.number_input(
        "Steel Plate Thickness",
        min_value=40,
        max_value=300,
        value=100
    )

with col2:

    length_of_conveyer = st.number_input(
        "Length of Conveyer",
        min_value=1227,
        max_value=1715,
        value=1400
    )

    maximum_of_luminosity = st.number_input(
        "Maximum of Luminosity",
        min_value=71,
        max_value=252,
        value=150
    )

    minimum_of_luminosity = st.number_input(
        "Minimum of Luminosity",
        min_value=0,
        max_value=196,
        value=50
    )

    sum_of_luminosity = st.number_input(
        "Sum of Luminosity",
        min_value=250,
        max_value=11591414,
        value=500000
    )

with col3:

    x_maximum = st.number_input(
        "X Maximum",
        min_value=4,
        max_value=1713,
        value=500
    )

    y_maximum = st.number_input(
        "Y Maximum",
        min_value=6724,
        max_value=12987692,
        value=100000
    )

    steel_type = st.selectbox(
        "Steel Type",
        ["A300", "A400"]
    )



    submitted = st.form_submit_button(
    "Predict Defect",
    use_container_width=True
    )

# ---------------- PREDICTION ---------------- #

if submitted:

    input_data = {
        "Pixels_Areas": pixels_areas,
        "X_Perimeter": x_perimeter,
        "Y_Perimeter": y_perimeter,
        "Steel_Plate_Thickness": steel_plate_thickness,
        "Length_of_Conveyer": length_of_conveyer,
        "Maximum_of_Luminosity": maximum_of_luminosity,
        "Minimum_of_Luminosity": minimum_of_luminosity,
        "Sum_of_Luminosity": sum_of_luminosity,
        "X_Maximum": x_maximum,
        "Y_Maximum": y_maximum,
        "Steel_Type": steel_type
    }

    try:

        with st.spinner("Generating Prediction..."):
            result = predict_defect(input_data)

        st.divider()

        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Defect",
            value=result
        )

        if result == "Bumps":
            st.warning(
                "Surface irregularities detected on the steel sheet."
            )

        elif result == "Dirtiness":
            st.info(
                "Contamination detected on the steel surface."
            )

        else:
            st.success(
                "Prediction generated successfully."
            )

        with st.expander("Input Summary"):
            st.json(input_data)

        with st.expander("Model Information"):
            st.write("Algorithm: XGBoost")
            st.write("Features Used: 11")
            st.write("Prediction Type: Multi-Class Classification")

    except Exception as e:
        st.error(f"Prediction Error: {e}")