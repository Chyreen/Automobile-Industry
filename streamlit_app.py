import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# ---------------------------------------------------
# Page config
# ---------------------------------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    layout="wide"
)

# ---------------------------------------------------
# Custom styling
# ---------------------------------------------------
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.2rem;
    }

    .prediction-card {
        background: linear-gradient(135deg, #111827, #1f2937);
        color: white;
        padding: 1.4rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }

    .prediction-label {
        font-size: 1rem;
        opacity: 0.85;
        margin-bottom: 0.4rem;
    }

    .prediction-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    .soft-box {
        background-color: #f7f7f8;
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid #ececec;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "auto_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "adverts.csv")
IMAGE_PATH = os.path.join(BASE_DIR, "sports_car.png")

# ---------------------------------------------------
# Load model
# ---------------------------------------------------
model = joblib.load(MODEL_PATH)

# ---------------------------------------------------
# Build make -> model mapping
# ---------------------------------------------------
df = pd.read_csv(DATA_PATH)

make_model_map = (
    df.groupby("standard_make")["standard_model"]
    .apply(lambda x: sorted(x.dropna().astype(str).unique().tolist()))
    .to_dict()
)

for make in make_model_map:
    if "Other" not in make_model_map[make]:
        make_model_map[make].append("Other")

make_model_map["Other"] = ["Other"]

# ---------------------------------------------------
# Helper function
# ---------------------------------------------------
def derive_status(mileage: float, age: int) -> str:
    """
    Recreates the same logic used during training.
    """
    if age > 10 and mileage >= (10000 * age):
        return "scrab"
    return "non_scrab"

# ---------------------------------------------------
# Hero section
# ---------------------------------------------------
if os.path.exists(IMAGE_PATH):
    img = Image.open(IMAGE_PATH)
    width, height = img.size

    # Crop image to half height for a banner effect
    img_cropped = img.crop((0, 0, width, height // 2))
    st.image(img_cropped)

st.markdown('<div class="hero-title">Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Estimate the likely market value of a vehicle from its details.</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# Main layout
# ---------------------------------------------------
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown("### Vehicle Details")

    col1, col2 = st.columns(2)

    with col1:
        mileage = st.number_input(
            "Mileage",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            help="Total mileage driven by the vehicle."
        )

        age = st.number_input(
            "Age (years)",
            min_value=0,
            value=5,
            step=1,
            help="Vehicle age in years."
        )

        make = st.selectbox(
            "Make",
            options=sorted(make_model_map.keys())
        )

        model_name = st.selectbox(
            "Model",
            options=make_model_map.get(make, ["Other"])
        )

    with col2:
        condition = st.selectbox(
            "Condition",
            options=["USED", "NEW"]
        )

        colour = st.selectbox(
            "Colour",
            options=[
                "Black", "White", "Blue", "Silver", "Grey",
                "Red", "Green", "Yellow", "Brown", "Orange",
                "Beige", "Purple", "Unknown"
            ]
        )

        body = st.selectbox(
            "Body Type",
            options=[
                "Hatchback", "SUV", "Saloon", "Estate", "Coupe",
                "Convertible", "MPV", "Pickup", "Van", "Other", "Unknown"
            ]
        )

        fuel = st.selectbox(
            "Fuel Type",
            options=[
                "Petrol", "Diesel", "Hybrid", "Electric",
                "Petrol Hybrid", "Diesel Hybrid", "Unknown"
            ]
        )

    predict_clicked = st.button("Predict Price", use_container_width=True)

with right:
    st.markdown("### About This App")
    st.markdown(
        """
        <div class="soft-box">
        This app uses a trained machine learning model to estimate a car's price
        from the details you provide.

        <br><br>
        <b>Derived internally:</b>
        <ul>
            <li><b>status</b> from mileage and age</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if predict_clicked:
    status = derive_status(mileage, age)

    input_df = pd.DataFrame([{
        "mileage": mileage,
        "make": make,
        "model": model_name,
        "condition": condition,
        "colour": colour,
        "body": body,
        "fuel": fuel,
        "age": age,
        "status": status
    }])

    pred_log = model.predict(input_df)[0]
    predicted_price = float(np.expm1(pred_log))

    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">Estimated Price</div>
            <p class="prediction-value">£{predicted_price:,.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Show processed input"):
        st.write("Derived values used by the model:")
        st.write({
            "status": status
        })
        st.dataframe(input_df, use_container_width=True)




# import os
# import json
# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st


# # ---------------------------------------------------
# # Page config
# # ---------------------------------------------------
# st.set_page_config(
#     page_title="Car Price Predictor",
#     layout="wide"
# )

# # ---------------------------------------------------
# # Custom styling
# # ---------------------------------------------------
# st.markdown("""
# <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}

#     .block-container {
#         padding-top: 1.2rem;
#         padding-bottom: 2rem;
#         max-width: 1100px;
#     }

#     .hero-title {
#         font-size: 2.4rem;
#         font-weight: 700;
#         margin-bottom: 0.2rem;
#     }

#     .hero-subtitle {
#         font-size: 1rem;
#         color: #666;
#         margin-bottom: 1.2rem;
#     }

#     .prediction-card {
#         background: linear-gradient(135deg, #111827, #1f2937);
#         color: white;
#         padding: 1.4rem;
#         border-radius: 16px;
#         text-align: center;
#         margin-top: 1rem;
#         box-shadow: 0 10px 25px rgba(0,0,0,0.15);
#     }

#     .prediction-label {
#         font-size: 1rem;
#         opacity: 0.85;
#         margin-bottom: 0.4rem;
#     }

#     .prediction-value {
#         font-size: 2rem;
#         font-weight: 700;
#         margin: 0;
#     }

#     .soft-box {
#         background-color: #f7f7f8;
#         padding: 1rem;
#         border-radius: 14px;
#         border: 1px solid #ececec;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ---------------------------------------------------
# # Paths
# # ---------------------------------------------------
# BASE_DIR = os.path.dirname(__file__)
# MODEL_PATH = os.path.join(BASE_DIR, "auto_model.pkl")
# DATA_PATH = os.path.join(BASE_DIR, "adverts.csv")
# IMAGE_PATH = os.path.join(BASE_DIR, "sports_car.png")  # rename your image to this if you want
# FALLBACK_IMAGE_PATH = os.path.join(
#     "/mnt/data",
#     "a_digital_photograph_features_a_sleek_silver_sport.png"
# )

# # ---------------------------------------------------
# # Load model
# # ---------------------------------------------------
# model = joblib.load(MODEL_PATH)

# # ---------------------------------------------------
# # Build make -> model mapping from dataset
# # ---------------------------------------------------
# df = pd.read_csv(DATA_PATH)

# make_model_map = (
#     df.groupby("standard_make")["standard_model"]
#     .apply(lambda x: sorted(x.dropna().astype(str).unique().tolist()))
#     .to_dict()
# )

# for make in make_model_map:
#     if "Other" not in make_model_map[make]:
#         make_model_map[make].append("Other")

# make_model_map["Other"] = ["Other"]

# # ---------------------------------------------------
# # Helper functions
# # ---------------------------------------------------
# def derive_status(mileage: float, age: int) -> str:
#     if age > 10 and mileage >= (10000 * age):
#         return "scrab"
#     return "non_scrab"


# def derive_state(age: int) -> str:
#     if age > 45:
#         return "vin_ant"
#     elif age > 20:
#         return "classics"
#     return "modern"


# # ---------------------------------------------------
# # Hero section
# # ---------------------------------------------------
# image_used = None
# if os.path.exists(IMAGE_PATH):
#     image_used = IMAGE_PATH
# elif os.path.exists(FALLBACK_IMAGE_PATH):
#     image_used = FALLBACK_IMAGE_PATH

# # if image_used:
# #     st.image(image_used)

# from PIL import Image  # make sure this is at the top of your script

# if image_used:
#     img = Image.open(image_used)

#     width, height = img.size

#     # Crop top half (banner style)
#     img_cropped = img.crop((0, 0, width, height // 2))

#     st.image(img_cropped)

# st.markdown('<div class="hero-title">Car Price Predictor</div>', unsafe_allow_html=True)
# st.markdown(
#     '<div class="hero-subtitle">Estimate the likely market value of a vehicle from its details.</div>',
#     unsafe_allow_html=True
# )

# # ---------------------------------------------------
# # Main layout
# # ---------------------------------------------------
# left, right = st.columns([1.15, 0.85], gap="large")

# with left:
#     st.markdown("### Vehicle Details")

#     col1, col2 = st.columns(2)

#     with col1:
#         mileage = st.number_input(
#             "Mileage",
#             min_value=0.0,
#             value=50000.0,
#             step=1000.0,
#             help="Total mileage driven by the vehicle."
#         )

#         age = st.number_input(
#             "Age (years)",
#             min_value=0,
#             value=5,
#             step=1,
#             help="Vehicle age in years."
#         )

#         make = st.selectbox(
#             "Make",
#             options=sorted(make_model_map.keys())
#         )

#         model_name = st.selectbox(
#             "Model",
#             options=make_model_map.get(make, ["Other"])
#         )

#     with col2:
#         condition = st.selectbox(
#             "Condition",
#             options=["USED", "NEW"]
#         )

#         colour = st.selectbox(
#             "Colour",
#             options=[
#                 "Black", "White", "Blue", "Silver", "Grey",
#                 "Red", "Green", "Yellow", "Brown", "Orange",
#                 "Beige", "Purple", "Unknown"
#             ]
#         )

#         body = st.selectbox(
#             "Body Type",
#             options=[
#                 "Hatchback", "SUV", "Saloon", "Estate", "Coupe",
#                 "Convertible", "MPV", "Pickup", "Van", "Other", "Unknown"
#             ]
#         )

#         fuel = st.selectbox(
#             "Fuel Type",
#             options=[
#                 "Petrol", "Diesel", "Hybrid", "Electric",
#                 "Petrol Hybrid", "Diesel Hybrid", "Unknown"
#             ]
#         )

#     predict_clicked = st.button("Predict Price", use_container_width=True)

# with right:
#     st.markdown("### About This App")
#     st.markdown(
#         """
#         <div class="soft-box">
#         This app uses a trained machine learning model to estimate a car's price
#         from the details you provide.

#         <br><br>
#         <b>Derived internally:</b>
#         <ul>
#             <li><b>status</b> from mileage and age</li>
#             <li><b>state</b> from age band</li>
#         </ul>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# # ---------------------------------------------------
# # Prediction
# # ---------------------------------------------------
# if predict_clicked:
#     status = derive_status(mileage, age)
#     state = derive_state(age)

#     input_df = pd.DataFrame([{
#         "mileage": mileage,
#         "make": make,
#         "model": model_name,
#         "condition": condition,
#         "colour": colour,
#         "body": body,
#         "fuel": fuel,
#         "age": age,
#         "status": status,
#         "state": state
#     }])

#     pred_log = model.predict(input_df)[0]
#     predicted_price = float(np.expm1(pred_log))

#     st.markdown(
#         f"""
#         <div class="prediction-card">
#             <div class="prediction-label">Estimated Price</div>
#             <p class="prediction-value">£{predicted_price:,.2f}</p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     with st.expander("Show processed input"):
#         st.write("Derived values used by the model:")
#         st.write({
#             "status": status
#         })
#         st.dataframe(input_df, use_container_width=True)