import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Airline Passenger Satisfaction",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# LOAD XGBOOST PIPELINE
# ============================================================

@st.cache_resource
def load_model():

    # Get the directory where app.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Model path
    model_path = os.path.join(
        base_dir,
        "models",
        "xgboost_tuned_model.pkl"
    )

    # Check whether model exists
    if not os.path.exists(model_path):

        st.error("XGBoost model file was not found.")

        st.code(model_path)

        st.stop()

    # Load model using joblib
    try:

        model = joblib.load(model_path)

        return model

    except Exception as e:

        st.error("Unable to load the XGBoost model.")

        st.exception(e)

        st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title(
    "✈️ Airline Passenger Satisfaction Prediction"
)

st.write(
    "XGBoost Classification Model"
)

st.write(
    "Enter the passenger details below to predict passenger satisfaction."
)

st.divider()


# ============================================================
# PASSENGER INFORMATION
# ============================================================

st.subheader("Passenger Information")


# ============================================================
# ROW 1
# Gender | Age | Customer Type
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )


with col2:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )


with col3:

    customer_type = st.selectbox(
        "Customer Type",
        [
            "First-time",
            "Returning"
        ]
    )


# ============================================================
# ROW 2
# Type of Travel | Class | Flight Distance
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    type_of_travel = st.selectbox(
        "Type of Travel",
        [
            "Business",
            "Personal"
        ]
    )


with col2:

    passenger_class = st.selectbox(
        "Class",
        [
            "Business",
            "Economy",
            "Economy Plus"
        ]
    )


with col3:

    flight_distance = st.number_input(
        "Flight Distance",
        min_value=0,
        max_value=10000,
        value=1000
    )


# ============================================================
# ROW 3
# Departure Delay | Arrival Delay |
# Departure and Arrival Time Convenience
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    departure_delay = st.number_input(
        "Departure Delay",
        min_value=0,
        max_value=1000,
        value=0
    )


with col2:

    arrival_delay = st.number_input(
        "Arrival Delay",
        min_value=0,
        max_value=1000,
        value=0
    )


with col3:

    departure_arrival_convenience = st.slider(
        "Departure and Arrival Time Convenience",
        min_value=0,
        max_value=5,
        value=3
    )


# ============================================================
# ROW 4
# Ease of Online Booking | Check-in Service | Online Boarding
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    ease_online_booking = st.slider(
        "Ease of Online Booking",
        min_value=0,
        max_value=5,
        value=3
    )


with col2:

    checkin_service = st.slider(
        "Check-in Service",
        min_value=0,
        max_value=5,
        value=3
    )


with col3:

    online_boarding = st.slider(
        "Online Boarding",
        min_value=0,
        max_value=5,
        value=3
    )


# ============================================================
# ROW 5
# Gate Location | On-board Service | Seat Comfort
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    gate_location = st.slider(
        "Gate Location",
        min_value=0,
        max_value=5,
        value=3
    )


with col2:

    onboard_service = st.slider(
        "On-board Service",
        min_value=0,
        max_value=5,
        value=3
    )


with col3:

    seat_comfort = st.slider(
        "Seat Comfort",
        min_value=0,
        max_value=5,
        value=3
    )


# ============================================================
# ROW 6
# Leg Room Service | Cleanliness | Food and Drink
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    leg_room_service = st.slider(
        "Leg Room Service",
        min_value=0,
        max_value=5,
        value=3
    )


with col2:

    cleanliness = st.slider(
        "Cleanliness",
        min_value=0,
        max_value=5,
        value=3
    )


with col3:

    food_drink = st.slider(
        "Food and Drink",
        min_value=0,
        max_value=5,
        value=3
    )


# ============================================================
# ROW 7
# In-flight Service | In-flight Wifi Service |
# In-flight Entertainment
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    inflight_service = st.slider(
        "In-flight Service",
        min_value=0,
        max_value=5,
        value=3
    )


with col2:

    inflight_wifi = st.slider(
        "In-flight Wifi Service",
        min_value=0,
        max_value=5,
        value=3
    )


with col3:

    inflight_entertainment = st.slider(
        "In-flight Entertainment",
        min_value=0,
        max_value=5,
        value=3
    )


# ============================================================
# ROW 8
# Baggage Handling
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    baggage_handling = st.slider(
        "Baggage Handling",
        min_value=0,
        max_value=5,
        value=3
    )


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔮 Predict Satisfaction",
    type="primary",
    use_container_width=True
):

    # ========================================================
    # CREATE RAW INPUT DATAFRAME
    # EXACT DATASET COLUMN SEQUENCE
    # ========================================================

    new_passenger = pd.DataFrame({

        "Gender": [gender],

        "Age": [age],

        "Customer Type": [customer_type],

        "Type of Travel": [type_of_travel],

        "Class": [passenger_class],

        "Flight Distance": [flight_distance],

        "Departure Delay": [departure_delay],

        "Arrival Delay": [arrival_delay],

        "Departure and Arrival Time Convenience": [
            departure_arrival_convenience
        ],

        "Ease of Online Booking": [
            ease_online_booking
        ],

        "Check-in Service": [
            checkin_service
        ],

        "Online Boarding": [
            online_boarding
        ],

        "Gate Location": [
            gate_location
        ],

        "On-board Service": [
            onboard_service
        ],

        "Seat Comfort": [
            seat_comfort
        ],

        "Leg Room Service": [
            leg_room_service
        ],

        "Cleanliness": [
            cleanliness
        ],

        "Food and Drink": [
            food_drink
        ],

        "In-flight Service": [
            inflight_service
        ],

        "In-flight Wifi Service": [
            inflight_wifi
        ],

        "In-flight Entertainment": [
            inflight_entertainment
        ],

        "Baggage Handling": [
            baggage_handling
        ]

    })


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    try:

        # Pipeline handles preprocessing automatically
        prediction = model.predict(
            new_passenger
        )


        # ====================================================
        # PREDICTION PROBABILITY
        # ====================================================

        confidence = None

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(
                new_passenger
            )

            confidence = probability[0].max()


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "Prediction Result"
        )


        # ----------------------------------------------------
        # SATISFIED
        # ----------------------------------------------------

        if prediction[0] == 1:

            st.success(
                "✅ Passenger is predicted to be SATISFIED"
            )


        # ----------------------------------------------------
        # DISSATISFIED
        # ----------------------------------------------------

        else:

            st.error(
                "❌ Passenger is predicted to be DISSATISFIED"
            )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        if confidence is not None:

            st.metric(
                "Prediction Confidence",
                f"{confidence * 100:.2f}%"
            )


        # ====================================================
        # PREDICTION VALUE
        # ====================================================

        st.write(
            f"Prediction value: `{prediction[0]}`"
        )


        # ====================================================
        # PASSENGER DETAILS
        # ====================================================

        with st.expander(
            "View Passenger Details"
        ):

            st.dataframe(
                new_passenger,
                use_container_width=True
            )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.write(
            "Error details:"
        )

        st.exception(e)