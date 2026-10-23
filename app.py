import streamlit as st
import pandas as pd
import joblib


# -------------------------
# Load model and resources
# -------------------------

model = joblib.load("models/mental_health_model.joblib")

links = pd.read_csv("links.csv", skiprows=1)

# Remove the separator row from links.csv
links = links[links["cntry"] != "---"]
links = links.dropna(subset=["country"])


# -------------------------
# Page setup
# -------------------------

st.set_page_config(
    page_title="Mental Well-being Quiz",
    page_icon="🧠",
    layout="centered"
)

st.title("Mental Well-being Quiz")

st.write(
    """
    This quiz uses a machine-learning model trained on
    European Social Survey data.

    The result shows patterns found in survey data and is
    **not a medical diagnosis**.
    """
)

st.divider()


# -------------------------
# Quiz
# -------------------------

st.header("Your answers")


generation = st.selectbox(
    "Which generation are you?",
    ["Gen Z", "Millennial"]
)


country = st.selectbox(
    "Which country do you live in?",
    sorted(links["country"].tolist())
)


internet_hours = st.number_input(
    "About how many hours per day do you use the internet?",
    min_value=0.0,
    max_value=24.0,
    value=4.0,
    step=0.5
)


sleep_options = {
    "None or almost none of the time": 1,
    "Some of the time": 2,
    "Most of the time": 3,
    "All or almost all of the time": 4
}

sleep_answer = st.selectbox(
    "During the past week, how often was your sleep restless?",
    list(sleep_options.keys())
)


ctrlife = st.slider(
    "How much control do you feel you have over your life?",
    min_value=0,
    max_value=10,
    value=5,
    help="0 = none, 10 = complete"
)


health_options = {
    "Very good": 1,
    "Good": 2,
    "Fair": 3,
    "Bad": 4,
    "Very bad": 5
}

health_answer = st.selectbox(
    "How is your health in general?",
    list(health_options.keys())
)


gender_options = {
    "Male": 1,
    "Female": 2
}

gender_answer = st.selectbox(
    "What is your gender assigned at birth?",
    list(gender_options.keys())
)


unemployment_options = {
    "Yes": 1,
    "No": 2
}

unemployment_answer = st.selectbox(
    "Have you ever been unemployed and seeking work for more than three months?",
    list(unemployment_options.keys())
)


support_options = {
    "None": 0,
    "1 person": 1,
    "2 people": 2,
    "3 people": 3,
    "4–6 people": 4,
    "7–9 people": 5,
    "10 or more people": 6
}

support_answer = st.selectbox(
    "How many people can you discuss intimate or personal matters with?",
    list(support_options.keys())
)


# -------------------------
# Prediction
# -------------------------

if st.button("See my result"):

    country_row = links[links["country"] == country]

    country_code = country_row["cntry"].iloc[0]
    resource_link = country_row["link"].iloc[0]

    # Model uses minutes, not hours
    netustm = internet_hours * 60

    user_data = pd.DataFrame(
        {
            "generation": [generation],
            "cntry": [country_code],
            "netustm": [netustm],
            "slprl": [sleep_options[sleep_answer]],
            "ctrlife": [ctrlife],
            "health": [health_options[health_answer]],
            "gndr": [gender_options[gender_answer]],
            "uemp3m": [unemployment_options[unemployment_answer]],
            "inprdsc": [support_options[support_answer]],
        }
    )

    probability = model.predict_proba(user_data)[0][1]
    risk = probability * 100

    st.divider()
    st.header("Your result")

    st.metric(
        "Estimated probability of reporting depressive feelings",
        f"{risk:.1f}%"
    )

    st.write(
        """
        This percentage represents patterns found among people
        with similar answers in the survey data. It does not
        determine whether you have depression or any other
        mental-health condition.
        """
    )

    st.subheader("Mental-health resources")

    st.markdown(
        f"[View resources for {country}]({resource_link})"
    )