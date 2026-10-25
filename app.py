import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib


# -------------------------
# Load model and resources
# -------------------------

model = joblib.load("models/mental_health_model.joblib")

links = pd.read_csv("links.csv", skiprows=1)

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

@st.dialog(" ")
def html_popup():
    with open("./article.html", "r", encoding="utf-8") as f:
        raw_html = f.read()
        
    # The '*' applies the rule to all elements inside the body.
    # '!important' overrides any existing colors in your HTML.
    custom_css = """
    <style>
        body, body * { 
            color: white !important; 
        }
        body {
            background-color: #0e1117 !important; /* Streamlit's default dark mode color */
        }
    </style>
    """
    
    html_data = f"{custom_css}\n{raw_html}"
        
    components.html(html_data, height=500, scrolling=True)
    
    if st.button("Close"):
        st.session_state.html_popup_shown = True
        st.rerun()

if "html_popup_shown" not in st.session_state:
    html_popup()


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
    ["Gen Z", "Millennial"],
    index=None,
    placeholder="Select an option"
)


country = st.selectbox(
    "Which country do you live in?",
    sorted(links["country"].tolist()),
    index=None,
    placeholder="Select a country"
)


internet_hours = st.number_input(
    "About how many hours per day do you use the internet?",
    min_value=0.0,
    max_value=24.0,
    value=None,
    step=0.5,
    placeholder="Enter hours"
)


sleep_options = {
    "None or almost none of the time": 1,
    "Some of the time": 2,
    "Most of the time": 3,
    "All or almost all of the time": 4
}

sleep_answer = st.selectbox(
    "During the past week, how often was your sleep restless?",
    list(sleep_options.keys()),
    index=None,
    placeholder="Select an option"
)


ctrlife = st.selectbox(
    "How much control do you feel you have over your life?",
    list(range(0, 11)),
    index=None,
    placeholder="Select a value from 0 to 10"
)

st.caption("0 = no control at all, 10 = complete control")


health_options = {
    "Very good": 1,
    "Good": 2,
    "Fair": 3,
    "Bad": 4,
    "Very bad": 5
}

health_answer = st.selectbox(
    "How is your health in general?",
    list(health_options.keys()),
    index=None,
    placeholder="Select an option"
)


gender_options = {
    "Male": 1,
    "Female": 2
}

gender_answer = st.selectbox(
    "What is your gender assigned at birth?",
    list(gender_options.keys()),
    index=None,
    placeholder="Select an option"
)


unemployment_options = {
    "Yes": 1,
    "No": 2
}

unemployment_answer = st.selectbox(
    "Have you ever been unemployed and seeking work for more than three months?",
    list(unemployment_options.keys()),
    index=None,
    placeholder="Select an option"
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
    list(support_options.keys()),
    index=None,
    placeholder="Select an option"
)

# -------------------------
# Prediction
# -------------------------

if st.button("See my result"):

    required_answers = [
        generation,
        country,
        internet_hours,
        sleep_answer,
        ctrlife,
        health_answer,
        gender_answer,
        unemployment_answer,
        support_answer
    ]

    if any(answer is None for answer in required_answers):
        st.warning("Please answer all questions before viewing your result.")
        st.stop()

    country_row = links[links["country"] == country]

    country_code = country_row["cntry"].iloc[0]
    resource_link = country_row["link"].iloc[0]

    netustm = internet_hours * 60

    # etc...

    # Find country code and country-specific resource link
    country_row = links[links["country"] == country]

    country_code = country_row["cntry"].iloc[0]
    resource_link = country_row["link"].iloc[0]

    # ESS stores internet use in minutes
    netustm = internet_hours * 60

    # Build input in exactly the same format as the model training data
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

    # Prediction
    probability = model.predict_proba(user_data)[0][1]
    risk = probability * 100

    st.divider()
    st.header("Your result")

    st.metric(
        "Estimated probability of reporting depressive feelings",
        f"{risk:.1f}%"
    )

    # Overall interpretation
    if risk < 25:
        st.success(
            "People with similar answers had a relatively low "
            "estimated likelihood of reporting depressive feelings."
        )

    elif risk <= 60:
        st.warning(
            "People with similar answers had a moderate "
            "estimated likelihood of reporting depressive feelings."
        )

    else:
        st.error(
            "People with similar answers had a relatively high "
            "estimated likelihood of reporting depressive feelings."
        )

    # Personalized information
    st.subheader("Based on your answers")

    personalized_feedback = False

    if ctrlife < 6:
        st.write(
            "• You reported relatively low perceived control over your life. "
            "In the survey data, lower perceived control was associated with "
            "more frequent depressive feelings."
        )
        personalized_feedback = True

    if sleep_options[sleep_answer] >= 2:
        st.write(
            "• You reported some level of restless sleep. Restless sleep "
            "was one of the strongest factors associated with depressive "
            "feelings in the model."
        )
        personalized_feedback = True

    health_value = health_options[health_answer]

    if health_value == 3:
        st.write(
            "• You rated your general health as fair. General health was "
            "associated with depressive feelings in the survey data."
        )
        personalized_feedback = True

    elif health_value >= 4:
        st.write(
            "• You rated your general health as bad or very bad. General "
            "health was strongly associated with depressive feelings in "
            "the survey data."
        )
        personalized_feedback = True

    if not personalized_feedback:
        st.write(
            "Your answers did not trigger any of the additional factors "
            "highlighted in this educational summary."
        )

    # Disclaimer
    st.info(
        "This result describes statistical patterns found in survey data. "
        "It is not a diagnosis and cannot determine whether you have "
        "depression or any other mental-health condition."
    )

    # Resources
    st.subheader("Mental-health resources")

    st.markdown(
        f"[View mental-health resources for {country}]({resource_link})"
    )

    st.write(
        "If you are concerned about your mental health, consider talking "
        "to a qualified healthcare professional."
    )

    st.subheader("Additional educational resources")

    st.markdown(
        """
        - [Mental Health Foundation](https://www.mentalhealth.org.uk/)
        - [Mental Health First Aid International](https://mhfainternational.org/)
        - [International Mental Health Association](https://imha.ngo/)
        """
    )