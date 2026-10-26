import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="Mental Well-being Quiz",
    page_icon="🧠",
    layout="centered"
)


# ---------------------------------------------------------
# LOAD MODEL + COUNTRY RESOURCE LINKS
# ---------------------------------------------------------

model = joblib.load("models/mental_health_model.joblib")

links = pd.read_csv("links.csv", skiprows=1)
links = links[links["cntry"] != "---"]
links = links.dropna(subset=["country"])


# ---------------------------------------------------------
# GENERAL STYLING
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* Main page width */
    .block-container {
        max-width: 850px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    /* Make dialog wider */
    div[data-testid="stDialog"] > div {
        width: 85vw !important;
        max-width: 950px !important;
    }

    /* Dialog background */
    div[data-testid="stDialog"] div[role="dialog"] {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border-radius: 16px !important;
    }

    /* More space inside dialog */
    div[data-testid="stDialog"] div[role="dialog"] > div {
        padding-left: 28px !important;
        padding-right: 28px !important;
    }

    /* Announcement text */
    .announcement-content {
        color: #374151;
        font-size: 1rem;
        line-height: 1.7;
        padding: 8px 12px 30px 12px;
    }

    .announcement-content p {
        margin-top: 0;
        margin-bottom: 18px;
    }

    .announcement-content h1 {
        color: #111827;
        margin-bottom: 24px;
    }

    .announcement-content a {
        color: #2563eb;
        text-decoration: none;
    }

    .announcement-content a:hover {
        text-decoration: underline;
    }

    .important-box {
        background-color: #f3f4f6;
        padding: 16px 18px;
        border-radius: 10px;
        margin-top: 18px;
        margin-bottom: 18px;
        color: #1f2937;
    }

    .privacy-box {
        background-color: #eef6f1;
        padding: 16px 18px;
        border-radius: 10px;
        margin-top: 18px;
        margin-bottom: 22px;
        color: #1f2937;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# ANNOUNCEMENT POPUP
# ---------------------------------------------------------

@st.dialog("Important Announcement", width="large")
def show_announcement():

    st.markdown(
        """
        <div class="announcement-content">

        <p>
        Worldwide there are over 1 billion people living with a mental health
        condition. This means 1 in 7 people in the world is currently mentally ill.
        </p>

        <p>
        Especially depression and anxiety have been continuously rising, being at
        an all-time high for the age group of 15–19 year olds. Depression has risen
        by around 30%, while anxiety disorders have also increased considerably.
        </p>

        <p>
        Not only are the numbers of people experiencing mental-health problems
        rising, but there is also a shortage of professional help. Access to
        mental-health professionals can be particularly limited for young people
        and in lower-income countries.
        </p>

        <p>
        If left untreated, mental-health problems can severely disrupt daily life
        and lead to long-term suffering. Mental and physical health are also closely
        connected.
        </p>

        <p>
        Many people experiencing mental-health problems remain undiagnosed.
        Possible reasons include stigma, lack of awareness, and barriers to
        accessing appropriate resources and professional help.
        </p>

        <p>
        If you are struggling with your mental health, consider reaching out to a
        friend, family member or qualified healthcare professional. You do not need
        to have a diagnosed mental-health condition to ask for support.
        </p>

        <p>
        If you think you are an immediate danger to yourself or others, please
        contact an emergency service or crisis hotline.
        Country-specific crisis resources can be found at
        <a href="https://worldhotlines.org/countries" target="_blank">
        World Hotlines
        </a>.
        </p>

        <p>
        This quiz was designed to provide educational information about mental
        well-being and help users find relevant mental-health resources.
        </p>

        <div class="important-box">
        <strong>
        This quiz cannot determine your mental state or diagnose any mental-health
        condition. The result is based on statistical patterns in survey data and
        is not a substitute for professional assessment or care.
        </strong>
        </div>

        <div class="privacy-box">
        <strong>Your privacy matters.</strong><br><br>
        Your answers are used only to calculate the result shown to you.
        They are not saved to a database, CSV file, email service or other
        persistent storage by this application.
        </div>

        <p><strong>Sources</strong></p>

        <p>
        <a href="https://cordialpsychiatry.com/how-much-mental-illness-goes-undiagnosed/"
        target="_blank">
        How Much Mental Illness Goes Undiagnosed? – Cordial Psychiatry
        </a>
        </p>

        <p>
        <a href="https://www.thelancet.com/journals/lanpub/article/PIIS2468-2667(25)00261-0/fulltext"
        target="_blank">
        Mental health: a public health crisis unfolding – The Lancet Public Health
        </a>
        </p>

        <p>
        <a href="https://www.who.int/news-room/fact-sheets/detail/mental-disorders"
        target="_blank">
        Mental disorders – World Health Organization
        </a>
        </p>

        <br>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Close"):
        st.rerun()


# Show announcement once per browser session
if "announcement_seen" not in st.session_state:
    st.session_state.announcement_seen = True
    show_announcement()


# Optional button to reopen it
if st.button("ℹ️ Read important information"):
    show_announcement()


# ---------------------------------------------------------
# INTRODUCTION
# ---------------------------------------------------------

st.title("Mental Well-being Quiz")

st.write(
    """
    This quiz uses a machine-learning model trained on European Social Survey
    data to explore patterns associated with depressive feelings.

    The result is an educational estimate based on survey patterns and is
    **not a medical diagnosis**.
    """
)

st.divider()


# ---------------------------------------------------------
# QUIZ
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# RESULT
# ---------------------------------------------------------

if st.button("See my result", type="primary"):

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
        st.warning(
            "Please answer all questions before viewing your result."
        )
        st.stop()


    # Country code + resource page
    country_row = links[links["country"] == country]

    country_code = country_row["cntry"].iloc[0]
    resource_link = country_row["link"].iloc[0]


    # Training data stores internet use in minutes
    netustm = internet_hours * 60


    # Model input
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


    # -----------------------------------------------------
    # OVERALL RESULT
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # PERSONALIZED FEEDBACK
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # DISCLAIMER
    # -----------------------------------------------------

    st.info(
        "This result describes statistical patterns found in survey data. "
        "It is not a diagnosis and cannot determine whether you have "
        "depression or another mental-health condition."
    )


    # -----------------------------------------------------
    # RESOURCES
    # -----------------------------------------------------

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