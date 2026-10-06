import pandas as pd
import joblib


# Load the trained Random Forest model
model = joblib.load("models/mental_health_model.joblib")


print("=" * 55)
print("Mental Well-being Quiz")
print("=" * 55)

print(
    "\nThis quiz uses a machine-learning model trained on "
    "European Social Survey data."
)

print(
    "It is an educational project and NOT a medical diagnosis.\n"
)


# 1. Generation
print("Generation:")
print("1. Gen Z")
print("2. Millennial")

generation_input = input("Choose 1 or 2: ")

if generation_input == "1":
    generation = "Gen Z"
else:
    generation = "Millennial"


# 2. Internet use
internet_hours = float(
    input("\nAbout how many hours per day do you use the internet? ")
)

netustm = internet_hours * 60


# 3. Restless sleep
print("\nDuring the past week, how often was your sleep restless?")
print("1. None or almost none of the time")
print("2. Some of the time")
print("3. Most of the time")
print("4. All or almost all of the time")

slprl = int(input("Choose 1-4: "))


# 4. Control over life
ctrlife = int(
    input(
        "\nHow much control do you feel you have over your life "
        "(0 = none, 10 = complete)? "
    )
)


# 5. General health
print("\nHow is your health in general?")
print("1. Very good")
print("2. Good")
print("3. Fair")
print("4. Bad")
print("5. Very bad")

health = int(input("Choose 1-5: "))


# 6. Gender
print("\nGender:")
print("1. Male")
print("2. Female")

gndr = int(input("Choose 1 or 2: "))


# 7. Unemployment
print(
    "\nHave you ever been unemployed and looking for work "
    "for more than three months?"
)

print("1. Yes")
print("2. No")

uemp3m = int(input("Choose 1 or 2: "))


# 8. Social support
print(
    "\nHow many people can you discuss intimate or personal matters with?"
)

print("0. None")
print("1. 1 person")
print("2. 2 people")
print("3. 3 people")
print("4. 4-6 people")
print("5. 7-9 people")
print("6. 10 or more people")

inprdsc = int(input("Choose 0-6: "))


# Create one row in the same format as the model training data
user_data = pd.DataFrame(
    {
        "generation": [generation],
        "netustm": [netustm],
        "slprl": [slprl],
        "ctrlife": [ctrlife],
        "health": [health],
        "gndr": [gndr],
        "uemp3m": [uemp3m],
        "inprdsc": [inprdsc],
    }
)


# Get probability for class 1
probability = model.predict_proba(user_data)[0][1]


print("\n" + "=" * 55)
print("RESULT")
print("=" * 55)

print(
    "\nEstimated probability of reporting depressive feelings "
    f"at least some of the time: {probability:.1%}"
)


if probability < 0.35:
    print(
        "\nYour answers are associated with a relatively lower "
        "estimated likelihood in this model."
    )

elif probability < 0.60:
    print(
        "\nYour answers are associated with a moderate "
        "estimated likelihood in this model."
    )

else:
    print(
        "\nYour answers are associated with a relatively higher "
        "estimated likelihood in this model."
    )


print(
    "\nThis result describes patterns found in survey data. "
    "It does not determine whether you have depression or "
    "any other mental-health condition."
)