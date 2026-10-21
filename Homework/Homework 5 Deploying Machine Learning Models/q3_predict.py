
import pickle

# Load the frozen model artifact supplied by the course
with open("pipeline.bin", "rb") as file:
    model = pickle.load(file)

# Lead supplied in Question 3
lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41
}

# Predict the probability of conversion
probability = model.predict_proba([lead])[0, 1]

# Display the result to three decimal places
print("Conversion probability:", round(float(probability), 3))
