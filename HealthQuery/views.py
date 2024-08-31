from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework import views
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json

# Load the dataset
data = pd.read_csv("heart.csv")
y = data["target"] 
X = data.drop("target", axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Debug: Print the shape of the train and test sets
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Evaluate the model accuracy
accuracy = model.score(X_test, y_test)

# Debug: Print the model accuracy
print(f"Model accuracy: {accuracy * 100:.2f}%")


def predict_target(test_data, model):
    predictions = model.predict(test_data)
    # Debug: Print the predictions
    print(f"Predicted targets: {predictions}")
    return predictions


class HeartAPIView(views.APIView):
    def post(self, request):
        try:
            data = json.loads(request.body)
            keys = [
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal",
            ]

            point = [data.get(key, 0) for key in keys]
            test_data = [point]

            predicted_targets = predict_target(test_data, model)
            result = predicted_targets[0]

            return JsonResponse({"data": result})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def submit_form(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            keys = [
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal",
            ]

            # Ensure all required fields are provided and default to 0 if missing
            point = [data.get(key, 0) for key in keys]
            test_data = [point]  # Convert to 2D array as required by the model

            # Make prediction
            predicted_targets = predict_target(test_data, model)
            result = predicted_targets[0] if predicted_targets else 0

            return JsonResponse({"message": result})
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Only POST requests are allowed"}, status=405)


def home(request):
    return render(request, "HealthQuery/home.html", {})
