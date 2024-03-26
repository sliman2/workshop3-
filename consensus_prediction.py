from flask import Blueprint, request, jsonify
from models.model1 import model1
from models.model2 import train_model as model2
import pandas as pd
import numpy as np

api = Blueprint('api', __name__)

# function to calculate model accuracy
def calculate_accuracy(predictions, true_labels):
    predictions = np.array(predictions)
    num_correct = np.sum(predictions == true_labels)
    total_predictions = len(true_labels)
    accuracy = num_correct / total_predictions
    return accuracy


@api.route('/predict', methods=['GET'])
def predict():
    parameters = request.args.to_dict()

    # Individual models predictions
    prediction1 = model1.predict(pd.DataFrame(parameters, index=[0]))
    prediction2 = model2.predict(pd.DataFrame(parameters, index=[0]))

    # Calculate individual models accuracies 
    accuracy_model1 = 1.00  # accuracy model 1 
    accuracy_model2 = 1.00  # Accuracy model 2 SVM
    

    # Adjust weights based on models accuracies
    weights = np.array([accuracy_model1, accuracy_model2])
    consensus_accuracy = np.mean(weights)
    adjusted_weights = weights / consensus_accuracy

    # Calculate weighted prediction
    weighted_prediction = (prediction1 * adjusted_weights[0] + prediction2 * adjusted_weights[1]) / 2

    response = {
        'consensus_prediction': weighted_prediction.tolist(),
        'individual_predictions': {
            'model1': prediction1.tolist(),
            'model2': prediction2.tolist(),
        }
    }

    return jsonify(response)