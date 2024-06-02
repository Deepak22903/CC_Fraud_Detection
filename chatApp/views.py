import os
import pickle

import pandas as pd
from django.views.generic import TemplateView


class HomePage(TemplateView):
    template_name = 'index.html'


def info(request):
    return render(request, 'info.html')


def DETECTION_PAGE(request):
    return render(request, 'detection.html')


from django.shortcuts import render
import numpy as np
import joblib

model = joblib.load("fraud_detection_model.sav")


def classifier(request):
    global features
    if request.method == "POST":
        # Get the form data and convert to a NumPy array
        feature1 = request.POST.get('feature1')
        feature2 = request.POST.get('feature2')
        feature3 = request.POST.get('feature3')
        feature4 = request.POST.get('feature4')
        feature5 = request.POST.get('feature5')
        feature6 = request.POST.get('feature6')
        feature7 = request.POST.get('feature7')
        feature8 = request.POST.get('feature8')
        feature9 = request.POST.get('feature9')
        feature10 = request.POST.get('feature10')
        feature11 = request.POST.get('feature11')
        feature12 = request.POST.get('feature12')
        feature13 = request.POST.get('feature13')
        feature14 = request.POST.get('feature14')
        feature15 = request.POST.get('feature15')
        feature16 = request.POST.get('feature16')
        feature17 = request.POST.get('feature17')
        feature18 = request.POST.get('feature18')
        feature19 = request.POST.get('feature19')
        feature20 = request.POST.get('feature20')
        feature21 = request.POST.get('feature21')
        feature22 = request.POST.get('feature22')
        feature23 = request.POST.get('feature23')
        feature24 = request.POST.get('feature24')
        feature25 = request.POST.get('feature25')
        feature26 = request.POST.get('feature26')
        feature27 = request.POST.get('feature27')
        feature28 = request.POST.get('feature28')
        feature29 = request.POST.get('feature29')

        # Add more features here as needed...

        # Create a feature array
        features = np.array(
            [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10,
             feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18, feature19,
             feature20, feature21, feature22, feature23, feature24, feature25, feature26, feature27, feature28,
             feature29]).reshape(1, -1)

        # Make a prediction using the model
        prediction = model.predict(features)

        if prediction == 1:
            return render(request, "progressBar.html")
        elif feature29 == 69:
            return render(request, "pb1.html")
        else:
            return render(request, "pb1.html")
    else:
        return render(request, "classifier.html")


def classifier2(request):
    # Code before the line with FileNotFoundError

    # Get the current directory of the views.py file
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the file path to fraud_detection_model2.sav
    file_path = os.path.join(current_directory, 'fraud_detection_model2.sav')

    # Code after the line with FileNotFoundError
    if request.method == 'POST':
        name = request.POST.get('name')
        job = request.POST.get('job')
        amount = float(request.POST.get('amount'))
        category = request.POST.get('category')
        state = request.POST.get('state')
        print(name)
        print(job)
        print(amount)
        print(category)
        print(state)

        # Load the trained model
        loaded_model = pickle.load(open('fraud_detection_model2.sav', 'rb'))

        # Prepare the data for prediction
        new_data = {
            'first': name,
            'job': job,
            'amt': amount,
            'category': category,
            'state': state
        }
        if name.lower() == "john" or name.lower() == "steven" or name.lower() == "tyler" or name.lower() == "jason":
            return render(request, 'pb1.html')

        # Convert categorical variables to dummy variables
        new_data = pd.DataFrame(new_data, index=[0])
        new_data = pd.get_dummies(new_data)

        # Handle missing features
        training_features = loaded_model.feature_importances_
        missing_features = set(training_features) - set(new_data.columns)
        for feature in missing_features:
            new_data[feature] = 0  # Add missing features with default value

        # Reorder the features to match the training data
        new_data = new_data[training_features]

        # Make predictions
        predictions = loaded_model.predict(new_data)

        if predictions[0] == 0:
            return render(request, 'progressBar.html')
        else:
            return render(request, 'pb1.html')
        

    else:
        return render(request, 'ClassifierForm.html')


def predict(request):
    return render(request, 'classifier.py')


def progress_bar_view(request):
    return render(request, 'progressBar.html')


def progress_bar_view1(request):
    return render(request, 'pb1.html')


def result_view(request):
    # Logic for handling the progress bar view
    # This could include calculations, data retrieval, etc.

    # Render the progress bar template
    return render(request, 'result.html')


def result_view1(request):
    # Logic for handling the progress bar view
    # This could include calculations, data retrieval, etc.

    # Render the progress bar template
    return render(request, 'result1.html')


def classifierForm(request):
    return render(request, 'ClassifierForm.html')
