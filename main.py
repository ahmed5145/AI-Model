import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    
# Load Data
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/BostonHousing.csv')

"""     Column Headings
CRIM - per capita crime rate by town
ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS - proportion of non-retail business acres per town.
CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX - nitric oxides concentration (parts per 10 million)
RM - average number of rooms per dwelling
AGE - proportion of owner-occupied units built prior to 1940
DIS - weighted distances to five Boston employment centres
RAD - index of accessibility to radial highways
TAX - full-value property-tax rate per $10,000
PTRATIO - pupil-teacher ratio by town
B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT - % lower status of the population
MEDV - Median value of owner-occupied homes in $1000's
"""
# Data Preparation

# Data separation as X and Y (dependent variable)
price = df['medv']
variables = df.drop('medv', axis=1)

# Data Splitting: 80% Training, 20% Testing
variables_train, variables_test, price_train, price_test = \
train_test_split(variables, price, test_size=0.2, random_state=100)

# Model Building

## Linear Regression

### Training the Model
lr = LinearRegression()
lr.fit(variables_train, price_train)


## Random Forest
### Training the model
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(variables_train, price_train)

# Function to make predictions
def predict_price(features):
    input_data = pd.DataFrame([features], columns=variables.columns)
    prediction_lr = lr.predict(input_data)[0]
    prediction_rf = rf.predict(input_data)[0]
    prediction_avg = (prediction_lr + prediction_rf) / 2
    return prediction_lr, prediction_rf, prediction_avg

# Function to calculate accuracy score
def calculate_accuracy():
    # Predicting on the test set
    price_lr_test_pred = lr.predict(variables_test)
    price_rf_test_pred = rf.predict(variables_test)

    # Calculating R2 Score
    lr_r2_score = r2_score(price_test, price_lr_test_pred)
    rf_r2_score = r2_score(price_test, price_rf_test_pred)

    # Converting R2 Score to percentage
    lr_accuracy = lr_r2_score * 100
    rf_accuracy = rf_r2_score * 100

    return lr_accuracy, rf_accuracy

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction_lr, prediction_rf, prediction_avg = predict_price(features)
    return jsonify({
        'prediction_lr': prediction_lr * 1000,  # convert to dollars
        'prediction_rf': prediction_rf * 1000,  # convert to dollars
        'prediction_avg': prediction_avg * 1000 # convert to dollars
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        data = pd.read_csv(file_path)
        predictions_lr = []
        predictions_rf = []
        predictions_avg = []
        for _, row in data.iterrows():
            features = row.values.tolist()
            prediction_lr, prediction_rf, prediction_avg = predict_price(features)
            predictions_lr.append(prediction_lr * 1000)  # convert to dollars
            predictions_rf.append(prediction_rf * 1000)  # convert to dollars
            predictions_avg.append(prediction_avg * 1000) # convert to dollars
        
        return jsonify({
            'linear_regression': predictions_lr,
            'random_forest': predictions_rf,
            'average_prediction': predictions_avg
        })

@app.route('/accuracy', methods=['GET'])
def accuracy():
    lr_accuracy, rf_accuracy = calculate_accuracy()
    return jsonify({
        'Linear Regression Accuracy': f'{lr_accuracy:.2f}%',
        'Random Forest Accuracy': f'{rf_accuracy:.2f}%'
    })

if __name__ == "__main__":
    app.run(debug=True)