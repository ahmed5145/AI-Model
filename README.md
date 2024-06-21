# Boston House Price Prediction Web App

This repository contains a Flask-based web application for predicting Boston house prices using Linear Regression and Random Forest models. Users can input house features manually or upload a CSV file for batch predictions. The web app also provides model accuracy scores.

## Project Structure

/ML-Model-Housing-Prices-in-Boston
├── main.py
├── /templates
│ └── index.html
└── /static
└── styles.css

- `main.py`: The main Flask application file containing model training, prediction, and routes for the web app.
- `templates/index.html`: The HTML file for the web interface.
- `static/styles.css`: The CSS file for styling the web interface.

## Features

1. **Manual Prediction**: Enter house features manually to get predictions from both models and their average.
2. **Batch Prediction**: Upload a CSV file with house features to get batch predictions.
3. **Model Accuracy**: Fetch the accuracy scores of both models as percentages.

## How to Run

### Prerequisites

- Python 3.6 or higher
- Flask
- scikit-learn
- pandas

### Installation

1. Clone this repository:

```bash
git clone https://github.com/ahmed5145/ML-Model-Housing-Prices-in-Boston.git
cd ML-Model-Housing-Prices-in-Boston
```
2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the required packages:
`pip install -r requirements.txt`

Open a web browser and navigate to http://127.0.0.1:5000/

## Usage
### Manual Prediction
1. Enter the house features in the form provided.
2. Click "Predict" to get predictions from both models and their average.

### Batch Prediction
1. Prepare a CSV file with house features. The file should have the following columns:
CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT
(Their abbreviations are commented in the main.py file.)

2. Upload the CSV file using the "Upload and Predict" form.

3. The web app will display the predictions for each row in the CSV file.

## Model Accuracy
1. Click on the "Fetch Accuracy" button.
2. The web app will display the accuracy scores of both models as percentages.

## Example CSV Format
CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT
0.00632,18.0,2.31,0,0.538,6.575,65.2,4.0900,1,296.0,15.3,396.90,4.98
0.02731,0.0,7.07,0,0.469,6.421,78.9,4.9671,2,242.0,17.8,396.90,9.14
...

## Model Training
The models are trained on the Boston Housing dataset. The dataset is loaded from a URL (find in main.py) and split into training and testing sets. Linear Regression and Random Forest models are trained on the training set.

## Contact
For any questions or issues, please contact Ahmed at ahmedmohamed200354@gmail.com

