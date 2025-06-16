from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model dan scaler
knn_model = joblib.load("model_knn.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = 1 if request.form['gender'] == 'Male' else 0
    age = float(request.form['age'])
    income = float(request.form['income'])
    score = float(request.form['score'])

    features = np.array([[age, income, score]])
    features_scaled = scaler.transform(features)
    cluster = knn_model.predict(features_scaled)[0]

    return render_template('index.html', prediction_text=f"Pelanggan termasuk dalam segmen: Cluster {cluster}")

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_table="File tidak ditemukan.")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    df = pd.read_csv(filepath)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    X_scaled = scaler.transform(X)
    predictions = knn_model.predict(X_scaled)
    df['Predicted Cluster'] = predictions

    return render_template('index.html', prediction_table=df.to_html(classes="table table-striped"))

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
