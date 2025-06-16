# app.py
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model dan scaler
knn_model = joblib.load("model_knn.pkl")
scaler = joblib.load("scaler.pkl")

def generate_plot_from_input(cluster):
    fig, ax = plt.subplots()
    ax.bar(["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"], [0, 0, 0, 0], color='lightgray')
    ax.bar(f"Cluster {cluster}", 1, color='orange')
    ax.set_title("Hasil Prediksi Input Manual")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return image_base64

def generate_plot(df):
    count = df['Predicted Cluster'].value_counts().sort_index()
    fig, ax = plt.subplots()
    count.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Distribusi Cluster Pelanggan')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Jumlah')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return image_base64

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
    manual_plot = generate_plot_from_input(cluster)

    return render_template('index.html', prediction_text=f"Pelanggan termasuk dalam segmen: Cluster {cluster}", manual_plot=manual_plot)

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
    plot_url = generate_plot(df)

    return render_template('index.html', prediction_table=df.to_html(classes="table table-striped table-sm table-responsive"), plot_url=plot_url)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
