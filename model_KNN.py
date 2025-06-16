import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load data
df = pd.read_csv("Mall_Customers.xls")

# Preprocessing
df.drop("CustomerID", axis=1, inplace=True)
df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering untuk membuat label
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters

# Simpan model clustering
joblib.dump(kmeans, "model_kmeans.pkl")

# Melatih KNN berdasarkan hasil clustering
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, clusters)

# Simpan model dan scaler
joblib.dump(knn, "model_knn.pkl")
joblib.dump(scaler, "scaler.pkl")

# Tampilkan prediksi sample pertama
sample_preds = knn.predict(X_scaled[:10])
df_sample = df.head(10).copy()
df_sample["Predicted Cluster"] = sample_preds
print(df_sample[["Age", "Annual Income (k$)", "Spending Score (1-100)", "Cluster", "Predicted Cluster"]])
