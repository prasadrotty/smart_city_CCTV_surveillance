import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple

class MLAgent:
    """
    K-Means Clustering & Random Forest Core Modeling Agent.
    Handles mathematical pipelines for unsupervised profiling and individual capacity predictions.
    """
    def __init__(self, data: pd.DataFrame):
        self.df = data
        self.rf_model = None
        self.le_district = LabelEncoder()
        self.le_street = LabelEncoder()
        
        # Pre-fit encoders to protect dynamic web inferences
        self.df['District_Enc'] = self.le_district.fit_transform(self.df['Район'])
        self.df['Street_Enc'] = self.le_street.fit_transform(self.df['Street_Type'])

    def run_district_clustering(self) -> pd.DataFrame:
        """Clusters urban territories using K-Means into 3 primary capacity categories."""
        district_stats = self.df.groupby('Район').agg(
            total_cameras=('Количество видеокамер', 'sum'),
            mean_cameras=('Количество видеокамер', 'mean'),
            max_cameras=('Количество видеокамер', 'max'),
            address_count=('Адрес', 'count')
        ).reset_index()

        scaler = StandardScaler()
        features = ['total_cameras', 'mean_cameras', 'max_cameras', 'address_count']
        scaled_features = scaler.fit_transform(district_stats[features])
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        district_stats['Tier_Cluster'] = kmeans.fit_predict(scaled_features)
        
        # Translate analytical cluster assignments into descriptive user-facing classifications
        cluster_mapping = {
            0: "Standard Infrastructure Zone", 
            1: "Emerging Infrastructure Zone", 
            2: "High-Density Core Hub"
        }
        district_stats['Hub_Classification'] = district_stats['Tier_Cluster'].map(cluster_mapping)
        return district_stats

    def train_predictive_regressor(self) -> Tuple[float, float]:
        """Trains a Random Forest Regressor to forecast asset scale requirements."""
        X = self.df[['District_Enc', 'Street_Enc', 'Address_Length']]
        y = self.df['Количество видеокамер']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
        self.rf_model.fit(X_train, y_train)
        
        preds = self.rf_model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        return rmse, r2

    def estimate_capacity(self, district_name: str, street_type: str, string_length: int) -> float:
        """Infers safety camera infrastructure volume demands for any specified address mock criteria."""
        if self.rf_model is None:
            self.train_predictive_regressor()
            
        try:
            dist_val = self.le_district.transform([district_name])[0]
            street_val = self.le_street.transform([street_type])[0]
            features = np.array([[dist_val, street_val, string_length]])
            return float(self.rf_model.predict(features)[0])
        except Exception:
            # Fall back to a safe statistical baseline if unexpected features occur
            return float(self.df['Количество видеокамер'].median())
