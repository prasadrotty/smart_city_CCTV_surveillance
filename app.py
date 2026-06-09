import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# MULTI-AGENT ARCHITECTURE DEFINITIONS
# ==========================================

class DataIngestionAgent:
    """Agent responsible for cleaning, profiling, and feature engineering raw text."""
    def __init__(self, file_path):
        self.file_path = file_path

    def process(self):
        df = pd.read_csv(self.file_path)
        # Feature Engineering: Categorize address infrastructure scale
        df['Street_Type'] = df['Адрес'].apply(self._extract_street_type)
        df['Address_Length'] = df['Адрес'].apply(lambda x: len(str(x)))
        return df

    @staticmethod
    def _extract_street_type(addr):
        addr = str(addr).lower()
        if 'улица' in addr: return 'Улица (Street)'
        elif 'проспект' in addr: return 'Проспект (Avenue)'
        elif 'набережная' in addr: return 'Набережная (Embankment)'
        elif 'шоссе' in addr: return 'Шоссе (Highway)'
        elif 'переулок' in addr: return 'Переулок (Lane)'
        elif 'линия' in addr: return 'Линия (Line)'
        else: return 'Другое (Other)'


class AnalyticsAgent:
    """Agent specialized in high-level descriptive statistical computing."""
    def __init__(self, data):
        self.df = data

    def get_macro_metrics(self):
        metrics = {
            "total_addresses": len(self.df),
            "total_cameras": int(self.df['Количество видеокамер'].sum()),
            "avg_cameras": float(self.df['Количество видеокамер'].mean()),
            "max_cameras": int(self.df['Количество видеокамер'].max()),
            "total_districts": int(self.df['Район'].nunique())
        }
        return metrics

    def get_district_leaderboard(self):
        return self.df.groupby('Район')['Количество видеокамер'].agg(['sum', 'count', 'mean']).reset_index()


class VisualEngineAgent:
    """Agent dedicated to crafting enterprise-grade aesthetic interactive figures."""
    def __init__(self, data):
        self.df = data

    def plot_district_distribution(self):
        summary = self.df.groupby('Район')['Количество видеокамер'].sum().reset_index()
        summary = summary.sort_values(by='Количество видеокамер', ascending=False)
        fig = px.bar(
            summary, x='Количество видеокамер', y='Район', orientation='h',
            title='<b>Total Camera Deployment by District</b>',
            labels={'Количество видеокамер': 'Camera Count', 'Район': 'District'},
            color='Количество видеокамер', color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500, template='plotly_dark')
        return fig

    def plot_street_type_distribution(self):
        summary = self.df.groupby('Street_Type')['Количество видеокамер'].mean().reset_index()
        summary = summary.sort_values(by='Количество видеокамер', ascending=False)
        fig = px.pie(
            summary, values='Количество видеокамер', names='Street_Type',
            title='<b>Average Density Share per Location Category</b>',
            hole=0.4, color_discrete_sequence=px.colors.sequential.MediumWarm
        )
        fig.update_layout(template='plotly_dark')
        return fig


class MLAgent:
    """Agent controlling Unsupervised Stratification & Supervised Capacity Estimation."""
    def __init__(self, data):
        self.df = data
        self.rf_model = None
        self.le_district = LabelEncoder()
        self.le_street = LabelEncoder()

    def run_district_clustering(self):
        # Aggregate features for territorial partitioning
        district_stats = self.df.groupby('Район').agg(
            total_cameras=('Количество видеокамер', 'sum'),
            mean_cameras=('Количество видеокамер', 'mean'),
            max_cameras=('Количество видеокамер', 'max'),
            address_count=('Адрес', 'count')
        ).reset_index()

        scaler = StandardScaler()
        scaled = scaler.fit_transform(district_stats[['total_cameras', 'mean_cameras', 'max_cameras', 'address_count']])
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        district_stats['Tier_Cluster'] = kmeans.fit_predict(scaled)
        
        # Format labels dynamically
        cluster_mapping = {0: "Standard Infrastructure Zone", 1: "Emerging Infrastructure Zone", 2: "High-Density Core Hub"}
        district_stats['Hub_Classification'] = district_stats['Tier_Cluster'].map(cluster_mapping)
        return district_stats

    def train_predictive_regressor(self):
        df_encoded = self.df.copy()
        df_encoded['District_Enc'] = self.le_district.fit_transform(df_encoded['Район'])
        df_encoded['Street_Enc'] = self.le_street.fit_transform(df_encoded['Street_Type'])
        
        X = df_encoded[['District_Enc', 'Street_Enc', 'Address_Length']]
        y = df_encoded['Количество видеокамер']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
        self.rf_model.fit(X_train, y_train)
        
        preds = self.rf_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        return rmse, r2

    def estimate_capacity(self, district_name, street_type, string_length):
        if self.rf_model is None:
            self.train_predictive_regressor()
            
        try:
            dist_val = self.le_district.transform([district_name])[0]
            street_val = self.le_street.transform([street_type])[0]
            features = np.array([[dist_val, street_val, string_length]])
            return float(self.rf_model.predict(features)[0])
        except Exception:
            return float(self.df['Количество видеокамер'].median())

# ==========================================
# STREAMLIT UI PRESENTATION LAYER
# ==========================================

st.set_page_config(page_title="SmartCity Multi-Agent Analytics", layout="wide", page_icon="🏙️")

# Custom UI styling injection
st.markdown("""
    <style>
    .metric-card {background-color: #1e2430; padding: 20px; border-radius: 10px; border-left: 5px solid #4f46e5; margin-bottom: 10px;}
    .agent-header {color: #a7f3d0; font-family: monospace; font-size: 0.95rem; margin-bottom: 15px;}
    </style>
""", unsafe_allow_html=True)

st.title("🏙️ Smart City Surveillance Analytics & Modeling Engine")
st.markdown("An Multi-Agent Autonomous Workflow powered by Machine Learning for public safety assessment.")

# Instantiate Agents
@st.cache_data
def run_ingestion_pipeline():
    ingestion_agent = DataIngestionAgent('v10_Addresses_camera_installation.csv')
    return ingestion_agent.process()

df_clean = run_ingestion_pipeline()
analytics_agent = AnalyticsAgent(df_clean)
visual_agent = VisualEngineAgent(df_clean)
ml_agent = MLAgent(df_clean)

# Sidebar Configuration
st.sidebar.title("🤖 Multi-Agent Console")
st.sidebar.markdown("<div class='agent-header'>STATUS: ALL AGENTS ACTIVE</div>", unsafe_allow_html=True)
selected_tab = st.sidebar.radio("Navigate Workspace:", ["Macro Analytics Hub", "Territorial Stratification (Clustering)", "Asset Requirement Estimation"])

# --- TAB 1: MACRO ANALYTICS HUB ---
if selected_tab == "Macro Analytics Hub":
    st.subheader("📊 Strategic Fleet Overview")
    metrics = analytics_agent.get_macro_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h4>Total Managed Nodes</h4><h2>{metrics['total_addresses']:,}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h4>Active Camera Fleet</h4><h2>{metrics['total_cameras']:,}</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h4>Mean Intensity per Node</h4><h2>{metrics['avg_cameras']:.2f}</h2></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><h4>Max Deployment Peak</h4><h2>{metrics['max_cameras']}</h2></div>", unsafe_allow_html=True)

    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns([3, 2])
    with col_chart1:
        st.plotly_chart(visual_agent.plot_district_distribution(), use_container_width=True)
    with col_chart2:
        st.plotly_chart(visual_agent.plot_street_type_distribution(), use_container_width=True)

# --- TAB 2: TERRITORIAL STRATIFICATION ---
elif selected_tab == "Territorial Stratification (Clustering)":
    st.subheader("🤖 Urban Structural Segmentation via Unsupervised K-Means")
    st.markdown("The **ML Pipeline Agent** compiled district level attributes to group administrative districts into $K=3$ distinct security infrastructure tiers.")
    
    cluster_df = ml_agent.run_district_clustering()
    
    fig_cluster = px.scatter(
        cluster_df, x='address_count', y='total_cameras', size='mean_cameras',
        color='Hub_Classification', hover_name='Район',
        title="<b>District Structural Partitioning Map</b>",
        labels={'address_count': 'Total Address Coordinates', 'total_cameras': 'Total Deployed Cameras'},
        color_discrete_sequence=px.colors.qualitative.G10
    )
    fig_cluster.update_layout(template='plotly_dark')
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    st.write("### Detailed Tier Allocation Summary")
    st.dataframe(cluster_df[['Район', 'address_count', 'total_cameras', 'mean_cameras', 'Hub_Classification']].sort_values(by='total_cameras', ascending=False), use_container_width=True)

# --- TAB 3: ASSET REQUIREMENT ESTIMATION ---
elif selected_tab == "Asset Requirement Estimation":
    st.subheader("🔮 Predictive Asset Infrastructure Capacity Modeling")
    st.markdown("This module triggers a **Supervised Random Forest Regressor** to predict structural safety sensor counts needed at any hypothetical location node.")
    
    with st.spinner("Training predictive models..."):
        rmse, r2 = ml_agent.train_predictive_regressor()
        
    st.success(f"Model trained successfully. Optimization Metrics Evaluation: RMSE = ${rmse:.4f}$ | $R^2$ Score = ${r2:.4f}$")
    
    st.markdown("#### Dynamic Real-Time Parameter Inference Simulator")
    input_col1, input_col2, input_col3 = st.columns(3)
    
    with input_col1:
        selected_district = st.selectbox("Select Target District (Район):", sorted(df_clean['Район'].unique()))
    with input_col2:
        selected_street = st.selectbox("Select Built Environment Category:", sorted(df_clean['Street_Type'].unique()))
    with input_col3:
        addr_text = st.text_input("Enter Target Local Address String for Feature Parsing:", "Невский проспект, дом 100")
        
    if st.button("Execute Estimator Inference"):
        calculated_pred = ml_agent.estimate_capacity(selected_district, selected_street, len(addr_text))
        st.markdown("---")
        st.markdown(f"### 🎯 Optimal Infrastructure Allocation Estimate: **{int(np.ceil(calculated_pred))}** Cameras")
        st.info("💡 Insight: The calculation evaluates historical administrative trends alongside textual variations in addressing schemas to assign spatial baseline metrics.")
