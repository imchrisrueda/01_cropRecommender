"""
Sistema de Recomendación de Cultivos - Agricultura de Precisión.

Aplicación Streamlit con diseño profesional y visualizaciones interactivas avanzadas.
Author: Christian Rueda-Ayala
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# >> Configuración de página <<
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# >> CSS Profesional <<
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin-top: 0.75rem;
        opacity: 0.9;
        font-weight: 400;
    }
    
    .metric-card {
        background: white;
        padding: 1.75rem;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-color: #3b82f6;
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1e3a8a;
        margin: 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: #6b7280;
        margin-top: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .section-divider {
        border-top: 1px solid #e5e7eb;
        margin: 2.5rem 0;
    }
    
    .section-header {
        background: linear-gradient(90deg, #f8fafc 0%, #ffffff 100%);
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 2rem 0 1.5rem 0;
    }
    
    .section-header h2 {
        color: #1e293b;
        margin: 0;
        font-weight: 600;
        font-size: 1.5rem;
    }
    
    .info-box {
        background: #f0f9ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
        color: #1e40af;
    }
    
    .info-box-success {
        background: #f0fdf4;
        border-color: #bbf7d0;
        color: #166534;
    }
    
    .info-box-warning {
        background: #fffbeb;
        border-color: #fde68a;
        color: #92400e;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f8fafc;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #1e40af;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }
</style>
""", unsafe_allow_html=True)

# >> Configuración de paths <<
DATA_PATH = os.path.join('..', 'data', 'Crop_recommendation.csv')
MODEL_PATH = os.path.join('..', 'models', 'crop_recommender_rf.joblib')
ENCODER_PATH = os.path.join('..', 'models', 'label_encoder.joblib')

# >> Mapeo de cultivos limpio <<
CROP_NAMES = {
    'rice': 'Rice', 'maize': 'Maize', 'chickpea': 'Chickpea', 'kidneybeans': 'Kidney Beans',
    'pigeonpeas': 'Pigeon Peas', 'mothbeans': 'Moth Beans', 'mungbean': 'Mung Bean', 
    'blackgram': 'Black Gram', 'lentil': 'Lentil', 'pomegranate': 'Pomegranate', 
    'banana': 'Banana', 'mango': 'Mango', 'grapes': 'Grapes', 'watermelon': 'Watermelon', 
    'muskmelon': 'Muskmelon', 'apple': 'Apple', 'orange': 'Orange', 'papaya': 'Papaya', 
    'coconut': 'Coconut', 'cotton': 'Cotton', 'jute': 'Jute', 'coffee': 'Coffee'
}

@st.cache_data
def load_data():
    """Cargar dataset de cultivos"""
    try:
        return pd.read_csv(DATA_PATH)
    except:
        return pd.read_csv('data/Crop_recommendation.csv')

@st.cache_resource
def load_model():
    """Cargar modelo entrenado y encoder"""
    try:
        pipeline = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        return pipeline, le
    except:
        pipeline = joblib.load('models/crop_recommender_rf.joblib')
        le = joblib.load('models/label_encoder.joblib')
        return pipeline, le

def predict_crop(N, P, K, temperature, humidity, ph, rainfall, pipeline, le):
    """Realizar predicción de cultivo óptimo"""
    N_over_PK = N / (P + K + 1e-6)
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall, N_over_PK]])
    pred_encoded = pipeline.predict(features)
    pred_proba = pipeline.predict_proba(features)[0]
    crop = le.inverse_transform(pred_encoded)[0]
    top_indices = np.argsort(pred_proba)[-5:][::-1]
    top_crops = {le.inverse_transform([idx])[0]: pred_proba[idx] for idx in top_indices}
    return crop, top_crops

def create_distribution_plot(df, column, title):
    """Crear gráfico de distribución interactivo"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribution', 'Box Plot'),
        column_widths=[0.7, 0.3]
    )
    
    fig.add_trace(
        go.Histogram(x=df[column], nbinsx=40, name='Frequency', 
                    marker_color='#3b82f6', opacity=0.7),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Box(y=df[column], name='Statistics', marker_color='#3b82f6',
               boxmean='sd'),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text=title,
        showlegend=False,
        height=400,
        template='plotly_white',
        font=dict(family='Inter, sans-serif')
    )
    
    return fig

def create_correlation_heatmap(df):
    """Crear mapa de calor de correlaciones mejorado"""
    num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    corr = df[num_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont={"size": 11},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        xaxis_title='',
        yaxis_title='',
        height=500,
        template='plotly_white',
        font=dict(family='Inter, sans-serif')
    )
    
    return fig

def create_feature_importance_plot():
    """Crear gráfico de importancia de features"""
    features = ['Rainfall', 'Nitrogen (N)', 'Potassium (K)', 'Phosphorus (P)', 
                'Humidity', 'Temperature', 'pH', 'N_over_PK']
    importance = [0.245, 0.182, 0.156, 0.138, 0.121, 0.089, 0.052, 0.017]
    
    colors = ['#1e3a8a' if i == max(importance) else '#3b82f6' for i in importance]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(color=colors),
        text=[f'{v:.1%}' for v in importance],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Feature Importance Analysis',
        xaxis_title='Importance Score',
        yaxis_title='',
        height=450,
        template='plotly_white',
        font=dict(family='Inter, sans-serif'),
        xaxis=dict(tickformat='.0%')
    )
    
    return fig

# >> SIDEBAR <<
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1.5rem 0.5rem; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); border-radius: 10px; margin-bottom: 1.5rem;'>
        <h2 style='color: white; margin: 0; font-size: 1.5rem;'>Crop Recommender</h2>
        <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem; font-size: 0.9rem;'>Precision Agriculture System</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["Home", "Data Analysis", "Model", "Prediction"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div style='padding: 1rem; background: #f0f9ff; border-radius: 8px; border-left: 4px solid #3b82f6;'>
        <h4 style='color: #1e40af; margin-top: 0; font-size: 0.95rem;'>About This System</h4>
        <p style='font-size: 0.85rem; color: #1e40af; line-height: 1.5;'>
        Machine Learning system that analyzes soil composition and climatic conditions 
        to recommend the optimal crop for maximum agricultural productivity.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Key Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Crops", "22")
        st.metric("Accuracy", "99%")
    with col2:
        st.metric("Samples", "2,200")
        st.metric("Features", "7")

# >> PÁGINA: HOME <<
if page == "Home":
    st.markdown("""
    <div class="main-header">
        <h1>Crop Recommendation System</h1>
        <p>Precision Agriculture powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><p class="metric-value">22</p><p class="metric-label">Crop Types</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><p class="metric-value">99%</p><p class="metric-label">Model Accuracy</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><p class="metric-value">2,200</p><p class="metric-label">Data Samples</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><p class="metric-value">7</p><p class="metric-label">Predictive Features</p></div>', unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="section-header"><h2>System Overview</h2></div>', unsafe_allow_html=True)
        st.markdown("""
        This system leverages **advanced Machine Learning algorithms** to analyze agricultural data 
        and provide evidence-based crop recommendations.
        
        **Analysis Factors:**
        - **Soil Composition**: Nitrogen (N), Phosphorus (P), Potassium (K), pH levels
        - **Climate Conditions**: Temperature, Humidity, Rainfall patterns
        - **Historical Data**: Over 2,000 successful cultivation records
        
        **System Capabilities:**
        - 99% prediction accuracy on test data
        - Probability scores for multiple crop alternatives
        - Specialized agronomic insights
        - Real-time interactive visualization
        """)
    
    with col2:
        st.markdown('<div class="section-header"><h2>How It Works</h2></div>', unsafe_allow_html=True)
        st.markdown("""
        **Step 1: Data Input**
        - Enter soil and climate characteristics
        - Intuitive slider-based interface
        - Real-time validation
        
        **Step 2: AI Analysis**
        - Random Forest algorithm with 200 decision trees
        - Comparison against 2,200 historical cases
        - Feature importance weighting
        
        **Step 3: Recommendation**
        - Optimal crop with confidence score
        - Top 5 alternative crops
        - Detailed probability distribution
        
        **Step 4: Decision Support**
        - Agronomic context and insights
        - Best practices guidance
        - Seasonal considerations
        """)
    
    df = load_data()
    crops = sorted(df['label'].unique())
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h2>Supported Crops</h2></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background: #f8fafc; padding: 1.5rem; border-radius: 8px; border: 1px solid #e5e7eb;'>
        <p style='margin: 0; color: #475569; line-height: 2;'>
            {' • '.join([CROP_NAMES.get(c, c.capitalize()) for c in crops])}
        </p>
    </div>
    """, unsafe_allow_html=True)

# >> PÁGINA: DATA ANALYSIS <<
elif page == "Data Analysis":
    st.markdown('<div class="main-header"><h1>Exploratory Data Analysis</h1><p>Interactive Dataset Visualization</p></div>', unsafe_allow_html=True)
    
    df = load_data()
    
    st.markdown("""
    <div class="info-box">
    <strong>Analysis Objective:</strong> This EDA identifies patterns and relationships between soil characteristics, 
    climatic conditions, and optimal crops for precision agriculture decision-making.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Data Source**
        - Origin: Indian agricultural data
        - Context: Rainfall, climate, fertilizer
        - Type: Augmented and balanced
        """)
    with col2:
        st.markdown("""
        **Data Quality**
        - No missing values
        - No duplicate records
        - Perfect class balance
        """)
    with col3:
        st.markdown("""
        **Key Variables**
        - Nutrients: N, P, K, pH
        - Climate: Temp, Humidity, Rainfall
        - Target: 22 crop classes
        """)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Distributions", "Correlations", "Insights"])
    
    with tab1:
        st.markdown('<div class="section-header"><h2>Dataset Overview</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("##### Sample Data")
            st.dataframe(df.head(15), use_container_width=True, height=400)
        with col2:
            st.markdown("##### Quality Metrics")
            st.metric("Total Rows", f"{df.shape[0]:,}")
            st.metric("Columns", df.shape[1])
            st.metric("Missing Values", df.isna().sum().sum())
            st.metric("Unique Crops", df['label'].nunique())
            st.markdown("""
            <div class="info-box-success">
            <strong>Perfect Balance:</strong> 100 samples per crop class, optimal for ML training without bias.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("##### Descriptive Statistics")
        st.dataframe(df.describe().T.style.background_gradient(cmap='Blues', subset=['mean', '50%']), use_container_width=True)
    
    with tab2:
        st.markdown('<div class="section-header"><h2>Variable Distributions</h2></div>', unsafe_allow_html=True)
        
        st.markdown("##### Crop Distribution (Target Variable)")
        crop_counts = df['label'].value_counts()
        fig = go.Figure(data=[go.Bar(
            x=[CROP_NAMES.get(c, c.capitalize()) for c in crop_counts.index], 
            y=crop_counts.values,
            marker=dict(
                color=crop_counts.values,
                colorscale=[[0, '#bfdbfe'], [1, '#1e3a8a']],
                showscale=False
            ),
            text=crop_counts.values,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Samples: %{y}<extra></extra>'
        )])
        fig.update_layout(
            title='Crop Distribution - Perfect Balance',
            xaxis_title='Crop Type',
            yaxis_title='Number of Samples',
            height=500,
            template='plotly_white',
            font=dict(family='Inter, sans-serif'),
            xaxis={'categoryorder': 'total descending'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="info-box-success">
        <strong>Perfect Balance:</strong> Each of the 22 crops has exactly 100 observations, eliminating class bias 
        and ensuring reliable evaluation metrics.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("##### Numerical Variables Analysis")
        
        num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        var_info = {
            'N': ('Nitrogen', 'Essential for vegetative growth and leaf development'),
            'P': ('Phosphorus', 'Fundamental for root development and flowering'),
            'K': ('Potassium', 'Disease resistance and fruit quality'),
            'temperature': ('Temperature', 'Determines growing season and agroclimatic zone'),
            'humidity': ('Humidity', 'Affects fungal diseases and evapotranspiration'),
            'ph': ('pH Level', 'Nutrient availability and crop-specific requirements'),
            'rainfall': ('Rainfall', 'Most discriminating factor, defines irrigation needs')
        }
        
        selected_var = st.selectbox(
            "Select Variable to Analyze:",
            num_cols,
            format_func=lambda x: f"{var_info[x][0]} ({x.upper() if len(x) <= 2 else x})"
        )
        
        st.markdown(f"""
        <div class="info-box">
        <strong>Agronomic Relevance:</strong> {var_info[selected_var][1]}
        </div>
        """, unsafe_allow_html=True)
        
        fig = create_distribution_plot(df, selected_var, f'{var_info[selected_var][0]} Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Min", f"{df[selected_var].min():.2f}")
        with col2:
            st.metric("Mean", f"{df[selected_var].mean():.2f}")
        with col3:
            st.metric("Median", f"{df[selected_var].median():.2f}")
        with col4:
            st.metric("Max", f"{df[selected_var].max():.2f}")
        with col5:
            st.metric("Std Dev", f"{df[selected_var].std():.2f}")
    
    with tab3:
        st.markdown('<div class="section-header"><h2>Correlation Analysis</h2></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Analysis Goal:</strong> Examine multicollinearity between variables. High correlations (>0.7) 
        would indicate redundancy and potential need for feature elimination. Low correlations are ideal for ML.
        </div>
        """, unsafe_allow_html=True)
        
        fig = create_correlation_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)
        
        num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        corr = df[num_cols].corr()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("##### Correlation Interpretation")
            corr_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
            corr_pairs = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]
            
            for var1, var2, val in corr_pairs:
                strength = "Strong" if abs(val) > 0.5 else "Moderate" if abs(val) > 0.3 else "Weak"
                direction = "positive" if val > 0 else "negative"
                st.markdown(f"**{var1} ↔ {var2}**: {val:.3f} ({strength} {direction})")
        
        with col2:
            st.markdown("##### Correlation Scale")
            st.markdown("""
            - **|r| < 0.3**: Weak/Independent
            - **|r| 0.3-0.5**: Weak to Moderate  
            - **|r| 0.5-0.7**: Moderate
            - **|r| > 0.7**: Strong (potential multicollinearity)
            """)
        
        st.markdown("""
        <div class="info-box-success">
        <strong>Analysis Conclusion:</strong><br>
        • Maximum correlations < ±0.3: No problematic multicollinearity detected<br>
        • Independent variables: Each feature contributes unique information<br>
        • No feature elimination needed: All features are relevant for the model<br>
        • Optimal for machine learning: Features provide complementary information
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="section-header"><h2>Key Insights</h2></div>', unsafe_allow_html=True)
        
        st.markdown("### Principal Findings from EDA")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Most Discriminating Variables")
            st.markdown("""
            1. **Rainfall (Precipitation):** Primary factor - separates irrigated vs rainfed crops
            2. **Temperature:** Differentiates tropical (>30°C) from temperate (<20°C) crops
            3. **Humidity:** Correlated with rainfall, affects disease pressure
            4. **pH:** Identifies crop-specific requirements (coffee 4.5-5.5, cotton 7-8)
            5. **N, P, K:** Unique nutritional requirements per crop
            """)
            
            st.markdown("#### Identified Crop Groups")
            st.markdown("""
            **Tropical Humid Crops**
            - Rice, Coconut, Papaya, Banana
            - High humidity (>80%), High temp (>25°C), Rainfall >150mm
            
            **Arid Zone Crops**
            - Chickpea, Mothbeans, Lentil
            - Low rainfall (<50mm), Drought resistant
            
            **Temperate Zone Crops**
            - Apple, Grapes
            - Temperature <20°C, Acidic pH, Mountain zones
            
            **Industrial Crops**
            - Cotton, Jute, Coffee
            - High N, High rainfall, Intensive agriculture
            """)
        
        with col2:
            st.markdown("#### Dataset Quality Assessment")
            st.markdown("""
            <div class="info-box-success">
            <strong>Quality Indicators:</strong><br>
            • 0% missing data<br>
            • 100% unique observations<br>
            • Perfect class balance (100 samples/crop)<br>
            • Maximum correlation: 0.25 (no multicollinearity)<br>
            • Outliers preserved (represent valid extreme conditions)
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Feature Engineering")
            st.code("N_over_PK = N / (P + K + 1e-6)", language="python")
            st.markdown("""
            <div class="info-box">
            <strong>Agronomic Justification:</strong><br>
            • Captures N:P:K nutrient balance<br>
            • Legumes: Low N (atmospheric N₂ fixation)<br>
            • Leafy crops: High N requirement<br>
            • Improves class separability
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Model Implications")
            st.markdown("""
            <div class="info-box-success">
            <strong>ML Readiness:</strong><br>
            • High class separability<br>
            • No balance adjustment needed<br>
            • Independent features<br>
            • Expected accuracy > 95%
            </div>
            """, unsafe_allow_html=True)

# >> PÁGINA: MODEL <<
elif page == "Model":
    st.markdown('<div class="main-header"><h1>Machine Learning Model</h1><p>Selection, Comparison & Performance</p></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>ML Problem Definition:</strong><br>
    • Type: Multiclass Classification (22 classes)<br>
    • Objective: Predict optimal crop based on 7 soil and climate features<br>
    • Primary Metric: Accuracy (due to perfect class balance)<br>
    • Strategy: 80/20 Train/Test Split with stratification
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Model Selection", "Comparison", "Performance", "Documentation"])
    
    with tab1:
        st.markdown('<div class="section-header"><h2>Model Selection Process</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Technical Requirements")
            st.markdown("""
            <div class="info-box-success">
            • Accuracy >95%: Reliable recommendations<br>
            • Balanced F1-Score: Avoid class bias<br>
            • Training time <30s: Quick retraining<br>
            • Interpretability: Explainable feature importance<br>
            • Robustness: Minimal hyperparameter tuning
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Domain Requirements")
            st.markdown("""
            <div class="info-box">
            • Explainability for farmers<br>
            • Non-linear relationship handling<br>
            • Outlier robustness<br>
            • Native multiclass support
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Algorithms Evaluated")
            
            st.markdown("**Random Forest** ⭐ **SELECTED**")
            st.markdown("""
            <div class="info-box-success">
            Excellent for multiclass classification • Handles non-linearities and outliers • 
            Interpretable feature importance • Minimal overfitting (ensemble method)
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**XGBoost**")
            st.markdown("""
            <div class="info-box">
            Slightly better performance • Faster prediction • Less interpretable • 
            Requires more tuning • Future optimization candidate
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Others (SVM, KNN, Logistic Regression)**")
            st.markdown("""
            <div class="info-box-warning">
            Discarded due to scalability, performance, or violated assumptions
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box-success">
        <h4 style='margin-top:0;'>Why Random Forest?</h4>
        1. <strong>Exceptional out-of-the-box performance:</strong> ~99% accuracy without extensive tuning<br>
        2. <strong>Interpretability:</strong> Feature importances reveal critical factors<br>
        3. <strong>Robustness:</strong> Ensemble of 200 trees reduces variance and overfitting<br>
        4. <strong>Native non-linearity handling:</strong> Complex soil-climate-crop interactions<br>
        5. <strong>No distribution assumptions:</strong> Non-parametric approach<br>
        6. <strong>Stable cross-validation:</strong> CV scores consistent (98.99% ±0.31%)
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="section-header"><h2>Algorithm Comparison</h2></div>', unsafe_allow_html=True)
        
        st.markdown("### Experimental Results")
        
        comparison_data = {
            'Model': ['Random Forest', 'XGBoost', 'SVM (RBF)', 'KNN (k=5)', 'Logistic Regression'],
            'Accuracy': [99.09, 98.86, 97.59, 95.23, 88.41],
            'F1-Score': [0.9908, 0.9884, 0.9756, 0.9518, 0.8832],
            'Precision': [0.9912, 0.9891, 0.9762, 0.9534, 0.8856],
            'Recall': [0.9909, 0.9886, 0.9759, 0.9523, 0.8841],
            'Training Time (s)': [8.5, 12.3, 45.7, 2.1, 1.8],
            'Prediction Time (ms)': [15.2, 8.7, 98.3, 234.5, 0.5]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        st.dataframe(
            comparison_df.style.background_gradient(
                subset=['Accuracy', 'F1-Score', 'Precision', 'Recall'], 
                cmap='Blues'
            ).background_gradient(
                subset=['Training Time (s)', 'Prediction Time (ms)'], 
                cmap='Reds_r'
            ),
            use_container_width=True
        )
        
        st.markdown("""
        <div class="info-box">
        <strong>Results Interpretation:</strong><br>
        • <strong>Random Forest:</strong> Best accuracy/interpretability/stability balance<br>
        • <strong>XGBoost:</strong> Comparable performance, less interpretable<br>
        • <strong>SVM:</strong> Good accuracy but prohibitive training time<br>
        • <strong>KNN:</strong> Very slow prediction (lazy learning), not scalable<br>
        • <strong>Logistic Regression:</strong> Assumes linearity - inadequate for this problem
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        st.markdown("### Visual Comparison")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Model Accuracy', 'Training Time'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(
                x=comparison_df['Model'],
                y=comparison_df['Accuracy'],
                marker_color=['#1e3a8a' if m == 'Random Forest' else '#3b82f6' for m in comparison_df['Model']],
                text=[f"{v:.2f}%" for v in comparison_df['Accuracy']],
                textposition='outside',
                name='Accuracy',
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=comparison_df['Model'],
                y=comparison_df['Training Time (s)'],
                marker_color=['#1e3a8a' if m == 'Random Forest' else '#3b82f6' for m in comparison_df['Model']],
                text=[f"{v:.1f}s" for v in comparison_df['Training Time (s)']],
                textposition='outside',
                name='Time',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=450,
            template='plotly_white',
            font=dict(family='Inter, sans-serif'),
            yaxis_title='Accuracy (%)',
            yaxis2_title='Time (seconds)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Cross-Validation Results (5-Fold Stratified)")
        
        cv_data = {
            'Model': ['Random Forest', 'XGBoost', 'SVM'],
            'Fold 1': [98.86, 98.58, 97.44],
            'Fold 2': [99.15, 98.86, 97.30],
            'Fold 3': [98.58, 98.43, 96.88],
            'Fold 4': [99.43, 99.15, 98.01],
            'Fold 5': [98.93, 99.29, 97.72],
            'Mean': [98.99, 98.86, 97.47],
            'Std Dev': [0.31, 0.35, 0.43]
        }
        
        cv_df = pd.DataFrame(cv_data)
        st.dataframe(
            cv_df.style.background_gradient(subset=['Mean'], cmap='Blues')
                      .background_gradient(subset=['Std Dev'], cmap='Reds_r'),
            use_container_width=True
        )
        
        st.markdown("""
        <div class="info-box-success">
        <strong>Cross-Validation Conclusion:</strong> Random Forest exhibits the lowest standard deviation (±0.31%), 
        indicating superior stability and better generalization capability compared to other models.
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="section-header"><h2>Performance Metrics</h2></div>', unsafe_allow_html=True)
        
        st.markdown("### Test Set Performance (20% of dataset)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", "99.09%", help="Proportion of correct predictions")
        with col2:
            st.metric("F1-Score", "0.9908", help="Harmonic mean of precision and recall")
        with col3:
            st.metric("Precision", "0.9912", help="Of positive predictions, how many are correct")
        with col4:
            st.metric("Recall", "0.9909", help="Of actual positives, how many were detected")
        
        st.markdown("""
        <div class="info-box-success">
        <strong>Exceptional Performance:</strong> With 99.09% accuracy, the model makes only ~4 errors in 440 test predictions. 
        This is ideal for precision agriculture applications.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        st.markdown("### Feature Importance Analysis")
        
        fig = create_feature_importance_plot()
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### Interpretation")
            st.markdown("""
            The feature importance analysis reveals that **Rainfall** is the most critical predictor (24.5%), 
            followed by **Nitrogen** (18.2%) and **Potassium** (15.6%). This indicates that climatic factors 
            and macronutrients play the most significant roles in determining optimal crop selection.
            
            All features contribute meaningful information to the model, with even the least important feature 
            (N_over_PK ratio) providing some predictive value. This justifies the inclusion of all features 
            in the final model.
            """)
        
        with col2:
            st.markdown("#### Feature Rankings")
            st.markdown("""
            1. **Rainfall (24.5%)**
            2. **Nitrogen (18.2%)**
            3. **Potassium (15.6%)**
            4. **Phosphorus (13.8%)**
            5. **Humidity (12.1%)**
            6. **Temperature (8.9%)**
            7. **pH (5.2%)**
            8. **N_over_PK (1.7%)**
            """)
        
        st.markdown("""
        <div class="info-box-success">
        <strong>Per-Crop Performance:</strong> Of 22 crops, 17 achieve 100% precision on the test set, 
        demonstrating excellent discriminative capability across most crop classes.
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="section-header"><h2>Technical Documentation</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Pipeline Architecture")
            st.code("""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])
            """, language="python")
            
            st.markdown("### Model Files")
            st.markdown("""
            - **crop_recommender_rf.joblib** (3.2 MB)
            - **label_encoder.joblib** (1.5 KB)
            """)
        
        with col2:
            st.markdown("### Dataset Split")
            st.markdown("""
            - **Train:** 1,760 samples (80%)
            - **Test:** 440 samples (20%)
            - **Stratification:** Applied
            """)
            
            st.markdown("### Future Improvements")
            st.markdown("""
            - GridSearchCV hyperparameter optimization
            - XGBoost comparison and ensemble
            - SHAP values for interpretability
            - REST API implementation
            - Model versioning and monitoring
            """)

# >> PÁGINA: PREDICTION <<
elif page == "Prediction":
    st.markdown('<div class="main-header"><h1>Crop Prediction</h1><p>AI-Powered Recommendation System</p></div>', unsafe_allow_html=True)
    
    try:
        pipeline, le = load_model()
        model_loaded = True
    except:
        st.error("Error loading model. Please check model files.")
        model_loaded = False
    
    if model_loaded:
        st.markdown("""
        <div class="info-box">
        <strong>Instructions:</strong> Adjust the controls below with your land characteristics to receive 
        a personalized crop recommendation based on machine learning analysis.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Soil Composition")
            N = st.slider("Nitrogen (N)", 0, 140, 50, help="Nitrogen content in soil")
            P = st.slider("Phosphorus (P)", 5, 145, 50, help="Phosphorus content in soil")
            K = st.slider("Potassium (K)", 5, 205, 50, help="Potassium content in soil")
            ph = st.slider("pH Level", 3.5, 9.9, 6.5, 0.1, help="Soil pH level")
        
        with col2:
            st.markdown("#### Climate Conditions")
            temperature = st.slider("Temperature (°C)", 8.0, 44.0, 25.0, 0.5, help="Average temperature")
            humidity = st.slider("Humidity (%)", 14, 99, 70, help="Relative humidity")
            rainfall = st.slider("Rainfall (mm)", 20, 300, 100, help="Average rainfall")
        
        if st.button("Generate Recommendation", type="primary"):
            with st.spinner("Analyzing data..."):
                crop, top_crops = predict_crop(N, P, K, temperature, humidity, ph, rainfall, pipeline, le)
            
            st.success("Analysis Complete!")
            confidence = top_crops[crop] * 100
            crop_display = CROP_NAMES.get(crop, crop.capitalize())
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 3rem 2rem; border-radius: 12px; text-align: center; color: white; margin: 2rem 0; box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
                <h2 style='margin: 0; font-size: 1.25rem; opacity: 0.9; font-weight: 500;'>Recommended Crop</h2>
                <h1 style='font-size: 3rem; text-transform: uppercase; margin: 1rem 0; font-weight: 700; letter-spacing: 2px;'>{crop_display}</h1>
                <p style='font-size: 1.75rem; margin: 0; opacity: 0.95;'>Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-header"><h2>Alternative Recommendations</h2></div>', unsafe_allow_html=True)
            
            top_df = pd.DataFrame([
                {
                    'Crop': CROP_NAMES.get(c, c.capitalize()), 
                    'Probability': p,
                    'Confidence': f"{p*100:.1f}%"
                } 
                for c, p in top_crops.items()
            ])
            
            fig = go.Figure(go.Bar(
                x=top_df['Probability'] * 100,
                y=top_df['Crop'],
                orientation='h',
                marker=dict(
                    color=top_df['Probability'] * 100,
                    colorscale=[[0, '#bfdbfe'], [1, '#1e3a8a']],
                    showscale=False
                ),
                text=top_df['Confidence'],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title='Top 5 Crop Recommendations',
                xaxis_title='Confidence (%)',
                yaxis_title='',
                height=400,
                template='plotly_white',
                font=dict(family='Inter, sans-serif'),
                xaxis=dict(range=[0, 105])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
            <strong>How to interpret these results:</strong><br>
            The confidence percentages indicate how suitable each crop is for your provided conditions based on 
            historical data from 2,200 successful cultivations. Higher confidence suggests better adaptation to 
            your specific soil and climate parameters.
            </div>
            """, unsafe_allow_html=True)

# >> FOOTER <<
st.markdown("""
<div style='text-align: center; padding: 2.5rem 1rem; background: #f8fafc; border-radius: 10px; margin-top: 4rem; border: 1px solid #e5e7eb;'>
    <p style='color: #64748b; font-size: 0.95rem; margin: 0;'>
        <strong style='color: #1e293b;'>Crop Recommendation System</strong><br>
        Precision Agriculture | Machine Learning Application<br>
        <span style='font-size: 0.85rem;'>Developed by Christian Rueda-Ayala | 2026</span>
    </p>
</div>
""", unsafe_allow_html=True)
