"""
Sistema de Recomendación de Cultivos - Agricultura de Precisión.

Aplicación Streamlit con diseño mejorado, secciones claras y visualización atractiva.
"""

# >> Imports <<
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
    page_title="🌾 Recomendador de Cultivos",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# >> CSS Personalizado <<
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #2ecc71;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2ecc71;
        margin: 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .section-divider {
        border-top: 3px solid #27ae60;
        margin: 2rem 0;
        opacity: 0.3;
    }
    
    .section-header {
        background: linear-gradient(90deg, #ecf0f1 0%, #f8f9fa 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1.5rem 0;
    }
    
    .section-header h2 {
        color: #2c3e50;
        margin: 0;
        font-weight: 600;
    }
    
    .info-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box-blue {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
    }
    
    .info-box-yellow {
        background: #fff3cd;
        border: 1px solid #ffeeba;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2ecc71;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# >> Configuración de paths <<
DATA_PATH = os.path.join('data', 'Crop_recommendation.csv')
MODEL_PATH = os.path.join('models', 'crop_recommender_rf.joblib')
ENCODER_PATH = os.path.join('models', 'label_encoder.joblib')

# >> Iconos de cultivos <<
CROP_ICONS = {
    'rice': '🌾', 'maize': '🌽', 'chickpea': '🫘', 'kidneybeans': '🫘',
    'pigeonpeas': '🫘', 'mothbeans': '��', 'mungbean': '🫘', 'blackgram': '🫘',
    'lentil': '🫘', 'pomegranate': '🥭', 'banana': '🍌', 'mango': '🥭',
    'grapes': '🍇', 'watermelon': '🍉', 'muskmelon': '🍈', 'apple': '🍎',
    'orange': '🍊', 'papaya': '🫐', 'coconut': '🥥', 'cotton': '☁️',
    'jute': '🌿', 'coffee': '☕'
}

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    pipeline = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    return pipeline, le

def predict_crop(N, P, K, temperature, humidity, ph, rainfall, pipeline, le):
    N_over_PK = N / (P + K + 1e-6)
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall, N_over_PK]])
    pred_encoded = pipeline.predict(features)
    pred_proba = pipeline.predict_proba(features)[0]
    crop = le.inverse_transform(pred_encoded)[0]
    top_indices = np.argsort(pred_proba)[-5:][::-1]
    top_crops = {le.inverse_transform([idx])[0]: pred_proba[idx] for idx in top_indices}
    return crop, top_crops

# >> SIDEBAR <<
with st.sidebar:
    st.title("🌾 Recomendación de Cultivos")
    st.markdown("---")
    page = st.radio("Navegación", ["🏠 Inicio", "📊 EDA", "🤖 Modelo", "🔮 Predicción"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style='padding: 1rem; background: #e8f5e9; border-radius: 10px;'>
        <h4 style='color: #2e7d32; margin-top: 0;'>💡 Sistema de IA</h4>
        <p style='font-size: 0.85rem; color: #558b2f;'>
        Agricultura de precisión que utiliza Machine Learning para recomendar 
        el cultivo óptimo según características del suelo y clima.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 📈 Métricas")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cultivos", "22")
        st.metric("Accuracy", "99%")
    with col2:
        st.metric("Muestras", "2,200")
        st.metric("Features", "7")

# >> PÁGINA: INICIO <<
if page == "🏠 Inicio":
    st.markdown("""
    <div class="main-header">
        <h1>🌾 Sistema de Recomendación de Cultivos</h1>
        <p>Agricultura de Precisión - Upgrade Bootcamp</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><p class="metric-value">22</p><p class="metric-label">Cultivos Analizados</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><p class="metric-value">99%</p><p class="metric-label">Accuracy del Modelo</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><p class="metric-value">2,200</p><p class="metric-label">Observaciones</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><p class="metric-value">7</p><p class="metric-label">Variables Predictivas</p></div>', unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="section-header"><h2>🎯 ¿Qué Hace Este Sistema?</h2></div>', unsafe_allow_html=True)
        st.markdown("""
        Este sistema utiliza **Machine Learning avanzado** para analizar las características de tu terreno.
        
        **Analiza:**
        - 🌱 **Composición del Suelo**: N, P, K, pH
        - 🌡️ **Condiciones Climáticas**: Temperatura, Humedad, Precipitación
        - 📊 **Patrones Históricos**: +2,000 siembras exitosas
        
        **Te Ofrece:**
        - ✅ Recomendación óptima con 99% de precisión
        - 📈 Probabilidades para múltiples cultivos
        - 💡 Información agronómica especializada
        """)
    
    with col2:
        st.markdown('<div class="section-header"><h2>🚀 Cómo Funciona</h2></div>', unsafe_allow_html=True)
        st.markdown("""
        1. **📝 Ingresa los Datos**
           - Características de suelo y clima
           - Controles deslizantes intuitivos
        
        2. **🤖 Análisis con IA**
           - Random Forest con 200 árboles
           - Comparación con 2,200 casos históricos
        
        3. **🎯 Recibe Recomendación**
           - Cultivo óptimo con confianza
           - Top-5 alternativas
        
        4. **📚 Consulta la Guía**
           - Épocas de siembra
           - Manejo de fertilizantes
        """)
    
    df = load_data()
    crops = sorted(df['label'].unique())
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h2>🌾 Cultivos Disponibles</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    for i, crop in enumerate(crops):
        icon = CROP_ICONS.get(crop, '🌱')
        if i % 4 == 0:
            col1.markdown(f"{icon} {crop.capitalize()}")
        elif i % 4 == 1:
            col2.markdown(f"{icon} {crop.capitalize()}")
        elif i % 4 == 2:
            col3.markdown(f"{icon} {crop.capitalize()}")
        else:
            col4.markdown(f"{icon} {crop.capitalize()}")

# >> PÁGINA: EDA <<
elif page == "📊 EDA":
    st.markdown('<div class="main-header"><h1>📊 Análisis Exploratorio de Datos</h1><p>Visualización Interactiva del Dataset</p></div>', unsafe_allow_html=True)
    
    df = load_data()
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset", "📈 Distribuciones", "🔗 Correlaciones", "🌾 Perfiles"])
    
    with tab1:
        st.markdown('<div class="section-header"><h2>Vista General</h2></div>', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df.head(15), use_container_width=True, height=400)
        with col2:
            st.metric("Total Filas", f"{df.shape[0]:,}")
            st.metric("Columnas", df.shape[1])
            st.metric("Valores Nulos", df.isna().sum().sum())
            st.metric("Cultivos Únicos", df['label'].nunique())
            st.success("✅ Dataset balanceado")
        st.dataframe(df.describe().T.style.background_gradient(cmap='Greens'), use_container_width=True)
    
    with tab2:
        st.markdown('<div class="section-header"><h2>Distribuciones</h2></div>', unsafe_allow_html=True)
        crop_counts = df['label'].value_counts()
        fig = go.Figure(data=[go.Bar(x=crop_counts.index, y=crop_counts.values, marker=dict(color=crop_counts.values, colorscale='Greens', showscale=True), text=crop_counts.values, textposition='auto')])
        fig.update_layout(title='Distribución de Cultivos', xaxis_title='Cultivo', yaxis_title='Cantidad', height=500, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        selected_var = st.selectbox("Variable:", num_cols)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=selected_var, nbins=40, marginal='box', color_discrete_sequence=['#2ecc71'])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.violin(df, y=selected_var, box=True, color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="section-header"><h2>Matriz de Correlación</h2></div>', unsafe_allow_html=True)
        num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="info-box"><h4>✅ Conclusión</h4><p>Correlaciones bajas (<±0.3) indican independencia - ideal para ML.</p></div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="section-header"><h2>Perfiles de Cultivos</h2></div>', unsafe_allow_html=True)
        selected_crop = st.selectbox("Cultivo:", sorted(df['label'].unique()), format_func=lambda x: f"{CROP_ICONS.get(x, '🌱')} {x.capitalize()}")
        crop_data = df[df['label'] == selected_crop]
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(crop_data[num_cols].describe().T.style.background_gradient(cmap='Greens'), use_container_width=True)
        with col2:
            comparison = pd.DataFrame({'Global': df[num_cols].median(), selected_crop: crop_data[num_cols].median()})
            st.dataframe(comparison, use_container_width=True)
        
        normalized = (crop_data[num_cols].median() - df[num_cols].min()) / (df[num_cols].max() - df[num_cols].min())
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=normalized.values, theta=num_cols, fill='toself', line=dict(color='#2ecc71', width=2)))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])), height=500)
        st.plotly_chart(fig, use_container_width=True)

# >> PÁGINA: MODELO <<
elif page == "🤖 Modelo":
    st.markdown('<div class="main-header"><h1>🤖 Modelos de Machine Learning</h1><p>Tecnología y Performance</p></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Comparación", "🎯 Performance", "📚 Docs"])
    
    with tab1:
        st.markdown('<div class="section-header"><h2>Comparación de Algoritmos</h2></div>', unsafe_allow_html=True)
        comparison_data = {
            'Modelo': ['Random Forest', 'XGBoost', 'SVM'],
            'Accuracy (CV)': [0.9899, 0.9886, 0.9759],
            'F1-Score': [0.9898, 0.9884, 0.9756],
            'Tiempo (s)': [8.5, 12.3, 45.7]
        }
        st.dataframe(pd.DataFrame(comparison_data).style.background_gradient(subset=['Accuracy (CV)', 'F1-Score'], cmap='Greens'), use_container_width=True)
        st.markdown('<div class="info-box"><h4>🏆 Seleccionado: Random Forest</h4><p>Performance equivalente, mayor estabilidad, mejor interpretabilidad.</p></div>', unsafe_allow_html=True)
    
    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", "99.09%")
        with col2:
            st.metric("F1-Score", "0.9908")
        with col3:
            st.metric("Precision", "0.9912")
        with col4:
            st.metric("Recall", "0.9909")
        
        feature_importance = {'rainfall': 0.245, 'N': 0.182, 'K': 0.156, 'P': 0.138, 'humidity': 0.121, 'temperature': 0.089, 'ph': 0.052}
        importance_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)
        fig = go.Figure(go.Bar(x=importance_df['Importance'], y=importance_df['Feature'], orientation='h', marker=dict(color=importance_df['Importance'], colorscale='Viridis'), text=[f"{v:.1%}" for v in importance_df['Importance']], textposition='auto'))
        fig.update_layout(title='Feature Importance', xaxis_title='Importancia', yaxis_title='Feature', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("""
        ### 📄 Pipeline
        ```python
        Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
        ])
        ```
        ### 📦 Archivos
        - `crop_recommender_rf.joblib` (3.2 MB)
        - `label_encoder.joblib` (1.5 KB)
        """)

# >> PÁGINA: PREDICCIÓN <<
elif page == "🔮 Predicción":
    st.markdown('<div class="main-header"><h1>🔮 Predicción de Cultivo Óptimo</h1><p>Recomendación Personalizada con IA</p></div>', unsafe_allow_html=True)
    
    try:
        pipeline, le = load_model()
        model_loaded = True
    except:
        st.error("❌ Error al cargar modelo")
        model_loaded = False
    
    if model_loaded:
        st.markdown('<div class="info-box-blue"><h4>💡 Instrucciones</h4><p>Ajusta los controles con las características de tu terreno.</p></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🌱 Composición del Suelo")
            N = st.slider("Nitrógeno (N)", 0, 140, 50)
            P = st.slider("Fósforo (P)", 5, 145, 50)
            K = st.slider("Potasio (K)", 5, 205, 50)
            ph = st.slider("pH", 3.5, 9.9, 6.5, 0.1)
        
        with col2:
            st.markdown("#### 🌡️ Condiciones Climáticas")
            temperature = st.slider("Temperatura (°C)", 8.0, 44.0, 25.0, 0.5)
            humidity = st.slider("Humedad (%)", 14, 99, 70)
            rainfall = st.slider("Precipitación (mm)", 20, 300, 100)
        
        if st.button("🌾 Predecir Cultivo Recomendado", type="primary"):
            with st.spinner("🔬 Analizando..."):
                crop, top_crops = predict_crop(N, P, K, temperature, humidity, ph, rainfall, pipeline, le)
            
            st.success("✅ Completado!")
            icon = CROP_ICONS.get(crop, '🌱')
            confidence = top_crops[crop] * 100
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white; margin: 2rem 0;'>
                <h1 style='font-size: 3rem; margin: 0;'>{icon}</h1>
                <h2>Cultivo Recomendado</h2>
                <h1 style='font-size: 2.5rem; text-transform: uppercase;'>{crop}</h1>
                <p style='font-size: 1.5rem;'>Confianza: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            top_df = pd.DataFrame([(c, p) for c, p in top_crops.items()], columns=['Cultivo', 'Probabilidad'])
            top_df['Probabilidad (%)'] = (top_df['Probabilidad'] * 100).round(2)
            top_df['Cultivo'] = top_df['Cultivo'].apply(lambda x: f"{CROP_ICONS.get(x, '🌱')} {x.capitalize()}")
            
            fig = go.Figure(go.Bar(x=top_df['Probabilidad (%)'], y=top_df['Cultivo'], orientation='h', marker=dict(color=top_df['Probabilidad (%)'], colorscale='Greens'), text=[f"{v:.2f}%" for v in top_df['Probabilidad (%)']], textposition='auto'))
            fig.update_layout(title='Top 5 Cultivos', xaxis_title='Probabilidad (%)', height=350)
            st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style='text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 3rem;'>
    <p style='color: #7f8c8d; font-size: 0.9rem;'>
        <b>🌾 Sistema de Recomendación de Cultivos</b><br>
        Agricultura de Precisión | Versión 2.0 | 2025<br>
        Desarrollado por Christian Rueda-Ayala
    </p>
</div>
""", unsafe_allow_html=True)
