# 🌾 Sistema de Recomendación de Cultivos - Agricultura de Precisión

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)](https://streamlit.io/)
[![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen)](/)

Sistema inteligente de recomendación de cultivos basado en Machine Learning que ayuda a agricultores a seleccionar el cultivo óptimo según las características del suelo y condiciones climáticas.

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Dataset](#-dataset)
- [Modelo](#-modelo)
- [Resultados](#-resultados)
- [Documentación](#-documentación)
- [Contribución](#-contribución)

---

## 🎯 Descripción

Este proyecto utiliza **Machine Learning** para recomendar cultivos basándose en:

- **Composición del Suelo:** Nitrógeno (N), Fósforo (P), Potasio (K), pH
- **Condiciones Climáticas:** Temperatura, Humedad, Precipitación

El sistema analiza **22 tipos de cultivos** diferentes y proporciona recomendaciones con **99% de precisión**.

---

## ✨ Características

✅ **Análisis Exploratorio Completo (EDA):** Visualizaciones interactivas y conclusiones del dominio  
✅ **Modelo de ML Optimizado:** Random Forest con 99% accuracy  
✅ **Aplicación Web Interactiva:** Interfaz Streamlit fácil de usar  
✅ **Documentación Exhaustiva:** EDA.md y MODEL.md detallados  
✅ **Código PEP 8:** Cumple con estándares de Python  
✅ **Reproducibilidad:** Semillas fijas y pipeline serializado

---

## 📁 Estructura del Proyecto

```
01_proyecto/
│
├── app.py                          # >> Aplicación Streamlit principal <<
├── requirements.txt                # >> Dependencias del proyecto <<
├── README.md                       # >> Este archivo <<
│
├── data/
│   ├── Crop_recommendation.csv     # >> Dataset principal <<
│   └── datacard.md                 # >> Descripción del dataset <<
│
├── notebooks/
│   ├── eda_full.ipynb              # >> Análisis exploratorio completo <<
│   └── eda.ipynb                   # >> Versión resumida <<
│
├── functions/
│   ├── __init__.py                 # >> Inicialización del módulo <<
│   └── func_util.py                # >> Funciones utilitarias (PEP 8) <<
│
├── models/
│   ├── crop_recommender_rf.joblib  # >> Modelo entrenado <<
│   └── label_encoder.joblib        # >> Codificador de etiquetas <<
│
├── reports/
│   ├── EDA.md                      # >> Documentación del EDA <<
│   ├── MODEL.md                    # >> Documentación del modelo <<
│   └── eda_report.md               # >> Reporte automático <<
│
└── prompt_style.md                 # >> Guía de estilo del proyecto <<
```

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/imchrisrueda/Bootcamp_Data_e_IA.git
cd Bootcamp_Data_e_IA/01_proyecto
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 💻 Uso

### Ejecutar Aplicación Streamlit

```bash
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`

### Ejecutar Notebook de EDA

```bash
jupyter notebook notebooks/eda_full.ipynb
```

### Usar el Modelo Directamente

```python
import joblib
import numpy as np

# >> cargar modelo <<
pipeline = joblib.load('models/crop_recommender_rf.joblib')
le = joblib.load('models/label_encoder.joblib')

# >> hacer predicción <<
# [N, P, K, temp, humidity, ph, rainfall, N_over_PK]
features = np.array([[90, 42, 43, 20.8, 82, 6.5, 202, 0.56]])
pred = pipeline.predict(features)
crop = le.inverse_transform(pred)[0]

print(f"Cultivo recomendado: {crop}")  # >> Output: rice <<
```

---

## 📊 Dataset

- **Fuente:** Datos de agricultura de India (rainfall, climate, fertilizer)
- **Tamaño:** 2,200 observaciones × 8 columnas
- **Balance:** Perfectamente balanceado (100 muestras por cultivo)
- **Calidad:** Sin valores nulos, sin duplicados

### Variables

| Variable | Tipo | Descripción | Rango |
|----------|------|-------------|-------|
| N | Numérica | Ratio de Nitrógeno | 0 - 140 |
| P | Numérica | Ratio de Fósforo | 5 - 145 |
| K | Numérica | Ratio de Potasio | 5 - 205 |
| temperature | Numérica | Temperatura (°C) | 8.8 - 43.7 |
| humidity | Numérica | Humedad (%) | 14 - 99 |
| ph | Numérica | pH del suelo | 3.5 - 9.9 |
| rainfall | Numérica | Precipitación (mm) | 20 - 298 |
| label | Categórica | Cultivo (22 clases) | - |

### Cultivos

rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee

---

## 🤖 Modelo

### Algoritmo: Random Forest Classifier

**Parámetros:**
- `n_estimators`: 200
- `random_state`: 42
- `n_jobs`: -1

**Pipeline:**
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier())
])
```

### Feature Importance

| Feature | Importancia | Interpretación |
|---------|-------------|----------------|
| rainfall | 24.5% | Factor más crítico |
| N | 18.2% | Nutriente principal |
| K | 15.6% | Resistencia |
| P | 13.8% | Desarrollo radicular |
| humidity | 12.1% | Condición climática |
| temperature | 8.9% | Zona agroclimática |
| ph | 5.2% | Cultivos específicos |

---

## 📈 Resultados

### Métricas de Evaluación

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 99.09% |
| **Precision (macro)** | 99% |
| **Recall (macro)** | 99% |
| **F1-Score (macro)** | 99% |

### Validación Cruzada (5-Fold)

- **Media:** 98.99%
- **Desviación:** ±0.31%

**Conclusión:** Excelente generalización, bajo overfitting.

---

## 📚 Documentación

### Documentos Principales

- **[EDA.md](reports/EDA.md):** Análisis exploratorio completo con insights agronómicos
- **[MODEL.md](reports/MODEL.md):** Metodología, arquitectura y evaluación del modelo
- **[datacard.md](data/datacard.md):** Descripción del dataset

### Notebooks

- **[eda_full.ipynb](notebooks/eda_full.ipynb):** Análisis completo con visualizaciones
- **[eda.ipynb](notebooks/eda.ipynb):** Versión resumida

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **Pandas:** Manipulación de datos
- **NumPy:** Operaciones numéricas
- **Scikit-learn:** Machine Learning
- **Matplotlib/Seaborn/Plotly:** Visualización
- **Streamlit:** Aplicación web
- **Jupyter:** Notebooks interactivos

---

## 🌟 Características de la Aplicación Streamlit

### 📖 Página de Inicio
- Resumen del sistema
- Métricas clave
- Lista de cultivos disponibles

### 📊 Análisis EDA
- Vista del dataset
- Distribuciones de variables
- Matriz de correlación
- Perfiles de cultivos (radar charts)

### 🤖 Información del Modelo
- Métricas de performance
- Feature importance interactivo
- Documentación técnica

### 🔮 Predicción
- Interfaz con sliders para inputs
- Predicción en tiempo real
- Top-5 cultivos con probabilidades
- Visualizaciones interactivas

---

## 🔄 Flujo de Trabajo

```
1. Carga de datos → data/Crop_recommendation.csv
2. EDA → notebooks/eda_full.ipynb
3. Feature Engineering → N_over_PK
4. Entrenamiento → Random Forest
5. Evaluación → 99% accuracy
6. Serialización → models/*.joblib
7. Deployment → app.py (Streamlit)
```

---

## 📝 Estilo de Código

Este proyecto sigue las directrices de **[prompt_style.md](prompt_style.md)**:

- ✅ Comentarios con formato: `# >> descripción <<`
- ✅ Funciones con docstrings PEP 8
- ✅ Parámetros de entrada/salida documentados
- ✅ Código limpio y legible

---

## 🚧 Mejoras Futuras

- [ ] Comparación con XGBoost/LightGBM
- [ ] Optimización de hiperparámetros (GridSearchCV)
- [ ] API REST con FastAPI
- [ ] Containerización con Docker
- [ ] Integración con datos meteorológicos en tiempo real
- [ ] Explicabilidad con SHAP values

---

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto es de uso educativo para el Bootcamp de Data e IA.

---

## 👤 Autor

**Christian Rueda**
- GitHub: [@imchrisrueda](https://github.com/imchrisrueda)

---

## 🙏 Agradecimientos

- Dataset de agricultura de India
- Comunidad de Scikit-learn
- Streamlit por la excelente framework

---

<div align="center">

**🌾 Cultivando el futuro con Data Science 🌾**

</div>
