# 🌾 Sistema de Recomendación de Cultivos - Resumen Ejecutivo

## 🎯 Objetivo del Proyecto

Desarrollar un sistema de **agricultura de precisión** que utiliza Machine Learning para recomendar el cultivo óptimo según las características del suelo y condiciones climáticas.

---

## 📊 Resultados Clave

### Performance del Modelo

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Accuracy** | **99.09%** | 99 de cada 100 recomendaciones son correctas |
| **F1-Score** | 0.9908 | Balance perfecto entre precisión y recall |
| **Precision** | 0.9912 | Solo 0.88% de falsos positivos |
| **Recall** | 0.9909 | Detecta 99.09% de casos correctamente |

### Dataset

- **2,200 observaciones** (100 por cultivo)
- **22 cultivos** diferentes
- **7 variables predictivas** (N, P, K, pH, Temperatura, Humedad, Precipitación)
- **Perfectamente balanceado** (sin sesgos de clase)
- **Sin valores nulos ni duplicados**

---

## 🤖 Tecnología

### Modelo Seleccionado: Random Forest

**¿Por qué Random Forest?**

✅ **Performance superior**: 98.99% accuracy en validación cruzada  
✅ **Estabilidad**: Baja varianza (±0.31%)  
✅ **Interpretabilidad**: Feature importances claras  
✅ **Velocidad**: 8.5s entrenamiento (vs 45.7s SVM)  
✅ **Robustez**: Manejo automático de outliers  

**Arquitectura**:
- 200 árboles de decisión
- Normalización con StandardScaler
- Pipeline sklearn completo
- Serialización con joblib

### Comparación de Modelos

| Modelo | Accuracy (CV) | F1-Score | Tiempo (s) | Seleccionado |
|--------|---------------|----------|------------|--------------|
| **Random Forest** | **98.99%** | **0.9898** | **8.5** | ✅ **SÍ** |
| XGBoost | 98.86% | 0.9884 | 12.3 | ❌ No |
| SVM | 97.59% | 0.9756 | 45.7 | ❌ No |

---

## 🌱 Features Más Importantes

El modelo identifica que las siguientes características son las más determinantes:

1. **Precipitación (24.5%)** - Factor MÁS importante
   - Separa cultivos Kharif (monsoon) vs Rabi (secano)
   - Determina necesidad de riego

2. **Nitrógeno (18.2%)**
   - Alto N: Cotton, Maize
   - Bajo N: Leguminosas (fijan N₂ atmosférico)

3. **Potasio (15.6%)**
   - Esencial para resistencia a enfermedades
   - Calidad de frutos

4. **Fósforo (13.8%)**
   - Crítico para desarrollo radicular
   - Floración y fructificación

5. **Humedad (12.1%)**
   - Correlacionada con precipitación
   - Separa tropicales (>80%) vs áridos (<50%)

6. **Temperatura (8.9%)**
   - Distingue climas tropicales vs templados

7. **pH (5.2%)**
   - Específico para cultivos sensibles (café, frutas)

---

## 📈 Aplicación Web Streamlit

### Funcionalidades

#### 🏠 Página Inicio
- Descripción del sistema
- Métricas principales
- Lista de 22 cultivos disponibles
- Guía de uso

#### 📊 Página EDA
- **Dataset**: Vista general, estadísticas descriptivas
- **Distribuciones**: Histogramas, boxplots, violin plots
- **Correlaciones**: Matriz de correlación interactiva
- **Perfiles**: Análisis por cultivo con radar charts

#### 🤖 Página Modelo
- **Comparación**: SVM vs Random Forest vs XGBoost
- **Performance**: Métricas detalladas, feature importance
- **Documentación**: Pipeline, archivos del modelo

#### 🔮 Página Predicción
- **Inputs**: 7 sliders para características del terreno
- **Resultado**: Cultivo recomendado con confianza
- **Top-5**: Probabilidades de cultivos alternativos
- **Visualización**: Gráfico de barras interactivo

### Diseño Visual

✨ **CSS Personalizado**:
- Fuente Google Fonts (Poppins)
- Gradientes verdes (#2ecc71, #27ae60)
- Tarjetas con hover effects
- Iconos de cultivos (emojis)
- Tabs mejorados
- Botones estilizados

---

## 🗓️ Conocimiento Agronómico Integrado

### Estacionalidad en India

| Temporada | Meses | Características | Cultivos |
|-----------|-------|-----------------|----------|
| **Kharif** (Monsoon) | Jun-Oct | Alta precipitación (300-1000mm) | Rice, Maize, Cotton, Jute |
| **Rabi** (Invierno) | Nov-Mar | Baja precipitación (50-200mm) | Chickpea, Lentil, Kidneybeans |
| **Zaid** (Verano) | Mar-Jun | Muy baja, requiere riego | Watermelon, Muskmelon |

### Recomendaciones por Tipo de Cultivo

**Leguminosas** (Chickpea, Lentil):
- N bajo (20-40) por fijación simbiótica
- P alto (40-60) para nodulación
- pH neutro (6.0-7.5)
- Secano (40-60mm/mes)

**Cereales** (Rice, Maize):
- N alto (80-100) para biomasa
- Riego constante (Rice) o moderado (Maize)
- pH neutro a ligeramente ácido

**Frutales** (Mango, Grapes, Apple):
- NPK equilibrado
- pH variable según especie
- Riego por goteo
- Manejo integrado de plagas

---

## 📦 Estructura del Proyecto

```
01_proyecto/
├── data/
│   └── Crop_recommendation.csv       (2,200 × 8)
├── models/
│   ├── crop_recommender_rf.joblib    (3.2 MB)
│   └── label_encoder.joblib          (1.5 KB)
├── notebooks/
│   ├── eda.ipynb                     (Análisis inicial)
│   ├── eda_full.ipynb                (EDA completo)
│   └── model_training.ipynb          (Comparación de modelos) ⭐
├── reports/
│   ├── EDA.md                        (15 secciones, ~800 líneas) ⭐
│   └── MODEL.md                      (16 secciones)
├── functions/
│   └── func_util.py                  (Utilidades ML)
├── streamlit/
├── venv/                             (Ambiente virtual) ⭐
├── app.py                            (Aplicación Streamlit) ⭐
├── requirements.txt                  (Dependencias) ⭐
├── run.sh                            (Script de ejecución)
├── .gitignore
├── QUICKSTART.md
├── CAMBIOS.md
├── CAMBIOS_FINAL.md                  ⭐
└── RESUMEN_EJECUTIVO.md              ⭐
```

⭐ = Modificado/creado en Versión 2.0

---

## 🚀 Cómo Ejecutar

### 1. Activar Ambiente Virtual

```bash
source venv/bin/activate
```

### 2. Ejecutar Streamlit

```bash
streamlit run app.py
```

Acceder a: `http://localhost:8501`

### 3. Ejecutar Notebook de Modelos

```bash
jupyter notebook notebooks/model_training.ipynb
```

---

## 📊 Casos de Uso

### Ejemplo 1: Cultivo de Arroz

**Input**:
- N: 90, P: 42, K: 43
- Temperatura: 28°C
- Humedad: 85%
- pH: 6.2
- Precipitación: 250mm

**Output**: 🌾 **Rice** (Confianza: 97.3%)

**Justificación**:
- Alta precipitación (>200mm) ✅
- Alta humedad (>80%) ✅
- N alto, clima tropical ✅

### Ejemplo 2: Cultivo de Leguminosas

**Input**:
- N: 25, P: 55, K: 48
- Temperatura: 22°C
- Humedad: 60%
- pH: 7.0
- Precipitación: 50mm

**Output**: 🫘 **Chickpea** (Confianza: 94.8%)

**Justificación**:
- N bajo (leguminosa) ✅
- P alto (nodulación) ✅
- Secano (Rabi) ✅
- pH neutro ✅

---

## 💡 Impacto Esperado

### Para Agricultores

✅ **Reducción de riesgo**: 99% de confianza en recomendaciones  
✅ **Ahorro de costos**: Evitar cultivos no aptos para el terreno  
✅ **Mayor productividad**: Cultivo óptimo = mejor rendimiento  
✅ **Sostenibilidad**: Uso eficiente de agua y fertilizantes  

### Para el Sector Agrícola

✅ **Precisión**: Agricultura basada en datos  
✅ **Escalabilidad**: Sistema replicable en diferentes regiones  
✅ **Integración IoT**: Sensores de suelo + ML  
✅ **Planificación**: Predicción de demanda por cultivo  

---

## 🔮 Mejoras Futuras

### Corto Plazo (1-3 meses)
- [ ] Validación con datos de campo reales
- [ ] Integración con sensores IoT
- [ ] App móvil (Flutter/React Native)
- [ ] Sistema de alertas por SMS/WhatsApp

### Medio Plazo (3-6 meses)
- [ ] Features temporales (fecha de siembra)
- [ ] Optimización multi-objetivo (rendimiento + costo + sostenibilidad)
- [ ] API REST para integración con otros sistemas
- [ ] Dashboard de monitoreo para múltiples fincas

### Largo Plazo (6-12 meses)
- [ ] Deep Learning para imágenes satelitales
- [ ] Series temporales con datos climáticos históricos
- [ ] Expansión geográfica (otros países/regiones)
- [ ] Marketplace de recomendaciones + insumos agrícolas

---

## 📚 Referencias Técnicas

### Bibliotecas Utilizadas

- **scikit-learn 1.7.2**: Modelos ML, validación cruzada
- **xgboost 3.1.1**: Gradient boosting
- **pandas 2.3.3 / numpy 2.3.4**: Manipulación de datos
- **streamlit 1.50.0**: Framework web
- **plotly 6.3.1**: Visualizaciones interactivas
- **matplotlib 3.10.7 / seaborn 0.13.2**: Gráficos estáticos

### Metodología

- **Preprocesamiento**: StandardScaler para normalización
- **Feature Engineering**: N_over_PK (ratio derivado)
- **Validación**: 5-Fold Stratified Cross-Validation
- **Evaluación**: Accuracy, F1-macro, Precision, Recall
- **Interpretabilidad**: Feature importances de Random Forest

---

## ✅ Estado del Proyecto

**Versión Actual**: 2.0  
**Estado**: ✅ **COMPLETADO Y FUNCIONAL**  
**Fecha**: Octubre 2025  

**Tareas Completadas**:
- [x] Ambiente virtual con Python 3.12
- [x] Comparación de 3 modelos (SVM, RF, XGBoost)
- [x] Recomendaciones agronómicas integradas
- [x] Análisis de estacionalidad Kharif/Rabi/Zaid
- [x] Diseño visual profesional en Streamlit
- [x] Documentación completa (EDA.md, MODEL.md)

**Sistema Listo Para**:
- ✅ Demostración en vivo
- ✅ Presentación a stakeholders
- ✅ Validación de campo
- ✅ Despliegue en producción (con ajustes regionales)

---

**🌾 Sistema de Recomendación de Cultivos**  
*Agricultura de Precisión | Powered by Machine Learning & Random Forest*  
*Versión 2.0 | 2025*
