# Análisis Exploratorio de Datos (EDA) - Sistema de Recomendación de Cultivos

## 1. Resumen Ejecutivo

Este documento presenta el análisis exploratorio de datos realizado sobre el conjunto de datos de recomendación de cultivos para agricultura de precisión. El objetivo es identificar patrones, relaciones y características clave que permitan desarrollar un sistema de recomendación efectivo basado en condiciones del suelo y clima.

---

## 2. Descripción del Dataset

### 2.1 Contexto del Negocio
La **agricultura de precisión** es fundamental para optimizar el rendimiento agrícola mediante decisiones informadas. Este dataset permite a los agricultores seleccionar el cultivo más adecuado según las características específicas de su terreno y condiciones ambientales.

### 2.2 Características del Dataset

**Fuente de datos:** Datos aumentados de India (rainfall, climate, fertilizer)

**Dimensiones:**
- Total de registros: 2,200 observaciones
- Variables: 8 columnas (7 features + 1 target)

**Variables del dataset:**

| Variable | Tipo | Descripción | Unidad |
|----------|------|-------------|--------|
| `N` | Numérica | Ratio de Nitrógeno en el suelo | Proporción |
| `P` | Numérica | Ratio de Fósforo en el suelo | Proporción |
| `K` | Numérica | Ratio de Potasio en el suelo | Proporción |
| `temperature` | Numérica | Temperatura ambiente | °C |
| `humidity` | Numérica | Humedad relativa | % |
| `ph` | Numérica | pH del suelo | Escala 0-14 |
| `rainfall` | Numérica | Precipitación | mm |
| `label` | Categórica | Cultivo recomendado | 22 clases |

---

## 3. Calidad de los Datos

### 3.1 Valores Faltantes
**No se detectaron valores nulos** en ninguna columna del dataset.

### 3.2 Duplicados
**No se encontraron registros duplicados**.

### 3.3 Conclusión de Calidad
El dataset presenta una **excelente calidad**, lo que facilita el análisis y reduce la necesidad de imputación o limpieza extensiva.

---

## 4. Análisis Univariado

### 4.1 Variable Objetivo: Distribución de Cultivos

**Observaciones:**
- El dataset contiene **22 tipos de cultivos diferentes**
- Cada cultivo tiene exactamente **100 observaciones**
- **Balance perfecto de clases** (distribución uniforme)

**Implicaciones:**
- No es necesario aplicar técnicas de balanceo (SMOTE, undersampling)
- Los modelos no estarán sesgados hacia ninguna clase
- Las métricas de evaluación serán confiables

**Cultivos incluidos:**
rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee

### 4.2 Variables Numéricas

#### Nitrógeno (N)
- **Rango:** 0 - 140
- **Distribución:** Relativamente uniforme con picos en valores bajos y medios
- **Relevancia agronómica:** Esencial para crecimiento vegetativo

#### Fósforo (P)
- **Rango:** 5 - 145
- **Distribución:** Uniforme en todo el rango
- **Relevancia agronómica:** Clave para desarrollo radicular y floración

#### Potasio (K)
- **Rango:** 5 - 205
- **Distribución:** Uniforme
- **Relevancia agronómica:** Fundamental para resistencia a enfermedades

#### Temperatura (°C)
- **Rango:** 8.8 - 43.7 °C
- **Distribución:** Multimodal (varios picos)
- **Relevancia agronómica:** Determina la estación de siembra

#### Humedad (%)
- **Rango:** 14 - 99%
- **Distribución:** Concentración en valores altos (60-100%)
- **Relevancia agronómica:** Afecta enfermedades fúngicas y evapotranspiración

#### pH
- **Rango:** 3.5 - 9.9
- **Distribución:** Relativamente normal centrada en 6.5
- **Relevancia agronómica:** Afecta disponibilidad de nutrientes

#### Precipitación (mm)
- **Rango:** 20 - 298 mm
- **Distribución:** Multimodal (varios picos)
- **Relevancia agronómica:** Determina necesidades de riego

---

## 5. Detección de Outliers

### 5.1 Análisis por Variable (Método IQR con k=1.5)

| Variable | Outliers Detectados | % del Total |
|----------|---------------------|-------------|
| N | ~150 | 6.8% |
| P | ~100 | 4.5% |
| K | ~120 | 5.5% |
| temperature | ~80 | 3.6% |
| humidity | ~90 | 4.1% |
| ph | ~85 | 3.9% |
| rainfall | ~110 | 5.0% |

### 5.2 Interpretación de Outliers

**Decisión:** **NO eliminar outliers**

**Justificación agronómica:**
1. **Diversidad real:** Los "outliers" representan condiciones extremas pero válidas
2. **Cultivos especializados:** Algunos cultivos prosperan en condiciones extremas
   - Arroz: alta humedad (>90%)
   - Café: alta precipitación (>200mm)
   - Manzana: temperaturas bajas (<15°C)
3. **Agricultura de precisión:** El modelo debe predecir para TODAS las condiciones

---

## 6. Análisis Multivariado

### 6.1 Matriz de Correlación

**Correlaciones significativas detectadas:**

| Par de Variables | Correlación | Interpretación |
|------------------|-------------|----------------|
| K - N | 0.25 | Débil positiva |
| P - K | -0.10 | Muy débil negativa |
| Resto | < ±0.15 | No correlacionados |

**Conclusión:**
- **No hay multicolinealidad problemática**
- Las variables son independientes entre sí
- No es necesario eliminar features por redundancia

### 6.2 Patrones por Cultivo

#### Análisis de Perfiles (Medianas por Cultivo)

**Cultivos de Alta Precipitación (>200mm):**
- Rice (arroz)
- Coconut (coco)
- Papaya

**Cultivos de Baja Precipitación (<50mm):**
- Chickpea (garbanzo)
- Kidneybeans (frijoles)
- Mothbeans

**Cultivos de Alta Temperatura (>30°C):**
- Papaya
- Coconut
- Cotton (algodón)

**Cultivos de Baja Temperatura (<20°C):**
- Apple (manzana)
- Grapes (uvas)
- Lentil (lentejas)

**Cultivos de Alto Nitrógeno (>100):**
- Cotton
- Sugarcane (caña de azúcar)
- Maize (maíz)

**Cultivos de Bajo pH (<6.0):**
- Coffee (café)
- Grapes
- Apple

---

## 7. Feature Engineering

### 7.1 Nueva Variable Creada

**`N_over_PK`**: Ratio de Nitrógeno sobre Fósforo+Potasio

**Fórmula:**
```
N_over_PK = N / (P + K + 1e-6)
```

**Justificación agronómica:**
- Captura el **balance de macronutrientes** (N-P-K)
- Los cultivos tienen requerimientos específicos de proporción N:P:K
- Ejemplo: Leguminosas requieren bajo N (fijan nitrógeno atmosférico)

---

## 8. Insights del Dominio Agronómico

### 8.1 Grupos de Cultivos por Requerimientos

#### Grupo 1: Cultivos de Climas Húmedos y Cálidos
- **Cultivos:** Rice, Coconut, Papaya, Banana
- **Características:** Alta humedad (>80%), Alta temperatura (>25°C), Alta precipitación (>150mm)
- **Región típica:** Zonas tropicales costeras

#### Grupo 2: Cultivos de Climas Áridos
- **Cultivos:** Chickpea, Mothbeans, Lentil
- **Características:** Baja precipitación (<50mm), Temperatura moderada (20-30°C)
- **Región típica:** Zonas semiáridas, secano

#### Grupo 3: Cultivos de Climas Templados
- **Cultivos:** Apple, Grapes
- **Características:** Temperatura baja-moderada (<20°C), pH ácido (<6.5)
- **Región típica:** Zonas de montaña, valles

#### Grupo 4: Cultivos Industriales de Alto Consumo
- **Cultivos:** Cotton, Jute, Coffee
- **Características:** Alto N, Alta precipitación (>100mm)
- **Región típica:** Zonas de agricultura intensiva

### 8.2 Variables Más Discriminantes

Según análisis visual y perfil de cultivos:

1. **Rainfall (Precipitación)** - Máxima separabilidad entre cultivos
2. **Temperature** - Separa cultivos tropicales de templados
3. **Humidity** - Diferencia cultivos de secano vs riego
4. **pH** - Identifica cultivos específicos (café, frutas)
5. **N, P, K** - Requerimientos nutricionales específicos

---

## 9. Preparación para Modelado

### 9.1 División del Dataset

- **Train Set:** 80% (1,760 observaciones)
- **Test Set:** 20% (440 observaciones)
- **Estratificación:** Aplicada (mantiene proporción de clases)

### 9.2 Escalado de Features

**Método:** StandardScaler (normalización z-score)

**Justificación:**
- Las variables tienen diferentes escalas (N:0-140, rainfall:20-300, pH:3-10)
- Random Forest se beneficia de features normalizadas
- Mejora convergencia de algoritmos de distancia

---

## 10. Modelo Baseline

### 10.1 Algoritmo Seleccionado

**Random Forest Classifier**
- n_estimators: 200
- random_state: 42
- n_jobs: -1 (paralelización)

### 10.2 Métricas de Desempeño

**Accuracy (Exactitud):** ~99%

**Interpretación:**
- Excelente separabilidad entre cultivos
- Las features son altamente informativas
- No hay overlap significativo entre clases

### 10.3 Feature Importance

**Top 5 Features más importantes:**

1. **Rainfall** (~25% importance)
2. **N** (~18% importance)
3. **K** (~16% importance)
4. **P** (~14% importance)
5. **humidity** (~12% importance)

**Validación:**
- Coincide con el conocimiento agronómico
- Rainfall es el factor más limitante en agricultura
- Macronutrientes (N-P-K) determinan el cultivo

---

## 11. Análisis de Estacionalidad

### 11.1 Calendario Agrícola de India (Base del Dataset)

#### Temporadas de Cultivo

**1. Kharif (Monsoon) - Junio a Octubre**
- **Precipitación:** Alta (300-1000mm total)
- **Temperatura:** 25-35°C
- **Cultivos:** Rice, Maize, Cotton, Jute, Papaya, Banana

**2. Rabi (Invierno) - Noviembre a Marzo**
- **Precipitación:** Baja (50-200mm total)
- **Temperatura:** 15-25°C
- **Cultivos:** Chickpea, Lentil, Kidney Beans

**3. Zaid (Verano) - Marzo a Junio**
- **Precipitación:** Muy baja, requiere riego
- **Temperatura:** 30-40°C
- **Cultivos:** Watermelon, Muskmelon, Mungbean

---

## 12. Conclusiones Finales

### 12.1 Conclusiones Técnicas

1. **Dataset de alta calidad:** Sin nulos, sin duplicados, balanceado
2. **Features independientes:** No hay multicolinealidad
3. **Alta separabilidad:** Los cultivos tienen perfiles bien diferenciados
4. **Modelo viable:** Random Forest baseline alcanza ~99% accuracy

### 12.2 Conclusiones del Dominio

1. **Precipitación como factor crítico:** Es la variable más importante para selección de cultivos
2. **Grupos climáticos claros:** Los cultivos se agrupan en zonas agroclimáticas específicas
3. **Balance NPK relevante:** Los requerimientos nutricionales son discriminantes
4. **Aplicabilidad práctica:** El sistema puede recomendar cultivos con alta confianza

### 12.3 Recomendaciones para Producción

1. **Sistema de alertas:** Implementar umbrales de confianza para predicciones
2. **Interfaz intuitiva:** Mostrar top-3 cultivos recomendados con probabilidades
3. **Información adicional:** Incluir rangos óptimos de cada feature por cultivo
4. **Validación continua:** Reentrenar modelo con datos reales de campo

---

## 13. Recomendaciones Agronómicas por Cultivo

### 13.1 Cereales

**Rice (Arroz)**
- **Época de siembra:** Monsoon (mayo-julio)
- **Requerimientos:** Precipitación >200mm/mes, Temperatura 25-35°C, Humedad >80%
- **N:** Alto (80-100), **P:** Moderado (40-60), **K:** Moderado (40-60)
- **pH óptimo:** 5.5-7.0

**Maize (Maíz)**
- **Época de siembra:** Kharif (junio-julio) o Rabi (octubre-noviembre)
- **Requerimientos:** Precipitación 50-100mm/mes, Temperatura 20-30°C
- **N:** Alto (80-120), **P:** 40-60, **K:** 40-60
- **pH óptimo:** 5.5-7.5

### 13.2 Leguminosas

**Chickpea (Garbanzo)**
- **Época de siembra:** Rabi (octubre-noviembre)
- **Requerimientos:** Precipitación 40-60mm/mes (secano), Temperatura 20-25°C
- **N:** Bajo (20-40) - fija N atmosférico
- **pH óptimo:** 6.0-7.5

**Lentil (Lentejas)**
- **Época de siembra:** Rabi (octubre-noviembre)
- **Requerimientos:** Precipitación 40-60mm/mes, Temperatura 15-25°C
- **N:** Bajo (20-40)
- **pH óptimo:** 6.0-7.5

### 13.3 Frutales Tropicales

**Banana (Plátano)**
- **Época de siembra:** Todo el año (clima tropical)
- **Requerimientos:** Precipitación 100-200mm/mes, Temperatura 25-35°C, Humedad >75%
- **N:** Alto (100-120), **P:** 60-80, **K:** 100-120
- **pH óptimo:** 6.0-7.5

**Mango**
- **Época de siembra:** Junio-julio (monsoon)
- **Requerimientos:** Precipitación 75-200mm/mes, Temperatura 24-30°C
- **pH óptimo:** 5.5-7.5

### 13.4 Frutales Templados

**Apple (Manzana)**
- **Época de siembra:** Diciembre-enero (invierno)
- **Requerimientos:** Precipitación 100-125mm/mes, Temperatura 15-20°C
- **pH óptimo:** 5.5-6.5 (ligeramente ácido)

**Grapes (Uvas)**
- **Época de siembra:** Enero-febrero
- **Requerimientos:** Precipitación 50-75mm/mes, Temperatura 15-25°C
- **pH óptimo:** 5.5-7.0

### 13.5 Cultivos Industriales

**Coffee (Café)**
- **Época de siembra:** Monsoon (mayo-junio)
- **Requerimientos:** Precipitación 150-250mm/mes, Temperatura 15-25°C, Humedad 70-85%
- **pH óptimo:** 4.5-6.0 (ácido)

**Cotton (Algodón)**
- **Época de siembra:** Kharif (mayo-junio)
- **Requerimientos:** Precipitación 50-100mm/mes, Temperatura 25-35°C
- **N:** Alto (100-140)
- **pH óptimo:** 6.0-7.5

---

## 14. Mejoras Futuras e Investigación

### 14.1 Limitaciones Actuales

1. **Falta de estacionalidad explícita:** No incluye mes de siembra
2. **Datos estáticos:** No considera variabilidad temporal
3. **Escala local:** Dataset de India, requiere calibración para otras regiones
4. **Variables omitidas:** Tipo de suelo, profundidad, salinidad, altitud

### 14.2 Mejoras Propuestas

**Corto Plazo:**
- Incorporar `month` como feature categórica
- Validar con expertos agrónomos locales
- Recopilar datos de campo para reentrenamiento
- Desarrollar sistema de alertas estacionales

**Mediano Plazo:**
- Integración con APIs meteorológicas
- Módulo de rotación de cultivos
- Calculadora de requerimientos de fertilizantes
- Análisis de rentabilidad económica

**Largo Plazo:**
- Imágenes satelitales (NDVI, humedad del suelo)
- IoT sensors en campo
- Gemelos digitales (simulación de rendimientos)
- Blockchain para trazabilidad

---

**Fecha del análisis:** Febrero 2026  
**Versión del documento:** 2.0 Professional  
**Analista:** Christian Rueda-Ayala

