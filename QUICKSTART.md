# 🚀 Guía Rápida de Inicio

## ⚡ Inicio Rápido (Automático)

```bash
cd 01_proyecto
./run.sh
```

El script automáticamente:
1. ✅ Verifica Python
2. ✅ Crea entorno virtual
3. ✅ Instala dependencias
4. ✅ Te permite elegir entre Streamlit o Jupyter

---

## 📝 Inicio Manual

### 1. Crear Entorno Virtual

```bash
cd 01_proyecto
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar la Aplicación

**Opción A: Streamlit (Recomendado)**
```bash
streamlit run app.py
```
Abre el navegador en `http://localhost:8501`

**Opción B: Jupyter Notebook**
```bash
jupyter notebook notebooks/eda_full.ipynb
```

---

## 🔍 Verificar que Todo Funcione

### Verificar Modelos

```bash
ls -lh models/
```

Deberías ver:
- `crop_recommender_rf.joblib` (~3.2 MB)
- `label_encoder.joblib` (~1.5 KB)

### Probar Importaciones

```python
python3 -c "import pandas, numpy, sklearn, streamlit; print('✅ Todo OK')"
```

---

## 🎯 Flujo de Trabajo Recomendado

### Para Exploración/Desarrollo:

1. **Ejecutar EDA Notebook:**
   ```bash
   jupyter notebook notebooks/eda_full.ipynb
   ```
   - Ejecuta todas las celdas (Cell → Run All)
   - Esto generará los modelos en `models/`

2. **Lanzar Streamlit:**
   ```bash
   streamlit run app.py
   ```

### Para Demo/Presentación:

```bash
streamlit run app.py
```

Navega por las tabs:
- 🏠 Inicio
- 📊 Análisis EDA
- 🤖 Modelo
- 🔮 Predicción

---

## 🐛 Solución de Problemas

### Error: "No module named 'streamlit'"

```bash
pip install -r requirements.txt
```

### Error: "FileNotFoundError: crop_recommender_rf.joblib"

1. Ejecuta el notebook completo:
   ```bash
   jupyter notebook notebooks/eda_full.ipynb
   ```
2. Ejecuta todas las celdas hasta la celda 14 (guardar modelos)

### Error: "No such file or directory: '../reports/eda_report.md'"

✅ Ya corregido en el notebook. Ejecuta de nuevo la celda 16.

---

## 📊 Uso de la Aplicación Streamlit

### Página: Predicción

1. Ajusta los sliders con los valores de tu terreno:
   - **Suelo:** N, P, K, pH
   - **Clima:** Temperatura, Humedad, Precipitación

2. Haz clic en "🌾 Predecir Cultivo Recomendado"

3. Verás:
   - Cultivo principal recomendado
   - Top-5 cultivos con probabilidades
   - Gráfico de barras interactivo

---

## 📚 Documentación Adicional

- **EDA completo:** [reports/EDA.md](reports/EDA.md)
- **Modelo técnico:** [reports/MODEL.md](reports/MODEL.md)
- **Dataset info:** [data/datacard.md](data/datacard.md)

---

## 🎨 Personalización

### Cambiar el Puerto de Streamlit

```bash
streamlit run app.py --server.port 8080
```

### Modo de Desarrollo (Auto-reload)

```bash
streamlit run app.py --server.runOnSave true
```

---

## 🔧 Comandos Útiles

### Ver estructura del proyecto
```bash
tree -L 2
```

### Ver tamaño de archivos
```bash
du -sh *
```

### Limpiar cache de Python
```bash
find . -type d -name __pycache__ -exec rm -rf {} +
```

---

## 💡 Tips

1. **Primero ejecuta el notebook** para generar los modelos
2. **Luego lanza Streamlit** para ver la app
3. **Revisa EDA.md y MODEL.md** para entender el proyecto
4. **Modifica app.py** para personalizar la interfaz

---

## 🌟 Siguientes Pasos

1. ✅ Ejecutar EDA notebook completo
2. ✅ Verificar que los modelos se hayan guardado
3. ✅ Lanzar Streamlit y probar predicciones
4. ⏳ Leer la documentación (EDA.md, MODEL.md)
5. ⏳ Experimentar con diferentes valores
6. ⏳ Modificar y mejorar el código

---

**¡Disfruta cultivando el futuro con Data Science! 🌾**
