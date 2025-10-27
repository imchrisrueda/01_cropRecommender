"""
Funciones utilitarias para análisis de Machine Learning.

Este módulo contiene funciones para evaluación de modelos de clasificación,
regresión, visualización de métricas y fronteras de decisión.
"""

# >> Imports <<
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, mean_squared_error, 
    r2_score, mean_absolute_error, precision_recall_curve,
    roc_curve, roc_auc_score
)


def obtencion_metricas_clasificacion(Entradas, modelo, Salidas_verdaderas):
    """
    Calcula y visualiza métricas de clasificación binaria.
    
    Parámetros de entrada:
        Entradas (array-like): Características del conjunto de datos.
        modelo (estimator): Modelo de clasificación entrenado.
        Salidas_verdaderas (array-like): Etiquetas verdaderas.
    
    Variables de proceso:
        Salidas_predichas: Predicciones del modelo.
        matriz_confusion: Matriz de confusión calculada.
        exactitud, precision, sensibilidad_recall, puntuacion_f1: Métricas.
    
    Salida:
        dict: Diccionario con métricas redondeadas a 4 dígitos.
    """
    # >> predicciones del modelo <<
    Salidas_predichas = modelo.predict(Entradas)
    matriz_confusion = confusion_matrix(Salidas_verdaderas, Salidas_predichas)
    
    print("La matriz de confusión es:")
    
    # >> visualización matriz de confusión <<
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        matriz_confusion, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Predicho 0', 'Predicho 1'],
        yticklabels=['Verdadero 0', 'Verdadero 1']
    )
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.show()
    
    # >> cálculo de métricas <<
    exactitud = accuracy_score(Salidas_verdaderas, Salidas_predichas)
    precision = precision_score(Salidas_verdaderas, Salidas_predichas, average='weighted')
    sensibilidad_recall = recall_score(Salidas_verdaderas, Salidas_predichas, average='weighted')
    puntuacion_f1 = f1_score(Salidas_verdaderas, Salidas_predichas, average='weighted')
    
    digitos = 4
    return {
        'exactitud': round(exactitud, digitos),
        'precision': round(precision, digitos),
        'sensibilidad_recall': round(sensibilidad_recall, digitos),
        'puntuacion_f1': round(puntuacion_f1, digitos)
    }




def dibujar_curva_roc(prob_falsa_alarma, ratio_verdaderos_positivos):
    """
    Dibuja la curva ROC.
    
    Parámetros de entrada:
        prob_falsa_alarma (array-like): Tasa de falsos positivos.
        ratio_verdaderos_positivos (array-like): Tasa de verdaderos positivos.
    
    Salida:
        None: Muestra gráfico de la curva ROC.
    """
    plt.plot(prob_falsa_alarma, ratio_verdaderos_positivos, "b-")
    plt.plot([0, 1], [0, 1], 'k--')  # >> diagonal de referencia <<
    
    plt.title('Curva ROC')
    plt.xlabel('Probabilidad de falsa alarma (PFA)')
    plt.ylabel('Sensibilidad o ratio verdaderos positivos (TRP)')
    plt.grid()
    plt.show()




def obtencion_metricas_regresion(y, y_sal, titulo_tipo_datos):
    """
    Calcula y muestra métricas de regresión.
    
    Parámetros de entrada:
        y (array-like): Valores verdaderos.
        y_sal (array-like): Valores predichos.
        titulo_tipo_datos (str): Descripción del tipo de datos (ej: 'de entrenamiento').
    
    Variables de proceso:
        error_MSE: Error cuadrático medio.
        error_MAE: Error absoluto medio.
        R2: Coeficiente de determinación.
    
    Salida:
        None: Imprime las métricas calculadas.
    """
    # >> error cuadrático medio <<
    error_MSE = mean_squared_error(y, y_sal)
    mensaje = f'El error cuadrático medio {titulo_tipo_datos} es:'
    print(mensaje)
    print(round(error_MSE, 4))
    
    # >> error absoluto medio <<
    error_MAE = mean_absolute_error(y, y_sal)
    mensaje = f'El error absoluto medio {titulo_tipo_datos} es:'
    print(mensaje)
    print(round(error_MAE, 4))
    
    # >> coeficiente de determinación <<
    R2 = r2_score(y, y_sal)
    mensaje = f'El valor del coeficiente de determinación R2 {titulo_tipo_datos} es:'
    print(mensaje)
    print(round(R2, 3))




def dibuja_ajuste_datos_regresion(x, y, x_entrada, y_sal):
    """
    Visualiza datos reales vs predicciones de regresión.
    
    Parámetros de entrada:
        x (array-like): Características originales.
        y (array-like): Valores verdaderos.
        x_entrada (array-like): Características para predicción.
        y_sal (array-like): Valores predichos.
    
    Variables de proceso:
        indices: Índices para ordenar los datos.
    
    Salida:
        None: Muestra gráfico con datos y predicciones.
    """
    # >> ordenar valores <<
    indices = np.argsort(x[:, 0], axis=0)
    x = x[indices]
    y = y[indices]
    
    indices = np.argsort(x_entrada[:, 0])
    x_entrada = x_entrada[indices]
    y_sal = y_sal[indices]
    
    # >> visualización <<
    plt.plot(x, y, "b-", label="Datos")
    plt.plot(x_entrada, y_sal, "m--", label="Predicción")
    
    plt.title('Curva de datos y predicción de la técnica de Machine Learning')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()




def dibuja_frontera_decision(Entradas, modelo, Salidas, clase_0, clase_1,
                              colores, etiqueta_leyenda_cero,
                              etiqueta_leyenda_uno, titulo, titulo_eje_x, 
                              titulo_eje_y, simbolo, tamano_simbolo, 
                              adicion_frontera):
    """
    Dibuja las clases y la frontera de decisión del modelo.
    
    Parámetros de entrada:
        Entradas (array-like): Características (2D).
        modelo (estimator): Modelo de clasificación entrenado.
        Salidas (array-like): Etiquetas verdaderas.
        clase_0, clase_1 (int/str): Identificadores de las clases.
        colores (list): Lista de colores para cada clase.
        etiqueta_leyenda_cero, etiqueta_leyenda_uno (str): Etiquetas para la leyenda.
        titulo, titulo_eje_x, titulo_eje_y (str): Títulos del gráfico.
        simbolo (str): Marcador para los puntos.
        tamano_simbolo (int): Tamaño de los marcadores.
        adicion_frontera (int): 1 para dibujar frontera, 0 para no dibujarla.
    
    Variables de proceso:
        X1, X2: Grillas de coordenadas para la frontera.
        Y: Predicciones en la grilla.
    
    Salida:
        None: Muestra gráfico con las clases y frontera de decisión.
    """
    # >> generar frontera de decisión <<
    if adicion_frontera == 1:
        minX1 = min(Entradas[:, 0])
        maxX1 = max(Entradas[:, 0])
        minX2 = min(Entradas[:, 1])
        maxX2 = max(Entradas[:, 1])
        marginX1 = (maxX1 - minX1) * 0.2
        marginX2 = (maxX2 - minX2) * 0.2
        
        x1 = np.linspace(minX1 - marginX1, maxX1 + marginX1, 1000)
        x2 = np.linspace(minX2 - marginX2, maxX2 + marginX2, 1000)
        X1, X2 = np.meshgrid(x1, x2)
        
        Y = modelo.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)
        plt.contourf(X1, X2, Y, levels=2, alpha=0.3)
    
    # >> graficar clase 0 <<
    condicion = Salidas == clase_0
    entradas_clase_0 = Entradas[condicion]
    x1_0 = entradas_clase_0[:, 0]
    x2_0 = entradas_clase_0[:, 1]
    color_cero = colores[0]
    plt.scatter(x1_0, x2_0, marker=simbolo, s=tamano_simbolo, 
                color=color_cero, label=etiqueta_leyenda_cero)
    
    # >> graficar clase 1 <<
    condicion = Salidas == clase_1
    entradas_clase_1 = Entradas[condicion]
    x1_1 = entradas_clase_1[:, 0]
    x2_1 = entradas_clase_1[:, 1]
    color_uno = colores[1]
    plt.scatter(x1_1, x2_1, marker=simbolo, s=tamano_simbolo,
                color=color_uno, label=etiqueta_leyenda_uno)
    
    plt.title(titulo)
    plt.xlabel(titulo_eje_x)
    plt.ylabel(titulo_eje_y)
    plt.legend()
    plt.show()
