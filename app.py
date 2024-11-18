import streamlit as st
import matplotlib.pyplot as plt
import pydicom
import tensorflow as tf
import tensorflow.keras.layers as tfl
import keras
import matplotlib.pyplot as plt
import io
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from tqdm import tqdm
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from joblib import load


# Cargar el modelo
model = load_model('modelo_fracturas.h5')
rf_model = load('random_forest_model.joblib')


def preprocess_image(image):
    '''Preprocesar la imagen cargada para el modelo CNN.'''
    img = np.array(image)
    img = cv2.resize(img, (128, 128))  # Redimensionar a 128x128
    img = img / 255.0  # Normalizar
    img = np.expand_dims(img, axis=0)  # Añadir dimensión batch
    return img


# Preprocesamiento de datos CSV para Random Forest
def preprocess_csv(data):
    #caracteristicas a tomar
    features = ['Rows', 'Columns', 'SliceThickness', 'SliceRatio',
                'ImagePositionPatientX', 'ImagePositionPatientY', 'ImagePositionPatientZ']
    return data[features]


def load_dicom(uploaded_file):
    '''Función para cargar y transformar imágenes DICOM desde un archivo en memoria'''
    # Leemos el archivo cargado en bytes
    file_bytes = uploaded_file.read()  # Lee los bytes del archivo
    dicom_data = pydicom.dcmread(io.BytesIO(file_bytes))  # Leer los datos DICOM desde los bytes
    
    # Obtén el array de píxeles
    data = dicom_data.pixel_array   
    
    # Normalizar la imagen al rango [0, 255] si es necesario
    data = data.astype(np.float32)
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    data = data.astype(np.uint8)  # Convertir a uint8
    
    # Si la imagen es en escala de grises (es probable para imágenes médicas)
    if len(data.shape) == 2:  # Imagen 2D, convertir a 3 canales si es necesario
        return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    
    return data

def ImgDataGenerator(train_df,base_path):
    '''Function to read dicom image path and store the images as numpy arrays'''
    trainset = []
    trainlabel = []
    for i in tqdm(range(len(train_df))):
        study_id = train_df.loc[i,'StudyInstanceUID']
        slice_id = train_df.loc[i,'slice']+'.dcm'
        study_path = study_id+'/'+slice_id
        
        path = os.path.join(base_path, study_path)
    
        #dc = dicom.read_file(os.path.join(path,im))
        #if dc.file_meta.TransferSyntaxUID.name =='JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])':
        #    continue
        img = load_dicom(path)
        img = cv2.resize(img, (128 , 128))
        image = img_to_array(img)
        image = image / 255.0
        trainset += [image]
        cur_label = [train_df.loc[i,f'C{j}'] for j in range(1,8)]
        trainlabel += [cur_label]

                        
            
    return np.array(trainset), np.array(trainlabel)

# Cambiar el ancho del contenedor principal
st.markdown(
    """
    <style>
    .stMainBlockContainer  {
        max-width: 90%; /* Ajusta el ancho máximo aquí */
        margin: 0 auto; /* Centra la app */
    }
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666666;
    }
    .stMarkdown, .stPlotlyChart, .stPyplot {
        max-width: 700px; /* Cambia este valor según el ancho deseado */
        margin: 0 auto;  /* Centrar el contenido */
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<p class="title">Detección de Fracturas Cervicales</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sube una imagen o un archivo CSV para realizar la predicción</p>', unsafe_allow_html=True)

# Carga de imagen o CSV
tab1, tab2, tab3 = st.tabs(["CNN: Imágenes DICOM", "Random Forest: CSV Metadata", "Comparación entre modelos"])

# Subida de imagen
with tab1:
    uploaded_file = st.file_uploader("Sube una imagen DICOM", type=["jpg", "png", "jpeg", "dcm"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.dcm'):
            image = load_dicom(uploaded_file)
            st.image(image, caption='Imagen DICOM cargada', width=500)
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption='Imagen cargada', width=500)
        
        if st.button('Predecir (CNN)'):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)[0]
            
            if all(prob <= 1e-4 for prob in prediction):
                st.markdown(
                    """
                    <h3 style="text-align: center;">Resultados de Predicción</h3>
                    <p style="text-align: center;">No se detectaron fracturas.</p>
                    """,
                    unsafe_allow_html=True,
                )
            else:            
                st.markdown(
                    """
                    <h3 style="text-align: center;">Resultados de Predicción</h3>
                    """,
                    unsafe_allow_html=True,
                )
        
                # Crear columnas
                col1, col2 = st.columns(2)
                
                # Columna 1: Porcentajes por vértebra
                with col1:
                    st.markdown("#### Probabilidades de fractura por vértebra (C1-C7):")
                    for i, prob in enumerate(prediction):
                        st.write(f"Vértebra C{i+1}: {prob:.4f}")
                
                # Columna 2: Gráfica
                with col2:
                    st.markdown("#### Distribución de Probabilidades por Vértebra")
                    fig, ax = plt.subplots(figsize=(6, 4))  # Ajustar tamaño de la gráfica
                    vertebra_labels = [f'C{i+1}' for i in range(7)]
                    ax.bar(vertebra_labels, prediction, color='skyblue')
                    ax.set_ylim(0, 1)
                    ax.set_xlabel('Vértebras')
                    ax.set_ylabel('Probabilidad')
                    ax.set_title('Probabilidades de fractura por vértebra (C1-C7)')
                    st.pyplot(fig)

                    

# Tab para CSV con Random Forest
with tab2:
    csv_file = st.file_uploader("Sube un archivo CSV con metadata", type=["csv"])
    if csv_file is not None:
        data = pd.read_csv(csv_file)
        st.write("Vista previa de los datos cargados:")
        st.dataframe(data.head())
        
        if st.button('Predecir (Random Forest)'):
            X = preprocess_csv(data)
            predictions = rf_model.predict_proba(X)  # Asegúrate de usar `predict_proba` si tienes probabilidades
            
            st.markdown("### Resultados de Predicción:")
            
            for i, probs in enumerate(predictions):
                st.write(f"Fila {i+1} - Probabilidades de fractura por vértebra:")
                
                # Mostrar las probabilidades para cada vértebra
                vertebra_labels = [f'C{i+1}' for i in range(7)]
                prob_df = pd.DataFrame({
                    'Vértebra': vertebra_labels, 
                    'Probabilidad': probs[0]  # Utiliza la primera fila de predicciones
                })
                
                # Visualizar tabla de probabilidades
                st.dataframe(prob_df)
                
                # Gráfico interactivo de las probabilidades
                fig, ax = plt.subplots()
                ax.bar(prob_df['Vértebra'], prob_df['Probabilidad'], color='lightcoral')
                ax.set_xlabel('Vértebras')
                ax.set_ylabel('Probabilidad')
                ax.set_title(f'Fila {i+1}: Distribución de Probabilidades')
                st.pyplot(fig)
                
                # Agregar un separador
                st.markdown("---")
        
        # Comparación de Importancia de Características
        if st.checkbox("Mostrar importancia de características"):
            st.markdown("### Importancia de las Características en el Modelo Random Forest")
            
            # Extraer importancia de las características
            importances = rf_model.feature_importances_
            feature_labels = ['Rows', 'Columns', 'SliceThickness', 'SliceRatio',
                              'ImagePositionPatientX', 'ImagePositionPatientY', 'ImagePositionPatientZ']
            
            # Crear DataFrame para visualización
            importance_df = pd.DataFrame({
                'Características': feature_labels,
                'Importancia': importances
            }).sort_values(by='Importancia', ascending=False)
            
            # Mostrar tabla y gráfico
            st.dataframe(importance_df)
            fig, ax = plt.subplots()
            ax.barh(importance_df['Características'], importance_df['Importancia'], color='seagreen')
            ax.set_xlabel('Importancia')
            ax.set_title('Importancia de las Características')
            st.pyplot(fig)

with tab3:
    # Pestaña Comparativa (rendimiento)
    st.markdown("### Comparación de Rendimiento CNN vs Random Forest")

    metrics = {
        'Modelo': ['CNN', 'Random Forest'],
        'Precisión': [0.98, 0.94],  # Valores de ejemplo, reemplazar con los reales
        'Recall': [0.97, 0.92],
        'F1-Score': [0.975, 0.93]
    }

    metrics_df = pd.DataFrame(metrics)
    st.dataframe(metrics_df)

    fig, ax = plt.subplots()
    metrics_df.set_index('Modelo').plot(kind='bar', ax=ax)
    ax.set_ylabel('Valor')
    ax.set_title('Comparación de Métricas entre Modelos')
    st.pyplot(fig)
