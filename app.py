import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import pydicom
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



# Cargar el modelo
model = load_model('modelo_fracturas.h5')

def preprocess_image(image):
    '''Preprocesar la imagen cargada para el modelo CNN.'''
    img = np.array(image)
    img = cv2.resize(img, (128, 128))  # Redimensionar a 128x128
    img = img / 255.0  # Normalizar
    img = np.expand_dims(img, axis=0)  # Añadir dimensión batch
    return img


def load_dicom(path):
    '''Función para cargar y transformar imágenes DICOM'''
    img = pydicom.dcmread(path)
    
    # Obtén el array de píxeles
    data = img.pixel_array
    
    # Si la imagen tiene más de 8 bits por píxel, normalizamos a uint8
    if data.dtype != np.uint8:
        data = data - np.min(data)  # Escalar valores entre 0 y max
        data = (data / np.max(data)) * 255  # Normalizar a 255
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




st.title("Detección de Fracturas Cervicales")
st.write("Sube una imagen de tomografía para realizar la predicción.")

# Subida de imagen
uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "png", "jpeg", "dcm"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada', use_column_width=True)

    if st.button('Predecir'):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        st.write("Predicciones por vértebra (C1-C7):")
        for i, prob in enumerate(prediction[0]):
            st.write(f"C{i+1}: {prob:.4f}")
