import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import cv2

def img_prepoces(ruta, tupla):
    img = cv2.imread(ruta)
    if img is None:
        print(f"Couldn't read the image at {ruta}.")
        return None
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (tupla[0], tupla[1])) / 255


# Cargar el modelo desde el archivo
modelo = load_model('prueba/modelos/con_data/mejor_modelo.keras')

# Cargar los datos de prueba
test = pd.read_csv('prueba/archive/Testing_set.csv')

# Preprocesar las imágenes de prueba y las etiquetas de la misma manera que lo hiciste para los datos de entrenamiento y validación
test_img = [img_prepoces("prueba/archive/test/"+name,(224,224)) for name in tqdm(test['filename'])]
test_names = test['label']

# Convertir a arrays de NumPy y redimensionar
X_test_img = np.array(test_img).reshape(-1, 224, 224, 3)

# Convertir las etiquetas a formato one-hot
y_test_names = to_categorical(LabelEncoder().fit_transform(test_names))

# Evaluar el modelo en los datos de prueba
test_loss, test_accuracy = modelo.evaluate(X_test_img, y_test_names)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
