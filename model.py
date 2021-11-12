import cv2
from keras.models import load_model
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time

# Load model
model = load_model('model_5class_resnet_87%.h5')

labels = ['G&M', 'Organic', 'Other', 'Paper', 'Plastic']
le = LabelEncoder()
labels = le.fit_transform(labels)

def pre(img_path):
    img = cv2.resize(img_path, (224, 224))
    img = imagenet_utils.preprocess_input(img)
    img = img_to_array(img)
    img = np.expand_dims(img / 255, 0)
    return img

cap = cv2.VideoCapture(0)

# Using IP camera
'''
address = 'https://192.168.0.122:8080/video' 
cap.open(address)
'''

success, img = cap.read()

font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)
org1 = (50, 150)

# fontScale
fontScale = 3

# Green color in RGB
color = (0, 255, 0)

# Line thickness of 2 px
thickness = 2

def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

while True:
    start_time = time.time()
    success, img = cap.read()
    img1 = pre(img)
    p = model.predict(img1)
    confidences = max(np.squeeze(model.predict(img1)))
    conf = round(confidences, 3)
    predicted_class = le.classes_[np.argmax(p[0], axis=-1)]
    print('FPS:', 1.0 / (time.time() - start_time))
    if float(to_str(conf)) > 0.85:
        cv2.putText(img, predicted_class, org, font,
                       fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(img, to_str(conf), org1, font,
                    fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Model1', img)
    k = cv2.waitKey(1)

