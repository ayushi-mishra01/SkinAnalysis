import cv2
import os
from typing import List
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from models.skin_tone.skin_tone_knn import identify_skin_tone
from flask import Flask, request
from flask_restful import Api, Resource, reqparse
import werkzeug
from models.recommender.rec import recs_essentials, makeup_recommendation
import base64
from io import BytesIO
import matplotlib.pyplot as plt

app = Flask(__name__)
api = Api(app)

class_names1 = ['Dry_skin', 'Normal_skin', 'Oil_skin']
class_names2 = ['Low', 'Moderate', 'Severe']
skin_tone_dataset = 'models/skin_tone/skin_tone_dataset.csv'

def get_model():
    global model1, model2, age
    model1 = load_model('./models/skin_model')
    print('Model 1 loaded')
    model2 = load_model('./models/acne_model')
    print("Model 2 loaded!")

    age1 = "age_deploy.prototxt"
    age2 = "age_net.caffemodel"
    age = cv2.dnn.readNet(age2, age1)
    print("Age model loaded!")

def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0
    return img_tensor

def predict_age(img_path):
    model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0-2)', '(4-6)', '(15-20)', '(21-24)','(25-32)', '(38-43)', '(48-53)', '(60-100)']

    img = cv2.imread(img_path)
    face_blob = cv2.dnn.blobFromImage(image=img, scalefactor=1.0, size=(227, 227), mean=model_mean_values, swapRB=False)
    
    age.setInput(face_blob)
    age_preds = age.forward()
    age_label = age_list[age_preds[0].argmax()]

    return age_label


def prediction_skin(img_path):
    new_image = load_image(img_path)
    pred1 = model1.predict(new_image)
    if len(pred1[0]) > 1:
        pred_class1 = class_names1[tf.argmax(pred1[0])]
    else:
        pred_class1 = class_names1[int(tf.round(pred1[0]))]
    return pred_class1

def prediction_acne(img_path):
    new_image = load_image(img_path)
    pred2 = model2.predict(new_image)
    if len(pred2[0]) > 1:
        pred_class2 = class_names2[tf.argmax(pred2[0])]
    else:
        pred_class2 = class_names2[int(tf.round(pred2[0]))]
    return pred_class2

def wrinkle_analysis(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (421, 612))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 10, 100)
    ret, thresh = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    wrinkle_count = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 5 and w < 50 and h < 20 and w / h > 2:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            wrinkle_count += 1
    wrinkle_grade = min(100, wrinkle_count)
    return 100-wrinkle_grade

def redness_grade(img_path):
    image = cv2.imread(img_path)
 
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
 
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
 
    mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
 
    mask = mask1 + mask2
 
    total_pixels = image.shape[0] * image.shape[1]
    red_pixels = np.sum(mask > 0)
    redness_grade = round((red_pixels / total_pixels) * 100)
 
    return 100-redness_grade

def dark_circle_analysis(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (421, 612))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    circles = cv2.HoughCircles(
        blur, 
        cv2.HOUGH_GRADIENT, 
        1, 
        20, 
        param1=50, 
        param2=30, 
        minRadius=10, 
        maxRadius=50
    )
    dark_circle_count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            if 10 < i[2] < 50:
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                dark_circle_count += 1
    if dark_circle_count > 5: 
        dark_circle_grade = 60
    else:
        dark_circle_grade = 100 - (dark_circle_count * 10)
    dark_circle_grade = min(max(dark_circle_grade, 0), 100)
    
    return dark_circle_grade



get_model()

img_put_args = reqparse.RequestParser()
img_put_args.add_argument("file", help="Please provide a valid image file", required=True)

rec_args = reqparse.RequestParser()
rec_args.add_argument("tone", type=int, help="Argument required", required=True)
rec_args.add_argument("type", type=str, help="Argument required", required=True)
rec_args.add_argument("features", type=dict, help="Argument required", required=True)

class Recommendation(Resource):
    def put(self):
        args = rec_args.parse_args()
        features = args['features']
        tone = args['tone']
        skin_type = args['type'].lower()
        skin_tone = 'light to medium' if tone <= 2 else 'fair to light' if tone >= 4 else 'medium to dark'
        fv = [int(value) for value in features.values()]
        general = recs_essentials(fv, None)
        makeup = makeup_recommendation(skin_tone, skin_type)
        return {'general': general, 'makeup': makeup}

class SkinMetrics(Resource):
    def put(self):
        args = img_put_args.parse_args()
        file = args['file']
        starter = file.find(',')
        image_data = file[starter + 1:]
        image_data = bytes(image_data, encoding="ascii")
        im = Image.open(BytesIO(base64.b64decode(image_data)))
        filename = 'image.png'
        file_path = os.path.join('./static', filename)
        im.save(file_path)
        skin_type = prediction_skin(file_path).split('_')[0]
        acne_type = prediction_acne(file_path)
        tone = identify_skin_tone(file_path, dataset=skin_tone_dataset)
        wrinkle_grade = wrinkle_analysis(file_path)
        redness_grade_value = redness_grade(file_path)
        darkCircle_grade = dark_circle_analysis(file_path)
        age = predict_age(file_path) 
        return {'type': skin_type, 'tone': str(tone), 'acne': acne_type, 'wrinkle_grade': wrinkle_grade, 'redness_grade': redness_grade_value, 'darkCircle_grade':darkCircle_grade, 'age': age}, 200
        # return {'type': skin_type, 'tone': str(tone), 'acne': acne_type, 'wrinkle_grade': wrinkle_grade}, 200

api.add_resource(SkinMetrics, "/upload")
api.add_resource(Recommendation, "/recommend")

if __name__ == "__main__":
    app.run(debug=False)
