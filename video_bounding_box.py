from statistics import mode

import cv2
from tensorflow.python.keras.models import load_model
from keras.preprocessing.image import img_to_array
from tensorflow.python.keras.preprocessing import image
import numpy as np
#import matplotlib.pyplot as plt
import random


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def get_labels():
        return {0:'Angry',1:'Disgusted',2:'Fear',3:'Happy',
                4:'Sad',5:'Surprised',6:'Neutral'}

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
       
def show_boxes(inputfile):
    # parameters for loading data and images
    detection_model_path = 'haar/haarcascade_frontalface_default.xml'
    emotion_model_path = 'model/emotion_detection_bounding_boxes.hdf5'
    emotion_labels = get_labels()
    gender_model_path = 'model/gender_detection.model'
    gender_classes = ['man','woman']
    # hyper-parameters for bounding boxes shape
    frame_window = 1
    emotion_offsets = (10, 20)

    # loading models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    gender_classification = load_model(gender_model_path, compile=False)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    #the mode of the list is the item that occurs most often.
    emotion_window = []
    COLOR = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
    cap =cv2.VideoCapture(inputfile)
    print(cap.isOpened())
    ret, frame = cap.read()
    count = 0 
    feat_array = list()
    gender_array = list()
    time_array = list()
    print("Detecting gender and emotion confidence")
    while(ret):
            #cv2.imshow('video',frame)
            if ret is True:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                continue

            faces = detect_faces(face_detection,gray_image)        
            #print(f'frame {count}')

            #print(len(faces))
            if len(faces) == 0 and count%30==0:
                feat_array.append(np.zeros(7))
                gender_index = random.randint(0,100)%2
                gender_array.append(gender_index)
                time_array.append(count//30)
            for face_coordinates in faces:
                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]
                gender_face = np.copy(frame[y1:y2, x1:x2])
                try:
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                    #cv2.imshow('Haar_Output',gray_face)
                except:
                    continue
                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = emotion_classifier.predict(gray_face)
                emotion_prediction = np.mean(emotion_prediction,axis=0)
                #print(emotion_prediction.shape)
                #emotion_prediction = np.reshape(emotion_prediction,(1,7))
                #print(emotion_prediction)
                face_crop = cv2.resize(gender_face, (96,96))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)
                gender_conf = gender_classification.predict(face_crop)[0]
                gender_label = ""
                idx = -1
                try:
                    idx = np.argmax(gender_conf)
                    gender_label = gender_classes[idx]
                except:
                    gender_label = "false_detection"
                if count%30 ==0:
                    feat_array.append(emotion_prediction)
                    time_array.append(count//30)
                    gender_array.append(idx)
                #print(emotion_prediction)  #Show Model Prediction --4
                emotion_probability = np.max(emotion_prediction)
                #print(emotion_probability) # Show Maximum Probability --5
                emotion_label_arg = np.argmax(emotion_prediction)   
                #print(emotion_label_arg)   #Determine the index of max emotion_prediction array  --6
                emotion_text = emotion_labels[emotion_label_arg]
                emotion_mode =emotion_text
            
                if emotion_text == 'angry':
                    color = emotion_probability * np.asarray((0, 0, 255)) #Red
                elif emotion_text == 'sad':
                    color = emotion_probability * np.asarray((255, 0, 0)) #Blue
                elif emotion_text == 'happy':
                    color = emotion_probability * np.asarray((0, 255, 0)) #Green
                elif emotion_text == 'surprise':
                    color = emotion_probability * np.asarray((0, 255, 255)) #bluish green
                elif emotion_text == 'neutral':
                    color = emotion_probability * np.asarray((255, 255, 0))#Yellow
                elif emotion_text == 'fear':
                    color = emotion_probability * np.asarray((255, 255, 255))#White
                else:
                    color = emotion_probability * np.asarray((0, 0, 0))#Black

                color = color.astype(int)
                color = color.tolist()
                #draw_bounding_box(face_coordinates, frame, color)
                #draw_text(face_coordinates, frame, emotion_mode, color, 0, -45, 1, 1)    
                cv2.rectangle(frame,(x1,y1),(x2,y2),(36,255,12), 2)
                cv2.putText(frame, emotion_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color)
                cv2.putText(frame, gender_label, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color)
                #cv2.putText(frame, count//30, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR)
            count += 1
            cv2.imshow('Video box', frame)#Show Final Output --7
            ret, frame = cap.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # feat_in_second = list()
    # for i in range(0,len(feat_array),30):
    #     base = i 
    #     end = base + 30 if base + 30 < len(feat_array) else len(feat_array)
    #     nparr = np.array(feat_array[base:end][:])
    #     feat_in_second.append(np.mean(nparr,axis=0))

    return feat_array,gender_array,time_array
    cap.release()
    cv2.destroyAllWindows()

#show_boxes()