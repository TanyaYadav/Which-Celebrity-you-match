import streamlit as st
from PIL import Image
from keras_vggface import VGGFace
import os
import pickle
import cv2
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

detector=MTCNN()
model=VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
feature_list=pickle.load(open('embedding.pkl','rb'))
filenames=pickle.load(open('filenames.pkl','rb'))
def save_uploaded_image(uploaded_img):
    try:
        with open(os.path.join('uploads',uploaded_img.name),'wb') as f:
            f.write(uploaded_img.getbuffer())
        return True
    except:
        return False
def extract_features(img_path,model,detector):
    img=cv2.imread(img_path)
    results=detector.detect_faces(img)
    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image)
    face_array = face_array.astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result


def recommend(feature_list,features):
    similarity = []

    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.title('Which bollywood celebrity are you?')
uploaded_img=st.file_uploader('Choose an image')
if uploaded_img is not None:
    if save_uploaded_image(uploaded_img):
        display_img=Image.open(uploaded_img)
        st.image(display_img,200)
        features=extract_features(os.path.join('uploads',uploaded_img.name),model,detector)
        index_pos=recommend(feature_list,features)
        predicted_actor=" ".join(filenames[index_pos].split('\\')[1].split('_'))
        col1,col2=st.columns(2)
        with col1:
            st.header('Your uploaded image')
            st.image(display_img,width=250)
        with col2:
            st.header('Seems like '+predicted_actor)
            st.image(filenames[index_pos],width=300)



