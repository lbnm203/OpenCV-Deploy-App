import cv2 as cv
import numpy as np
import os
import streamlit as st
from PIL import Image

# Thiết lập cấu hình trang để full width

def load_dataset():
    face = []
    non_face = []
    face_dataset = []
    
    dir_face = os.listdir('Haar/faces_24x24')
    dir_non_face = os.listdir('Haar/non_faces_24x24')

    for i in range(100):
        path_face = 'Haar/faces_24x24/' + dir_face[i]
        path_non = 'Haar/non_faces_24x24/' + dir_non_face[i]
        
        image_face = cv.imread(path_face)
        if image_face is not None:
            image_face = cv.cvtColor(image_face, cv.COLOR_BGR2GRAY)
            face.append(image_face)
        else:
            print(f"Không thể đọc ảnh: {path_face}")

        image_non = cv.imread(path_non)
        if image_non is not None:
            image_non = cv.cvtColor(image_non, cv.COLOR_BGR2GRAY)
            if len(non_face) < 50:
                non_face.append(image_non)
        else:
            print(f"Không thể đọc ảnh: {path_non}")    

    for i in range(len(face)):
        face_dataset.append(face[i])
    for i in range(len(non_face)):
        face_dataset.append(non_face[i])
    
    labels = np.array([1] * 100 + [0] * 50)
    face_dataset = np.array(face_dataset)

    return face_dataset, labels


def face_detection_app():
    st.title('✨ Ứng dụng Face Detection Haar')

    st.markdown(""" 
        * ####  Hướng dẫn sử dụng:
            - Chọn ảnh phía bên thanh trái để tải ảnh lên
            - Nhấn nút "Detect" để tiến hành nhận dạng ảnh tải lên
    """)

    st.divider()

    st.sidebar.title('Upload your image')
    image_upload = st.sidebar.file_uploader(" ", type=["png", "jpg", "jpeg"])

    if image_upload is not None:
        # Tạo 2 cột
        col1, col2 = st.columns(2)

        col1.markdown('### Ảnh trước khi nhận dạng')
        img = Image.open(image_upload)
        img = np.array(img)
        col1.image(img, channels="RGB")

        if col1.button('Detect', type="primary"):
            col2.markdown('### Ảnh sau khi nhận dạng')
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

            # haar_cascade = cv.CascadeClassifier('assets/haarcascade_frontalface_default.xml') 

            haar_cascade_path = 'assets/haarcascade_frontalface_default.xml'
            if not os.path.exists(haar_cascade_path):
                st.error("Tệp Haar Cascade không tồn tại. Vui lòng kiểm tra đường dẫn.")
                return
            
            haar_cascade = cv.CascadeClassifier(haar_cascade_path)
            if haar_cascade.empty():
                st.error("Không thể tải tệp Haar Cascade. Vui lòng kiểm tra tệp và đường dẫn.")
                return

            faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
            
            for (x, y, w, h) in faces_rect:
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            col2.image(img, channels="RGB")

            if len(faces_rect) > 0:
                col2.markdown('#### Kết quả: Có chứa khuôn mặt</span>', unsafe_allow_html=True)
            else:
                col2.markdown('#### Kết quả: Không chứa khuôn mặt', unsafe_allow_html=True)

