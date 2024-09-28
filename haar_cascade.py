# Khởi tạo dữ liệu
# face_data_dir = 'D:/OpenCV/Haar/faces_and_non_faces_data/faces_24x24'  # Thư mục chứa ảnh khuôn mặt
# non_face_data_dir = 'D:/OpenCV/Haar/faces_and_non_faces_data/non_faces_24x24'  # Thư mục chứa ảnh không phải khuôn mặt

import streamlit as st
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.title('Face Detection using Haar Cascade and kNN')

# Hàm tải và xử lý ảnh từ webcam hoặc tệp tin
def process_image(image, model, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (24, 24)).flatten()  # Resizing ảnh thành 24x24
            label = model.predict([face_resized])[0]  # Dự đoán nhãn với mô hình kNN
            
            if label == 1:
                cv2.putText(image, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.putText(image, 'Non-Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return image

# Khởi tạo dữ liệu
face_data_dir = 'D:/OpenCV/Haar/faces_and_non_faces_data/faces_24x24'  # Thư mục chứa ảnh khuôn mặt
non_face_data_dir = 'D:/OpenCV/Haar/faces_and_non_faces_data/non_faces_24x24'  # Thư mục chứa ảnh không phải khuôn mặt

faces = []
non_faces = []

# Tải ảnh khuôn mặt
for file in os.listdir(face_data_dir):
    if file.endswith('.jpg') or file.endswith('.png'):
        img = cv2.imread(os.path.join(face_data_dir, file), cv2.IMREAD_GRAYSCALE)
        faces.append(img.flatten())  # Chuyển ảnh thành vector

# Tải ảnh không phải khuôn mặt
for file in os.listdir(non_face_data_dir):
    if file.endswith('.jpg') or file.endswith('.png'):
        img = cv2.imread(os.path.join(non_face_data_dir, file), cv2.IMREAD_GRAYSCALE)
        non_faces.append(img.flatten())  # Chuyển ảnh thành vector

# Gán nhãn cho dữ liệu: 1 cho khuôn mặt, 0 cho không phải khuôn mặt
face_labels = [1] * len(faces)
non_face_labels = [0] * len(non_faces)

# Gộp dữ liệu và nhãn lại thành một bộ dataset hoàn chỉnh
X = np.array(faces + non_faces)
y = np.array(face_labels + non_face_labels)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình kNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Non-Face', 'Face'])
conf_matrix = confusion_matrix(y_test, y_pred)

# Hiển thị các chỉ số đánh giá trên Streamlit
st.subheader("Evaluation Metrics")
st.text(f"Accuracy: {accuracy * 100:.2f}%")
st.text("Classification Report")
st.text(report)

# Vẽ confusion matrix
st.text("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Face', 'Face'], yticklabels=['Non-Face', 'Face'], ax=ax)
st.pyplot(fig)

# Khởi tạo Haar Cascade và webcam
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# Hiển thị webcam
st.header("Webcam Live Feed")
frame_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Dự đoán khuôn mặt hoặc không khuôn mặt
    frame = process_image(frame, knn, face_cascade)
    frame_placeholder.image(frame, channels="BGR")

cap.release()
