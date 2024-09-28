import cv2 as cv
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import grabcut as ga
import watershed_app as ws

choice = st.sidebar.selectbox("Chọn thuật toán", ['GrabCut', 'WaterShed', 'Harr'])

if choice == 'GrabCut':
    st.sidebar.write("## Tải ảnh lên")
    uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])

    st.write("# ✨ Ứng dụng GrabCut")

    st.divider()

    st.markdown("""
        ## Hướng dẫn cách dùng
        - Chọn ảnh muốn tách nền
        - Chọn chế độ vẽ bên thanh trái
        - Giữ chuột phải vào ảnh để vẽ hình chữ nhật quanh đối tượng cần tách nền
            - Lưu ý: vẽ hình chữ nhật đầu tiên và chỉ vẽ 1 lần. 
        - Để xóa thao tác, click vào biểu tượng thùng rác 🗑️ dưới ảnh 
        - Ấn vào biểu tượng "Chạy Grabcut để phân đoạn" để tiến hành chạy

    """)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.resize((700, 500))
        # st.image(image, caption='Ảnh được tải lên', use_column_width=True) 

        img_np = np.array(image)
        app = ga.grabcut_app(img_np)
        # app = grabcut_app(img_np)


if choice == 'WaterShed':
    st.sidebar.write("## Tải lên ảnh")
    uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])

    st.write("# ✨ Ứng dụng WaterShed")

    




