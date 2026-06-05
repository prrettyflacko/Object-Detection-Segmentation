import os
import io
import requests
import urllib.request  # ДОБАВЛЕН ИМПОРТ
import streamlit as st
import cv2
import numpy as np
import torch           # ДОБАВЛЕН ДЛЯ ОПРЕДЕЛЕНИЯ GPU
from PIL import Image
from ultralytics import YOLO

# Убран st.set_page_config, так как он уже есть в app.py

WEIGHTS_URL = "https://github.com/Expat777/Object-Detection-Segmentation/releases/download/v01/best.pt"
WEIGHTS_PATH = '/home/vitaliy/runs/detect/train-28/weights/best.pt'

# Авто-определение: если есть GPU, используем его (0), иначе CPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_models():
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    
    if not os.path.exists(WEIGHTS_PATH) and "your-cloud-storage" not in WEIGHTS_URL:
        with st.spinner("Downloading model weights from cloud storage... Please wait..."):
            try:
                tmp_path = WEIGHTS_PATH + ".tmp"
                urllib.request.urlretrieve(WEIGHTS_URL, tmp_path)
                os.rename(tmp_path, WEIGHTS_PATH)
                st.success("Weights downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download weights from cloud: {e}")
                return None

    if os.path.exists(WEIGHTS_PATH):
        return YOLO(WEIGHTS_PATH)
    else:
        st.error(f"Weights file not found at {WEIGHTS_PATH}. Please check the path.")
        return None

detect_model = load_models()
if detect_model is None:
    st.stop()

page = st.sidebar.selectbox('Please choose pages', ['Detect and Blur Face', 'About model and process of fit'])

if page == 'Detect and Blur Face':
    st.title('Detection and Anonymazing Faces')
    st.write('Load pic`s from your PC or use URL on pic`s')
    uploaded_files = st.file_uploader(
        'please choose pic for fit(one or more)',
        type=['jpeg', 'jpg', 'png'],
        accept_multiple_files=True
    )
    url_input = st.text_input('put Here you URL')

    images_to_process = []

    if uploaded_files:
        for file in uploaded_files:
            images_to_process.append((file.name, Image.open(file)))

    if url_input:
        try:
            response = requests.get(url_input, timeout=10)
            if response.status_code == 200:
                img_name = url_input.split('/')[-1] or 'url_image.jpg'
                images_to_process.append((img_name, Image.open(io.BytesIO(response.content))))
            else:
                st.error(f' cannot download files by URL')
        except Exception as e:
            st.error(f'URL problem {e}')

    if images_to_process:
        st.success(f' Ready to fit {len(images_to_process)}')

        conf_threshold = st.slider('Confidence', 0.1, 1.0, 0.15)
        for name, img in images_to_process:
            st.subheader(f' File {name}')

            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            anon_img = img_bgr.copy()

            # Заменено на динамический DEVICE вместо жесткого device=0
            results = detect_model.predict(source=img_bgr, conf=conf_threshold, imgsz=800, device=DEVICE, verbose=False)[0]
            
            if len(results.boxes) > 0:
                for box in results.boxes:
                    # Безопасное извлечение координат без избыточного squeeze
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    face_roi = anon_img[y1:y2, x1:x2]

                    if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
                        ksize_x = int(face_roi.shape[1] * 0.4) | 1
                        ksize_y = int(face_roi.shape[0] * 0.4) | 1
                        blur_face = cv2.GaussianBlur(face_roi, (ksize_x, ksize_y), 0)
                        anon_img[y1:y2, x1:x2] = blur_face

                annot_img = results.plot(labels=True, conf=True, img=anon_img)
                final_img = cv2.cvtColor(annot_img, cv2.COLOR_BGR2RGB)
            else:
                final_img = img_np

            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption='Original', use_container_width=True)
            with col2:
                st.image(final_img, caption='Blur_Face', use_container_width=True)

elif page == 'About model and process of fit':
    st.title('Info about model and quality of fit')
        
    yolo_run_dir = '/home/vitaliy/runs/detect/train-28'

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label='Epochs', value=25)
    with col2:
        st.metric(label='Value of train ', value = '13 386 Pic`s')
    with col3:
        st.metric(label='mAP50', value='0.883')
    with col4:
        st.metric(label='model', value ='yolo11x')
    st.write('---')

    st.header('Graphics fit quality')

    metrics_mapping = {
        "Матрица ошибок (Confusion Matrix)": "confusion_matrix.png",
        "PR-кривая (Precision-Recall Curve)": "BoxPR_curve.png",
        "F1-кривая (F1-Confidence Curve)": "BoxF1_curve.png",
        "Результаты по эпохам (Loss & Metrics)": "results.png"
    }

    col_g1, col_g2 = st.columns(2)

    for idx, (title, filename) in enumerate(metrics_mapping.items()):
        full_path = os.path.join(yolo_run_dir, filename)
        target_col = col_g1 if idx % 2 == 0 else col_g2
        with target_col:
            st.subheader(title)
            if os.path.exists(full_path):
                st.image(full_path, use_container_width=True)
            else:
                st.info(f'Graph {filename} not found in {yolo_run_dir}')
