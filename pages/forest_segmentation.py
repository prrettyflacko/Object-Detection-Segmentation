import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import requests
from io import BytesIO

# Настройка страницы
st.set_page_config(page_title="Сегментация леса", page_icon="🌲", layout="wide")

st.title("🌲 Сегментация лесных массивов (U-Net)")
st.markdown("В данном модуле используется нейросеть архитектуры U-Net с энкодером EfficientNet-B4 для автоматического попиксельного выделения лесных зон.")
st.markdown("---")

# 1. Функция загрузки весов модели с кэшированием
@st.cache_resource
def load_forest_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None, 
        in_channels=3,
        classes=1
    )
    
    try:
        model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("⚠️ Файл весов 'best_model.pth' не найден в папке 'models/'.")
        return None, device

# Инициализируем модель
model, device = load_forest_model()

# 2. Интерфейс загрузки файлов
st.subheader("📥 Загрузка снимка")
tab1, tab2 = st.tabs(["Загрузить файл", "Скачать по ссылке"])

image_np = None

with tab1:
    uploaded_file = st.file_uploader("Выберите файл с компьютера...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

with tab2:
    url = st.text_input("Введите прямую ссылку на изображение (JPG/PNG):")
    if st.button("Скачать и обработать"):
        if url:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image_np = np.array(image)
            except Exception as e:
                st.error(f"Не удалось загрузить изображение: {e}")
        else:
            st.warning("Пожалуйста, введите ссылку.")

# 3. Обработка изображения
if image_np is not None and model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Исходный снимок")
        st.image(image_np, use_container_width=True)
        
    with st.spinner("Нейросеть производит попиксельный анализ снимка..."):
        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        augmented = transform(image=image_np)
        img_tensor = augmented['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (prob > 0.5).astype(np.uint8) * 255

    with col2:
        st.subheader("🎭 Предсказанная маска")
        st.image(mask, caption="Белые зоны — обнаруженный лес", use_container_width=True)
    
    st.markdown("---")
    st.subheader("🗺️ Визуализация наложения маски")
    overlay = image_np.copy()
    overlay[mask == 255] = [0, 220, 0] 
    blended = cv2.addWeighted(image_np, 0.6, overlay, 0.4, 0)
    st.image(blended, caption="Зеленый индикатор указывает на границы лесных массивов", use_container_width=True)
