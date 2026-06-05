import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Настройка страницы
st.set_page_config(page_title="Сегментация леса", page_icon="🌲", layout="wide")

st.title("🌲 Сегментация лесных массивов (U-Net)")
st.markdown("В данном модуле используется нейросеть архитектуры U-Net с энкодером EfficientNet-B4 для автоматического попиксельного выделения лесных зон на аэрокосмических и спутниковых снимках.")
st.markdown("---")

# 1. Функция загрузки весов модели с кэшированием
@st.cache_resource
def load_forest_model():
    # Streamlit Cloud обычно работает на CPU, поэтому делаем умный выбор девайса
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Инициализируем структуру сети (как при обучении)
    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,  # Веса берем локальные
        in_channels=3,
        classes=1
    )
    
    try:
        # Загружаем веса из папки models/
        model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("⚠️ Файл весов 'best_model.pth' не найден в папке 'models/'. Пожалуйста, загрузите его туда на GitHub.")
        return None, device

# Инициализируем модель
model, device = load_forest_model()

# 2. Интерфейс загрузки файлов
uploaded_file = st.file_uploader("Загрузите аэрокосмический или спутниковый снимок...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Читаем картинку через PIL и переводим в RGB numpy массив
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    # Создаем две колонки для отображения "До / После"
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Исходный снимок")
        st.image(image, use_container_width=True)
        
    # Запускаем процесс предсказания с анимацией загрузки
    with st.spinner("Нейросеть производит попиксельный анализ снимка..."):
        # Нормализация ImageNet и перевод в тензор (как на валидации при обучении)
        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # Подготовка тензора
        augmented = transform(image=image_np)
        img_tensor = augmented['image'].unsqueeze(0).to(device)
        
        # Инференс модели
        with torch.no_grad():
            output = model(img_tensor)
            # Применяем Сигмоиду и пороговую фильтрацию > 0.5
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = (prob > 0.5).astype(np.uint8) * 255

    with col2:
        st.subheader("🎭 Предсказанная маска")
        st.image(mask, caption="Белые зоны — обнаруженный лес", use_container_width=True)
        
    # 3. Визуальное наложение (Overlay) для интерактивности
    st.markdown("---")
    st.subheader("🗺️ Визуализация наложения маски на местность")
    
    # Создаем зеленую маску-подсветку
    overlay = image_np.copy()
    overlay[mask == 255] = [0, 220, 0]  # Насыщенный зеленый цвет для лесных зон
    
    # Смешиваем оригинальную картинку и зеленую подсветку в пропорции 60/40
    blended = cv2.addWeighted(image_np, 0.6, overlay, 0.4, 0)
    
    st.image(blended, caption="Зеленый индикатор указывает на границы лесных массивов", use_container_width=True)