import os
import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# --- НАСТРОЙКА ПУТЕЙ ---
CURRENT_DIR = os.getcwd()
# Путь к вашей обученной модели (yolo11m согласно конфигу)
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "yolo11m_tumor_detection_best_config-2", "weights", "best.pt")
# Путь к папке с результатами обучения для отображения графиков
EXP_DIR = os.path.join(CURRENT_DIR, "models", "yolo11m_tumor_detection_best_config-2")

# Кэшируем загрузку модели, чтобы приложение не зависало при каждом клике
@st.cache_resource
def load_tumor_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    else:
        st.error(f"Файл модели не найден по пути: {MODEL_PATH}")
        return None

model = load_tumor_model()

# ==========================================
# ⚙️ НАСТРОЙКИ ДЕТЕКЦИИ (БОКОВАЯ ПАНЕЛЬ)
# ==========================================
st.sidebar.header("⚙️ Настройки детекции")
st.sidebar.write("Управляйте чувствительностью модели в реальном времени.")

# Ползунок для Confidence Threshold (Порог уверенности)
conf_threshold = st.sidebar.slider(
    "Порог уверенности (Conf)", 
    min_value=0.01, 
    max_value=1.00, 
    value=0.05,  # Режим бдительности по умолчанию
    step=0.01,
    help="Чем ниже порог, тем чаще модель будет реагировать на любые подозрительные области."
)

# Ползунок для IoU Threshold (Порог перекрытия рамок)
iou_threshold = st.sidebar.slider(
    "Порог перекрытия рамок (IoU)", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.80,  # Завышаем, чтобы не склеивать спорные близкие рамки
    step=0.05,
    help="Регулирует склеивание дублирующих рамок. Чем выше, тем больше альтернативных рамок останется."
)

# --- СТРАНИЦА STREAMLIT ---
st.title("🧠 Детекция опухолей головного мозга (YOLOv11m)")
st.write("Загрузите МРТ-снимки в аксиальной проекции для автоматического поиска патологий.")

# Разделяем интерфейс на вкладки: Инференс и Аналитика
tab_inference, tab_metrics = st.tabs(["🔍 Детекция", "📊 Информация о модели"])

# ==========================================
# ВКЛАДКА 1: ДЕТЕКЦИЯ (Инференс)
# ==========================================
with tab_inference:
    st.subheader("Загрузка данных")
    
    # 1. Загрузка нескольких файлов
    uploaded_files = st.file_uploader(
        "Выберите МРТ-снимки (можно несколько)", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )
    
    # 2. Подгрузка файла по прямой ссылке
    url_input = st.text_input("Или вставьте прямую ссылку на изображение МРТ:")
    
    images_to_process = []
    
    # Собираем картинки из загрузчика
    if uploaded_files:
        for file in uploaded_files:
            images_to_process.append((Image.open(file), file.name))
            
    # Собираем картинку по ссылке
    if url_input:
        try:
            response = requests.get(url_input, stream=True, timeout=10)
            if response.status_code == 200:
                img = Image.open(response.raw)
                images_to_process.append((img, "image_from_url.jpg"))
            else:
                st.error("Не удалось скачать изображение по ссылке. Проверьте URL.")
        except Exception as e:
            st.error(f"Ошибка при загрузке по ссылке: {e}")

    # Запуск предсказания
    if images_to_process:
        st.subheader("Результаты обработки:")
        
        if model is None:
            st.warning("Работа в демо-режиме: модель не загружена.")
        else:
            for img, name in images_to_process:
                st.write(f"**Файл:** {name}")
                
                # Передаем динамические значения из интерактивных ползунков
                results = model.predict(
                    img, 
                    imgsz=800, 
                    conf=conf_threshold, 
                    iou=iou_threshold
                )
                
                # Рендерим результат (рисуем bounding boxes)
                annotated_img = results[0].plot()
                
                # Выводим оригинальное и обработанное изображение в две колонки
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Оригинальный снимок", use_container_width=True)
                with col2:
                    st.image(annotated_img, channels="BGR", caption=f"Результат (Conf: {conf_threshold}, IoU: {iou_threshold})", use_container_width=True)
                st.markdown("---")
    else:
        st.info("Ожидание загрузки снимков для анализа...")

# ==========================================
# ВКЛАДКА 2: ИНФОРМАЦИЯ О МОДЕЛИ (По ТЗ)
# ==========================================
with tab_metrics:
    st.subheader("Параметры и конфигурация обучения")
    
    # Выводим РЕАЛЬНЫЕ параметры на основе вашего args.yaml в красивые карточки
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Эпохи (Epochs)", value="200")
    with col2:
        st.metric(label="Разрешение (Img Size)", value="800x800")
    with col3:
        st.metric(label="Оптимизатор", value="AdamW")
    with col4:
        st.metric(label="Архитектура", value="YOLOv11m (Medium)")
        
    st.markdown("""
    **Дополнительные детали конфигурации:**
    * **Объем выборки:** Папка `axial_t1wce_2_class` (~400 снимков).
    * **Аугментация:** Включены повороты (`degrees=15.0`), сдвиги (`translate=0.1`) и горизонтальное отражение (`fliplr=0.5`). 
    * **Mosaic & Mixup:** Отключены (`0.0`), чтобы не нарушать медицинскую анатомию снимков.
    * **Режим ранней остановки (Patience):** 50 эпох без улучшений.
    """)
    
    st.markdown("### 📊 Графики метрик качества")
    
    # Автоматический вывод графиков прямо из вашей папки эксперимента YOLO
    metrics_images = {
        "Матрица ошибок (Confusion Matrix)": "confusion_matrix.png",
        "Нормализованная матрица ошибок": "confusion_matrix_normalized.png",
        "PR-Кривая (Precision-Recall Curve)": "BoxPR_curve.png",
        "F1-Кривая (F1-Score по порогам уверенности)": "BoxF1_curve.png",
        "Результаты обучения (Функции потерь Loss и метрики по эпохам)": "results.png"
    }
    
    for title, img_name in metrics_images.items():
        full_img_path = os.path.join(EXP_DIR, img_name)
        if os.path.exists(full_img_path):
            st.write(f"**{title}**")
            st.image(full_img_path, use_container_width=True)
            st.markdown("---")
        else:
            st.caption(f"График '{img_name}' не найден в папке эксперимента `{EXP_DIR}`.")