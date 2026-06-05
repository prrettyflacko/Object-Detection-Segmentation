import os
import streamlit as st

# 1. Общая конфигурация приложения (СТРОГО ОДИН РАЗ ЗДЕСЬ)
st.set_page_config(
    page_title="• Computer Vision Project",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Функция-заглушка для отображения "В разработке"
def show_placeholder(module_name):
    st.title(f"🚧 Модуль «{module_name}» в разработке")
    st.info("Ожидаем интеграцию кода от разработчиков проекта. Этот раздел скоро обновится!")
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <span style="font-size: 70px;">⏳</span>
        <h3 style="color: gray;">Coming Soon...</h3>
    </div>
    """, unsafe_allow_html=True)

# Функция для безопасного создания объекта страницы
def create_safe_page(filename, module_name, title, icon, url_path):
    full_path = os.path.join("pages", filename)
    if os.path.exists(full_path):
        # Если файл есть, Streamlit сам правильно его запустит
        return st.Page(full_path, title=title, icon=icon, url_path=url_path)
    else:
        # Если файла нет — подсовываем функцию-заглушку
        return st.Page(lambda: show_placeholder(module_name), title=title, icon=icon, url_path=url_path)

# 2. Описание структуры страниц
pages = {
    "Главная": [
        st.Page(lambda: None, title="О проекте", icon="🏠", default=True)
    ],
    "Задачи Детекции (YOLO)": [
        create_safe_page("face_blur.py", "Маскировка лиц", "Маскировка лиц", "👤", "face-blur"),
        create_safe_page("tumor_detection.py", "Детекция опухолей мозга", "Детекция опухолей мозга", "🧠", "tumor-detection")
    ],
    "Задачи Сегментации (U-Net)": [
        create_safe_page("forest_segmentation.py, "Сегментация снимков Земли", "Сегментация снимков Земли", "", "satellite")
    ]
}

# 3. Инициализация навигации
pg = st.navigation(pages)

# 4. Проверяем, выбрана ли Главная страница
if pg == pages["Главная"][0]:
    st.markdown("""
        <div style="background-color:#1E1E1E; padding:20px; border-radius:10px; border-left: 8px solid #FF4B4B;">
            <h1 style="color:white; margin:0;">🚀 Computer Vision Project</h1>
            <p style="color:#FAFAFA; font-size:18px; margin-top:10px;">
                Многофункциональное веб-приложение для решения задач детекции и семантической сегментации.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("") 
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🤖 Реализованные модули")
        st.markdown("""
        - **Маскировка лиц (YOLO):** Обнаружение лиц на фото с последующим размытием (Blur) конфиденциальных зон.
        - **Детекция опухолей мозга (YOLOv11m):** Анализ МРТ-снимков головного мозга в аксиальной проекции (T1wce) с выводом графиков обучения модели.
        - **Сегментация снимков (U-Net):** Попиксельное выделение объектов на аэрокосмических снимках.
        """)
        
        st.subheader("📦 Использованный стек технологий")
        st.info("Python 3.13 • Ultralytics YOLOv11 • PyTorch • Streamlit • OpenCV / Pillow")

    with col2:
        st.subheader("👥 Наша Команда")
        st.success("👨‍💻 **Роман** — Детекция опухолей мозга (YOLOv11m) — *Готово*")
        st.success("👨‍💻 **Виталий** — Маскировка лиц (YOLO) — *Готово*")
        st.warning("👨‍💻 **** — Сегментация снимков (U-Net) — *В разработке*")
        
        st.markdown("---")
        st.caption("Используйте боковое меню слева 👈 для переключения между интерактивными страницами моделей.")
else:
    # Запуск выбранной страницы силами самого Streamlit
    pg.run()
