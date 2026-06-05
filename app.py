import os
import streamlit as st

# 1. Общая конфигурация приложения
st.set_page_config(
    page_title="Yolo Team • Computer Vision Project",
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

# Функция-обёртка для безопасного импорта страниц
def load_page(filename, module_name):
    full_path = os.path.join("pages", filename)
    if os.path.exists(full_path):
        # Если файл существует — Streamlit выполнит его код
        with open(full_path, "r", encoding="utf-8") as f:
            code = f.read()
            exec(code, globals())
    else:
        # Если файла на диске нет — выводим красивую заглушку
        show_placeholder(module_name)

# 2. Описание структуры страниц (Используем функции запуска, чтобы не было конфликтов с URL)
pages = {
    "Главная": [
        st.Page(lambda: None, title="О проекте", icon="🏠", default=True)
    ],
    "Задачи Детекции (YOLO)": [
        st.Page(lambda: load_page("face_blur.py", "Маскировка лиц"), title="Маскировка лиц", icon="👤", url_path="face-blur"),
        st.Page(lambda: load_page("tumor_detection.py", "Детекция опухолей мозга"), title="Детекция опухолей мозга", icon="🧠", url_path="tumor-detection")
    ],
    "Задачи Сегментации (U-Net)": [
        st.Page(lambda: load_page("satellite.py", "Сегментация снимков Земли"), title="Сегментация снимков Земли", icon="🛰️", url_path="satellite")
    ]
}

# 3. Инициализация навигации
pg = st.navigation(pages)

# 4. Проверяем, выбрана ли Главная страница
# В Streamlit это делается через сравнение объекта страницы с дефолтным
if pg == pages["Главная"][0]:
    # Красивый баннер/заголовок проекта
    st.markdown("""
        <div style="background-color:#1E1E1E; padding:20px; border-radius:10px; border-left: 8px solid #FF4B4B;">
            <h1 style="color:white; margin:0;">🚀 Computer Vision Project • Yolo Team</h1>
            <p style="color:#FAFAFA; font-size:18px; margin-top:10px;">
                Многофункциональное веб-приложение для решения задач детекции и семантической сегментации.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("") # Отступ
    
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
        st.warning("👨‍💻 **Коллега** — Маскировка лиц (YOLO) — *В разработке*")
        st.warning("👨‍💻 **Коллега 2** — Сегментация снимков (U-Net) — *В разработке*")
        
        st.markdown("---")
        st.caption("Используйте боковое меню слева 👈 для переключения между интерактивными страницами моделей.")

else:
    # Если выбрана любая другая страница — запускаем её обёртку через .run()
    pg.run()