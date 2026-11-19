# app.py
# Interfaz sencilla con Streamlit para probar Florence-2

import time
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Florence2ForConditionalGeneration

nombre_modelo = "florence-community/Florence-2-base"
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

# Nombres de las tareas en español y la instrucción que entiende el modelo
tareas = {
    "Descripción corta": "<CAPTION>",
    "Descripción detallada": "<MORE_DETAILED_CAPTION>",
    "Detección de objetos": "<OD>",
    "Leer texto (OCR)": "<OCR>",
}


# cache_resource hace que el modelo se cargue una sola vez
@st.cache_resource
def cargar_modelo():
    procesador = AutoProcessor.from_pretrained(nombre_modelo)
    modelo = Florence2ForConditionalGeneration.from_pretrained(nombre_modelo)
    modelo = modelo.to(dispositivo)
    modelo.eval()
    return procesador, modelo


def ejecutar_tarea(procesador, modelo, imagen, tarea, num_beams):
    entradas = procesador(text=tarea, images=imagen, return_tensors="pt")
    entradas = entradas.to(dispositivo)

    with torch.no_grad():
        ids = modelo.generate(**entradas, max_new_tokens=256, num_beams=num_beams)

    texto = procesador.batch_decode(ids, skip_special_tokens=False)[0]
    return procesador.post_process_generation(
        texto, task=tarea, image_size=imagen.size
    )


def dibujar_cajas(imagen, resultado):
    imagen_dibujo = imagen.copy()
    dibujo = ImageDraw.Draw(imagen_dibujo)
    fuente = ImageFont.load_default(size=30)

    for caja, etiqueta in zip(resultado["bboxes"], resultado["labels"]):
        x1, y1, x2, y2 = caja
        dibujo.rectangle([x1, y1, x2, y2], outline="red", width=4)

        # Medir el texto para dibujar un fondo rojo detrás
        izq, arriba, der, abajo = dibujo.textbbox((0, 0), etiqueta, font=fuente)
        ancho = der - izq
        alto = abajo - arriba

        # La etiqueta va encima de la caja, y si no cabe va adentro
        y_texto = y1 - alto - 10
        if y_texto < 0:
            y_texto = y1

        dibujo.rectangle(
            [x1, y_texto, x1 + ancho + 10, y_texto + alto + 10], fill="red"
        )
        dibujo.text((x1 + 5, y_texto + 5 - arriba), etiqueta, fill="white", font=fuente)
    return imagen_dibujo


# ---------- Interfaz ----------
st.title("Inferencia con Florence-2")
st.write("Dispositivo:", dispositivo)

archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
nombre_tarea = st.selectbox("Tarea", list(tareas.keys()))
num_beams = st.slider("num_beams (más alto = mejor calidad, más lento)", 1, 5, 1)

if archivo is not None:
    imagen = Image.open(archivo).convert("RGB")
    st.image(imagen, caption="Imagen original")

    if st.button("Ejecutar"):
        procesador, modelo = cargar_modelo()
        tarea = tareas[nombre_tarea]

        with st.spinner("Ejecutando el modelo..."):
            inicio = time.time()
            resultado = ejecutar_tarea(procesador, modelo, imagen, tarea, num_beams)
            fin = time.time()

        st.write("Tiempo:", round(fin - inicio, 1), "segundos")

        if tarea == "<OD>":
            st.image(dibujar_cajas(imagen, resultado[tarea]), caption="Detección")
            st.write(resultado[tarea]["labels"])
        else:
            st.text(resultado[tarea])