# prueba_inferencia.py
# Prueba rápida: cargar Florence-2 y describir una imagen

import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

# Nombre del modelo en Hugging Face
nombre_modelo = "florence-community/Florence-2-base"

# Usar GPU si hay, si no usar CPU
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo:", dispositivo)

# El procesador prepara la imagen y el texto, el modelo genera la respuesta
procesador = AutoProcessor.from_pretrained(nombre_modelo)
modelo = Florence2ForConditionalGeneration.from_pretrained(nombre_modelo)
modelo = modelo.to(dispositivo)
modelo.eval()

# Cargar la imagen y escoger la tarea
imagen = Image.open("prueba.jpg").convert("RGB")
import time

# Probar varias tareas sobre la misma imagen
tareas = ["<CAPTION>", "<OD>"]

for tarea in tareas:
    inicio = time.time()

    entradas = procesador(text=tarea, images=imagen, return_tensors="pt")
    entradas = entradas.to(dispositivo)

    with torch.no_grad():
        ids_generados = modelo.generate(**entradas, max_new_tokens=256, num_beams=1)

    texto = procesador.batch_decode(ids_generados, skip_special_tokens=False)[0]
    resultado = procesador.post_process_generation(
        texto, task=tarea, image_size=imagen.size
    )

    fin = time.time()
    print(tarea, "->", resultado)
    print("Tiempo:", round(fin - inicio, 1), "segundos")
    print()