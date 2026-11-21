# Inferencia con Florence-2

Proyecto final de Procesamiento de Datos Secuenciales, Especialización en Inteligencia Artificial, Universidad Autónoma de Occidente.

Integrantes: Juan Esteban Aristizabal, Juan Felipe Lopez y John Angel Posso.

## 1. Resumen

Pendiente. Se escribe al final, cuando estén los resultados de la interfaz y las pruebas en GPU.

## 2. Introducción

Este trabajo toma como base el artículo "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks" de Xiao et al., publicado en CVPR 2024. Los pesos preentrenados y el código del modelo están disponibles en https://huggingface.co/florence-community/Florence-2-base, y la implementación que usamos es la que viene integrada en la librería transformers de Hugging Face.

Normalmente describir una imagen, leer el texto que aparece en ella y detectar objetos son tareas que se resuelven con modelos distintos. Florence-2 las resuelve con un solo Transformer de tipo encoder-decoder, que recibe la imagen junto con una instrucción corta escrita en texto y devuelve la respuesta también en texto. El objetivo de este proyecto no es entrenar el modelo, sino cargar los pesos ya entrenados, ejecutar la inferencia, mostrar los resultados en una interfaz y explicar en detalle cómo funciona la arquitectura y su mecanismo de atención.

## 3. Marco teórico

Pendiente. Aquí van la arquitectura encoder-decoder, la atención y cómo se generan los tensores Q, K y V, y las innovaciones del modelo.

## 4. Metodología

Trabajamos con el modelo Florence-2 en su versión base, de 0.23 mil millones de parámetros, sin ningún entrenamiento adicional. Los pesos se descargan de Hugging Face la primera vez que se ejecuta el código. El entorno se maneja con uv, y las librerías principales son torch, transformers, pillow, matplotlib y streamlit. Usamos la clase Florence2ForConditionalGeneration de transformers, que ya incluye la arquitectura completa, en lugar del código remoto original de Microsoft, porque ese último dejó de funcionar con las versiones recientes de la librería.

## 5. Desarrollo e implementación

Para ejecutar el proyecto:

1. Clonar el repositorio y entrar a la carpeta.
2. Instalar las librerías con `make instalar` (equivale a `uv sync`).
3. Abrir la interfaz con `make app` (equivale a `uv run streamlit run app.py`).
4. Opcionalmente, probar el modelo por consola con `make prueba`, que corre el script `prueba_inferencia.py` sobre una imagen llamada `prueba.jpg` puesta en la carpeta del proyecto.

El comando `make` no viene instalado en Windows. En ese sistema se puede instalar con `winget install ezwinports.make`, o se pueden usar directamente los comandos de uv que aparecen entre paréntesis.

La primera vez que se ejecuta, el modelo descarga sus pesos de Hugging Face, que pesan unos 463 MB. Los pesos se cargan con la clase Florence2ForConditionalGeneration de la librería transformers, y el procesador (AutoProcessor) se encarga del preprocesamiento: ajusta el tamaño de la imagen, la normaliza y convierte la instrucción de texto en tokens. Después se llama a `generate`, donde el encoder procesa la imagen y la instrucción, y el decoder escribe la respuesta token por token. Por último, `post_process_generation` interpreta ese texto según la tarea, por ejemplo convirtiendo los tokens de posición en cajas con coordenadas reales.

La interfaz está hecha con Streamlit y está en el archivo `app.py`. Permite subir una imagen, escoger la tarea y el valor de `num_beams`, y ejecutar el modelo. Las tareas disponibles son descripción corta, descripción detallada, detección de objetos, lectura de texto (OCR) y lectura de texto por líneas con regiones. Según la tarea, la interfaz muestra el resultado como texto o dibuja cajas sobre la imagen. También muestra el dispositivo usado, el tiempo de ejecución y la fecha y hora de la prueba. El modelo se carga una sola vez gracias al caché de Streamlit, para no repetir la carga en cada clic.

## 6. Resultados y análisis

Todas las pruebas se hicieron en un portátil con procesador Intel Core i7-1065G7, sin GPU, es decir, solo en CPU. Las pruebas con GPU quedan pendientes y se agregarán aquí.

**Descripciones.** Con una imagen de un personaje en pleno vuelo sobre una ciudad, `<CAPTION>` devolvió una frase correcta pero corta, que no mencionaba al segundo personaje de la escena. `<MORE_DETAILED_CAPTION>` acertó en el fondo de la imagen, pero inventó un detalle que no existía: dijo que el segundo personaje sostenía una espada. Es un ejemplo de alucinación del modelo.

**Detección de objetos.** Con `<OD>` el modelo detectó a los dos personajes de esa imagen y, en otra prueba con una foto de un perro, devolvió una caja con la etiqueta "dog". Con `num_beams=3` encontró un elemento más (calzado) que con `num_beams=1`, lo que muestra que explorar más opciones mejora un poco la detección, a cambio de más tiempo. A diferencia de otros detectores, Florence-2 no entrega un porcentaje de confianza para cada caja, solo la caja y el nombre.

**Lectura de texto.** Con la portada impresa de un libro, `<OCR>` leyó todo el texto sin errores de letras, aunque devolvió las líneas pegadas en una sola cadena y perdió la tilde de "ASCENSIÓN". Con apuntes escritos a mano, en cambio, el resultado fue mucho peor: acertó algunas palabras sueltas, mezcló o inventó otras, y el texto se cortó antes de terminar la hoja, probablemente por el límite de tokens que fijamos al generar. Esto muestra que el modelo funciona bien con letra impresa y mal con letra manuscrita. Para resolver el problema de las líneas pegadas usamos la instrucción `<OCR_WITH_REGION>`, que devuelve cada línea por separado con su posición, y en la interfaz se dibuja un recuadro sobre cada una.

**Tiempos.** En CPU cada tarea tardó entre 35 y 58 segundos por imagen, sea cual fuera la tarea. Por ejemplo, `<CAPTION>` tardó 57.8 segundos con `num_beams=3` y unos 35 con `num_beams=1` en el script de prueba, y entre 47 y 54 segundos dentro de la interfaz. La mejora al reducir `num_beams` fue pequeña, y como el tiempo es parecido entre tareas muy distintas, buena parte se va en procesar la imagen y no en escribir la respuesta. Esta es una interpretación a partir de los tiempos medidos, no se comprobó directamente.

**Capturas.** Pendiente: aquí irán las capturas propias de la interfaz funcionando, con la fecha y hora visibles.

## 7. Conclusiones

Pendiente.

## 8. Referencias

[1] B. Xiao, H. Wu, W. Xu, X. Dai, H. Hu, Y. Lu, M. Zeng, C. Liu y L. Yuan, "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks," in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2024, pp. 4818-4829, doi: 10.1109/CVPR52733.2024.00461.

[2] Hugging Face, "Florence-2," Transformers documentation. [En línea]. Disponible en: https://huggingface.co/docs/transformers/model_doc/florence2

[3] Florence-2 Community, "Florence-2-base," Hugging Face. [En línea]. Disponible en: https://huggingface.co/florence-community/Florence-2-base