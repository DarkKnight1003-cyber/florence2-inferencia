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
2. Ejecutar `uv sync` para instalar las librerías.
3. Poner una imagen de prueba en la carpeta del proyecto con el nombre `prueba.jpg`.
4. Ejecutar `uv run python prueba_inferencia.py`.

La primera ejecución descarga los pesos del modelo, que pesan unos 463 MB. El script carga el procesador y el modelo, convierte la imagen y la instrucción en tensores, genera la respuesta con `generate` y la convierte de nuevo en texto. El procesador se encarga del preprocesamiento: ajusta la imagen al tamaño que espera el encoder visual y convierte el texto de la instrucción en tokens.

La interfaz con Streamlit está pendiente.

## 6. Resultados y análisis

Hicimos una primera prueba con una imagen de un personaje en pleno vuelo sobre una ciudad, con otro personaje parado en una plataforma. La prueba se corrió en un portátil con procesador Intel Core i7-1065G7, sin GPU, solo en CPU. Probamos cuatro instrucciones del modelo.

Con `<CAPTION>` el modelo devolvió una descripción corta y correcta del personaje principal, pero no mencionó al segundo personaje. Con `<MORE_DETAILED_CAPTION>` la descripción fue más larga y acertó en el fondo de la imagen, pero inventó un detalle que no existe: dijo que el segundo personaje sostenía una espada. Con `<OD>`, la detección de objetos, el modelo encontró tres elementos con `num_beams=3` (calzado y dos personas), incluido el personaje de la plataforma, y con `num_beams=1` perdió el calzado. Con `<OCR>` devolvió solo un guion, lo cual es coherente porque la imagen no tiene texto.

En cuanto a tiempo, en CPU cada instrucción tardó entre 42 y 58 segundos con `num_beams=3`. Al bajar a `num_beams=1` los tiempos fueron de unos 35 segundos para `<CAPTION>` y 38 para `<OD>`, así que la mejora fue pequeña y costó algo de calidad en la detección. Esto indica que buena parte del tiempo se va en el procesamiento de la imagen y no en la generación del texto. Las pruebas en GPU quedan pendientes y se reportarán aquí junto con las capturas de pantalla de la interfaz.

## 7. Conclusiones

Pendiente.

## 8. Referencias

[1] B. Xiao, H. Wu, W. Xu, X. Dai, H. Hu, Y. Lu, M. Zeng, C. Liu y L. Yuan, "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks," in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2024, pp. 4818-4829, doi: 10.1109/CVPR52733.2024.00461.

[2] Hugging Face, "Florence-2," Transformers documentation. [En línea]. Disponible en: https://huggingface.co/docs/transformers/model_doc/florence2

[3] Florence-2 Community, "Florence-2-base," Hugging Face. [En línea]. Disponible en: https://huggingface.co/florence-community/Florence-2-base