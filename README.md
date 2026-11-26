# Inferencia con Florence-2

Proyecto final de Procesamiento de Datos Secuenciales, Especialización en Inteligencia Artificial, Universidad Autónoma de Occidente.

Integrantes: Juan Esteban Aristizabal, Juan Felipe Lopez y John Angel Posso.

## 1. Resumen

Este trabajo implementa la inferencia de Florence-2, un modelo de visión de tipo Transformer encoder-decoder, a partir de los pesos preentrenados publicados por sus autores. Se construyó una interfaz en Streamlit donde se sube una imagen, se escoge una de cinco tareas (descripción corta, descripción detallada, detección de objetos, lectura de texto y lectura de texto por regiones) y se visualiza el resultado, junto con el tiempo de ejecución y el dispositivo usado. Las pruebas mostraron que el modelo describe bien lo principal de una imagen y lee con precisión el texto impreso grande, pero se equivoca con letra pequeña o manuscrita, puede inventar detalles en las descripciones largas y no entrega un porcentaje de confianza en la detección. También se encontró que cargar el modelo en media precisión (float16) lo hace muy lento en CPU, y que cargarlo en float32 reduce el tiempo de unos 35 a 58 segundos a unos 4 a 5 segundos por imagen. COMPLETAR: agregar aquí la comparación final entre CPU y GPU cuando estén todas las pruebas.

## 2. Introducción

Este trabajo toma como base el artículo "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks" de Xiao et al., publicado en CVPR 2024. Los pesos preentrenados y el código del modelo están disponibles en https://huggingface.co/florence-community/Florence-2-base, y la implementación que usamos es la que viene integrada en la librería transformers de Hugging Face.

Normalmente describir una imagen, leer el texto que aparece en ella y detectar objetos son tareas que se resuelven con modelos distintos. Florence-2 las resuelve con un solo Transformer de tipo encoder-decoder, que recibe la imagen junto con una instrucción corta escrita en texto y devuelve la respuesta también en texto. El objetivo de este proyecto no es entrenar el modelo, sino cargar los pesos ya entrenados, ejecutar la inferencia, mostrar los resultados en una interfaz y explicar en detalle cómo funciona la arquitectura y su mecanismo de atención.

## 3. Marco teórico

### 3.1 Arquitectura encoder-decoder

Florence-2 es un modelo secuencia a secuencia: recibe una secuencia de entrada y produce otra de salida. La entrada es una imagen junto con una instrucción en texto, y la salida es siempre texto. Para eso usa una arquitectura Transformer de tipo encoder-decoder. Primero, un encoder visual (DaViT) convierte la imagen en una secuencia de tokens visuales. Esos tokens se combinan con los tokens de la instrucción y pasan por el encoder del Transformer, que produce una representación de todo lo que el modelo necesita saber para responder. Después, el decoder escribe la respuesta un token a la vez, y cada token nuevo depende de lo que ya escribió y de lo que el encoder entendió.

### 3.2 Mecanismo de atención

La atención permite que cada elemento de una secuencia decida a cuáles otros elementos debe prestarle más importancia [4]. Se calcula con tres tensores: las queries (Q), las keys (K) y los values (V). Cada token genera su propia Q, K y V multiplicando su vector por matrices de pesos que el modelo aprendió durante el entrenamiento. La fórmula es:

`Atención(Q, K, V) = softmax(Q · Kᵀ / √d) · V`

Q · Kᵀ compara cada pregunta con todas las etiquetas y da una puntuación de qué tan relevante es cada una. La división por √d evita que esas puntuaciones crezcan demasiado, el softmax las convierte en pesos que suman 1, y al multiplicar por V se obtiene una mezcla de la información, con más peso en lo más relevante.

En el encoder se usa atención propia, donde Q, K y V salen de la misma secuencia (la imagen y la instrucción), así que los tokens se relacionan entre sí. En el decoder hay dos tipos. La atención propia solo mira lo que ya se escribió, y la atención cruzada mira al encoder: allí la Q sale del decoder, es decir, de lo que lleva escrito, y la K y la V salen del encoder, es decir, de la imagen y la instrucción. Así, cada vez que el decoder va a escribir una palabra, consulta las partes de la imagen que le sirven para eso.

### 3.3 Una sola representación para varias tareas

En Florence-2, todas las tareas se plantean como generar texto. Para la detección de objetos, por ejemplo, las coordenadas de cada caja se escriben como tokens especiales de posición que se agregaron al vocabulario del modelo, y por eso el mismo decoder que escribe frases también puede escribir ubicaciones. Qué tarea hace el modelo lo decide la instrucción que acompaña a la imagen, como `<CAPTION>`, `<OD>` u `<OCR>`.

### 3.4 Innovaciones

La primera innovación es esa representación unificada: un solo modelo, con una sola arquitectura y una sola forma de entrenarse, resuelve tareas que antes requerían modelos distintos. La segunda es el conjunto de datos FLD-5B, que los autores construyeron con 126 millones de imágenes y 5.4 mil millones de anotaciones, producidas con un proceso automático que se repite y se va refinando. La tercera es su tamaño: la versión base tiene unos 0.23 mil millones de parámetros, y aun así los autores reportan resultados competitivos frente a modelos mucho más grandes [1].

### 3.5 Limitaciones observadas

En nuestras pruebas notamos varias limitaciones. En la lectura de texto, el modelo se equivoca con letra pequeña y con letra manuscrita, e incluso llega a inventar texto que no está en la imagen. En las descripciones detalladas puede agregar elementos que no existen. La detección no entrega un porcentaje de confianza para cada objeto. Y en CPU, con la precisión por defecto del modelo (float16), tardaba entre 35 y 58 segundos por imagen. Al cargarlo en precisión normal (float32) bajó a unos 4 segundos, lo que muestra que el tipo de dato influye mucho en el rendimiento según el dispositivo.

### 3.6 Tamaños del modelo (Florence-2-base)

Estos datos salen de la configuración que carga la librería transformers. La parte de texto es un Transformer de tipo BART con 6 capas en el encoder y 6 en el decoder. Los vectores tienen 768 dimensiones y cada capa usa 12 cabezas de atención, es decir, 64 dimensiones por cabeza (768 dividido entre 12). La red feed-forward de cada capa tiene 3072 neuronas y el vocabulario es de 51328 tokens. Cada capa del decoder tiene atención propia y atención cruzada.

El encoder visual es un DaViT con cuatro etapas, con 128, 256, 512 y 1024 canales y con 1, 1, 9 y 1 bloques respectivamente. Reduce la imagen por un factor de 32 en total (pasos de 4, 2, 2 y 2) y proyecta sus características a 768 dimensiones para que tengan el mismo tamaño que los vectores del texto y se puedan combinar con ellos.

## 4. Metodología

Trabajamos con el modelo Florence-2 en su versión base, de 0.23 mil millones de parámetros, sin ningún entrenamiento adicional. Los pesos se descargan de Hugging Face la primera vez que se ejecuta el código. El entorno se maneja con uv, y las librerías principales son torch, transformers, pillow, matplotlib y streamlit. Usamos la clase Florence2ForConditionalGeneration de transformers, que ya incluye la arquitectura completa, en lugar del código remoto original de Microsoft, porque ese último dejó de funcionar con las versiones recientes de la librería.

Las pruebas se hicieron en dos equipos. El primero es un portátil con procesador Intel Core i7-1065G7, sin GPU dedicada. El segundo es un equipo con procesador AMD Ryzen 7 9700X, tarjeta NVIDIA GeForce RTX 5070 de 12 GB, 24 GB de RAM y Windows 11 Pro.

Una decisión importante fue el tipo de dato con el que se carga el modelo. Los pesos están guardados en float16 (media precisión), y con las versiones recientes de transformers se cargan así por defecto. Esto funciona bien en GPU, pero en CPU resultó muy lento. Por eso la interfaz carga el modelo en float16 cuando hay GPU y en float32 cuando solo hay CPU.

## 5. Desarrollo e implementación

Para ejecutar el proyecto:

1. Clonar el repositorio y entrar a la carpeta.
2. Instalar las librerías con `make instalar` (equivale a `uv sync`).
3. Abrir la interfaz con `make app` (equivale a `uv run streamlit run app.py`).
4. Opcionalmente, probar el modelo por consola con `make prueba`, que corre el script `prueba_inferencia.py` sobre una imagen llamada `prueba.jpg` puesta en la carpeta del proyecto.

El comando `make` no viene instalado en Windows. En ese sistema se puede instalar con `winget install ezwinports.make`, o se pueden usar directamente los comandos de uv que aparecen entre paréntesis.

La primera vez que se ejecuta, el modelo descarga sus pesos de Hugging Face, que pesan unos 463 MB. Los pesos se cargan con la clase Florence2ForConditionalGeneration, y el procesador (AutoProcessor) se encarga del preprocesamiento: ajusta el tamaño de la imagen, la normaliza y convierte la instrucción de texto en tokens. Después se llama a `generate`, donde el encoder procesa la imagen y la instrucción, y el decoder escribe la respuesta token por token. Por último, `post_process_generation` interpreta ese texto según la tarea, por ejemplo convirtiendo los tokens de posición en cajas con coordenadas reales.

La interfaz está hecha con Streamlit, se llama Florence-2 Vision Studio y está en el archivo `app.py`. Tiene cuatro páginas. En "Análisis de imagen" se sube una imagen, se escoge la tarea y el valor de `num_beams` y se ejecuta el modelo. En "Comparaciones" se ejecuta la misma tarea sobre dos imágenes. "Historial" guarda las inferencias de la sesión, y "Acerca del modelo" resume la arquitectura y el entorno. Según la tarea, el resultado se muestra como texto o como cajas dibujadas sobre la imagen, y se puede descargar en JSON o PNG. La barra lateral indica si el modelo corre en CPU o en GPU y, en ese caso, el nombre de la tarjeta y su memoria. Cada análisis muestra el tiempo de inferencia, el dispositivo y la fecha y hora. El modelo se carga una sola vez gracias al caché de Streamlit.

## 6. Resultados y análisis

**Descripciones.** Con una imagen de un personaje en pleno vuelo sobre una ciudad, `<CAPTION>` devolvió una frase correcta pero corta, que no mencionaba al segundo personaje de la escena. `<MORE_DETAILED_CAPTION>` acertó en el fondo de la imagen, pero inventó un detalle que no existía: dijo que el segundo personaje sostenía una espada. Con un anuncio de un videojuego, la descripción detallada agregó elementos como un caballero con armadura y un dragón que no aparecen en la imagen. Es el mismo tipo de alucinación en dos imágenes distintas.

**Detección de objetos.** Con `<OD>` el modelo detectó a los dos personajes de la primera imagen y, con la foto de un perro, devolvió una caja con la etiqueta "dog". En el anuncio del videojuego detectó seis personas, incluidas las figuras pequeñas de los recuadros, y un afiche. Con `num_beams=3` encontró un elemento más (calzado) que con `num_beams=1`, lo que muestra que explorar más opciones mejora un poco la detección, a cambio de más tiempo. A diferencia de otros detectores, Florence-2 no entrega un porcentaje de confianza para cada caja, solo la caja y el nombre.

**Lectura de texto.** Con la portada impresa de un libro, `<OCR>` leyó todo el texto sin errores de letras, aunque devolvió las líneas pegadas en una sola cadena y perdió la tilde de "ASCENSIÓN". Con apuntes escritos a mano el resultado fue mucho peor: acertó algunas palabras sueltas, mezcló o inventó otras, y el texto se cortó antes de terminar la hoja, probablemente por el límite de 256 tokens que fijamos al generar. Con el anuncio del videojuego, el OCR leyó bien el texto grande, pero se equivocó con el pequeño ("SEZE THE REINS OF CHARLOT", "EMRACE YOUR CURED SOUL", "BABYTON") e inventó direcciones web que no están en la imagen. Para resolver el problema de las líneas pegadas se usa `<OCR_WITH_REGION>`, que devuelve cada línea por separado con su posición, y la interfaz dibuja un recuadro sobre cada una. Esta tarea no detecta todas las líneas: en el anuncio omitió partes del texto pequeño.

**Tiempos y precisión numérica.** En el portátil con CPU Intel, con el modelo cargado en float16 (el valor por defecto), cada tarea tardó entre 35 y 58 segundos por imagen. Al cargarlo en float32, la descripción corta de la misma imagen tardó unos 4 segundos (4.1 y 4.0 segundos en dos ejecuciones seguidas, con `num_beams=1`), y dentro de la interfaz una descripción corta tardó 5.23 segundos con otra imagen. El resultado del modelo fue idéntico, por lo que el cambio solo afectó el tiempo. Esto indica que en CPU la media precisión es muy lenta porque el procesador no está optimizado para ese tipo de cálculo.

**Comparación entre CPU y GPU.** Los tiempos de inferencia con `num_beams=1`, en la segunda ejecución de cada tarea, para la misma imagen (COMPLETAR: nombre de la imagen):

| Tarea | CPU Intel i7-1065G7 (float32) | GPU RTX 5070 (float16) |
|---|---|---|
| Descripción corta | COMPLETAR | COMPLETAR |
| Detección de objetos | COMPLETAR | COMPLETAR |
| OCR con regiones | COMPLETAR | COMPLETAR |

COMPLETAR: un párrafo corto que interprete la tabla (cuántas veces más rápida es la GPU y si la diferencia es la misma en todas las tareas).

**Capturas.** COMPLETAR: aquí van las capturas de pantalla propias de la interfaz funcionando, con el tiempo, el dispositivo y la fecha y hora visibles.

## 7. Conclusiones

Este trabajo permitió entender cómo funciona un Transformer encoder-decoder aplicado a visión: cómo una imagen se convierte en tokens, cómo el decoder escribe la respuesta palabra por palabra y cómo la atención cruzada le permite consultar las partes de la imagen que necesita. Florence-2 resultó ser un modelo muy versátil, porque una sola arquitectura resuelve tareas distintas con solo cambiar la instrucción.

También vimos sus límites. Se equivoca con texto pequeño o manuscrito, puede inventar detalles que no existen, y no dice qué tan seguro está de cada detección. Además, descubrimos que el rendimiento depende mucho de cómo se carga el modelo: el mismo código pasó de casi un minuto a pocos segundos en CPU solo por cambiar el tipo de dato.

Como posibles mejoras, sería interesante visualizar los pesos de atención cruzada sobre la imagen para ver qué regiones usa el modelo al escribir cada palabra, probar la versión large del modelo, y comparar el OCR con herramientas especializadas en texto.

## 8. Referencias

[1] B. Xiao, H. Wu, W. Xu, X. Dai, H. Hu, Y. Lu, M. Zeng, C. Liu y L. Yuan, "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks," in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2024, pp. 4818-4829, doi: 10.1109/CVPR52733.2024.00461.

[2] Hugging Face, "Florence-2," Transformers documentation. [En línea]. Disponible en: https://huggingface.co/docs/transformers/model_doc/florence2

[3] Florence-2 Community, "Florence-2-base," Hugging Face. [En línea]. Disponible en: https://huggingface.co/florence-community/Florence-2-base

[4] A. Vaswani et al., "Attention is all you need," in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017.