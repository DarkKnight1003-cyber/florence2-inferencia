
# ============================================================
# FLORENCE-2 VISION STUDIO
# Procesamiento de Datos Secuenciales - UAO
# ============================================================

import json
import time
import hashlib

from datetime import datetime
from collections import Counter
from io import BytesIO

import streamlit as st
import torch

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Florence2ForConditionalGeneration


# ---------- Configuración general ----------

st.set_page_config(
    page_title="Florence-2 Vision Studio",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

NOMBRE_MODELO = "florence-community/Florence-2-base"

DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"

TAREAS = {
    "Descripción corta": "<CAPTION>",
    "Descripción detallada": "<MORE_DETAILED_CAPTION>",
    "Detección de objetos": "<OD>",
    "Lectura de texto (OCR)": "<OCR>",
    "OCR con regiones": "<OCR_WITH_REGION>",
}


# ---------- Estilos de la interfaz ----------

st.markdown("""
<style>

.stApp {
    background-color: #080e1c;
    color: #f1f5f9;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.hero {
    background: linear-gradient(115deg, #101d3b, #172554, #25104b);
    border: 1px solid #263b69;
    border-radius: 18px;
    padding: 30px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #ffffff;
    font-size: 34px;
    margin-bottom: 8px;
}

.hero p {
    color: #b7c6e5;
    font-size: 15px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #101a30;
    border: 1px solid #26385c;
    border-radius: 15px;
}

div[data-testid="stMetric"] {
    background-color: #14213b;
    border: 1px solid #26385c;
    padding: 15px;
    border-radius: 12px;
}

.stButton > button {
    background: linear-gradient(100deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    min-height: 45px;
}

.stButton > button:hover {
    border: 1px solid #93c5fd;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #0b1427;
    border-right: 1px solid #26385c;
}

hr {
    border-color: #26385c;
}

</style>
""", unsafe_allow_html=True)


# ---------- Inicialización del historial ----------

if "historial" not in st.session_state:
    st.session_state.historial = []

if "ultimo_analisis" not in st.session_state:
    st.session_state.ultimo_analisis = None

if "comparacion_actual" not in st.session_state:
    st.session_state.comparacion_actual = None


# ---------- Carga del modelo ----------

@st.cache_resource
def cargar_modelo():

    # El modelo se carga una sola vez durante la sesión.
    procesador = AutoProcessor.from_pretrained(NOMBRE_MODELO)

    # En GPU se usa media precisión y en CPU precisión normal,
    # porque float16 en CPU es muy lento.
    tipo_dato = torch.float16 if DISPOSITIVO == "cuda" else torch.float32

    modelo = Florence2ForConditionalGeneration.from_pretrained(
        NOMBRE_MODELO,
        dtype=tipo_dato
    )

    modelo = modelo.to(DISPOSITIVO)
    modelo.eval()

    return procesador, modelo

# ---------- Proceso de inferencia ----------

def ejecutar_tarea(procesador, modelo, imagen, tarea, num_beams):

    # Sincroniza la GPU para obtener una medición precisa.
    if DISPOSITIVO == "cuda":
        torch.cuda.synchronize()

    inicio = time.perf_counter()

    # Prepara la imagen y la instrucción para el modelo.
    entradas = procesador(
        text=tarea,
        images=imagen,
        return_tensors="pt"
    )

    entradas = entradas.to(DISPOSITIVO)

    # Genera la respuesta mediante el decoder.
    with torch.inference_mode():

        ids = modelo.generate(
            **entradas,
            max_new_tokens=256,
            num_beams=num_beams
        )

    # Convierte los tokens generados en texto.
    texto = procesador.batch_decode(
        ids,
        skip_special_tokens=False
    )[0]

    # Interpreta la salida según la tarea ejecutada.
    resultado = procesador.post_process_generation(
        texto,
        task=tarea,
        image_size=imagen.size
    )

    if DISPOSITIVO == "cuda":
        torch.cuda.synchronize()

    tiempo = round(time.perf_counter() - inicio, 3)

    return resultado, tiempo


# ---------- Identificación de imágenes ----------

def obtener_id_imagen(archivo):

    # Genera un identificador para distinguir imágenes con igual nombre.
    return hashlib.sha256(
        archivo.getvalue()
    ).hexdigest()


# ---------- Visualización de detecciones ----------

def dibujar_cajas(imagen, resultado):

    imagen_dibujo = imagen.copy()
    dibujo = ImageDraw.Draw(imagen_dibujo)
    fuente = ImageFont.load_default(size=20)

    for caja, etiqueta in zip(
        resultado["bboxes"],
        resultado["labels"]
    ):

        x1, y1, x2, y2 = caja

        dibujo.rectangle(
            [x1, y1, x2, y2],
            outline="#38bdf8",
            width=4
        )

        # Calcula el tamaño del fondo de cada etiqueta.
        texto = str(etiqueta)

        bbox_texto = dibujo.textbbox(
            (0, 0),
            texto,
            font=fuente
        )

        ancho = bbox_texto[2] - bbox_texto[0]
        alto = bbox_texto[3] - bbox_texto[1]

        y_texto = max(0, y1 - alto - 10)

        dibujo.rectangle(
            [
                x1,
                y_texto,
                x1 + ancho + 12,
                y_texto + alto + 10
            ],
            fill="#2563eb"
        )

        dibujo.text(
            (
                x1 + 6,
                y_texto + 4 - bbox_texto[1]
            ),
            texto,
            fill="white",
            font=fuente
        )

    return imagen_dibujo


# ---------- Visualización de regiones OCR ----------

def dibujar_lineas_texto(imagen, resultado):

    imagen_dibujo = imagen.copy()
    dibujo = ImageDraw.Draw(imagen_dibujo)

    for cuadro in resultado["quad_boxes"]:

        # Cada región contiene las coordenadas de sus cuatro esquinas.
        puntos = [
            (cuadro[i], cuadro[i + 1])
            for i in range(0, 8, 2)
        ]

        dibujo.line(
            puntos + [puntos[0]],
            fill="#38bdf8",
            width=4
        )

    return imagen_dibujo


# ---------- Procesamiento de imágenes ----------

def generar_imagen_procesada(imagen, resultado, tarea):

    if tarea == "<OD>":

        return dibujar_cajas(
            imagen,
            resultado[tarea]
        )

    elif tarea == "<OCR_WITH_REGION>":

        return dibujar_lineas_texto(
            imagen,
            resultado[tarea]
        )

    return imagen.copy()


# ---------- Conversión de resultados a JSON ----------

def convertir_json(datos):

    if isinstance(datos, dict):

        return {
            str(k): convertir_json(v)
            for k, v in datos.items()
        }

    if isinstance(datos, (list, tuple)):

        return [
            convertir_json(v)
            for v in datos
        ]

    if isinstance(datos, torch.Tensor):
        return datos.detach().cpu().tolist()

    if hasattr(datos, "tolist"):
        return datos.tolist()

    if isinstance(datos, (str, int, float, bool)) or datos is None:
        return datos

    return str(datos)


# ---------- Exportación de informes ----------

def generar_informe_json(registro):

    informe = {
        "proyecto": "Florence-2 Vision Studio",
        "modelo": NOMBRE_MODELO,
        "dispositivo": DISPOSITIVO,
        "fecha": registro["fecha"],
        "imagen": registro["imagen"],
        "tarea": registro["tarea"],
        "instruccion": registro["codigo_tarea"],
        "num_beams": registro["num_beams"],
        "tiempo_inferencia_segundos": registro["tiempo"],
        "resultado": convertir_json(registro["resultado"])
    }

    return json.dumps(
        informe,
        ensure_ascii=False,
        indent=4
    ).encode("utf-8")


# ---------- Exportación de imágenes ----------

def generar_imagen_png(imagen, resultado, tarea):

    imagen_final = generar_imagen_procesada(
        imagen,
        resultado,
        tarea
    )

    buffer = BytesIO()

    imagen_final.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return buffer.getvalue()


# ---------- Presentación de resultados ----------

def mostrar_resultado(imagen, resultado, tarea):

    datos = resultado[tarea]

    if tarea == "<OD>":

        imagen_procesada = dibujar_cajas(imagen, datos)

        st.image(
            imagen_procesada,
            caption="Detecciones de Florence-2",
            use_container_width=True
        )

        # Agrupa los objetos identificados por categoría.
        conteo = Counter(datos["labels"])

        st.markdown("#### Objetos identificados")

        if conteo:

            columnas = st.columns(min(len(conteo), 4))

            for i, (etiqueta, cantidad) in enumerate(conteo.items()):

                with columnas[i % len(columnas)]:
                    st.metric(etiqueta, cantidad)

        else:
            st.info("No se identificaron objetos.")

        with st.expander("Consultar coordenadas"):
            st.json(convertir_json(datos))

    elif tarea == "<OCR_WITH_REGION>":

        imagen_procesada = dibujar_lineas_texto(imagen, datos)

        st.image(
            imagen_procesada,
            caption="Regiones de texto identificadas",
            use_container_width=True
        )

        # Organiza los textos reconocidos por región.
        lineas = [
            etiqueta.replace("</s>", "").replace("<s>", "")
            for etiqueta in datos["labels"]
        ]

        st.markdown("#### Texto reconocido")
        st.text("\n".join(lineas))

        with st.expander("Consultar regiones"):
            st.json(convertir_json(datos))

    else:

        st.markdown("#### Resultado generado")
        st.info(str(datos))


# ---------- Registro del historial ----------

def registrar_historial(registro):

    st.session_state.historial.insert(
        0,
        registro
    )


def resumen_resultado(resultado, tarea):

    datos = resultado[tarea]

    if tarea == "<OD>":

        conteo = Counter(datos["labels"])

        return ", ".join(
            f"{nombre}: {cantidad}"
            for nombre, cantidad in conteo.items()
        )

    elif tarea == "<OCR_WITH_REGION>":

        return f"{len(datos['labels'])} regiones de texto"

    return str(datos)[:180]


# ---------- Barra lateral y navegación ----------

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:15px 0;">
            <h1 style="color:#60a5fa;">◉</h1>
            <h2>Florence-2</h2>
            <p style="color:#94a3b8;">Vision Studio</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    pagina = st.radio(
        "NAVEGACIÓN",
        [
            "Análisis de imagen",
            "Comparaciones",
            "Historial",
            "Acerca del modelo"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Hardware")

    if DISPOSITIVO == "cuda":

        nombre_gpu = torch.cuda.get_device_name(0)

        memoria = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        st.success("GPU en línea")
        st.markdown(f"**{nombre_gpu}**")
        st.caption(f"Memoria total: {memoria:.1f} GB")

    else:

        st.warning("Ejecutando en CPU")

    st.divider()

    st.caption("FLORENCE-2 BASE")
    st.caption("Computer Vision · Encoder-Decoder")


# ---------- Encabezado principal ----------

st.markdown(
    """
    <div class="hero">
        <h1>Florence-2 Vision Studio</h1>
        <p>
        Analiza, comprende y extrae información de tus imágenes
        mediante inteligencia artificial.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PÁGINA 1. ANÁLISIS DE IMAGEN
# ============================================================

if pagina == "Análisis de imagen":

    st.subheader("Análisis de imagen")

    columna_config, columna_resultado = st.columns(
        [0.9, 1.3],
        gap="large"
    )

    with columna_config:

        with st.container(border=True):

            st.markdown("### Configuración de inferencia")

            archivo = st.file_uploader(
                "Sube una imagen",
                type=["jpg", "jpeg", "png"],
                key="imagen_principal"
            )

            nombre_tarea = st.selectbox(
                "Selecciona la tarea",
                list(TAREAS.keys())
            )

            num_beams = st.slider(
                "num_beams",
                min_value=1,
                max_value=5,
                value=1,
                help="Valores mayores exploran más secuencias posibles."
            )

            ejecutar = st.button(
                "▶ Ejecutar análisis",
                use_container_width=True
            )

    if archivo is not None:

        imagen = Image.open(
            BytesIO(archivo.getvalue())
        ).convert("RGB")

        id_imagen = obtener_id_imagen(archivo)
        tarea = TAREAS[nombre_tarea]

        if ejecutar:

            with st.spinner("Procesando imagen con Florence-2..."):

                procesador, modelo = cargar_modelo()

                resultado, tiempo = ejecutar_tarea(
                    procesador,
                    modelo,
                    imagen,
                    tarea,
                    num_beams
                )

            registro = {
                "fecha": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "imagen": archivo.name,
                "id_imagen": id_imagen,
                "tarea": nombre_tarea,
                "codigo_tarea": tarea,
                "num_beams": num_beams,
                "tiempo": tiempo,
                "resultado": resultado
            }

            st.session_state.ultimo_analisis = registro
            registrar_historial(registro)

        with columna_resultado:

            with st.container(border=True):

                st.markdown("### Resultado del análisis")

                st.image(
                    imagen,
                    caption="Imagen original",
                    use_container_width=True
                )

                st.caption(
                    f"{archivo.name} · "
                    f"{imagen.width} × {imagen.height} px · "
                    f"{archivo.size / 1024:.1f} KB"
                )

                registro = st.session_state.ultimo_analisis

                if (
                    registro is not None
                    and registro["id_imagen"] == id_imagen
                    and registro["codigo_tarea"] == tarea
                    and registro["num_beams"] == num_beams
                ):

                    st.success("Inferencia completada")

                    metrica1, metrica2 = st.columns(2)

                    metrica1.metric(
                        "Tiempo de inferencia",
                        f"{registro['tiempo']:.2f} s"
                    )

                    metrica2.metric(
                        "Dispositivo",
                        DISPOSITIVO.upper()
                    )

                    st.caption(f"Fecha: {registro['fecha']}")

                    mostrar_resultado(
                        imagen,
                        registro["resultado"],
                        tarea
                    )

                    st.divider()

                    st.markdown("### Exportar resultados")

                    col_json, col_png = st.columns(2)

                    nombre_base = archivo.name.rsplit(".", 1)[0]

                    with col_json:

                        st.download_button(
                            label="📄 Descargar JSON",
                            data=generar_informe_json(registro),
                            file_name=f"{nombre_base}_informe.json",
                            mime="application/json",
                            use_container_width=True,
                            key="descargar_json_analisis"
                        )

                    with col_png:

                        st.download_button(
                            label="🖼️ Descargar PNG",
                            data=generar_imagen_png(
                                imagen,
                                registro["resultado"],
                                tarea
                            ),
                            file_name=f"{nombre_base}_procesada.png",
                            mime="image/png",
                            use_container_width=True,
                            key="descargar_png_analisis"
                        )

                else:

                    st.info(
                        "Ejecuta el análisis para visualizar "
                        "los resultados y habilitar las descargas."
                    )

    else:

        with columna_resultado:
            st.info("Carga una imagen para comenzar el análisis.")


# ============================================================
# PÁGINA 2. COMPARACIÓN DE IMÁGENES
# ============================================================

elif pagina == "Comparaciones":

    st.subheader("Comparación de imágenes")

    st.write(
        "Ejecuta la misma tarea sobre dos imágenes "
        "y compara sus resultados."
    )

    col1, col2 = st.columns(2)

    with col1:

        archivo1 = st.file_uploader(
            "Primera imagen",
            type=["jpg", "jpeg", "png"],
            key="comparacion_1"
        )

    with col2:

        archivo2 = st.file_uploader(
            "Segunda imagen",
            type=["jpg", "jpeg", "png"],
            key="comparacion_2"
        )

    nombre_tarea = st.selectbox(
        "Tarea para ambas imágenes",
        list(TAREAS.keys()),
        key="tarea_comparacion"
    )

    num_beams = st.slider(
        "num_beams",
        1,
        5,
        1,
        key="beams_comparacion"
    )

    if archivo1 is not None and archivo2 is not None:

        imagen1 = Image.open(
            BytesIO(archivo1.getvalue())
        ).convert("RGB")

        imagen2 = Image.open(
            BytesIO(archivo2.getvalue())
        ).convert("RGB")

        id1 = obtener_id_imagen(archivo1)
        id2 = obtener_id_imagen(archivo2)

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                imagen1,
                caption="Imagen 1",
                use_container_width=True
            )

        with col2:

            st.image(
                imagen2,
                caption="Imagen 2",
                use_container_width=True
            )

        tarea = TAREAS[nombre_tarea]

        if st.button(
            "▶ Comparar imágenes",
            use_container_width=True
        ):

            with st.spinner("Analizando ambas imágenes..."):

                procesador, modelo = cargar_modelo()

                resultado1, tiempo1 = ejecutar_tarea(
                    procesador,
                    modelo,
                    imagen1,
                    tarea,
                    num_beams
                )

                resultado2, tiempo2 = ejecutar_tarea(
                    procesador,
                    modelo,
                    imagen2,
                    tarea,
                    num_beams
                )

            fecha = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

            registro1 = {
                "fecha": fecha,
                "imagen": archivo1.name,
                "id_imagen": id1,
                "tarea": nombre_tarea,
                "codigo_tarea": tarea,
                "num_beams": num_beams,
                "tiempo": tiempo1,
                "resultado": resultado1
            }

            registro2 = {
                "fecha": fecha,
                "imagen": archivo2.name,
                "id_imagen": id2,
                "tarea": nombre_tarea,
                "codigo_tarea": tarea,
                "num_beams": num_beams,
                "tiempo": tiempo2,
                "resultado": resultado2
            }

            st.session_state.comparacion_actual = {
                "registro1": registro1,
                "registro2": registro2,
                "id1": id1,
                "id2": id2,
                "tarea": tarea,
                "num_beams": num_beams
            }

            registrar_historial(registro1)
            registrar_historial(registro2)

        comparacion = st.session_state.comparacion_actual

        if (
            comparacion is not None
            and comparacion["id1"] == id1
            and comparacion["id2"] == id2
            and comparacion["tarea"] == tarea
            and comparacion["num_beams"] == num_beams
        ):

            r1 = comparacion["registro1"]
            r2 = comparacion["registro2"]

            st.success("Comparación completada")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Tiempo imagen 1",
                    f"{r1['tiempo']:.2f} s"
                )

                mostrar_resultado(
                    imagen1,
                    r1["resultado"],
                    tarea
                )

            with col2:

                st.metric(
                    "Tiempo imagen 2",
                    f"{r2['tiempo']:.2f} s"
                )

                mostrar_resultado(
                    imagen2,
                    r2["resultado"],
                    tarea
                )

            st.divider()

            st.markdown("### Exportar comparación")

            base1 = archivo1.name.rsplit(".", 1)[0]
            base2 = archivo2.name.rsplit(".", 1)[0]

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.download_button(
                    "JSON imagen 1",
                    data=generar_informe_json(r1),
                    file_name=f"{base1}_informe.json",
                    mime="application/json",
                    use_container_width=True,
                    key="json_comp_1"
                )

            with col2:

                st.download_button(
                    "PNG imagen 1",
                    data=generar_imagen_png(
                        imagen1,
                        r1["resultado"],
                        tarea
                    ),
                    file_name=f"{base1}_procesada.png",
                    mime="image/png",
                    use_container_width=True,
                    key="png_comp_1"
                )

            with col3:

                st.download_button(
                    "JSON imagen 2",
                    data=generar_informe_json(r2),
                    file_name=f"{base2}_informe.json",
                    mime="application/json",
                    use_container_width=True,
                    key="json_comp_2"
                )

            with col4:

                st.download_button(
                    "PNG imagen 2",
                    data=generar_imagen_png(
                        imagen2,
                        r2["resultado"],
                        tarea
                    ),
                    file_name=f"{base2}_procesada.png",
                    mime="image/png",
                    use_container_width=True,
                    key="png_comp_2"
                )

    else:

        st.info("Carga dos imágenes para habilitar la comparación.")


# ============================================================
# PÁGINA 3. HISTORIAL DE INFERENCIAS
# ============================================================

elif pagina == "Historial":

    st.subheader("Historial de inferencias")

    historial = st.session_state.historial

    if historial:

        st.metric(
            "Inferencias registradas",
            len(historial)
        )

        for i, registro in enumerate(historial):

            with st.expander(
                f"{registro['imagen']} · {registro['tarea']}"
            ):

                st.write("**Fecha:**", registro["fecha"])

                st.write(
                    "**Tiempo:**",
                    f"{registro['tiempo']:.2f} segundos"
                )

                st.write("**Dispositivo:**", DISPOSITIVO.upper())

                st.write(
                    "**Resumen:**",
                    resumen_resultado(
                        registro["resultado"],
                        registro["codigo_tarea"]
                    )
                )

                st.download_button(
                    "📄 Descargar JSON",
                    data=generar_informe_json(registro),
                    file_name=(
                        registro["imagen"].rsplit(".", 1)[0]
                        + "_informe.json"
                    ),
                    mime="application/json",
                    key=f"hist_json_{i}",
                    use_container_width=True
                )

        if st.button("Limpiar historial"):

            st.session_state.historial = []
            st.rerun()

    else:

        st.info("Todavía no hay inferencias registradas.")


# ============================================================
# PÁGINA 4. INFORMACIÓN DEL MODELO
# ============================================================

elif pagina == "Acerca del modelo":

    st.subheader("Acerca de Florence-2")

    st.markdown("""
    Florence-2 es un modelo de visión artificial
    basado en una arquitectura Transformer encoder-decoder.

    Su arquitectura permite realizar diferentes tareas
    de visión mediante instrucciones textuales.

    ### Características

    - Arquitectura encoder-decoder.
    - Representación visual unificada.
    - Generación secuencial de tokens.
    - Mecanismos de atención.
    - Modelo preentrenado Florence-2-base.

    ### Tareas implementadas

    | Tarea | Instrucción |
    |---|---|
    | Descripción corta | CAPTION |
    | Descripción detallada | MORE_DETAILED_CAPTION |
    | Detección de objetos | OD |
    | Reconocimiento de texto | OCR |
    | OCR con regiones | OCR_WITH_REGION |

    ### Recursos

    Modelo:
    https://huggingface.co/florence-community/Florence-2-base

    Documentación:
    https://huggingface.co/docs/transformers/model_doc/florence2
    """)

    st.divider()

    st.markdown("### Información del entorno")

    st.write("**Modelo:**", NOMBRE_MODELO)
    st.write("**Dispositivo:**", DISPOSITIVO.upper())
    st.write("**PyTorch:**", torch.__version__)

    if DISPOSITIVO == "cuda":

        st.write("**CUDA:**", torch.version.cuda)

        st.write(
            "**GPU:**",
            torch.cuda.get_device_name(0)
        )

        memoria = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )

        st.write(
            "**Memoria GPU:**",
            f"{memoria:.1f} GB"
        )


# ---------- Pie de página ----------

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        padding:10px;
    ">
        FLORENCE-2 VISION STUDIO<br>
        Procesamiento de Datos Secuenciales · 2026
    </div>
    """,
    unsafe_allow_html=True
)
