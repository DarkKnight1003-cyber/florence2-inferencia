# Makefile
# Atajos para ejecutar el proyecto de Florence-2

.PHONY: instalar prueba app

# Instalar las librerías del proyecto
instalar:
	uv sync

# Ejecutar la prueba de inferencia por consola
prueba:
	uv run python prueba_inferencia.py

# Abrir la interfaz en Streamlit
app:
	uv run streamlit run app.py