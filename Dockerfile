# Utiliza una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo requirements.txt al directorio de trabajo
COPY requirements.txt .

# Instala las dependencias listadas en requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia todos los archivos del proyecto al directorio de trabajo
COPY . .

# Expone el puerto 7860, que es el puerto por defecto que utiliza Gradio
EXPOSE 7860

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]
