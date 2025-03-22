"""
Herramientas Computacionales: El Arte de la Programación

Evidencia de Proyecto

Miguel de Jesús Degollado Macías|A01255388.

Docente: Baldomero Olvera Villanueva

Fecha: 23/03/2025
 """
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import os

# Función para aplicar convolución a una imagen
def convolution(image: np.ndarray, kernel: np.ndarray, average: bool = False, verbose: bool = False) -> np.ndarray:

    if image.ndim == 3:  # Si la imagen tiene 3 dimensiones (color)
        channels = cv2.split(image)  # Separa los canales de color (R, G, B)
        processed_channels = [convolution(ch, kernel, average, verbose) for ch in channels]
        return cv2.merge(processed_channels)  # Une los canales procesados

    # Obtiene dimensiones de la imagen y el kernel
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape
    pad_h, pad_w = kernel_row // 2, kernel_col // 2  # Calcula el padding necesario
    
    # Aplica padding a la imagen para evitar pérdida de bordes
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=np.float32)  # Crea una matriz de salida vacía

    # Itera sobre cada píxel de la imagen y aplica la convolución
    for row in range(image_row):
        for col in range(image_col):
            region = padded_image[row:row + kernel_row, col:col + kernel_col]  # Extrae la región
            output[row, col] = np.sum(region * kernel)  # Aplica el filtro
            if average:
                output[row, col] /= kernel.size  # Normaliza si es necesario

    return np.clip(output, 0, 255).astype(np.uint8)  # Asegura valores entre 0 y 255

# Diccionario con diferentes filtros predefinidos
filters = {
    "Bordes": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),  # Resalta bordes
    "Desenfoque": np.ones((5, 5), dtype=np.float32) / 25,  # Suaviza la imagen
    "Realce": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),  # Realza detalles
    "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)  # Efecto de relieve
}

# Carpeta donde se guardarán las imágenes procesadas
output_dir = "Fotos_convolucion"
os.makedirs(output_dir, exist_ok=True)  # Crea la carpeta si no existe

# Solicita el nombre del archivo de imagen al usuario
image_name = input("Ingrese el nombre de la imagen (con extensión, por ejemplo, 'imagen.jpg'): ")
image = cv2.imread(image_name)  # Carga la imagen

if image is None:
    print("Error: No se pudo cargar la imagen.")  # Manejo de error si la imagen no se encuentra
else:
    plt.figure(figsize=(10, 10))  # Configura el tamaño de la figura para visualizar imágenes
    for i, (filter_name, kernel) in enumerate(filters.items(), 1):
        result = convolution(image, kernel)  # Aplica el filtro
        output_name = os.path.join(output_dir, f"imagen_{filter_name.lower()}.jpg")  # Nombre del archivo de salida
        cv2.imwrite(output_name, result)  # Guarda la imagen procesada
        print(f"Imagen con filtro {filter_name} guardada en {output_name}")
        
        # Muestra cada imagen con su respectivo filtro
        plt.subplot(2, 2, i)  # Organiza en una cuadrícula de 2x2
        plt.imshow(result, cmap='gray')  # Muestra la imagen en escala de grises
        plt.title(f"Filtro: {filter_name}")  # Título con el nombre del filtro
        plt.axis('off')  # Oculta los ejes para mejor visualización
    plt.show()  # Muestra las imágenes procesadas