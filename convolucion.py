import numpy as np
import cv2
import matplotlib.pyplot as plt

def convolution(image: np.ndarray, kernel: np.ndarray, average: bool = False, verbose: bool = False) -> np.ndarray:
    """
    Aplica una convolución a una imagen en escala de grises o a color procesando cada canal por separado.

    Parámetros:
        image (np.ndarray): Imagen de entrada (grayscale o color).
        kernel (np.ndarray): Kernel de convolución (matriz 2D).
        average (bool): Si es True, normaliza el kernel.
        verbose (bool): Si es True, muestra imágenes intermedias.

    Retorna:
        np.ndarray: Imagen procesada después de la convolución.
    """
    
    # Si la imagen tiene 3 canales (es a color), procesamos cada canal por separado
    if image.ndim == 3:
        print(f"Imagen con 3 canales detectada: {image.shape}")
        channels = cv2.split(image)  # Separamos los canales B, G y R
        # Aplicamos la convolución en cada canal por separado
        processed_channels = [convolution(ch, kernel, average, verbose) for ch in channels]
        return cv2.merge(processed_channels)  # Volvemos a unir los canales procesados

    print(f"Procesando imagen en escala de grises. Tamaño: {image.shape}")
    print(f"Kernel Shape: {kernel.shape}")

    # Muestra la imagen original si verbose está activado
    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Imagen Original")
        plt.show()

    # Obtiene dimensiones de la imagen y del kernel
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    # Calcula el tamaño del padding necesario para la convolución
    pad_h, pad_w = kernel_row // 2, kernel_col // 2
    # Agrega padding de ceros alrededor de la imagen para evitar pérdida de información en los bordes
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Muestra la imagen con padding si verbose está activado
    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Imagen con Padding")
        plt.show()

    # Crea una matriz vacía para almacenar la imagen de salida
    output = np.zeros_like(image, dtype=np.float32)

    # Aplica la convolución recorriendo cada píxel de la imagen original
    for row in range(image_row):
        for col in range(image_col):
            # Extrae la región de la imagen del mismo tamaño que el kernel
            region = padded_image[row:row + kernel_row, col:col + kernel_col]
            # Aplica la convolución sumando los valores multiplicados
            output[row, col] = np.sum(region * kernel)
            # Si average es True, normaliza dividiendo por el tamaño del kernel
            if average:
                output[row, col] /= kernel.size

    print(f"Tamaño de la imagen de salida: {output.shape}")

    # Muestra la imagen procesada si verbose está activado
    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title(f"Imagen Procesada con {kernel_row}x{kernel_col} Kernel")
        plt.show()

    # Asegura que los valores estén en el rango de 0 a 255 y convierte a entero sin signo
    return np.clip(output, 0, 255).astype(np.uint8)


# Solicita al usuario el nombre del archivo de la imagen
image_name = input("Ingrese el nombre de la imagen (con extensión, por ejemplo, 'imagen.jpg'): ")

# Carga la imagen desde el archivo ingresado
image = cv2.imread(image_name)

# Verifica si la imagen se cargó correctamente
if image is None:
    print("Error: No se pudo cargar la imagen. Verifique el nombre del archivo y la ubicación.")
else:
    # Define un kernel de ejemplo para detección de bordes
    kernel = np.array([[-1, -1, -1], 
                       [-1,  8, -1], 
                       [-1, -1, -1]], dtype=np.float32)

    # Aplica la convolución a la imagen
    result = convolution(image, kernel, verbose=True)

    # Guarda la imagen procesada con un nuevo nombre
    output_name = "imagen_procesada.jpg"
    cv2.imwrite(output_name, result)
    print(f"Imagen procesada guardada como {output_name}")

    # Muestra la imagen procesada con matplotlib
    plt.imshow(result, cmap='gray')
    plt.title("Imagen Procesada")
    plt.show()
