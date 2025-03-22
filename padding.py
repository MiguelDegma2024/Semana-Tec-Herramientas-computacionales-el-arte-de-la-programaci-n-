import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_padding(image: np.ndarray, padding_size: int, padding_type: str = 'constant', constant_value: int = 0) -> np.ndarray:
    # Mapear los tipos de padding a los de OpenCV
    padding_modes = {
        'constant': cv2.BORDER_CONSTANT,
        'reflect': cv2.BORDER_REFLECT,     # Refleja los bordes (abc|cba)
        'replicate': cv2.BORDER_REPLICATE, # Replica los bordes (aaaa|abcd|dddd)
        'wrap': cv2.BORDER_WRAP            # Envuelve la imagen (abcd|abcd|abcd)
    }
    
    if padding_type not in padding_modes:
        raise ValueError("Tipo de padding no válido. Usa: 'constant', 'reflect', 'replicate' o 'wrap'.")

    # Aplica el padding
    padded_image = cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, 
                                      borderType=padding_modes[padding_type], value=constant_value)
    
    return padded_image

# Solicita el nombre de la imagen al usuario
image_name = input("Ingrese el nombre de la imagen (con extensión, por ejemplo, 'imagen.jpg'): ")

# Carga la imagen
image = cv2.imread(image_name)

# Verifica si la imagen se cargó correctamente
if image is None:
    print("Error: No se pudo cargar la imagen. Verifique el nombre del archivo y la ubicación.")
else:
    # Solicita al usuario el tamaño del padding
    padding_size = int(input("Ingrese el tamaño del padding en píxeles: "))

    # Solicita el tipo de padding
    print("Tipos de padding disponibles: 'constant', 'reflect', 'replicate', 'wrap'")
    padding_type = input("Ingrese el tipo de padding: ")

    # Si el padding es constante, solicita el valor de relleno
    constant_value = 0
    if padding_type == 'constant':
        constant_value = int(input("Ingrese el valor del padding (0-255): "))

    # Aplica el padding
    padded_image = apply_padding(image, padding_size, padding_type, constant_value)

    # Muestra la imagen original y con padding
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Imagen Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Imagen con Padding ({padding_type})")
    plt.axis("off")

    plt.show()

    # Crear la carpeta Fotos_Padding si no existe
    if not os.path.exists("Fotos_Padding"):
        os.makedirs("Fotos_Padding")

    # Definir el nombre del archivo de salida
    output_name = os.path.join("Fotos_Padding", "imagen_con_padding.jpg")

    # Guarda la imagen con padding en la carpeta Fotos_Padding
    cv2.imwrite(output_name, padded_image)
    print(f"Imagen con padding guardada como {output_name}")

