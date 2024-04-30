import os
import re

def cargar_archivo(ruta_archivo):
    with open(ruta_archivo, "r", encoding="utf-8") as archivo:
        contenido = archivo.read()
    return contenido

def tokenizar_texto(texto):
    # Dividir el texto en palabras individuales
    palabras = re.findall(r'\b\d{2,}\b|\b\w+\b', texto)
    return palabras

def limpiar_texto(palabras):
    # Excluir signos de puntuación y convertir números en palabras
    palabras_limpias = [palabra.lower() for palabra in palabras if palabra.isalpha()]
    return palabras_limpias

def procesar_archivo(ruta_archivo):
    # Cargar el archivo de texto
    contenido = cargar_archivo(ruta_archivo)
    
    # Tokenizar el texto y limpiar las palabras
    palabras_tokenizadas = tokenizar_texto(contenido)
    palabras_limpias = limpiar_texto(palabras_tokenizadas)
    
    return palabras_limpias

def preparar_archivos_tokenizados(directorio):
    archivos_tokenizados = {}
    
    # Recorrer todos los archivos en el directorio
    for nombre_archivo in os.listdir(directorio):
        if nombre_archivo.endswith(".txt"):
            ruta_archivo = os.path.join(directorio, nombre_archivo)
            
            # Procesar el archivo y almacenar el texto procesado en una lista de palabras
            palabras_procesadas = procesar_archivo(ruta_archivo)
            archivos_tokenizados[nombre_archivo] = palabras_procesadas
    
    return archivos_tokenizados

# Directorio donde se encuentran los archivos de texto
directorio = r"D:\SEPTIMO SEMESTRE II\RECUPERACION INFORMACION\KevinMaldonado99\RETRIEVAL INFO\week01\TASK 1\Data"

# Preparar archivos tokenizados
archivos_tokenizados = preparar_archivos_tokenizados(directorio)

# Ejemplo de uso: Mostrar las palabras del primer archivo tokenizado
primer_archivo = next(iter(archivos_tokenizados.values()))
print(primer_archivo)

def crear_indice_invertido(archivos_tokenizados):
    indice_invertido = {}

    # Recorrer cada archivo tokenizado
    for nombre_archivo, palabras in archivos_tokenizados.items():
        # Obtener el ID del documento (en este caso, el nombre del archivo)
        id_documento = nombre_archivo

        # Contar la frecuencia de cada palabra en el documento
        frecuencia_palabra = {}
        for palabra in palabras:
            frecuencia_palabra[palabra] = frecuencia_palabra.get(palabra, 0) + 1

        # Actualizar el índice invertido con la información del documento actual
        for palabra, frecuencia in frecuencia_palabra.items():
            if palabra in indice_invertido:
                indice_invertido[palabra].append((id_documento, frecuencia))
            else:
                indice_invertido[palabra] = [(id_documento, frecuencia)]

    return indice_invertido

# Crear el índice invertido
indice_invertido = crear_indice_invertido(archivos_tokenizados)


def analizar_consulta(consulta):
    # Separar la consulta en términos y operadores booleanos
    terminos = consulta.split()
    operadores_booleanos = [term for term in terminos if term in ["AND", "OR", "NOT"]]
    terminos = [term for term in terminos if term not in ["AND", "OR", "NOT"]]
    
  
    return terminos, operadores_booleanos

def buscar_documentos(terminos, operadores_booleanos, indice_invertido):
    # Inicializar una lista de documentos coincidentes con el primer término de búsqueda
    documentos_coincidentes = set(indice_invertido.get(terminos[0], []))

    # Iterar sobre los términos de búsqueda y aplicar los operadores booleanos
    for i in range(1, len(terminos)):
        # Obtener los documentos que contienen el término actual, si existe
        documentos_termino = set(indice_invertido.get(terminos[i], []))
        
        if operadores_booleanos:
            operador = operadores_booleanos.pop(0)
            if operador == "AND":
                # Verificar si ambos términos están presentes en los mismos documentos
                documentos_coincidentes &= documentos_termino
            elif operador == "OR":
                documentos_coincidentes |= documentos_termino
            elif operador == "NOT":
                documentos_coincidentes -= documentos_termino
        else:
            # Si no hay más operadores booleanos, asumimos un operador "AND" implícito
            # Verificar si ambos términos están presentes en los mismos documentos
            documentos_coincidentes &= documentos_termino

    return documentos_coincidentes




def mostrar_resultados(documentos_coincidentes):
    if documentos_coincidentes:
        print("Documentos que coinciden con la consulta:")
        for documento in documentos_coincidentes:
            print("- ", documento)
    else:
        print("No se encontraron documentos que coincidan con la consulta.")

# Ejemplo de uso
def obtener_consulta_usuario():
    consulta = input("Ingrese la consulta de búsqueda: ")
    return consulta

# Obtener la consulta del usuario
consulta_usuario = obtener_consulta_usuario()

# Analizar la consulta del usuario
terminos, operadores_booleanos = analizar_consulta(consulta_usuario)

# Buscar documentos que coincidan con la consulta del usuario
documentos_coincidentes = buscar_documentos(terminos, operadores_booleanos, indice_invertido)

# Mostrar los resultados al usuario
mostrar_resultados(documentos_coincidentes)



#Para mostrar los indices invertidos de cada palabra
#for palabra, apariciones in indice_invertido.items():
 #   print(palabra, apariciones)
    # Limitar la cantidad de salida si es necesario
  #  if len(apariciones) > 5:
   #     break

