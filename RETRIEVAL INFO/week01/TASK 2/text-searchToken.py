import os
import re

def cargar_archivo(ruta_archivo):
    with open(ruta_archivo, "r", encoding="utf-8") as archivo:
        contenido = archivo.read()
    return contenido

def tokenizar_texto(texto):
    palabras = re.findall(r'\b\d{2,}\b|\b\w+\b', texto)
    return palabras

def limpiar_texto(palabras):
    palabras_limpias = [palabra.lower() for palabra in palabras if palabra.isalpha()]
    return palabras_limpias

def procesar_archivo(ruta_archivo):
    contenido = cargar_archivo(ruta_archivo)
    palabras_tokenizadas = tokenizar_texto(contenido)
    palabras_limpias = limpiar_texto(palabras_tokenizadas)
    return palabras_limpias

def preparar_archivos_tokenizados(directorio):
    archivos_tokenizados = {}
    for nombre_archivo in os.listdir(directorio):
        if nombre_archivo.endswith(".txt"):
            ruta_archivo = os.path.join(directorio, nombre_archivo)
            palabras_procesadas = procesar_archivo(ruta_archivo)
            archivos_tokenizados[nombre_archivo] = palabras_procesadas
    return archivos_tokenizados

def crear_indice_invertido(archivos_tokenizados):
    indice_invertido = {}
    for nombre_archivo, palabras in archivos_tokenizados.items():
        for palabra in palabras:
            if palabra in indice_invertido:
                indice_invertido[palabra].add(nombre_archivo)
            else:
                indice_invertido[palabra] = {nombre_archivo}
    return indice_invertido

def analizar_consulta(consulta):
    terminos = consulta.split()
    operadores_booleanos = [term for term in terminos if term in ["AND", "OR", "NOT"]]
    terminos = [term for term in terminos if term not in ["AND", "OR", "NOT"]]
    return terminos, operadores_booleanos

def buscar_documentos(terminos, operadores_booleanos, indice_invertido):
    documentos_coincidentes = None
    for i, termino in enumerate(terminos):
        documentos_termino = indice_invertido.get(termino, set())
        if operadores_booleanos and i < len(operadores_booleanos):
            operador = operadores_booleanos[i]
            if operador == "AND":
                documentos_coincidentes = documentos_termino if documentos_coincidentes is None else documentos_coincidentes & documentos_termino
            elif operador == "OR":
                documentos_coincidentes = documentos_termino if documentos_coincidentes is None else documentos_coincidentes | documentos_termino
            elif operador == "NOT":
                documentos_coincidentes -= documentos_termino
        else:
            documentos_coincidentes = documentos_termino if documentos_coincidentes is None else documentos_coincidentes & documentos_termino
    return documentos_coincidentes


def mostrar_resultados(documentos_coincidentes):
    if documentos_coincidentes:
        print("\nDocumentos que coinciden con la consulta:")
        for documento in documentos_coincidentes:
            print("- ", documento)
    else:
        print("No se encontraron documentos que coincidan con la consulta.")

def obtener_consulta_usuario():
    consulta = input("Ingrese la consulta de búsqueda: ")
    return consulta




directorio = r"D:\SEPTIMO SEMESTRE II\RI\KevinMaldonado99\RETRIEVAL INFO\week01\TASK 2\Data"
archivos_tokenizados = preparar_archivos_tokenizados(directorio)
indice_invertido = crear_indice_invertido(archivos_tokenizados)


consulta_usuario = obtener_consulta_usuario()
terminos, operadores_booleanos = analizar_consulta(consulta_usuario)
documentos_coincidentes = buscar_documentos(terminos, operadores_booleanos, indice_invertido)
mostrar_resultados(documentos_coincidentes)

def crear_matriz_indices(archivos_tokenizados):
    matriz_indices = {}
    for nombre_archivo, palabras in archivos_tokenizados.items():
        for palabra in palabras:
            if palabra in matriz_indices:
                matriz_indices[palabra][nombre_archivo] = 1
            else:
                matriz_indices[palabra] = {archivo: 0 for archivo in archivos_tokenizados}
                matriz_indices[palabra][nombre_archivo] = 1
    return matriz_indices


def mostrar_matriz_indices(matriz_indices):
    print("\nMatriz de índices:")
    print("{: <15}".format("Palabra"), end="")
    for nombre_archivo in matriz_indices[next(iter(matriz_indices))]:
        print("{: <10}".format(nombre_archivo), end="")
    print()
    for palabra, presencias in matriz_indices.items():
        print("{: <15}".format(palabra), end="")
        for nombre_archivo in matriz_indices[next(iter(matriz_indices))]:
            print("{: <10}".format(presencias.get(nombre_archivo, 0)), end="")
        print()


# Luego de obtener la consulta y los documentos coincidentes

# Preparar la matriz de índices
matriz_indices = crear_matriz_indices(archivos_tokenizados)

# Mostrar la matriz de índices
mostrar_matriz_indices(matriz_indices)
