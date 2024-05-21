import os
import numpy as np
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Define la ruta al directorio que contiene los archivos de texto
CORPUS_DIR = r"D:\SEPTIMO SEMESTRE II\RI\KevinMaldonado99\RETRIEVAL INFO\week01\TASK 2\Data"
documents = {}

# Contador de archivos de texto
txt_files_count = 0

# Paso 1: Cargar y preprocesar documentos
print("Paso 1: Cargar y preprocesar documentos")
for filename in os.listdir(CORPUS_DIR):
    if filename.endswith('.txt'):
        txt_files_count += 1
        file_path = os.path.join(CORPUS_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            # Tokenización y conteo de palabras
            word_count = collections.Counter(file.read().lower().split())
            documents[filename] = ' '.join(word_count.elements())

# Imprimir resultados del Paso 1
print("Número de archivos .txt listos para usar:", txt_files_count)
print("Documentos cargados y preprocesados.")

# Paso 2: Vector Space Model (VSM)
print("\nSTEP 2: Vector Space Model (SVM)")
query = input("Enter you query: ")
#query = "home happy"
print(f"Consulta: {query}\n")

# Construye el vocabulario único
all_texts = list(documents.values())
vectorizer = CountVectorizer()
document_vectors = vectorizer.fit_transform(all_texts)
vocabulary = vectorizer.get_feature_names_out().tolist()  # Convertir a lista

# Paso 2.1: Representación de cada palabra como vector
print("1. Representación de cada palabra como vector:")
print("Dimensiones del vocabulario:", len(vocabulary))
print("Vectores de cada palabra (columnas de la matriz):")
print(document_vectors.toarray(), "\n")

# Paso 2.2: Normalización de las columnas de la matriz de vectores de documentos
print("2. Normalización de las columnas de la matriz:")
document_vectors_normalized = normalize(document_vectors, axis=0)
print("Matriz de vectores de documentos normalizada:")
print(document_vectors_normalized.toarray(), "\n")

# Paso 2.3: Representación de la consulta como un vector unitario
print("3. Vector unitario de la consulta:")
query_vector = np.zeros((1, len(vocabulary)))
query_words = query.split()
for word in query_words:
    if word in vocabulary:
        query_index = vocabulary.index(word)
        query_vector[0, query_index] = 1
print("Consulta como vector unitario:")
print(query_vector, "\n")

# Paso 3: Calcular similitud del coseno entre la consulta y los documentos
print("Paso 3: Calcular similitud del coseno")
cosine_similarities = cosine_similarity(query_vector, document_vectors_normalized).flatten()

# Paso 4: Clasificación de documentos según la similitud del coseno
print("4. Resultados de la similitud de coseno:")
print("Documentos clasificados por similitud de coseno:")
document_scores = [(filename, score) for filename, score in zip(documents.keys(), cosine_similarities)]
document_scores.sort(key=lambda x: x[1], reverse=True)
for filename, score in document_scores:
    print(f"{filename}: {score}")
