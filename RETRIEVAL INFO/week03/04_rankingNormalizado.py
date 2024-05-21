import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# STEP 1: Data Preprocessing

# Define la ruta al directorio que contiene los archivos de texto
CORPUS_DIR = r"D:\SEPTIMO SEMESTRE II\RI\KevinMaldonado99\RETRIEVAL INFO\week01\TASK 2\Data"
documents = {}

# Count and preprocess documents
txt_files_count = 0
for filename in os.listdir(CORPUS_DIR):
    if filename.endswith('.txt'):
        txt_files_count += 1
        file_path = os.path.join(CORPUS_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            documents[filename] = file.read().lower()  # Leer y convertir a minúsculas

# Corpus is ready for use
print("STEP 1: Data Preprocessing")
print("Corpus is ready for use.")
print(f"Number of .txt files ready to use: {txt_files_count}\n")            

# STEP 2: Vector Space Model (SVM)

# Define la consulta
query = "home"
print("STEP 2: Vector Space Model (SVM)")
print(f"Consulta: {query}\n")

# Construye el vocabulario único
all_texts = list(documents.values())
vectorizer = CountVectorizer()
document_vectors = vectorizer.fit_transform(all_texts)
vocabulary = vectorizer.get_feature_names_out().tolist()  # Convertir a lista

print("1. Representación de cada palabra como vector:")
print("Dimensiones del vocabulario:", len(vocabulary))
print("Vectores de cada palabra (columnas de la matriz):")
print(document_vectors.toarray(), "\n")

# Normalizar las columnas de la matriz de vectores de documentos
document_vectors_normalized = normalize(document_vectors, axis=0)
print("2. Normalización de las columnas de la matriz:")
print("Matriz de vectores de documentos normalizada:")
print(document_vectors_normalized.toarray(), "\n")

# Representación de consulta como un vector unitario
query_vector = np.zeros((1, len(vocabulary)))
query_index = vocabulary.index(query) if query in vocabulary else -1
if query_index != -1:
    query_vector[0, query_index] = 1

print("3. Vector unitario de la consulta:")
print("Consulta como vector unitario:")
print(query_vector, "\n")

# Calcular la similitud del coseno entre la consulta y los vectores de documentos normalizados
cosine_similarities = cosine_similarity(query_vector, document_vectors_normalized).flatten()

# Clasificación de documentos según la similitud del coseno
document_scores = [(filename, score) for filename, score in zip(documents.keys(), cosine_similarities)]
document_scores.sort(key=lambda x: x[1], reverse=True)

# Imprime los resultados
print("4. Resultados de la similitud de coseno:")
print("Documentos clasificados por similitud de coseno:")
for filename, score in document_scores:
    print(f"{filename}: {score}")
