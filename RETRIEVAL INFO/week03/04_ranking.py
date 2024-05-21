import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#STEP 1: Data Preprocessing

# Define the path to the directory containing the text files
CORPUS_DIR = r"D:\SEPTIMO SEMESTRE II\RI\KevinMaldonado99\RETRIEVAL INFO\week01\TASK 2\Data"
documents = {}

# Count and preprocess documents
txt_files_count = 0
for filename in os.listdir(CORPUS_DIR):
    if filename.endswith('.txt'):
        txt_files_count += 1
        file_path = os.path.join(CORPUS_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            documents[filename] = file.read().lower()  # Read and convert to lowercase

# Corpus is ready for use
print("Corpus is ready for use.")
print(f"Number of .txt files ready to use: {txt_files_count}")            

#STEP 2: Vector Space Model (SVM)


# Define la consulta
query = "home"

# Construye el vocabulario único
all_texts = list(documents.values())
vectorizer = CountVectorizer()
document_vectors = vectorizer.fit_transform(all_texts)
vocabulary = vectorizer.get_feature_names_out()

# Representación de documentos y consulta como vectores
query_vector = vectorizer.transform([query])

# Cálculo de similitud del coseno
cosine_similarities = cosine_similarity(query_vector, document_vectors).flatten()

# Clasificación de documentos según similitud de coseno
document_scores = [(filename, score) for filename, score in zip(documents.keys(), cosine_similarities)]
document_scores.sort(key=lambda x: x[1], reverse=True)

# Imprime los resultados
print("\nDocumentos clasificados por similitud de coseno:")
for filename, score in document_scores:
    print(f"{filename}: {score}")
    # Contar coincidencias de palabras de la consulta en cada documento
    document_words = documents[filename].split()
    matches = sum(word in document_words for word in query.lower().split())
    print(f"Total de conteo: {matches} palabras coinciden")
    print()  # Imprime una línea en blanco entre cada documento
    
    # Obtener la matriz de conteo
count_matrix = document_vectors.toarray()

# Imprimir la matriz de conteo
print("\nMatriz de conteo:")
print(count_matrix)