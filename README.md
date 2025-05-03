# 🔍 Information Retrieval System - Reuters-21578

A university project developed for the **Information Retrieval** course (ICCD753) at the **Escuela Politécnica Nacional**, using the **Reuters-21578** corpus as the main dataset.

## 📚 Description

This project aims to design, build, and deploy an Information Retrieval System (IRS) capable of preprocessing, indexing, and retrieving documents based on similarity with user queries. The solution is implemented in **Python**, with a **Flask**-based web interface, and explores classical approaches like **Bag of Words (BoW)** and **TF-IDF** for vector representation.

## 👩‍💻 About Us

We are students from the **Computer Engineering** program at EPN:

- **Paola Aucapiña**
- **Kevin Maldonado**
- **Raquel Zumba**

**Instructor:** Iván Carrera, Ph.D.  
**Delivery Date:** June 19, 2024  
**Semester:** 2024-A

## ⚙️ Project Structure

### 📥 1. Data Acquisition
- Downloaded and organized the Reuters-21578 corpus.
- Included support files: `stopwords.txt`, `cats.txt`.

### 🧹 2. Preprocessing
- Normalization, tokenization, stopword removal, stemming.
- Preprocessed documents stored in a DataFrame.

### 📊 3. Vector Representation
- **BoW** and **TF-IDF** techniques applied.
- Document-term matrices created for each method.

### 🗂️ 4. Indexing
- Built inverted indexes for both BoW and TF-IDF.
- Enables efficient document lookup by term.

### 🔎 5. Search Engine
- Cosine similarity computed between queries and documents.
- Top matching results returned based on threshold values.

### 📈 6. Evaluation
- Precision, Recall, and F1-score metrics calculated.
- Ground truth derived from document categories.

### 🌐 7. Web Interface
- Developed with Flask.
- Users can enter queries and view relevant documents with ranked results.

## 🛠️ Technologies Used
- Python
- Flask
- Pandas, Scikit-learn
- HTML/CSS (for the frontend interface)

## 📌 Highlights
- Compared the performance of BoW vs TF-IDF.
- Achieved high recall scores in several categories.
- TF-IDF demonstrated better overall performance.

