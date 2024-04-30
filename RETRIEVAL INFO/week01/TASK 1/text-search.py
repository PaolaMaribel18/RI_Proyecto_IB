import os

def foundText(word, dir):
    results = {}
    
    # Search in all archivesin the dir
    for files in os.listdir(dir):
        if files.endswith(".txt"):  # Only txt
            ruta_files = os.path.join(dir, files)
            with open(ruta_files, "r", encoding="utf-8") as f:
                line_concidences = []
                total_text = 0
                for num_line, linea in enumerate(f, start=1):
                    if word in linea:
                        line_concidences.append(num_line)
                        total_text += linea.count(word)
                if line_concidences:
                    results[files] = {"lineas": line_concidences, "total_text": total_text}
    
    return results

def main():
    # Dir entered by me
    dir = r"D:\SEPTIMO SEMESTRE II\RECUPERACION INFORMACION\KevinMaldonado99\RETRIEVAL INFO\TASK 1\BooksAll"

    word_to_search = input("Enter text to search: ")

    results = foundText(word_to_search, dir)

    if results:
        print("The word '{}' appears in the next files:".format(word_to_search))
        for files, info in results.items():
            print("- files: {}".format(files))
            #print("  Lineas con coincidencia:", info["lineas"])
            print("  Total coincidences:", info["total_text"])
            print()
    else:
        print("The word '{}'dosen't appear in any of the files.".format(word_to_search))

if __name__ == "__main__":
    main()
