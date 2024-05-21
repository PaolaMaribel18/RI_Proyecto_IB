
import numpy as np
import csv

from sklearn.metrics.pairwise import cosine_similarity

# PASO 1: PREPROCESAMIENTO DE DE DATA

# Textos normalizados
texto1 = "Historia de Madrid: Madrid la capital de España tiene una rica historia que se remonta al siglo IX La ciudad fue fundada por los musulmanes y su nombre proviene del árabe Majrit que significa fuente de agua A lo largo de los siglos Madrid se ha convertido en un centro político económico y cultural de España"
texto2 = "El Palacio Real: El Palacio Real de Madrid es una de las residencias oficiales de la familia real española aunque ahora se usa principalmente para ceremonias de estado Con más de 3000 habitaciones es el palacio real más grande de Europa Occidental y un testimonio del esplendor y la historia de España"
texto3 = "Museo del Prado: El Museo del Prado es uno de los museos más famosos del mundo y un destino imprescindible para los amantes del arte que visitan Madrid Alberga una impresionante colección de obras de Velázquez Goya El Greco y otros maestros europeos"
texto4 = "Parque del Retiro: El Parque del Retiro es el pulmón verde de Madrid Antiguamente perteneciente a la monarquía española este parque de 118 hectáreas ofrece un respiro de la vida urbana con sus estanques jardines y el Palacio de Cristal"
texto5 = "Gastronomía Madrileña: Madrid es famosa por su vibrante escena gastronómica que incluye desde tapas en pequeños bares hasta restaurantes de alta cocina Platos típicos como el cocido madrileño y las patatas bravas son esenciales para cualquier visitante"
texto6 = "Gran Vía: La Gran Vía es una de las calles más famosas de Madrid conocida por sus tiendas teatros y arquitectura emblemática Este bullicioso bulevar es un punto de encuentro para turistas y locales por igual"
texto7 = "Real Madrid y el Estadio Santiago Bernabéu: El Real Madrid uno de los clubes de fútbol más famosos y exitosos del mundo juega sus partidos en el Estadio Santiago Bernabéu Este estadio es una meca para los aficionados al fútbol y ofrece tours que permiten a los visitantes explorar su rica historia"
texto8 = "Navidad en Madrid: Durante la temporada navideña Madrid se transforma con mercados festivos luces deslumbrantes y decoraciones La Plaza Mayor se convierte en un enorme mercado de Navidad donde se pueden encontrar todo tipo de regalos y delicias tradicionales"
texto9 = "Nochevieja en Puerta del Sol: La Puerta del Sol es el corazón de Madrid y el lugar donde miles se reúnen cada Nochevieja para dar la bienvenida al año nuevo Comer las doce uvas de la suerte al son de las campanadas del reloj es una tradición que todos deberían experimentar"
texto10 = "Madrid Río: Madrid Río es un parque lineal a lo largo del río Manzanares y un ejemplo perfecto de cómo la ciudad ha transformado áreas industriales en espacios verdes vibrantes Es un lugar popular para caminar andar en bicicleta y disfrutar de actividades al aire libre"
texto11 = "Historia de Dortmund: Dortmund situada en la región del Ruhr en Alemania tiene una historia que se remonta al año 882 Originalmente una pequeña villa Dortmund creció para convertirse en una importante ciudad industrial especialmente conocida por su producción de acero y cerveza"
texto12 = "Westfalenpark: Westfalenpark es uno de los parques urbanos más grandes de Europa y un lugar destacado para los visitantes en Dortmund El parque es famoso por su Rosarium que alberga una de las colecciones más extensas de rosas en el mundo y la torre Florian que ofrece vistas panorámicas de la ciudad"
texto13 = "Museo del Fútbol Alemán: Inaugurado en 2015 el Museo del Fútbol Alemán ofrece una experiencia interactiva que celebra la rica historia del fútbol alemán Ubicado en Dortmund el museo atrae a aficionados de todo el mundo y destaca por sus exposiciones que cuentan desde los primeros días del deporte hasta los triunfos en Copas del Mundo"
texto14 = "Mercado de Navidad de Dortmund: El Mercado de Navidad de Dortmund es uno de los más grandes y más visitados de Alemania Con su gigantesco árbol de Navidad posiblemente el más alto de los mercados navideños del país y más de 300 puestos que venden artesanías alimentos y bebidas tradicionales es un evento imperdible durante la temporada festiva"
texto15 = "U-Tower - Centro de Arte y Creatividad: El U-Tower una antigua cervecería reconvertida es ahora un centro dinámico de arte y creatividad Alberga el Museo Ostwall espacios de exposiciones y estudios de artistas y es un símbolo de la transformación cultural de Dortmund"
texto16 = "Borussia Dortmund y el Signal Iduna Park: Borussia Dortmund es uno de los clubes de fútbol más populares y exitosos de Alemania. El equipo juega sus partidos en casa en el Signal Iduna Park, conocido por su increíble atmósfera y el famoso “Muro Amarillo”, la tribuna sur que alberga a 25,000 aficionados apasionados."
texto17 = "La cervecería Dortmund: Dortmund tiene una larga tradición cervecera, y la visita a una de sus muchas cervecerías es una parada obligatoria. Las cervecerías ofrecen tours que incluyen degustaciones de cervezas locales como Dortmunder Export, una lager que es un orgullo de la ciudad."
texto18 = "La Reinoldikirche: La iglesia de San Reinold es el edificio más antiguo de Dortmund, datando del siglo XIII. Es un punto de referencia histórico importante y ofrece a los visitantes una mirada al pasado medieval de la ciudad."
texto19 = "Dortmund durante la Segunda Guerra Mundial: Dortmund fue un objetivo significativo durante los bombardeos de la Segunda Guerra Mundial debido a su industria pesada. Hoy en día, varios monumentos y museos ofrecen reflexiones sobre este período y sus efectos en la ciudad."
texto20 = "Zoológico de Dortmund: El Zoológico de Dortmund es un lugar excelente para familias y amantes de la naturaleza. Con más de 1,500 animales y un enfoque especial en la conservación de especies amenazadas, el zoológico proporciona una experiencia educativa y entretenida."

# Textos normalizados y tokenizados
texto1_tokens = [palabra.lower() for palabra in texto1.split() if palabra.isalnum()]
texto2_tokens = [palabra.lower() for palabra in texto2.split() if palabra.isalnum()]
texto3_tokens = [palabra.lower() for palabra in texto3.split() if palabra.isalnum()]
texto4_tokens = [palabra.lower() for palabra in texto4.split() if palabra.isalnum()]
texto5_tokens = [palabra.lower() for palabra in texto5.split() if palabra.isalnum()]
texto6_tokens = [palabra.lower() for palabra in texto6.split() if palabra.isalnum()]
texto7_tokens = [palabra.lower() for palabra in texto7.split() if palabra.isalnum()]
texto8_tokens = [palabra.lower() for palabra in texto8.split() if palabra.isalnum()]
texto9_tokens = [palabra.lower() for palabra in texto9.split() if palabra.isalnum()]
texto10_tokens = [palabra.lower() for palabra in texto10.split() if palabra.isalnum()]
texto11_tokens = [palabra.lower() for palabra in texto11.split() if palabra.isalnum()]
texto12_tokens = [palabra.lower() for palabra in texto12.split() if palabra.isalnum()]
texto13_tokens = [palabra.lower() for palabra in texto13.split() if palabra.isalnum()]
texto14_tokens = [palabra.lower() for palabra in texto14.split() if palabra.isalnum()]
texto15_tokens = [palabra.lower() for palabra in texto15.split() if palabra.isalnum()]
texto16_tokens = [palabra.lower() for palabra in texto16.split() if palabra.isalnum()]
texto17_tokens = [palabra.lower() for palabra in texto17.split() if palabra.isalnum()]
texto18_tokens = [palabra.lower() for palabra in texto18.split() if palabra.isalnum()]
texto19_tokens = [palabra.lower() for palabra in texto19.split() if palabra.isalnum()]
texto20_tokens = [palabra.lower() for palabra in texto20.split() if palabra.isalnum()]

# Vectores tokenizados
print("Texto 1:", texto1_tokens)
print("\n Texto 2:", texto2_tokens)
print("\n Texto 3:", texto3_tokens)
print("\n Texto 4:", texto4_tokens)
print("\n Texto 5:", texto5_tokens)
print("\n Texto 6:", texto6_tokens)
print("\n Texto 7:", texto7_tokens)
print("\n Texto 8:", texto8_tokens)
print("\n Texto 9:", texto9_tokens)
print("\n Texto 10:", texto10_tokens)
print("\n Texto 11:", texto11_tokens)
print("\n Texto 12:", texto12_tokens)
print("\n Texto 13:", texto13_tokens)
print("\n Texto 14:", texto14_tokens)
print("\n Texto 15:", texto15_tokens)
print("\n Texto 16:", texto16_tokens)
print("\n Texto 17:", texto17_tokens)
print("\n Texto 18:", texto18_tokens)
print("\n Texto 19:", texto19_tokens)
print("\n Texto 20: ", texto20_tokens)

textos = [
    "Historia de Madrid: Madrid, la capital de España, tiene una rica historia que se remonta al siglo IX. La ciudad fue fundada por los musulmanes, y su nombre proviene del árabe “Majrit” que significa “fuente de agua”. A lo largo de los siglos, Madrid se ha convertido en un centro político, económico y cultural de España.",
    "El Palacio Real: El Palacio Real de Madrid es una de las residencias oficiales de la familia real española, aunque ahora se usa principalmente para ceremonias de estado. Con más de 3,000 habitaciones, es el palacio real más grande de Europa Occidental y un testimonio del esplendor y la historia de España.",
    "Museo del Prado: El Museo del Prado es uno de los museos más famosos del mundo y un destino imprescindible para los amantes del arte que visitan Madrid. Alberga una impresionante colección de obras de Velázquez, Goya, El Greco y otros maestros europeos.",
    "Parque del Retiro: El Parque del Retiro es el pulmón verde de Madrid. Antiguamente perteneciente a la monarquía española, este parque de 118 hectáreas ofrece un respiro de la vida urbana con sus estanques, jardines y el Palacio de Cristal.",
    "Gastronomía Madrileña: Madrid es famosa por su vibrante escena gastronómica que incluye desde tapas en pequeños bares hasta restaurantes de alta cocina. Platos típicos como el cocido madrileño y las patatas bravas son esenciales para cualquier visitante.",
    "Gran Vía: La Gran Vía es una de las calles más famosas de Madrid, conocida por sus tiendas, teatros y arquitectura emblemática. Este bullicioso bulevar es un punto de encuentro para turistas y locales por igual.",
    "Real Madrid y el Estadio Santiago Bernabéu: El Real Madrid, uno de los clubes de fútbol más famosos y exitosos del mundo, juega sus partidos en el Estadio Santiago Bernabéu. Este estadio es una meca para los aficionados al fútbol y ofrece tours que permiten a los visitantes explorar su rica historia.",
    "Navidad en Madrid: Durante la temporada navideña, Madrid se transforma con mercados festivos, luces deslumbrantes y decoraciones. La Plaza Mayor se convierte en un enorme mercado de Navidad, donde se pueden encontrar todo tipo de regalos y delicias tradicionales.",
    "Nochevieja en Puerta del Sol: La Puerta del Sol es el corazón de Madrid y el lugar donde miles se reúnen cada Nochevieja para dar la bienvenida al año nuevo. Comer las doce uvas de la suerte al son de las campanadas del reloj es una tradición que todos deberían experimentar.",
    "Madrid Río: Madrid Río es un parque lineal a lo largo del río Manzanares y un ejemplo perfecto de cómo la ciudad ha transformado áreas industriales en espacios verdes vibrantes. Es un lugar popular para caminar, andar en bicicleta y disfrutar de actividades al aire libre.",
    "Historia de Dortmund: Dortmund, situada en la región del Ruhr en Alemania, tiene una historia que se remonta al año 882. Originalmente una pequeña villa, Dortmund creció para convertirse en una importante ciudad industrial, especialmente conocida por su producción de acero y cerveza.",
    "Westfalenpark: Westfalenpark es uno de los parques urbanos más grandes de Europa y un lugar destacado para los visitantes en Dortmund. El parque es famoso por su Rosarium, que alberga una de las colecciones más extensas de rosas en el mundo, y la torre Florian, que ofrece vistas panorámicas de la ciudad.",
    "Museo del Fútbol Alemán: Inaugurado en 2015, el Museo del Fútbol Alemán ofrece una experiencia interactiva que celebra la rica historia del fútbol alemán. Ubicado en Dortmund, el museo atrae a aficionados de todo el mundo y destaca por sus exposiciones que cuentan desde los primeros días del deporte hasta los triunfos en Copas del Mundo.",
    "Mercado de Navidad de Dortmund: El Mercado de Navidad de Dortmund es uno de los más grandes y más visitados de Alemania. Con su gigantesco árbol de Navidad, posiblemente el más alto de los mercados navideños del país, y más de 300 puestos que venden artesanías, alimentos y bebidas tradicionales, es un evento imperdible durante la temporada festiva.",
    "U-Tower - Centro de Arte y Creatividad: El U-Tower, una antigua cervecería reconvertida, es ahora un centro dinámico de arte y creatividad. Alberga el Museo Ostwall, espacios de exposiciones y estudios de artistas, y es un símbolo de la transformación cultural de Dortmund.",
    "Borussia Dortmund y el Signal Iduna Park: Borussia Dortmund es uno de los clubes de fútbol más populares y exitosos de Alemania. El equipo juega sus partidos en casa en el Signal Iduna Park, conocido por su increíble atmósfera y el famoso “Muro Amarillo”, la tribuna sur que alberga a 25,000 aficionados apasionados.",
    "La cervecería Dortmund: Dortmund tiene una larga tradición cervecera, y la visita a una de sus muchas cervecerías es una parada obligatoria. Las cervecerías ofrecen tours que incluyen degustaciones de cervezas locales como Dortmunder Export, una lager que es un orgullo de la ciudad.",
    "La Reinoldikirche: La iglesia de San Reinold es el edificio más antiguo de Dortmund, datando del siglo XIII. Es un punto de referencia histórico importante y ofrece a los visitantes una mirada al pasado medieval de la ciudad.",
    "Dortmund durante la Segunda Guerra Mundial: Dortmund fue un objetivo significativo durante los bombardeos de la Segunda Guerra Mundial debido a su industria pesada. Hoy en día, varios monumentos y museos ofrecen reflexiones sobre este período y sus efectos en la ciudad.",
    "Zoológico de Dortmund: El Zoológico de Dortmund es un lugar excelente para familias y amantes de la naturaleza. Con más de 1,500 animales y un enfoque especial en la conservación de especies amenazadas, el zoológico proporciona una experiencia educativa y entretenida."
]

# Crear un conjunto para almacenar todas las palabras únicas normalizadas
palabras_unicas_normalizadas = set()

# Agregar todas las palabras normalizadas al conjunto
for texto in textos:
    palabras = texto.split()
    for palabra in palabras:
        # Eliminar caracteres especiales y convertir a minúsculas
        palabra = palabra.strip('":,.').lower()
        # Agregar la palabra normalizada al conjunto
        palabras_unicas_normalizadas.add(palabra)

# Convertir el conjunto a una lista para obtener un vector
diccionario_palabras_normalizado = list(palabras_unicas_normalizadas)

# Mostrar el vector de palabras normalizadas
print("\nVector de termino-diccionario: \n \n" ,diccionario_palabras_normalizado)


##############
# Número total de términos en el diccionario
total_terminos = len(diccionario_palabras_normalizado)

# Mostrar el número total de términos en el diccionario
print("\nNúmero total de términos en el diccionario:", total_terminos,"\n")


# Número total de palabras en todo el corpus
total_palabras_corpus = sum(len(texto.split()) for texto in textos)

# Mostrar el número total de palabras en todo el corpus
#print("\nNúmero total de palabras en todo el corpus:", total_palabras_corpus)


#Matriz de categorias# Inicializar matriz de conteo de palabras por texto
matriz_conteo = []

# Recorrer cada texto
for texto_tokens in [texto1_tokens, texto2_tokens, texto3_tokens, texto4_tokens, texto5_tokens,
                     texto6_tokens, texto7_tokens, texto8_tokens, texto9_tokens, texto10_tokens,
                     texto11_tokens, texto12_tokens, texto13_tokens, texto14_tokens, texto15_tokens,
                     texto16_tokens, texto17_tokens, texto18_tokens, texto19_tokens, texto20_tokens]:
    # Inicializar vector de conteo para el texto actual
    conteo_texto = []
    # Recorrer cada término en el diccionario
    for palabra_diccionario in diccionario_palabras_normalizado:
        # Contar la frecuencia de la palabra en el texto actual
        frecuencia = texto_tokens.count(palabra_diccionario)
        # Agregar el conteo al vector de conteo del texto actual
        conteo_texto.append(frecuencia)
    # Agregar el vector de conteo del texto actual a la matriz de conteo
    matriz_conteo.append(conteo_texto)
# Nombre del archivo CSV de salida
nombre_archivo_csv = 'matriz_conteo.csv'

# Escribir los datos en el archivo CSV
with open(nombre_archivo_csv, 'w', newline='', encoding='utf-8') as archivo_csv:
    escritor_csv = csv.writer(archivo_csv)
    
    # Escribir los encabezados
    escritor_csv.writerow(['Texto'] + diccionario_palabras_normalizado)
    
    # Escribir los datos de la matriz de conteo
    for i, conteo_texto in enumerate(matriz_conteo, start=1):
        escritor_csv.writerow([f'Texto {i}'] + conteo_texto)

# Imprimir encabezado de las columnas (primeros 10 términos del diccionario)
print("  ".join(diccionario_palabras_normalizado[:10]) + "  ...  " + "  ".join(diccionario_palabras_normalizado[-10:]))

# Recorrer cada texto
for i, conteo_texto in enumerate(matriz_conteo, start=1):
    # Convertir el conteo de palabras a una cadena de texto
    conteo_texto_str = "  ".join(map(str, conteo_texto))
    # Imprimir la fila de conteo de palabras para el texto actual
    print(f"Texto {i}: {conteo_texto_str[:50]} ... {conteo_texto_str[-50:]}")



#PASO 2:  Query de las 20 consultas (Q)

# Consultas
consultas = [
    "Historia medieval de las ciudades europeas",
    "Principales destinos turísticos en Europa",
    "Influencia de la realeza en la cultura europea",
    "Importancia de los parques urbanos en las ciudades",
    "Gastronomía típica en capitales europeas",
    "Eventos deportivos icónicos en Europa",
    "Celebraciones de Navidad en ciudades europeas",
    "Museos de arte importantes en Europa",
    "Efectos de la Segunda Guerra Mundial en ciudades europeas",
    "Arquitectura histórica en ciudades europeas",
    "Clubes de fútbol famosos y sus estadios en Europa",
    "Transformación urbana y regeneración de espacios",
    "Tradición cervecera en ciudades europeas",
    "Mercados y comercio tradicional en Europa",
    "Centros de arte y creatividad en ciudades modernas",
    "Actividades de ocio y entretenimiento en ciudades metropolitanas",
    "Conservación de la naturaleza y la vida silvestre en zonas urbanas",
    "Planificación de eventos culturales y festivales en ciudades",
    "Desarrollo del transporte y la infraestructura urbana",
    "Impacto de la tecnología en la vida urbana"
]

# Crear la matriz de consultas
matriz_consultas = []

# Crear la matriz
for i, consulta in enumerate(consultas, start=1):
    # Normalizar la consulta y contar términos
    consulta_normalizada = consulta.lower()
    numero_terminos = len(consulta.split())
    # Agregar la consulta y el número de términos a la matriz
    matriz_consultas.append([f"q{i}", #consulta_normalizada, 
                             numero_terminos])

# Mostrar la matriz de consultas
print("Matriz de consultas:")
for fila in matriz_consultas:
    print(fila)
#PASO 3: Matriz 20 x n  (Q')

print("\n\nMatriz Q'\n")

# Crear la matriz de consultas con conteos de términos
matriz_consultas_conteo = []

# Mostrar los términos del diccionario en las columnas
print("\nTérminos del diccionario:")
print("  ".join(diccionario_palabras_normalizado))

# Recorrer cada consulta
for i, consulta in enumerate(consultas, start=1):
    # Tokenizar la consulta
    consulta_tokens = [palabra.lower() for palabra in consulta.split() if palabra.isalnum()]
    # Inicializar el vector de conteo para la consulta actual
    conteo_consulta = []
    # Recorrer cada término en el diccionario de palabras normalizado
    for palabra_diccionario in diccionario_palabras_normalizado:
        # Contar la frecuencia de la palabra en la consulta actual
        frecuencia = consulta_tokens.count(palabra_diccionario)
        # Agregar el conteo al vector de conteo de la consulta actual
        conteo_consulta.append(frecuencia)
    # Agregar el vector de conteo de la consulta actual a la matriz de consultas
    matriz_consultas_conteo.append(conteo_consulta)

# Mostrar la matriz de consultas con conteos de términos
print("\nMatriz de consultas con conteos de términos:")
for i, conteo_consulta in enumerate(matriz_consultas_conteo, start=1):
    # Convertir el conteo de términos a una cadena de texto
    conteo_consulta_str = "  ".join(map(str, conteo_consulta))
    # Imprimir la fila de conteo de términos para la consulta actual
    print(f"Consulta {i}: {conteo_consulta_str[:50]} ... {conteo_consulta_str[-50:]}")


#PASO 4: Matriz de distancias.

print("\n\nMatriz de distancias\n\n") 


# Convertir las matrices de conteo de términos a matrices numpy para facilitar los cálculos
matriz_conteo_np = np.array(matriz_conteo)
matriz_consultas_conteo_np = np.array(matriz_consultas_conteo)

# Calcular la distancia del coseno entre cada par de consulta y documento
distancias_cos = cosine_similarity(matriz_consultas_conteo_np, matriz_conteo_np)

# Mostrar la matriz de distancias del coseno
print("\nMatriz de distancias del coseno entre consultas y documentos:")
for i in range(len(consultas)):
    for j in range(len(textos)):
        print(f"Distancia entre Consulta {i+1} y Texto {j+1}: {distancias_cos[i][j]}")


# PASO 5 : Rnaking de distancias mas relevantes

# Función para encontrar las distancias más cercanas a 0 para cada consulta
def distancias_cercanas_a_cero(distancias, consultas, textos):
    resultados = []  # Lista para almacenar los resultados
    for i, consulta in enumerate(consultas):
        # Filtrar las distancias para excluir las que sean 0.0
        distancias_filtradas = [d for d in distancias[i] if d > 0.0]
        # Ordenar las distancias filtradas
        distancias_ordenadas = sorted(distancias_filtradas)
        # Tomar las tres distancias más cercanas a 0
        distancias_cercanas = distancias_ordenadas[:3]
        resultados.append((consulta, distancias_cercanas))
    
    return resultados

# Calcula las distancias del coseno entre consultas y textos
distancias_cos = cosine_similarity(matriz_consultas_conteo_np, matriz_conteo_np)

# Obtener los resultados de las distancias cercanas a 0 para cada consulta
resultados = distancias_cercanas_a_cero(distancias_cos, consultas, textos)

# Guardar los resultados en un archivo de texto
with open('resultados.csv', 'w') as archivo:
    for consulta, distancias_cercanas in resultados:
        archivo.write(f"Para la consulta {consulta}, las distancias más cercanas a 0 son:\n")
        for distancia in distancias_cercanas:
            indice_texto = np.where(distancias_cos[consultas.index(consulta)] == distancia)[0][0]
            archivo.write(f"Distancia entre Consulta {consultas.index(consulta)+1} y Texto {indice_texto+1}: {distancia}\n")


