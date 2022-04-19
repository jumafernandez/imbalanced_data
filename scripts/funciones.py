# -*- coding: utf-8 -*-
"""
Archivo con funciones propias para el procesamiento de textos para la clasificación automática

Autor: Juan Fernández
"""

def limpiar_texto(texto): # Estrategia tomada de UNSL (https://github.com/hjthomp/tf-thompson)
    '''
    toma un texto y lo devuelve limpio (pasa a minúsculas, elimina símbolos, 
    dobles ocurrencias de letras, palabras y espacios).
    
    Returns
    -------
    
    '''
    import re
    import unicodedata
    
    # Pasamos todo a minúsculas
    texto = texto.lower()
    
    # Se Elimina TODO menos: \w = alphanum y ¿?%/
    texto = re.sub(r'[^\w %/]', " ", texto)   
        
    # Elimina acentos
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn') 
    
    #Remover letras con 2 ocurrencias (con excepciones)      
    letras_dobles = "abdfghijkmnñpqstuvwxyz" # Excepciones: ee-cc-ll-rr-oo (y mayus)
    letras_dobles += letras_dobles.upper()          
    texto = re.sub("(?P<char>[" + re.escape(letras_dobles) + "])(?P=char)+", r"\1", texto) 
    
    #remover caracteres que se repiten al menos 3 veces
    texto = re.sub(r'([\w\W])\1{2,}', r'\1', texto) 
    
    #remover palabras que se repiten 
    texto = re.sub(r'\b(\w+)(\b\W+\b\1\b)*', r'\1', texto) 
    
    #Eliminar repetición de espacios
    texto = re.sub(r"\s{2,}", " ", texto) 
    
    return texto

def preprocesar_correo(correo, remove_stopwords=True):
    '''
    Se eliminan las stopwords del texto del correo así como los acentos
        
    Returns
    -------
    texto_preprocesado: string con un correo limpio sin stopwords ni acentos
    '''
    import unicodedata
        
    texto = limpiar_texto(correo) 
        
    tokens = texto.split(' ')
    tkns_limpios = [] 
    stopwords = []
      
    if remove_stopwords:
        import nltk 
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        from nltk.corpus import stopwords
        stop_words_set = set(stopwords.words('spanish')) 
      
    #Eliminar acentos de stopwords
    stopwords_final = []
    
    for st in list(stop_words_set):
        stopwords_final.append(''.join(c for c in unicodedata.normalize('NFD', st) if unicodedata.category(c) != 'Mn'))        
        stopwords = set(stopwords_final)
    
        for tk in tokens: 
            if tk not in stopwords:
                if tk.count('/')>0 or tk.count('%')>0 or tk.isdigit() or (tk.isalpha() and (len(tk)>1)): 
                    tkns_limpios.append(tk) 
            
        texto_preprocesado = ' '.join(tkns_limpios)
      
    return texto_preprocesado


def preprocesar_correos(correos):
    '''
    Esta función toma los correos y los va preprocesando 
    uno a uno para devolverlos limpios de acuerdo a la función preprocesar_correo
      
    Returns
    -------
    correos_limpios: lista de correo preprocesados con la función preprocesar_correo 
    '''
    correos_limpios = []

    for correo in correos:
        correo_limpio = preprocesar_correo(correo, True)
        correos_limpios.append(correo_limpio)
        
    return correos_limpios


def representacion_documentos(textos, estrategia, MAX_TKS=None):
    ''' 
    Esta función recibe las consultas y genera las features dinámicas en base a 
    5 estrategias = {BASELINE, BOW, TFIDF, 3-4-NGRAM-CHAR, 1-2-NGRAM-WORD}
  
    Returns
    -------
    df_vectorizado: dataframe con las consultas vectorizadas  
    '''
    # Vamos a probar 4 estrategias de representación de documentos
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    import pandas as pd
      
    # Inicializamos el vectorizer de acuerdo a la estrategia de representación  
    if(estrategia=="BOW"):
        vectorizer = CountVectorizer(token_pattern = '[\w\/\%]+', max_features=MAX_TKS)
        
    elif(estrategia=="TFIDF"):
        vectorizer = TfidfVectorizer(token_pattern = '[\w\/\%]+', max_features=MAX_TKS)
                 
    elif(estrategia=="3-4-NGRAM-CHARS"):
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,4), token_pattern = '[\w\/\%]+', max_features=MAX_TKS)
        
    elif(estrategia=="1-2-NGRAM-WORDS"):
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), token_pattern = '[\w\/\%]+', max_features=MAX_TKS)
    else:
        # estrategia=="BASELINE"
        vectorizer = CountVectorizer(max_features=MAX_TKS)
    
    # Entrenamos el vectorizer para train y test
    df_vectorizado = vectorizer.fit_transform(textos)

    return df_vectorizado


def load_correos_etiquetados(TIPO='train', ESTATICAS=False, REPRESENTACION=False):
    """
    Carga los train y test set de correos electrónicos etiquetados

    Returns
    -------
    X: dataframe con características
    y: vector numpy con clases
    """
    # Se importan las librerías necesarias
    import pandas as pd
    import numpy as np
    import warnings
    import wget
    warnings.filterwarnings("ignore")

    # Se descargan los archivos del dataset
    DS_DIR = 'https://raw.githubusercontent.com/jumafernandez/imbalanced_data/main/data/'
    
    if TIPO=='train':
        FILE = 'correos-train-80.csv'
    elif TIPO=='test':
        FILE = 'correos-test-20.csv'

    # Genero el enlace completo y descargo los archivos
    URL_file = DS_DIR + FILE
    print(f'Se inicia descarga del dataset: {FILE}.')
    wget.download(URL_file)

    # Se levanta el archivo en un dataframe
    df = pd.read_csv(FILE)

    print(f"El conjunto de datos tiene la dimensión: {df.shape}")
    
    # Se preprocesan las consultas eliminando stopwords, acentos y otras cuestiones
    df['consulta'] = pd.Series(preprocesar_correos(df['consulta']))

    # Se separa el df en el vector de características y la clase (X e y)
    y = df['clase'].to_numpy()

    # Elimino la columna clase    
    df.drop('clase', inplace=True, axis=1)
    
    # Se escalan los datos con el método MinMax    
    from sklearn.preprocessing import MinMaxScaler
    min_max_scaler = MinMaxScaler()
    X = df
    columnas_numericas = list(X.select_dtypes(include=np.number).columns)

    X[columnas_numericas] = min_max_scaler.fit_transform(df[columnas_numericas])

    return X, y

# X, y = load_correos_etiquetados(REPRESENTACION='BOW')
