



import streamlit as st
import time
import pandas as pd
#from textblob import TextBlob
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download("punkt")
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
#from gensim.models import Word2Vec

st.markdown('<style>body{background-color:black;}</style>',unsafe_allow_html=True)
st.markdown('<img class="preload-me lazyloaded"  src="https://nanos.ai/wp-content/uploads/2020/09/nanos_logo_WEaB.png" width="200" height="300" alt="Nanos" sizes="512px"  srcset="https://nanos.ai/wp-content/uploads/2020/09/nanos_logo_WEaB.png 54w, https://nanos.ai/wp-content/uploads/2020/10/nanos_logo.png 417w" data-ll-status="loaded">',unsafe_allow_html=True)

html_temp = """
<div style="background-color:#5eb782;padding:10px">
<h2 style="color:white;text-align:center;">Word_Similarity.app </h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)

#Importing packages
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.common.exceptions import InvalidArgumentException
import pandas as pd




url=st.text_input("Enter Your URL")
#ht="""<text-style="background-color:#5eb782;padding:10px">"""
#st.markdown(ht,unsafe_allow_html=True)
try:
    


    submit = st.button('Submit')

    @st.cache
    def func(url):
    
        driver = webdriver.Chrome(r"C:\Users\MONSTER\Desktop\driver\chromedriver.exe")
        driver.get(url)
        time.sleep(3)
        text = driver.find_elements_by_xpath("/html/body")[0]
        all_text = text.text


        #from textblob import TextBlob
        from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn import decomposition, ensemble
        from nltk.tokenize import sent_tokenize, word_tokenize
        import nltk
        nltk.download("punkt")
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords
        from warnings import filterwarnings
        filterwarnings('ignore')
        from gensim.models import Word2Vec
        import numpy as np
    
        text=word_tokenize(all_text.lower())
        text = [w for w in text if w.isalpha()]
        stop_words = stopwords.words('english')
        text = set([t for t in text if t not in stop_words])
    
        #glove has 100 dimension and 6B words
        embeddings_index = dict()
        f = open('./glove.6B.100d.txt', encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
    
        import numpy as np
        from tensorflow.keras.models import Sequential,Model
        from tensorflow.keras.layers import Embedding
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        MAX_NB_WORDS = 100

        tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(text)
        sequences = tokenizer.texts_to_sequences(text)

        word_index = tokenizer.word_index
        #print('Found %s unique tokens.' % len(word_index))
    
        #glove yükle
        vocab_size = len(tokenizer.word_index) + 1
        embedding_matrix = np.zeros((vocab_size, 100))
        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        sim_mat = cosine_similarity(embedding_matrix,embedding_matrix)
        return sim_mat, text, word_index




    def word_sim(word):
        index = func(url)[2].get(word)
        df = pd.DataFrame.from_dict(func(url)[2], orient='index')
        df=df.rename(columns={0: "similarity"})
        df.similarity=func(url)[0][index][1:]
        df = df.sort_values('similarity', ascending = False)[1:10]
        return df.style.apply(lambda x : ['background:white'],axis=1)

   
    word = st.selectbox('Select a word', sorted(func(url)[1]))
    if st.button('submit'):
        table = func(url)[0]
        if word: 
            st.table(word_sim(word))
           
           


except InvalidArgumentException as ex:
    st.write('Oops URL?')



    

    

