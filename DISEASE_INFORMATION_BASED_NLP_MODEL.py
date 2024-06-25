#!/usr/bin/env python
# coding: utf-8

# In[ ]:


cd Desktop/NLP_MODEL_5


# In[ ]:


#Importing important libraries related to NLP.

import pandas as pd
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


#Reading the text file containing medical question and answer pairs in order to train the model.

df = pd.read_csv('Question_Answer.txt', sep='\t')


# In[ ]:


df.head(10)


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


print('original df length: ',len(df))
df.dropna(subset=['question'], inplace=True)
df.dropna(subset=['answer'], inplace=True)
df.dropna(subset=['focus_area'], inplace=True)
df.dropna(subset=['source'])
df = df[~df.question.str.contains('#')] # remove badly formatted questions
df = df[~df.answer.isin(['no','yes','Yes','No','No,','Yes,','No.','Yes.','yes.','no.'])] # remove yes/no questions
print('new df length: ',len(df))


# In[ ]:


df = df.drop_duplicates( subset='question' )
df.head(10)


# In[ ]:


df['question'] = df['focus_area'] + ' ' + df['question']
df.head()


# In[ ]:


df01 = df[['question', 'answer']]
df01.head()


# In[ ]:


df01.shape


# In[ ]:


print( type( tuple( df['question'] ) ) )


# In[ ]:


stopwords_list = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def my_tokenizer(doc):
    words = word_tokenize(doc)
    pos_tags = pos_tag(words)
    non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list]
    non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation]
    
    lemmas = []
    for w in non_punctuation:
        if w[1].startswith('J'):
            pos = wordnet.ADJ
        elif w[1].startswith('V'):
            pos = wordnet.VERB
        elif w[1].startswith('N'):
            pos = wordnet.NOUN
        elif w[1].startswith('R'):
            pos = wordnet.ADV
        else:
            pos = wordnet.NOUN
        
        lemmas.append(lemmatizer.lemmatize(w[0], pos))
        
    return lemmas


# In[ ]:


tfidf_vectorizer = TfidfVectorizer(tokenizer=my_tokenizer)
#tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(tuple(df01['question']))
print(tfidf_matrix.shape)


# In[ ]:


def question_answer(question):
    
    query_vect = tfidf_vectorizer.transform([question])
    similarity = cosine_similarity(query_vect, tfidf_matrix)
    similarity_max = np.argmax(similarity)
    
    print('> Answer:\n- {}'.format(df01.iloc[similarity_max]['answer']))


# In[ ]:


question_user = input('Type your question here: ')
question_answer(question_user)

