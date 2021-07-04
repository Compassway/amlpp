from sklearn.preprocessing import OrdinalEncoder
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

from gensim.models import Word2Vec
from AMLpp.additional.secondary_functions import *
from typing import List, Callable
import pandas as pd
import numpy as np
import pymorphy2

snowball = SnowballStemmer(language="russian")
morph = pymorphy2.MorphAnalyzer()
stop_words = stopwords.words("russian")

##############################################################################

class CategoricalEncoder():
    """ Класс кодирования категориальных данных, с заполнение пропусков на некоторое значение определенное сратегией

    Parameters
    ----------
    columns : List[str]
        Названия столбцов, которые будут подвегнуты обработке

    straegy : str
        Строка указывающая на используемую стратегию заполнения пропусков
    
    fill_value : float or str
        Заполнитель, которым будут заполняться пропущенные значения,
        при использование стратегии const
        
    """
    encoder = {}

    def __init__(self, columns:List[str], strategy:str='mean', fill_value:float or str = np.nan): # strategy in mean, median, most_frequency, const, iterative inputer?
        self.columns = columns
        self.fill_value = {'mean':np.mean, 'median':np.median, 'most_freq':most_frequency, 'const':(lambda x:fill_value)}
        self.fill_value = self.fill_value[strategy]

    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        for column in self.columns:
            self.encoder[column] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
            X_fit = pd.DataFrame(X[column].loc[~X[column].isnull()])
            self.encoder[column].fit(X_fit)
            X_transform = self.encoder[column].transform(pd.DataFrame(X_fit))
            self.encoder[column].unknown_value = self.fill_value(X_transform)
        shelve_save(self.encoder, 'CategoricalEncoder')
        return self

    def transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series = None):
        self.encoder = shelve_load('CategoricalEncoder')
        for column in self.columns:
            X[column] = self.encoder[column].transform(pd.DataFrame(X[column].fillna('NAN')))
        return X

# class Word2Vectorization():

#     len_sentence = {} 
#     mean_word = {} 
#     word2 = {} 

#     def __init__(self, columns:List[str], level_formatting:int = 0):
#         self.columns = columns

#     def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
#         for column in self.columns: 
#             filtered = [self.refactor_string(str(i)) for i in X[column]]
#             self.word2[column] = Word2Vec(sentences=filtered, epochs=5000, 
#                                     min_count=1, window=5, vector_size=1,
#                                     sg=1, cbow_mean=1, alpha=0.1,
#                                     seed=self.seed)

#             word_vac = [self.word2[column].wv[i] for i in self.word2[column].wv.key_to_index.values()]
#             self.len_sentence[column] = max([len(sentence) for sentence in filtered]) 
#             self.mean_word[column] = np.mean(word_vac)                              
#         return self
#     def transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series = None):
        
#         return X

#     def refactor_string(self, string:str)->List[str]:
#         string = string if str(string) != 'nan' else ""                          # Проверка на NAN
#         string = word_tokenize(str(string).lower())                              # Нижний регистр и токенизация
#         if self.level_formatting > 0:
#             string = [i for i in string if i.isalpha()]                              # Избавления от знаков пунктуации
#             if self.level_formatting > 1:
#                 string = [i for i in string if not i in stop_words]                      # Избавления от стоп слов
#                 if self.level_formatting > 2:
#                     string = [snowball.stem(morph.parse(i)[0].normal_form) for i in string]  # СТЭММИНГ и ЛЕММАТИЗАЦИЯ
#         return string

#     def mean_word2vec(self, sentence:str, column:str) ->List[float]:
#         vector = self.refactor_string(sentence)
#         vector = [self.word2[token] for token in vector if token in self.word2.wv.key_to_index.keys()]
#         return np.mean(vector) if vector != [] else 0
# ################################################################################################
#     ## Выборка слов из датасета, и подача их на вход 
#     def fit(self, X:pd.DataFrame, y:pd.Series or List[float]):
#         for column in self.columns: 
#             filtered = [self.refactor_string(str(i)) for i in X[column]]
#             self.word2[column] = Word2Vec(sentences=filtered, epochs=5000, 
#                                     min_count=1, window=5, vector_size=1,
#                                     sg=1, cbow_mean=1, alpha=0.1,
#                                     seed=self.seed)

#             word_vac = [self.word2[column].wv[i] for i in self.word2[column].wv.key_to_index.values()]
#             self.len_sentence[column] = max([len(sentence) for sentence in filtered]) 
#             self.mean_word[column] = np.mean(word_vac)                                  

#         if not os.path.exists('model_new_property'):
#             os.makedirs('model_new_property')
#         with open('model_new_property/len_sentence', 'wb') as file:
#             pickle.dump(self.len_sentence, file)
#         with open('model_new_property/mean_word', 'wb') as file:
#             pickle.dump(self.mean_word, file)
#         with open('model_new_property/word2', 'wb') as file:
#             pickle.dump(self.word2, file)

#         return self

#     def transform(self, X:pd.DataFrame, y = None)->pd.Series:
#         with open('model_new_property/len_sentence', 'rb') as file:
#             self.len_sentence = pickle.load(file)
#         with open('model_new_property/mean_word', 'rb') as file:
#             self.mean_word = pickle.load(file)
#         with open('model_new_property/word2', 'rb') as file:
#             self.word2 = pickle.load(file)

#         for column in self.columns:
#             X[column] = list([self.get_vector_sentence(val, column) for val in X[column]])
#         return X

# class Imputer():