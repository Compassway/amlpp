from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import IterativeImputer as IterativeImputer
from sklearn.impute import SimpleImputer as SimpleImputer
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk

from gensim.models import Word2Vec
from AMLpp.additional.secondary_functions import shelve_save, shelve_load, most_frequency
from typing import List, Callable
import pandas as pd
import numpy as np
import pymorphy2

nltk.download('stopwords')
nltk.download('punkt')

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

    def __init__(self, columns:List[str], fill_strategy:str='mean', fill_value:float or str = np.nan):
        self.columns = columns
        self.user_value = fill_value
        self.fill_value = {'mean':np.mean, 'median':np.median, 'most_freq':most_frequency, 'const':self.return_fill_value}
        self.fill_value = self.fill_value[fill_strategy]

    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        for column in self.columns:
            if column in X.columns:
                self.encoder[column] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
                X_fit = pd.DataFrame(X[column].loc[~X[column].isnull()])
                if len(X_fit) > 0:
                    self.encoder[column].fit(X_fit)
                    X_transform = self.encoder[column].transform(pd.DataFrame(X_fit))
                    self.encoder[column].unknown_value = self.fill_value(X_transform)
                else:
                    self.encoder[column] = False
        shelve_save(self.encoder, 'CategoricalEncoder')
        return self

    def transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series = None):
        self.encoder = shelve_load('CategoricalEncoder')
        for column in self.columns:
            if column in X.columns:
                if self.encoder[column]:
                    X[column] = self.encoder[column].transform(pd.DataFrame(X[column].fillna('NAN')))
                else:
                    del X[column]
        return X
    def return_fill_value(self, trash):
        return self.user_value

##############################################################################

class Word2Vectorization():

    fill = {} 
    word2 = {} 

    def __init__(self, columns:List[str], level_formatting:int = 1, fill_strategy:str='mean', fill_value:float or str = np.nan, **params_for_word2vec):
        self.columns = columns
        self.level_formatting = level_formatting
        self.fill_value = {'mean':np.mean, 'median':np.median, 'most_freq':most_frequency, 'const':(lambda x:fill_value)}
        self.fill_value = self.fill_value[fill_strategy]
        self.params_for_word2vec = params_for_word2vec if params_for_word2vec != {} else \
                {'epochs':5000, 'min_count':1, 'window':5, 'vector_size':1, 'sg':1, 'cbow_mean':1, 'alpha':0.1}

    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        for column in self.columns: 
            dictionary = [self.refactor_string(str(i)) for i in X[column]]
            self.word2[column] = Word2Vec(**self.params_for_word2vec, seed = 42)

            word_vac = [self.word2[column].wv[i] for i in self.word2[column].wv.key_to_index.values()]
            self.fill[column] = self.fill_value(word_vac)
        shelve_save(self.fill, 'fill_word2vec')
        shelve_save(self.word2, 'word2')
        return self
    def transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series = None):
        self.fill = shelve_load('fill_word2vec')
        self.word2 = shelve_load('word2')
        for column in self.columns:
            X[column] = list([self.mean_word2vec(val, column) for val in X[column]])
        return X

    def refactor_string(self, string:str)->List[str]:
        string = string if str(string) != 'nan' else ""                          # Проверка на NAN
        string = word_tokenize(str(string).lower())                              # Нижний регистр и токенизация
        if self.level_formatting > 0:
            string = [i for i in string if i.isalpha()]                              # Избавления от знаков пунктуации
            if self.level_formatting > 1:
                string = [i for i in string if not i in stop_words]                      # Избавления от стоп слов
                if self.level_formatting > 2:
                    string = [snowball.stem(morph.parse(i)[0].normal_form) for i in string]  # СТЭММИНГ и ЛЕММАТИЗАЦИЯ
        return string

    def mean_word2vec(self, sentence:str, column:str) ->List[float]:
        vector = self.refactor_string(sentence)
        vector = [self.word2[column].wv[token] for token in vector if token in self.word2[column].wv.key_to_index.keys()]
        return np.mean(vector) if vector != [] else self.fill[column]


##############################################################################

class ImputerValue():
    current_columns = None
    def __init__(self, columns:List[str] or None = None,  missing_values=np.nan, strategy='mean'):
        self.columns = columns 
        self.imputer = SimpleImputer(missing_values=missing_values, strategy=strategy)


    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        self.current_columns = self.columns if self.columns != None else X.columns
        self.imputer.fit(X[self.current_columns])
        shelve_save(self.imputer, 'SimpleImputer')
        return self

    def transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series = None):
        self.imputer = shelve_load('SimpleImputer')
        X_transform = pd.DataFrame(self.imputer.transform(X[self.current_columns]), columns = self.current_columns)
        for column in self.current_columns:
            X[column] = X_transform[column].values
        return X

class ImputerIterative():
    current_columns = None
    def __init__(self, columns:List[str] or None = None):
        self.columns = columns
        self.imputer = IterativeImputer()

    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        for column in X:
            if len(X.loc[X[column].isnull()]) == len(X):
                X[column] = [0 for i in range(len(X))]
        self.current_columns = self.columns if self.columns != None else X.columns
        self.imputer.fit(X[self.current_columns])
        shelve_save(self.imputer, 'IterativeImputer')
        return self

    def transform(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series = None):
        self.imputer = shelve_load('IterativeImputer')
        X_transform = pd.DataFrame(self.imputer.transform(X[self.current_columns]), columns = self.current_columns)
        for column in self.current_columns:
            X[column] = X_transform[column].values
        return X