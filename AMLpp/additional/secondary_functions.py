from typing import List
import numpy as np
import shelve
import time
import os 

##############################################################################
def most_frequency(x:List[List[int]]):
    x_ = [i[0] for i in x]
    return np.argmax(np.bincount(x_))

def lead_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result =  func(*args, **kwargs)
        print('lead time {} = {:.3f}'.
                format(func.__name__, time.time() - start_time))
        return result 
    return wrapper

def shelve_save(save_data:object, key:str, path:str = 'model_property/data'):
    if not os.path.exists("/".join(path.split('/'))):
        os.makedirs("/".join(path.split('/')))
    with shelve.open(path) as save:
        save[key] = save_data

def shelve_load(key:str, path:str = 'model_property/data'):
    with shelve.open(path) as load:
        return load[key]