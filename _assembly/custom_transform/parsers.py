from ast import literal_eval

import pandas as pd
import numpy as np
import json

def pars_user_agent(user_agent:str):
    try:
        browser, system, brand = (np.nan, np.nan, np.nan)
        with open('model_property/user_agent.json', 'r') as load_file:
            info = pd.DataFrame(json.load(load_file))
        list_user_agent = info['useragent'].values

        if user_agent  in list_user_agent:
            system = info[info['useragent'] == user_agent]['system'].values[0].lower().split(' ')
            browser = system[0]
            system = 'windows' if system[2].find('win') != -1 else system[2]
            brand = 'apple' if (system == 'macos') else np.nan
        else:
            user_agent = user_agent[user_agent.index('(')+1:user_agent.index(')')].lower().split(';')

            system = {'windows':('windows', np.nan), 'x11':('linux',np.nan), 
                        'iphone':('iphone', 'apple'), 'ipad':('ipad', 'apple'), 'macintosh':('macos', 'apple')}
            for key in system.copy().keys():
                for col in user_agent:
                    if col.find(key) != -1:
                        system, brand = system[key]
                        return browser, system, brand 
            else:
                isandroid = len([i for i in user_agent if i.find('android') != -1]) > 0            
                if isandroid:
                    brand_phone = {'samsung':'samsung', 'xiaomi':'xiaomi', 'huawei':'huawei', 'lenovo':'lenovo',
                                    'motorola':'motorola', 'nokia':'nokia', 'sony':'sony', 'honor':'huawei', 
                                    'tecno':'tecno', 'asus':'asus', 'meizu':'meizu', 'vivo':'vivo', 'neffos':'neffos',
                                    'ulefone':'ulefone', 'htc ':'htc', 'pocophone':'poco', 'pixel':'google',
                                    'lg':'lg', 'sm':'samsung', 'redmi':'xiaomi', 'oneplus':'huawei', 'htc':'htc',
                                    'zte':'zte', 'mi':'xiaomi', 'm200':'xiaomi', 'cph':'oppo', 'moto':'motorola',
                                    'rmx':'realme', 'jsn':'huawei','-lx':'huawei', 'yal-':'huawei', 'eml-':'huawei',
                                    '-l21':'huawei', '-l29':'huawei', '-l22':'huawei', '-l31':'huawei','psp':'prestigio',
                                    '-l09':'huawei', '-l19':'huawei', 'pra-':'huawei', '-l41':'huawei', '-u29':'huawei', 
                                    'mz':'meizu', 'u10':'meizu', 'm5':'xiaomi','m6':'xiaomi', 'note':'xiaomi',
                                    }
                    system = 'android'
                    for key in brand_phone.keys():
                        for col in user_agent:
                            if col.find(key) != -1:
                                brand = brand_phone[key]
                                return browser, system, brand 
    finally:
         return [browser, system, brand]

def pars_detections(detections:str):
    country, region, city, isp = np.nan, np.nan, np.nan, np.nan
    try:
        detections = literal_eval(detections)['geo']
        isp = detections['isp']
        country = detections['country']
        city = detections['city']
        region = int(detections['region'])
    finally:
        return [country, city, region, isp]