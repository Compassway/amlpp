
import pandas as pd
import numpy as np
from ast import literal_eval
from typing import List
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

class UserTransform_status56():
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        return self
        
    def transform(self, X:pd.DataFrame, y = None) -> pd.DataFrame:
        leave_columns = ['loan_amount', 'loan_days', 'gender_id', 'marital_status_id', 'children_count_id', 'education_id', 'addr_region_id',
                         'addr_owner_type_id', 'fact_addr_same', 'fact_addr_region_id', 'fact_addr_owner_type_id', 'has_immovables', 'has_movables',
                         'employment_type_id', 'position_id', 'organization_type_id', 'organization_branch_id', 'empoyees_count_id', 'seniority_years',
                         'has_prior_employment', 'monthly_income', 'income_frequency_id', 'income_source_id', 'monthly_expenses', 'other_loans_about_current', 
                         'other_loans_about_monthly', 'product_dpr', 'product_amount_from', 'product_amount_to', 'product_overdue_dpr', 'product_interest_min', 
                         'median_day_credit',	'mean_credit_summ',	'mean_credit_debt', 'last_cdolgn', 'last_wdohod', 'last_wstag', 'cgrag', 'sstate', 'family', 
                         'ceduc', 'ubki_balance_value', 'ubki_score', 'ubki_scorelast', 'ubki_scorelevel', 'ubki_all_credits', 'ubki_open_credits', 
                         'ubki_closed_credits', 'ubki_expyear', 'ubki_maxnowexp', 'ubki_phone_deltatime', 'ubki_email_deltatime', 'ubki_week_queries',
                         'rejected_applications_count', 'mean_loans', 'applied_at', 'purpose_other', 'birth_date', 'passport_date', 'email', 'position_other', 
                         'detections', 'user_agent' ]

        X = X[leave_columns]

        X = X.replace('[]', np.nan, regex=False)
        X['email'] = X['email'].str.split('@', expand=True)[1]

        X['passport_year'] = pd.to_datetime(X['passport_date'], format='%Y-%m-%d', errors='coerce').dt.year
        X['birth_year'] = pd.to_datetime(X['birth_date'], format='%Y-%m-%d', errors='coerce').dt.year

        X['applied_at'] = pd.to_datetime(X['applied_at'], format='%Y-%m-%d %H', errors='coerce')
        X['applied_day'] = X['applied_at'].dt.day
        X['applied_weekday'] = X['applied_at'].dt.weekday
        X['applied_hour'] = X['applied_at'].dt.hour
        
        X = X.drop(['passport_date', 'birth_date', 'applied_at'], axis = 1)

        X[['country_det', 'city_det', 'region_det', 'isp']] = [pars_detections(val) for val in X['detections']]
        X[['browser', 'system', 'brand']] = [pars_user_agent(val) for val in X['user_agent']]
        
        X = X.drop(['detections', 'user_agent'], axis = 1)
        return X

    def target_transform(self, Y:pd.DataFrame) -> pd.DataFrame or pd.Series or List[float or int]:
        Y = Y[['overdue_days', 'status_id']]
        Y['overdue_days'] = Y['overdue_days'].fillna(0)
        Y['overdue_days'].loc[Y['overdue_days'] == 0] = 0
        Y['overdue_days'].loc[Y['overdue_days'] > 0] = 1
        Y['overdue_days'].loc[Y['status_id'] == 2] = 1
        return Y['overdue_days'].replace({0: 1, 1: 0})