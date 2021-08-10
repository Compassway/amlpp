from typing import List
from .parsers import pars_detections, pars_user_agent

import pandas as pd
import numpy as np
import datetime

# Классическая версия трансформера, с user_agent и detections
class TransformClassicExt():
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
                         'detections', 'user_agent']
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

    def target_transform(self, Y:pd.DataFrame) -> pd.DataFrame:
        Y = Y[['overdue_days', 'status_id']]
        Y['overdue_days'] = Y['overdue_days'].fillna(0)
        Y['overdue_days'].loc[Y['overdue_days'] == 0] = 0
        Y['overdue_days'].loc[Y['overdue_days'] > 0] = 1
        Y['overdue_days'].loc[Y['status_id'] == 2] = 1
        return Y['overdue_days'].replace({0: 1, 1: 0})

# Версия трансформера 
# для нового убки (добавленны req_credit, quantity_rejection, max_cdolgn, max_wdohod, max_wstag)
# + user_agent и detections
class TransformNewubkiExt():
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
                         'detections', 'user_agent', "req_credit" , 'max_cdolgn', 'max_wdohod', 'max_wstag']

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

    def target_transform(self, Y:pd.DataFrame) -> pd.DataFrame:
        Y = Y[['overdue_days', 'status_id']]
        Y['overdue_days'] = Y['overdue_days'].fillna(0)
        Y['overdue_days'].loc[Y['overdue_days'] == 0] = 0
        Y['overdue_days'].loc[Y['overdue_days'] > 0] = 1
        Y['overdue_days'].loc[Y['status_id'] == 2] = 1
        return Y['overdue_days'].replace({0: 1, 1: 0})

# Версия трансформера 
# для анкет, c регрессором основанным на дате закрытия кредита 
# 1 - клиент закрылся вовремя (closed_at <= applied_at + loan_days)
# 0.8 - клиент закрылся в течении 90 дней (closed_at <= applied_at + loan_days + 90)
# 0.5 - клиент закрылся позже 90 дней (closed_at > applied_at_ + loan_days + 90)
# 0 - клиент в статусе 6
class TransformApp90():
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        return self
        
    def transform(self, X:pd.DataFrame, y = None) -> pd.DataFrame:
        leave_columns = ['loan_amount', 'loan_days', 'gender_id', 'marital_status_id', 'children_count_id', 'education_id', 'addr_region_id', 'addr_owner_type_id', 
                 'fact_addr_same', 'fact_addr_region_id', 'fact_addr_owner_type_id', 'has_immovables', 'has_movables', 'employment_type_id', 'position_id',
                 'organization_type_id', 'organization_branch_id', 'empoyees_count_id', 'seniority_years', 'has_prior_employment', 'monthly_income',
                 'income_frequency_id', 'income_source_id', 'monthly_expenses', 'other_loans_about_current', 'other_loans_about_monthly',
                 'product_dpr', 'product_amount_from', 'product_amount_to', 'product_overdue_dpr', 'product_interest_min', 'applied_at', 'purpose_other',
                 'birth_date', 'passport_date', 'email', 'position_other', 'organization_type_other']
        X = X[leave_columns]

        X = X.replace('[]', '', regex=False)
        X['email'] = X['email'].str.split('@', expand=True)[1]

        X['passport_year'] = pd.to_datetime(X['passport_date'], format='%Y-%m-%d', errors='coerce').dt.year
        del X['passport_date']

        X['birth_year'] = pd.to_datetime(X['birth_date'], format='%Y-%m-%d', errors='coerce').dt.year
        del X['birth_date']

        X['applied_at'] = pd.to_datetime(X['applied_at'], format='%Y-%m-%d %H', errors='coerce')
        X['applied_day'] = X['applied_at'].dt.day
        X['applied_weekday'] = X['applied_at'].dt.weekday
        X['applied_hour'] = X['applied_at'].dt.hour
        del X['applied_at']

        return X

    def target_transform(self, Y:pd.DataFrame) -> pd.DataFrame:
        Y['target'] = Y['loan_days']

        Y['closed_at'] = pd.to_datetime(Y['closed_at'], format='%Y-%m-%d', errors='coerce')
        Y['applied_at'] = pd.to_datetime(Y['applied_at'], format='%Y-%m-%d', errors='coerce')

        for iter, val in Y.iterrows():
            if val['closed_at'] <= val['applied_at'] + datetime.timedelta(days = val['loan_days']):
                Y.loc[iter, 'target'] = 1
            elif val['closed_at'] <= val['applied_at'] + datetime.timedelta(days = val['loan_days'] + 90):
                Y.loc[iter, 'target'] = 0.8
            elif val['closed_at'] > val['applied_at'] + datetime.timedelta(days = val['loan_days'] + 90):
                Y.loc[iter, 'target'] = 0.5
            elif val['status_id'] == 6:
                Y.loc[iter, 'target'] = 0
            print(f"iter ={iter}, {Y.loc[iter, 'target']}")

        return Y['target']


class TransformApps15():
    def __init__(self):
        pass

    def fit(self, X:pd.DataFrame, Y:pd.DataFrame or pd.Series):
        return self
        
    def transform(self, X:pd.DataFrame, y = None) -> pd.DataFrame:
        leave_columns = ['loan_amount', 'loan_days', 'gender_id', 'marital_status_id', 'children_count_id', 'education_id', 'addr_region_id', 'addr_owner_type_id', 
                 'fact_addr_same', 'fact_addr_region_id', 'fact_addr_owner_type_id', 'has_immovables', 'has_movables', 'employment_type_id', 'position_id',
                 'organization_type_id', 'organization_branch_id', 'empoyees_count_id', 'seniority_years', 'has_prior_employment', 'monthly_income',
                 'income_frequency_id', 'income_source_id', 'monthly_expenses', 'other_loans_about_current', 'other_loans_about_monthly',
                 'product_dpr', 'product_amount_from', 'product_amount_to', 'product_overdue_dpr', 'product_interest_min', 'applied_at', 'purpose_other',
                 'birth_date', 'passport_date', 'email', 'position_other', 'organization_type_other']
        X = X[leave_columns]

        X = X.replace('[]', '', regex=False)
        X['email'] = X['email'].str.split('@', expand=True)[1]

        X['passport_year'] = pd.to_datetime(X['passport_date'], format='%Y-%m-%d', errors='coerce').dt.year
        del X['passport_date']

        X['birth_year'] = pd.to_datetime(X['birth_date'], format='%Y-%m-%d', errors='coerce').dt.year
        del X['birth_date']

        X['applied_at'] = pd.to_datetime(X['applied_at'], format='%Y-%m-%d %H', errors='coerce')
        X['applied_day'] = X['applied_at'].dt.day
        X['applied_weekday'] = X['applied_at'].dt.weekday
        X['applied_hour'] = X['applied_at'].dt.hour
        del X['applied_at']

        return X

    def target_transform(self, Y:pd.DataFrame) -> pd.DataFrame:
        Y['target'] = Y['status_id']
        Y['target'].loc[Y['status_id'] == 1] = 0
        Y['target'].loc[Y['status_id'] == 5] = 1
        return Y['target']