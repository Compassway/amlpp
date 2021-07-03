from ast import literal_eval
from typing import List
from .querying import execute_cmd
import datetime

# 
def get_ids_cmd(cmd:str) -> List[int]:
    result = execute_cmd(cmd)
    return [i['id'] for i in result] if(result != []) else []

def get_application_ids(date_from: datetime.datetime, date_to: datetime.datetime,  limit:int = 20000) -> List[int]:
    result = execute_cmd("""
                            SELECT id FROM finplugs_pdl_applications
                            WHERE (applied_at between '{date_from}' AND '{date_to}') AND status_id IN (5,6,2) 
                            ORDER BY RAND() 
                            LIMIT {limit}
                         """.format(date_from=date_from, date_to=date_to, limit=limit))
    result = [dict(i) for i in result]
    return [i['id'] for i in result] if(result != []) else []

# Получение Анкеты по backend_application_id
def get_application_by_backend_id(backend_application_id: int) -> dict:
    result = execute_cmd("""
                            SELECT * FROM finplugs_pdl_applications
                            WHERE backend_application_id = {id} 
                         """.format(id=backend_application_id))
    result = [i for i in result]
    return dict(result[0]) if(result != []) else {}

# Получение Анкеты по id
def get_application_by_id(id: int) -> dict:
    result = execute_cmd("""
                            SELECT * FROM finplugs_pdl_applications
                            WHERE id = {id} 
                         """.format(id=id))
    result = [i for i in result]
    return dict(result[0]) if(result != []) else {}

# Получение истории по анкетам 
def get_application_history(user_id: int, historical_application_date: datetime.datetime) -> dict:
    result = execute_cmd("""
                            SELECT * FROM finplugs_pdl_applications
                            WHERE user_id = {user_id} AND applied_at <= '{applied_at}'
                         """.format(user_id=user_id, applied_at=historical_application_date))
    result = [dict(i) for i in result]
    return result if(result != []) else [{}]

# Получение убки
def get_ubki(social_number: int, application_id:int) -> dict:
    result = execute_cmd("""
                            SELECT response FROM finplugs_pdl_bki_calls
                            WHERE social_number = '{social_number}' AND application_id <= '{application_id}'
                            ORDER BY id DESC LIMIT 1
                         """.format(social_number=social_number, application_id=application_id))
    result = [i for i in result]
    return literal_eval(result[0]['response']) if(result != []) else {}