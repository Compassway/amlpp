from sqlalchemy import create_engine
from . import sql_config


def execute_cmd(execute:str):
    engine = create_engine('mysql://%s:%s@%s:%s/%s?charset=utf8' % (
        sql_config['decision']['user'],
        sql_config['decision']['password'],
        sql_config['decision']['host'],
        sql_config['decision']['port'],
        sql_config['decision']['catalog']
    ))
    with engine.connect() as connection:
        return connection.execute(execute)