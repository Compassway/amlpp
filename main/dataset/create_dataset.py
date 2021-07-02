from ast import literal_eval
import threading
from Work.SAAS.utils import ubki, application_history
from Work.SAAS.sql import methods
import datetime
import pandas as pd
import time
import os.path


class save_worker(threading.Thread):
    def __init__(self, date_from, date_to, limit, file_name, counter):
        threading.Thread.__init__(self)

        self.date_from = date_from
        self.date_to = date_to
        self.limit = limit

        self.file_name = file_name
        self.counter = counter

        self.result = list(methods.get_application_and_ubki(self.date_from, self.date_to, self.limit))
        self.save_data = pd.DataFrame()

    def run(self):
        social_number = -1
        iter = 0
        for row in self.result:
            if social_number != row['social_number']:
                self.add_row_data(row)
                iter += 1
            if iter % self.counter == 0:
                self.save_file()
            social_number = row['social_number']
        self.save_file()

    def add_row_data(self, row):
        user = dict(row)
        try:
            ubki_data = ubki.get_useful_ubki_fields(literal_eval(str(user['response'])), user['phone_mobile'],
                                                    user['email'], user['app_at'])
        except:
            ubki_data = ubki.get_useful_ubki_fields(None, None, None, None)
        try:
            hist_data = application_history.get_useful_history_fields(
                [hist for hist in self.result if (hist['social_number'] == row['social_number'])])
        except:
            hist_data = application_history.get_useful_history_fields(None)
        del user['response']
        self.save_data = self.save_data.append({**user, **ubki_data, **hist_data}, ignore_index=True)

    def save_file(self):
        if os.path.exists(self.file_name):
            self.save_data = self.save_data.append(pd.read_excel(self.file_name))
        self.save_data.to_excel(self.file_name, index=False)
        self.save_data = pd.DataFrame()


def create_dataset_worker(DATE_FROM: datetime.datetime, DATE_TO: datetime.datetime, LIMIT: int, worker: int = 1,
                          count: int = 50):
    threading = []
    for i in range(worker):
        threading.append(save_worker(DATE_FROM, DATE_TO, int(LIMIT / worker), "file" + str(i) + ".xlsx", count))
        threading[i].start()

    for thr in threading:
        thr.join()


def create_dataset(DATE_FROM: datetime.datetime, DATE_TO: datetime.datetime, LIMIT: int, count: int = 50,
                   file_name: str = "dataset.xlsx"):
    current_social_number, iter, full_data = (-1, 0, pd.DataFrame())
    result = list(methods.get_application_and_ubki(DATE_FROM, DATE_TO, LIMIT))
    for row in result:
        if current_social_number != row['social_number']:
            user = dict(row)
            try:
                ubki_data = ubki.get_useful_ubki_fields(literal_eval(str(user['response'])), user['phone_mobile'],
                                                        user['email'], user['app_at'])
            except:
                ubki_data = ubki.get_useful_ubki_fields(None, None, None, None)

            try:
                hist_data = application_history.get_useful_history_fields(
                    [hist for hist in result if (hist['social_number'] == row['social_number'])])
            except:
                hist_data = application_history.get_useful_history_fields(None)
            del user['response']
            full_data = full_data.append({**user, **ubki_data, **hist_data}, ignore_index=True)
            if iter % count == 0:
                full_data = save_file(full_data, file_name)
        current_social_number = row['social_number']

    save_file(full_data, file_name)


def save_file(save_data: pd.DataFrame, file_name: str) -> pd.DataFrame:
    if os.path.exists(file_name):
        save_data = save_data.append(pd.read_excel(file_name))
    save_data.to_excel(file_name, index=False)
    return pd.DataFrame()


if __name__ == '__main__':
    start_time = time.time()
    print(datetime.datetime.now().strftime("%d-%m-%Y %H:%M"))
    DATE_FROM = datetime.datetime(year=2020, month=1, day=1)
    DATE_TO = datetime.datetime.now()
    create_dataset_worker(DATE_FROM, DATE_TO, 8000, 2, 1000)
    print("--- %s seconds ---" % (time.time() - start_time))
