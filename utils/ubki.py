import datetime as dt
import numpy as np
import datetime


def format_date(attribute: str) -> datetime:
    return dt.datetime.strptime(attribute, "%Y-%m-%d")


def last_date_index(date_list: list):
    return np.argmax([format_date(d['@attributes']['vdate']) for d in date_list])


def check_null_value(value: str) -> bool:
    return value != '' and value != "NA" and value != [] and value != "null"


def coding_no_yes(arg: str) -> int:
    return 0 if (arg.lower() == "нет") else 1


def coding_maxnowexp(arg: str) -> int:
    return 0 if (arg.lower() == "нет") else arg.split(" ")[0]


def get_useful_ubki_fields(ubki: dict, phone: str, email: str, our_date: datetime) -> dict:
    # Словарь регрессоров которые которые определенны как полезные
    res_dict = {
        k: None
        for k in [
            "median_day_credit",  # Медианна по дате начала соглашения
            "mean_credit_summ",  # Средняя сумма по кредитам
            "mean_credit_debt",  # Средння задолжность по кредитам
            "last_cdolgn",  # Последяя зафиксированая должность на работе
            "last_wdohod",  # Последяя зафиксированый доход на работе
            "last_wstag",  # Послдний зафиксированый стаж на работе
            "cgrag",  # Код странны       (категориальный)
            "sstate",  # Социальный статус (категориальный)
            "family",  # Симейный статус   (категориальный)
            "ceduc",  # Образование       (категориальный)
            "ubki_balance_value",  # Текущий баланс
            "ubki_score",  # УБКИ очки
            "ubki_scorelast",  # УБКИ последние очки
            "ubki_scorelevel",  # УБКИ уровень очков
            "ubki_all_credits",  # Все кредиты
            "ubki_open_credits",  # Открытые кредиты
            "ubki_closed_credits",  # Закрытые кредиты
            "ubki_expyear",  # Просрочка по кредитам
            "ubki_maxnowexp",  #
            "ubki_phone_deltatime",  # Разница между первым упоминанием данного телефена и текущей датой
            "ubki_email_deltatime",  # Разница между первым упоминанием данного почты и текущей датой
            "ubki_week_queries"  # Количество запросов в неделю
        ]
    }
    if not ubki:
        return res_dict
    try:
        big_datetime = dt.datetime(dt.datetime.today().year + 1, 1, 1)
        ubki_response = ubki
        if "tech" in ubki_response.keys() and "billing" in ubki_response['tech'].keys():  # Системные данные
            res_dict["ubki_balance_value"] = int(
                float(ubki_response['tech']["billing"]["balance"]["@attributes"]["value"]))

        # Персональные данные
        if "comp" in ubki_response.keys():
            for comp in ubki_response["comp"]:
                comp_id = comp["@attributes"]["id"]
                # Блок персональных данных
                if comp_id == '2':
                    try:
                        ubki_crdeal = comp['crdeal']
                        credit_summ = quantity_credit = 0  # Сумма кредита и количество кредитов
                        credit_debt_cur = credit_debt_exp = 0  # Сумма задолжности по кредиту
                        median_days = [0, 0]
                        if not "@attributes" in ubki_crdeal:  # Проверка на количество кредитов
                            for cr in ubki_crdeal:
                                cr_summ, cr_debt, med_days = get_credit_ubki_fields(cr)
                                if cr_summ > 0:
                                    credit_summ += cr_summ
                                    quantity_credit += 1
                                if cr_debt > 0:
                                    credit_debt_cur += cr_debt
                                    credit_debt_exp += 1
                                median_days[0] += med_days[0]
                                median_days[1] += med_days[1]
                        else:  # Только один кредит
                            credit_summ, credit_debt_cur, median_days = get_credit_ubki_fields(ubki_crdeal)
                            quantity_credit += 1 if (credit_summ > 0) else 0
                            credit_debt_exp += 1 if (credit_debt_cur > 0) else 0
                        res_dict['median_day_credit'] = int(np.argmax(median_days))
                        res_dict['mean_credit_summ'] = int(credit_summ / quantity_credit)
                        res_dict['mean_credit_debt'] = int(credit_debt_cur / credit_debt_exp)
                    except:
                        pass
                elif comp_id == '1':

                    # Данные про субьект
                    try:
                        comp_ident = comp['cki']['ident']
                        attributes = comp_ident[last_date_index(comp_ident)]['@attributes']
                        for i in ["cgrag", 'sstate', 'family', 'ceduc']:
                            res_dict[i] = int(attributes[i]) if () else None
                    except:
                        pass
                    # Данные про роботу (учитывается последняя)
                    try:
                        comp_work = comp['cki']['work']
                        attributes = (comp_work[last_date_index(comp_work)]['@attributes'] if (len(comp_work) > 1)
                                      else comp_work['@attributes'])

                        for i in ['cdolgn', 'wdohod', 'wstag']:
                            res_dict["last_" + i] = int(float(attributes[i])) \
                                if (check_null_value(attributes[i])) else None
                    except:
                        pass

                # Кредитный рейтинг УБКИ
                elif comp_id == "8":
                    try:
                        ubki_rating = comp["urating"]

                        rating_attributes = ubki_rating["@attributes"]
                        for i in ["score", "scorelast", "scorelevel"]:
                            res_dict["ubki_" + i] = int(float(rating_attributes[i])) \
                                if (check_null_value(rating_attributes[i])) else None

                        dinfo_attributes = ubki_rating["dinfo"]["@attributes"]

                        res_dict["ubki_all_credits"] = int(float(dinfo_attributes["all"])) \
                            if (check_null_value(rating_attributes[i])) else None
                        res_dict["ubki_open_credits"] = int(float(dinfo_attributes["open"])) \
                            if (check_null_value(dinfo_attributes["open"])) else None
                        res_dict["ubki_closed_credits"] = int(float(dinfo_attributes["close"])) \
                            if (check_null_value(dinfo_attributes["close"])) else None
                        res_dict["ubki_expyear"] = int(coding_no_yes(dinfo_attributes["expyear"])) \
                            if (check_null_value(dinfo_attributes["expyear"])) else None
                        res_dict["ubki_maxnowexp"] = int(float(coding_maxnowexp(dinfo_attributes["maxnowexp"]))) \
                            if (check_null_value(dinfo_attributes["maxnowexp"])) else None
                    except:
                        pass

                # Блок регестрации запросов
                if comp_id == "4":
                    try:
                        res_dict["ubki_week_queries"] = int(
                            comp["reestrtime"]["@attributes"]["wk"])  # Количество запросов в неделю
                    except:
                        pass

                # Блок истории своих контактных данных
                if comp_id == "10":
                    try:
                        first_datetime_phone = first_datetime_email = big_datetime
                        contacts = comp["cont"]
                        for contact in contacts:  # Проход по контактам
                            attributes = contact["@attributes"]
                            cval = attributes["cval"]
                            vdate = attributes["vdate"]

                            # Нахождения самого раннего упоминания телефона
                            if cval[1:] == str(phone):
                                if format_date(vdate) < first_datetime_phone:
                                    first_datetime_phone = format_date(vdate)

                            # Нахождения самого раннего упоминания електронной почты
                            elif str(cval.lower()) == str(email):
                                if format_date(vdate) < first_datetime_email:
                                    first_datetime_email = format_date(vdate)

                        res_dict["ubki_phone_deltatime"] = int((our_date - first_datetime_phone).days) if (
                                first_datetime_phone != big_datetime) else None
                        res_dict["ubki_email_deltatime"] = int((our_date - first_datetime_email).days) if (
                                first_datetime_phone != big_datetime) else None
                    except:
                        pass
    except:
        pass
    finally:
        return res_dict


def get_credit_ubki_fields(cr: dict):
    credit_summ = credit_debt_cur = 0
    median_days = [0, 0]
    credit_value = float(cr["@attributes"]['dlamt'])
    if credit_value > 0:  # Проверка на наличие суммы кредита
        credit_summ += credit_value

    # Медианна дней начала кредитного соглашения
    cr_deallife = cr['deallife']
    credit_date = (format_date(cr_deallife[0]['@attributes']['dlds'])
                   if (len(cr_deallife) > 1) else format_date(cr_deallife['@attributes']['dlds']))
    median_days[int(credit_date.day > 15)] += 1

    # Нахождение максимального значения по задолжности кредита
    if len(cr_deallife) > 1:
        max_debt = 0
        for dl in cr_deallife:  # Проходка по кредитной истории конкретного кредитного соглашения
            dlamtcur = float(dl['@attributes']['dlamtcur'])
            if dlamtcur > max_debt:
                max_debt = dlamtcur
        if max_debt > 0:  # проверка на задолжности
            credit_debt_cur += max_debt
    else:
        dlamtcur = float(cr_deallife['@attributes']['dlamtcur'])
        if dlamtcur > 0:
            credit_debt_cur += dlamtcur
    return credit_summ, credit_debt_cur, median_days
