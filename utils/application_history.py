from typing import List, Generator


def values_in_list(application_history: List[dict], key: str) -> Generator:
    for application in application_history:
        yield application[key]


def avg(values_list: List[float or int] or None) -> float:
    try:
        return sum(values_list) / len(values_list)
    except:
        return 0


def get_useful_history_fields(application_history: List[dict]) -> dict:
    useful_fields = {
        'all_applications_count': None,
        'closed_credits_count': None,
        'rejected_applications_count': None,
        'mean_loans': None,
        'mean_overdue_days': None,
        'overdue_credits_count': None
    }
    try:
        if not application_history[0]:
            return useful_fields
        useful_fields['all_applications_count'] = \
            len(application_history)
        useful_fields['closed_credits_count'] = \
            len([status_id for status_id in values_in_list(application_history, 'status_id') if status_id == 5])
        useful_fields['rejected_applications_count'] = \
            len([status_id for status_id in values_in_list(application_history, 'status_id') if status_id == 2])
        useful_fields['mean_loans'] = \
            avg([loan_amount for loan_amount in values_in_list(application_history, 'loan_amount')])
        useful_fields['mean_overdue_days'] = \
            avg([overdue_days for overdue_days in values_in_list(application_history, 'overdue_days')])
        useful_fields['overdue_credits_count'] = \
            len([status_id for status_id in values_in_list(application_history, 'status_id') if status_id == 6])

        for useful_field in useful_fields:
            useful_fields[useful_field] = int(useful_fields[useful_field])

    finally:
        return useful_fields
