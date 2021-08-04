from pandas import DataFrame
import pandas as pd
import numpy as np

def interval_index(value:float, left_bound:float, interval:float, right_bound:float) -> int:
    for bound in np.arange(left_bound, right_bound + interval,  interval):
        if value < bound:
            return f"{round(bound - interval,2)} - {round(bound,2)}"

def get_scoring_table_statistic(df:DataFrame, 
                                left_bound:float = 0, 
                                interval:float = 0.5, 
                                right_bound:float = 1.05) -> DataFrame:

    df['intervals_values'] = df['result'].apply(lambda x: interval_index(x, left_bound, interval, right_bound))

    index = np.sort(df['intervals_values'].value_counts().index)
    statistic_df = DataFrame({key:index  for key in df['status_id'].value_counts().index}, index = index)
    statistic_df['count'] = index

    for interval in index:
        for status in [2, 5, 6]:
            count = len(df.loc[(df['intervals_values'] == interval) & (df['status_id'] == status)])
            statistic_df.at[interval, status] = count
        statistic_df.at[interval, 'count'] = sum(statistic_df.loc[interval, :].values[:-1])

    approved, default = [], []
    full_sum = sum(statistic_df['count'].iloc[:])
    for i in range(len(statistic_df)):
        approved.append(round((sum(statistic_df['count'].iloc[i:])) / full_sum , 2))
    statistic_df['approved'] = approved

    for i in range(len(statistic_df)):
        status_5 = sum(statistic_df[5].iloc[i:])
        status_6 = sum(statistic_df[6].iloc[i:])
        default.append(round(status_6 / (status_6 + status_5), 2))
    statistic_df['default'] = default 

    return statistic_df