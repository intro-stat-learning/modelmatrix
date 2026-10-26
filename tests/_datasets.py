# Carseats and Wage, from ISLP, with the dtypes ISLP's load_data gives them

from pathlib import Path

import pandas as pd

_DATA = Path(__file__).parent / 'data'

_CATEGORICAL = {'Carseats': ['ShelveLoc', 'Urban', 'US'],
                'Wage': ['maritl', 'race', 'region', 'jobclass',
                         'health', 'health_ins']}
_ORDERED = {'Wage': ['education']}


def load_data(name):
    df = pd.read_csv(_DATA / f'{name}.csv')
    for col in _CATEGORICAL.get(name, []):
        df[col] = pd.Categorical(df[col])
    for col in _ORDERED.get(name, []):
        df[col] = pd.Categorical(df[col], ordered=True)
    return df
