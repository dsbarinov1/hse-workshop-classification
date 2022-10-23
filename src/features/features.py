from numpy import diff
import pandas as pd

def get_hours(time:str) -> int:
    return int(time.split(':')[0])

def get_num(x:int) -> int:
    return x

def add_early_wakeup(df: pd.DataFrame) -> pd.DataFrame:
    df['Жаворонок'] = df['Время пробуждения']
    df['Жаворонок'] = df['Время пробуждения'].apply(lambda x: 1 if get_hours(x) <= 7 else 0)
    return df

def add_late_wakeup(df: pd.DataFrame) -> pd.DataFrame:
    df['Сова'] = df['Время засыпания']
    df['Сова'] = df['Время засыпания'].apply(lambda x: 1 if get_hours(x) > 23 else 0)
    return df

def add_sleep_time(df: pd.DataFrame) -> pd.DataFrame:
    df['Время пробуждения'] = df['Время пробуждения'].apply(lambda x: get_hours(x))
    df['Время засыпания'] = df['Время засыпания'].apply(lambda x: get_hours(x))
    df['Время сна'] = df['Время пробуждения'] - df['Время засыпания']
    df['Время сна'] = df['Время сна'].apply(lambda x: get_num(x) if get_num(x) >= 0 else 24+get_num(x))
    return df

def ciggaretes_for_life(df: pd.DataFrame) -> pd.DataFrame:
    for a,b in zip(df['Сигарет в день'].values, df['Возраст курения'].values):
        df["Сигарет за жизнь"] = a*(b*365)
    return df

    
def lifestyle(df: pd.DataFrame) -> pd.DataFrame:
    for sport, smoke, alco in zip(df['Спорт, клубы'].values, df['Статус Курения'].values, df['Алкоголь'].values):
        if sport and not smoke and not alco:
            df["Образ жизни"] = 2
        if sport and smoke and alco:
            df["Образ жизни"] = 1
        if not sport and smoke and alco:
            df["Образ жизни"] = 0
    return df 
