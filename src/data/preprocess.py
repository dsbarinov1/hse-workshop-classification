import pandas as pd
import numpy as np
import src.config as cfg
from sklearn.preprocessing import LabelEncoder

def drop_unnecesary_id(df: pd.DataFrame) -> pd.DataFrame:
    if 'ID_y' in df.columns:
        df = df.drop('ID_y', axis=1)
    return df


def fill_sex(df: pd.DataFrame) -> pd.DataFrame:
    most_freq = df[cfg.SEX_COL].value_counts().index[0]
    df[cfg.SEX_COL] = df[cfg.SEX_COL].fillna(most_freq)
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    ohe_int_cols = df[cfg.OHE_COLS].select_dtypes('number').columns  
    
    df[cfg.CAT_COLS] = df[cfg.CAT_COLS].astype('object')
    df[ohe_int_cols] = df[ohe_int_cols].astype(np.int8)
    df[cfg.REAL_COLS] = df[cfg.REAL_COLS].astype(np.float32)
    return df

def encode(df: pd.DataFrame) -> pd.DataFrame:
    encod = LabelEncoder()
    for i in cfg.OHE_COLS:
        df[i] = encod.fit_transform(df[i])
    for i in cfg.REAL_COLS:
        df[i] = encod.fit_transform(df[i])
    for i in cfg.CAT_COLS:
        df[i] = encod.fit_transform(df[i])
    return df

def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame:
    df = df.set_index(idx_col)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = set_idx(df, cfg.ID_COL)
    df = drop_unnecesary_id(df)
    df = fill_sex(df)
    df = cast_types(df)
    df = data_cleaning(df)
    return df


def preprocess_target(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.TARGET_COLS] = df[cfg.TARGET_COLS].astype(np.int8)
    return df


def extract_target(df: pd.DataFrame):
    df, target = df.drop(cfg.TARGET_COLS, axis=1), df[cfg.TARGET_COLS]
    return df, target

def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    if 'ID_y' in df.columns:
        df = df.drop('ID_y', axis=1)
    if 'ID' in df.columns:
        df = df.set_index('ID')
    
    # заполнение пропусков
    df['Возраст курения'] = np.where(df['Статус Курения'] == 'Никогда не курил(а)', 0, df['Возраст курения'])
    df['Сигарет в день'] = np.where(df['Статус Курения'] == 'Никогда не курил(а)', 0, df['Сигарет в день'])
    df['Сигарет в день'] = np.where(df['Сигарет в день'].isna(), 0, df['Сигарет в день'])
    df['Частота пасс кур'] = np.where(df['Пассивное курение'] == 0, 'Ни разу в день', df['Частота пасс кур'])
    df['Возраст алког'] = np.where(df['Алкоголь'] == 'никогда не употреблял', 0, df['Возраст алког'])

    df = df.drop(df[df['Возраст алког'].isna()].index)
    df = df.drop(df[df['Пол'].isna()].index)
    df = df.drop(df[df['Частота пасс кур'].isna()].index)
    
    return df