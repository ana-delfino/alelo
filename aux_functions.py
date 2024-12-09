import numpy as np
import pandas as pd

def transform_text(text):
    if text and isinstance(text, str):  
        return text.replace(' / ', '-')
    else:
        pass

def format_column_name(df:pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower().str.replace(' ', '_')  
    df.columns = df.columns.str.lower().str.replace('/', '_')
    df.columns = df.columns.str.lower().str.replace('-', '_')
    df.columns = df.columns.str.lower().str.replace('__', '_')
    return df
    
def format_object_columns(df:pd.DataFrame) -> pd.DataFrame:
    for column in df.select_dtypes("object").columns.tolist():
        df[column] = df[column].str.lower()
        df[column] = df[column].str.strip()
        df[column] = df[column].map(transform_text)
    return df   




    