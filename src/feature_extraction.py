import pandas as pd
import numpy as np

def extract_features(df):
    """
    Ham URL verisinden sayısal özellikler çıkarır.
    """
    # URL'nin uzunluğunu hesap etmek için yeni bir sütun ekledik
    df['length'] = df['URL'].apply(len)

    # URL'deki '.' sayısını belirlemek için yeni bir sütun ekledik
    df['dot_count'] = df['URL'].apply(lambda x : x.count('.'))

    # URL'deki 'http' ile 'https' kontrolünü yaptık (Binary Özellik : 1 veya 0)
    df['is_https'] = df['URL'].apply(lambda x : 1 if 'https' in str(x) else 0)

    # URL içindeki rakam sayısını toplam uzunluğa böl
    df['digit_ratio'] = df['URL'].apply(lambda x : sum(c.isdigit() for c in x) / len(x)) 
    # c.isdigit() karakterin rakam olup olmadığını kontrol eder

    return df