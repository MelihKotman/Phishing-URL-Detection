import pandas as pd
import numpy as np
import math
import re

# Özellik Çıkarımı Fonksiyonu
def extract_features(df):
    """
    Ham URL verisinden sayısal özellikler çıkarır.
    """
    # Başlangıçta orijinal DataFrame'in bir kopyasını alıyoruz
    features_df = df.copy()

    # URL'nin uzunluğunu hesap etmek için yeni bir sütun ekledik
    features_df['length'] = features_df['URL'].apply(len)

    # URL'deki '.' sayısını belirlemek için yeni bir sütun ekledik
    features_df['dot_count'] = features_df['URL'].apply(lambda x : x.count('.'))

    # URL'deki 'http' ile 'https' kontrolünü yaptık (Binary Özellik : 1 veya 0)
    features_df['is_https'] = features_df['URL'].apply(lambda x : 1 if 'https' in str(x) else 0)

    # URL içindeki rakam sayısını toplam uzunluğa böl
    features_df['digit_ratio'] = features_df['URL'].apply(lambda x : sum(c.isdigit() for c in x) / len(x)) 
    # c.isdigit() karakterin rakam olup olmadığını kontrol eder

    # URL içindeki özel karakter sayıları
    features_df['dot_count'] = features_df['URL'].apply(lambda x: x.count('.'))
    features_df['hyphen_count'] = features_df['URL'].apply(lambda x: x.count('-'))
    features_df['slash_count'] = features_df['URL'].apply(lambda x: x.count('/'))

    # URL'nin Shannon Entropisini hesapla
    features_df['Entropy'] = features_df['URL'].apply(calculate_entropy)

    # URL içinde IP adresi var mı kontrol et
    features_df['has_ip'] = features_df['URL'].apply(has_ip_address)

    # Şüpheli kelimelerin sayısını hesapla
    features_df['suspicious_word_count'] = features_df['URL'].apply(count_suspicious_words)

    # Etiketi (Label) Sayısala Çevirtme : bad = 1 , good = 0
    # Ama zaten 'bad' ve 'good' etiketleri varsa çevir yoksa dokunma
    label_map = {'bad': 1, 'good': 0, 'phishing': 1, 'benign': 0}
    features_df['target'] = features_df['Label'].map(label_map)

    # Eşleşmeyen etiket varsa NaN olur, bunları kaldırıyoruz
    features_df = features_df.dropna(subset=['target'])

    features_df['has_shortening_service'] = features_df['URL'].apply(has_shortening_service)

    print("Özellik çıkarma tamamlandı.")
    return features_df # Olan sütunları geri gönderiyoruz

# URL Entropisi Hesaplama
def calculate_entropy(url):
    """
    URL'in karmaşıklığını (Shannon Entropisi) hesaplar.
    Düzensiz, rastgele karakterler arttıkça skor yükselir.
    """

    url = str(url).strip() # Boşlukları kaldırır
    if not url: # Boş URL kontrolü
        return 0
    
    #Her karakterin olasılığını hesapla
    prob = [float(url.count(c) / len(url)) for c in dict.fromkeys(list(url))] # Her karakterin sayısını alır ve toplam uzunluğa böler

    #Shannon Formülü : -sum(p * log2(p))
    entropy = -sum([p * math.log(p) / math.log(2.0) for p in prob])

    return entropy

# URL içinde IP adresi kontrolü
def has_ip_address(url):
    """
    URL içinde IP adresi (örn: 192.168.0.1) var mı kontrol eder.
    """

    ip_pattern = r'(([01]?\d\∂?|2[0-4\d|25[0-5])\.([01]?\d\∂?|2[0-4\d|25[0-5])\.( [01]?\d\∂?|2[0-4\d|25[0-5])\.( [01]?\d\∂?|2[0-4\d|25[0-5]))' # IP Adresi deseni şöyle görünür: xxx.xxx.xxx.xxx 0-255 arası değerler alır.

    match = re.search(ip_pattern, str(url))
    return 1 if match else 0

def count_suspicious_words(url):
    """
    EDA aşamasında bulduğumuz tehlikeli kelimeleri sayar.
    """

    suspicious_words = [ # En tehlikeli 20 kelimelerden bazıları
                        'login', 
                        'wp',
                        'paypal',
                        'battle',
                        'en',
                        'us',
                        'images',
                        'cmd',
                        'content',
                        'co'
                        # Klasik Şüpheli kelimeler dizisi
                        'secure', 'account', 'verify', 'banking', 'signin', 'admin',
                        'confirm', 'password'
                        ]
    count = 0
    url_lower = str(url).lower() # Küçük harfe çeviriyoruz

    for word in suspicious_words:
        if word in url_lower:
            count += 1

    return count

# URL Kısaltma Servisi Kontrolü
def has_shortening_service(url):
    """
    URL bir kısaltma servisi kullanıyor mu kontrol eder.
    """

    shorteners = {
        'bit.ly', 'goo.gl', 'shorte.st', 'go2l.ink', 'x.co', 'ow.ly', 't.co', 
        'tinyurl', 'tr.im', 'is.gd', 'cli.gs', 'yfrog', 'migre.me', 'ff.im', 
        'tiny.cc', 'url4.eu', 'twit.ac', 'su.pr', 'twurl.nl', 'snipurl', 
        'short.to', 'BudURL', 'ping.fm', 'post.ly', 'Just.as', 'bkite.com', 
        'snipr.com', 'fic.kr', 'loopt.us', 'doiop.com', 'short.ie', 'kl.am', 
        'wp.me', 'rubyurl.com', 'om.ly', 'to.ly', 'bit.do', 't.co', 'lnkd.in', 
        'db.tt', 'qr.ae', 'adf.ly', 'goo.gl', 'bitly.com', 'cur.lv', 'tinyurl.com', 
        'ow.ly', 'bit.ly', 'ity.im', 'q.gs', 'is.gd', 'po.st', 'bc.vc', 
        'twitthis.com', 'u.to', 'j.mp', 'buzurl.com', 'cutt.us', 'u.bb', 'yourls.org', 
        'x.co', 'prettylinkpro.com', 'scrnch.me', 'filoops.info', 'vzturl.com', 
        'qr.net', '1url.com', 'tweez.me', 'v.gd', 'tr.im', 'link.zip.net'
    }
    url = str(url).lower()

    for service in shorteners:
        if service in url:
            return 1
    return 0