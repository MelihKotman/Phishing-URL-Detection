# src/db_manager.py
# Veritabanı yönetimi için gerekli fonksiyonlar ve sınıflar burada tanımlanır.

import sqlite3 # Importing the sqlite3 module to manage SQLite databases
import pandas as pd
import os

# Veritabanı dosyamızın adı
DB_NAME = 'data/phishing_final.db'

# Veritabanı bağlantısı fonksiyonu
def get_db_connection():
    """
    Veritabanı bağlantısı oluşturur ve döner.
    Eğer veritabanı dosyası yoksa, yeni bir tane oluşturur.
    """
    conn = sqlite3.connect(DB_NAME)
    return conn

# Veritabanı oluşturma fonksiyonu
def init_db():
    """
    Veritabanı tablolarını oluşturur.
    Burada sadece 'urls' tablosu oluşturulacak.
    """
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME) # Eski veritabanı dosyasını sil
        print(f"Eski veritabanı dosyası silindi: {DB_NAME}")

    conn = get_db_connection() # Veritabanı bağlantısı al
    cursor = conn.cursor() # İmleç (cursor) oluştur

    # SQL komutu: "urls" adında bir tablo yap.
    # id: Sıra No (Otomatik artan)
    # url: Site adresi (Metin)
    # label: Durum (bad/good)
    create_tabley_query = """
    CREATE TABLE urls(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        label TEXT NOT NULL
    );
    """

    cursor.execute(create_tabley_query) # Tabloyu oluştur
    conn.commit() # Değişiklikleri kaydet
    conn.close() # Bağlantıyı kapat

# CSV'den veritabanına veri ekleme fonksiyonu
def import_csv_to_db(csv_path):
    """
    ETL Süreci: CSV dosyasından verileri okuyup dönüştürür ve veritabanına ekler.
    """
    if not os.path.exists(csv_path):
        print(f"CSV dosyası bulunamadı: {csv_path}")
        return
    
    conn = get_db_connection() # Veritabanı bağlantısı al
    cursor = conn.cursor() # İmleç (cursor) oluştur

    # Tablo var mı kontrol et
    try:
        cursor.execute("SELECT COUNT(*) FROM urls;") # Tabloyu kontrol et
        count = cursor.fetchone()[0] # Kayıt sayısını al
    except sqlite3.OperationalError:
        count = 0 # Tablo yoksa kayıt sayısı 0
    
    if count > 0:
        print(f"Veritabanında zaten {count} kayıt var. Yükleme atlandı.")
        conn.close()
        return
    
    print("CSV dosyasından veritabanına veri yükleniyor...")

    # Pandas ile CSV dosyasını oku
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower() # Sütun isimlerini küçük harfe çevir
    df = df[['url', 'label']]

    # SQL'E dök 
    # if_exists='append' : Eğer tablo varsa ekle
    df.to_sql('urls', conn, if_exists='append', index=False) # Parametreler: tablo adı, bağlantı, var ise ekle, indeks ekleme

    conn.close() # Bağlantıyı kapat
    print("Veri yükleme tamamlandı.")

# Veritabanından veri çekme fonksiyonu
def load_data_from_db(limit=None):
    """
    Model eğitimi için veriyi SQL'den çeker.
    limit: Çekilecek kayıt sayısı (Varsayılan: None, tüm kayıtlar)
    """ 
    conn = get_db_connection() # Veritabanı bağlantısı al

    query = "SELECT * FROM urls"
    if limit and isinstance(limit, int):
        query += f" LIMIT {limit}"

    # SQL sorgusuyla veriyi çek ve DataFrame'e dönüştür
    df = pd.read_sql_query(query, conn) # Veriyi DataFrame olarak al

    conn.close()   # Bağlantıyı kapat
    return df  # DataFrame'i döndür

def inject_popular_domains():
    """
    Model kısa domainleri de öğrensin diye dünyanın en popüler sitelerini veritabanına 'good' olarak ekler.
    """
    conn = get_db_connection() # Veritabanı bağlantısı al
    cursor = conn.cursor() # İmleç (cursor) oluştur
    # Dünyanın en popüler siteleri (Örnek liste)
    popular_sites = [
        "google.com", "youtube.com", "facebook.com", "baidu.com", "wikipedia.org",
        "qq.com", "taobao.com", "yahoo.com", "tmall.com", "amazon.com",
        "twitter.com", "sohu.com", "jd.com", "live.com", "weibo.com",
        "sina.com.cn", "vk.com", "360.cn", "login.tmall.com", "blogspot.com",
        "yandex.ru", "instagram.com", "linkedin.com", "netflix.com", "twitch.tv",
        "microsoft.com", "ebay.com", "bing.com", "office.com", "apple.com",
        "msn.com", "wordpress.com", "stackoverflow.com", "github.com", "ibu.edu.tr"
    ]
    
    print("Popüler kısa domainler veritabanına aşılanıyor...")
    for site in popular_sites:
        # Önce var mı diye bak, yoksa ekle
        cursor.execute("SELECT count(*) FROM urls WHERE url = ?", (site,))
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO urls (url, label) VALUES (?, ?)", (site, 'good'))
            
    conn.commit()
    conn.close()
    print("Veri aşılama tamamlandı.")

    