# Veritabanı kullanarak model eğitme ve veri yönetimi

from json import load
import pandas as pd
import os
import sqlite3
import  sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src_db.database_connection.db_manager import init_db, import_csv_to_db, inject_popular_domains, load_data_from_db
from src_db.model.deep_model_db import train_deep_learning_model

DATA_PATH = 'data/phishing_site_urls.csv' # Veri dosyasının yolu
DB_PATH = 'phishing_data.db' # Veritabanı dosyasının yolu
SAMPLE_SIZE = None # Eğitim için örnek boyutu
ENABLE_DATA = False  # Grafikleri görmek istemiyorsak False yapabiliriz

def main():

    # Başta veri altyapısını hazırla

    print("Veritabanı başlatılıyor...")
    init_db() # Veritabanını başlat
    import_csv_to_db(DATA_PATH) # CSV'den veritabanına veri ekle (bu satırı ilk seferde açın)
    inject_popular_domains() # Popüler domainleri ekle

    print("Veritabanından veri çekiliyor...")
    df = load_data_from_db(limit = SAMPLE_SIZE) # Veritabanından veriyi çek
    print(f"    - Toplam Satır Sayısı: {len(df)}")

    # Eğer örneklem ayarı açıksa veriyi küçült
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"    - Veri örnekleniyor: {SAMPLE_SIZE} satır...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42).copy()

    # Makine Öğrenmesi (MODEL TRAINING)
    print("\n Yapay Zeka Modeli Eğitiliyor...")


    #model, accuracy = train_and_evaluate(processed_df)
    model, tokenizer = train_deep_learning_model(df)
        
    print(f"\nModel eğitimi tamamlandı. Model: {model}, Tokenizer: {tokenizer}")
    model.save('src_db/model/phishing_model.keras') # Modeli kaydet

    with open ('src_db/model/tokenizer.pickle', 'wb') as handle:
        import pickle
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

if __name__ == "__main__":
    main()