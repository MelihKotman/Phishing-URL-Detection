import os
import sys
import tensorflow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # TensorFlow uyarılarını bastırır

import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
import numpy as np

# --- AYARLAR (Eğitimdekiyle AYNI) ---
MAX_LEN = 75

def load_ai_assets():
    """
    Eğitim sırasında kaydedilen model ve tokenizer dosyalarını yükler.
    """

    print("Model Yükleniyor...")

    # Modeli yükle
    try: 
        model = (tf.keras.models.load_model('src_db/model/phishing_model.keras'))  # type: ignore
    except OSError as e:
        print("Model dosyası bulunamadı. Lütfen modeli eğitip kaydedin.")
        raise e
    
    # Tokenizer'ı yükle
    try:
        with open('src_db/model/tokenizer.pickle', 'rb') as handle: 
            tokenizer = pickle.load(handle)
    except FileNotFoundError as e:
        print("Tokenizer dosyası bulunamadı. Lütfen modeli eğitip kaydedin.")
        raise e
    return model, tokenizer
    
def predict_url(url, model, tokenizer):
    """Verilen URL'in Phishing olup olmadığını tahmin eder."""
    WHITELIST_DOMAINS = {
    "google.com", "www.google.com", "youtube.com", "www.youtube.com",
    "facebook.com", "www.facebook.com", "amazon.com", "twitter.com",
    "instagram.com", "linkedin.com", "wikipedia.org", "yahoo.com",
    "yandex.com", "yandex.ru", "whatsapp.com", "bing.com", "live.com",
    "microsoft.com", "apple.com", "github.com", "stackoverflow.com",
    "ibu.edu.tr", "www.ibu.edu.tr", "turkiye.gov.tr", "enabiz.gov.tr"
}   
    clean_url = url.replace("https://", "").replace("http://", "").replace("www.", "").strip("/")
    
    # 1. Beyaz Liste Kontrolü (Yapay Zekadan Önce)
    if clean_url in WHITELIST_DOMAINS:
        print(f"\n🔍 Analiz Edilen: {url}")
        print("🛡️  SONUÇ: GÜVENLİ (Beyaz Listede Mevcut)")
        print("   -> Yapay zeka yorulmadı, bilinen güvenli site.")
        print("-" * 40)
        return

    # 1. Ön İşleme (Preprocessing)
    # URL'i string yap, listeye koy (Tokenizer liste bekler)
    sequences = tokenizer.texts_to_sequences([str(clean_url)])
    
    # Uzunluğu sabitle (Padding)
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    
    # 2. Tahmin (Prediction)
    prediction = model.predict(padded, verbose=0)[0][0]
    
    # 3. Sonuç Yorumlama
    print(f"\n🔍 Analiz Edilen: {clean_url}")
    print(f"📊 Phishing Skoru: %{prediction * 100:.2f}")
    
    if prediction > 0.5:
        print("SONUÇ: TEHLİKELİ (PHISHING) ")
        print("   -> Bu site bilgilerinizi çalmaya çalışabilir!")
    else:
        print("SONUÇ: GÜVENLİ (BENIGN)")
        print("   -> Temiz görünüyor.")
    print("-" * 40)


def main():
    model, tokenizer = load_ai_assets()
    
    print("Çıkmak için 'q' veya 'exit' yazın.")
    print("-" * 40)

    while True:
        url = input("🔗 Kontrol edilecek URL'i girin: ")
        
        if url.lower() in ['q', 'exit', 'quit']:
            print("👋 Güle güle!")
            break
        
        if len(url.strip()) == 0:
            continue
            
        predict_url(url, model, tokenizer)

if __name__ == "__main__":
    main()