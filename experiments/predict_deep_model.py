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
    """
    Verilen URL'in Phishing olup olmadığını tahmin eder.
    """
    POPULAR_DOMAINS = [
    "google.com", "youtube.com", "github.com",
    "wikipedia.org", "spotify.com", "apple.com"
]
    domain = url.replace("https://", "").replace("http://", "").split("/")[0]

    if domain in POPULAR_DOMAINS:
        print("SONUÇ: GÜVENLİ (BENIGN)")
        print("-> Popular domain whitelist")
        return

    sequences = tokenizer.texts_to_sequences([str(url)])
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0][0]

    domain = url.replace("https://", "").replace("http://", "").split("/")[0]
    domain_len = len(domain)

    if domain_len <= 15:
        threshold = 0.85
    else:
        threshold = 0.65

    print(f"\n🔍 Analiz Edilen: {url}")
    print(f"📊 Phishing Skoru: %{prediction * 100:.2f}")

    if prediction > threshold:
        print("SONUÇ: TEHLİKELİ (PHISHING)")
    else:
        print("SONUÇ: GÜVENLİ (BENIGN)")
    print("-" * 40)


def main():
    model, tokenizer = load_ai_assets()
    
    print("Çıkmak için 'q' veya 'exit' yazın.")
    print("-" * 40)

    while True:
        url = input("Kontrol edilecek URL'i girin: ")
        
        if url.lower() in ['q', 'exit', 'quit']:
            print("Güle güle!")
            break
        
        if len(url.strip()) == 0:
            continue
            
        predict_url(url, model, tokenizer)

if __name__ == "__main__":
    main()