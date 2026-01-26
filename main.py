from aem import con
import pandas as pd
import os

from regex import W

from src.model import train_and_evaluate

from sympy import plot
from src.visualization import (
    plot_class_distribution,
    plot_url_length_distribution,
    plot_special_char_breakdown,
    plot_top_tlds,
    plot_word_frequency
)
from src.feature_extraction import extract_features

DATA_PATH = "data/phishing_site_urls.csv" # Veri dosyasının yolu
SAMPLE_SIZE = None # Eğitim için örnek boyutu
ENABLE_DATA = False  # Grafikleri görmek istemiyorsak False yapabiliriz

def main():

    # Veri Yükleme
    print(f"Veri Yükleniyor...")
    if not os.path.exists(DATA_PATH):
        print("HATA: Veri dosyası bulunamadı! Lütfen 'data' klasörünü kontrol et.")
        return
    
    df = pd.read_csv(DATA_PATH)

    # Veri Hakkında Kısa Bilgi 
    print(f"    - Toplam Satır Sayısı: {len(df)}")
    print(f"    - Etiket Dağılımı: {df['Label'].value_counts().to_dict()}")

    # Eğer örneklem ayarı açıksa veriyi küçült
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        print(f"    - Veri örnekleniyor: {SAMPLE_SIZE} satır...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
    
    # Görsel Analizler (EDA)
    if ENABLE_DATA:
        print("\nGörsel Analizler (EDA) Başlatılıyor...")
        
        if not os.path.exists("visualization"):
            os.makedirs("visualization")
        
        try:
            plot_class_distribution(df)
            plot_url_length_distribution(df)
            plot_special_char_breakdown(df)
            plot_top_tlds(df)
            plot_word_frequency(df)
        except Exception as e:
            print(f"Görsel analizlerde hata oluştu: {e}")
    else:
        print("\nGörsel Analizler (EDA) Devre Dışı Bırakıldı.")

    # Özellik Çıkarımı (Feature Extraction)
    print("\nÖzellik Çıkarımı Başlatılıyor...")
    processed_df = extract_features(df)

    # Meraklısı için kısa bir kontrol çıktısı
    print("\n    --- Çıkarılan Özelliklerin Ortalamaları (Good vs Bad) ---")
    print(processed_df.groupby('target')[['length','digit_ratio','suspicious_word_count']].mean())

    # Makine Öğrenmesi (MODEL TRAINING)
    print("\n Yapay Zeka Modeli Eğitiliyor...")

    try:
        model, accuracy = train_and_evaluate(processed_df)
        print(f"\n İŞLEM TAMAMLANDI! Model Başarı Oranı: %{accuracy * 100:.2f}")

        print("\n" + "="*50 + "\n")
        print("Phishing URL Tespit Sistemi Testing Moduna Geçilsin.")
        print("Çıkmak için q yazıp Enter'a basın.\n")

        WHITELIST = {"edu.tr",
                     "gov.tr",
                     "google.com",
                     "github.com",
                     "linkedin.com",
                     "youtube.com",
                     "X.com",
                     "instagram.com",
                     "facebook.com"
                    }

        while True:
    
            url_input = input("Test etmek istediğiniz URL'yi girin (Çıkmak için q): ").strip()
            if url_input.lower() == 'q':
                print("Çıkılıyor...")
                break
            if not url_input:
                print("Lütfen geçerli bir URL girin.")
                continue
            
            is_trusted = False # Güvenilir mi kontrolü en başta güvenilir değil dedik
            # Eğer güvenilir listede varsa kullanıcıya sor
            for domain in WHITELIST: 
                if domain in url_input:
                    is_trusted = True
                    break # Döngüden çık
            
            # Güvenilir ise kullanıcıya sor
            if is_trusted: # Eğer güvenilir ise
                continue_print = input("Girilen URL güvenilir listesinde. Yine de test etmek ister misiniz? (e/h): ").strip().lower()
                if continue_print != 'e': # Evet değilse atlasın
                    print("URL güvenilir kabul edildi, test atlandı.\n")
                    continue
                else: # Evet ise devam et ve test etsin
                    print("Devam ediliyor, URL test edilecek...\n")

        # Girilen URL'yi DataFrame'e çevirip özellik çıkarımı yap
            test_df = pd.DataFrame({'URL': [url_input], 'Label': ['good']}) # Label önemli değil, dummy değer
            test_features = extract_features(test_df)

            X_input = test_features.drop(['URL', 'Label', 'target'], axis=1) # Modelin beklediği formatta

            prediction = model.predict(X_input)[0] #  Tahmin (0 veya 1)
            probability = model.predict_proba(X_input)[0][prediction] #  Tahmin olasılığı

            if prediction == 1:
                print(f"Uyarı: Girilen URL 'Phishing' (Kötü Amaçlı) olarak tespit edildi! (Olasılık: %{probability * 100:.2f})\n")
            else:
                print(f"Girilen URL 'Güvenli' olarak tespit edildi. (Olasılık: %{probability * 100:.2f})\n")
        
    except ValueError as ve:
        print(f"     - Model eğitimi sırasında değer hatası oluştu: {ve}")

    except Exception as e:
        print(f"     - Model eğitimi sırasında hata oluştu: {e}")

if __name__ == "__main__":
    main()