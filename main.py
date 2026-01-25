import pandas as pd
from src.visualization import plot_class_distribution, plot_url_length_distribution, plot_special_char_breakdown,plot_top_tlds,plot_word_frequency
from src.feature_extraction import extract_features

print("Veri yükleniyor...")
df = pd.read_csv("data/phishing_site_urls.csv")

# --- DEBUG (HATA AYIKLAMA) KODU ---
#print("--- Veri Seti Dedektifi ---")
#print("1. Veri setindeki orjinal etiketler neler?:")
#print(df['Label'].unique()) 

#print(len(df[df['Label'] == 'bad']))
#print("\n2. 'bad' diye aratınca kaç satır buluyorum?:")

#print("\n3. 'good' diye aratınca kaç satır buluyorum?:")
#print(len(df[df['Label'] == 'good']))

print("Analiz yapılıyor...")
plot_class_distribution(df)
plot_url_length_distribution(df)

print("Özellikler çıkarılıyor...")
df_processed = extract_features(df)

print("Özel Karakter Analizi Yapılıyor...")
plot_special_char_breakdown(df)

print("Domain uzantı analizi hazırlanıyor...")
plot_top_tlds(df)

print("En Çok Geçen Kelimeler Analiz Ediliyor...")
plot_word_frequency(df)

print(df_processed.head())
