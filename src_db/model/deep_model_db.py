import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore (importing Sequential model)
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D # type: ignore (importing layers)
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore (importing Tokenizer)
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore (importing pad_sequences)
from sklearn.model_selection import train_test_split # Test ve eğitim verilerini ayırmak için
from sklearn.metrics import  classification_report # Model değerlendirme metrikleri

# --- Metrikler (Constants) ---
MAX_WORDS = 10000 # En sık kullanılan karakter/heceyi tanı
MAX_LEN = 75 # Her URL'i 75 karaktere sabitle (Kısa ise doldur, uzun ise kes)
EMBEDDING_DIM = 32 # Her karakteri 32 boyutlu vektörle temsil et

def train_deep_learning_model(df):
    """
    URL'leri ham metin olarak alır ve Derin Öğrenme CNN ile sınıflandırma modeli eğitir.
    """

    print("Veri hazırlama başladı...")

    # Veri ön işleme (X: URL'ler, y: etiketler)
    urls = df['url'].astype(str).tolist() # Pandas serisini listeye dönüştür

    # Hedef değişkeni hazırla
    # Veritabanondan 'label' sütununu al ve 0/1 formatına dönüştür
    y = df['label'].apply(lambda x: 1 if x == 'bad' else 0).values

    # Tokenization (Metni Sayıya dönüştür)
    # URL'leri harf harf veya parça parça sayılara dönüştürür.
    print("Tokenization işlemi başladı...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, char_level=True) # char_level=True ile karakter bazlı tokenization
    tokenizer.fit_on_texts(urls) # Tokenizer'ı URL'lere göre eğit
    sequences = tokenizer.texts_to_sequences(urls) # URL'leri sayısal diz

    # Padding (Dizileri Sabitleme)
    # Her URL farklı boyda olamayacağı için 75 karaktere sabitliyoruz.
    X = pad_sequences(sequences, maxlen=MAX_LEN) # Kısa URL'leri doldur, uzun URL'leri kes

    # Eğitim ve test verilerini ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Eğitim verisi boyutu: {X_train.shape}, Test verisi boyutu: {X_test.shape}")

    # Model Mimarisi
    model = Sequential()

    # Embedding katmanı: Her karakteri 32 boyutlu vektörle temsil et
    model.add(Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN))

    # Convolutional katmanı: Özellik çıkarımı için
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    
    # Pooling katmanı: Boyut indirgeme
    model.add(GlobalMaxPooling1D())

    # Dense katmanlar: Sınıflandırma için
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5)) # Aşırı öğrenmeyi önlemek için Dropout

    # Çıktı katmanı 0 veya 1 için
    model.add(Dense(1, activation='sigmoid')) # İkili sınıflandırma için sigmoid aktivasyonu

    # Modeli derleme
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # İkili sınıflandırma için binary_crossentropy kaybı

    print("Model eğitimi başladı...")
    # Epochs = 3 yani 3 kez tüm eğitim verisi üzerinde geçiş yap
    model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test)) # Eğitim sırasında doğrulama verisi kullan

    print("Model eğitimi tamamlandı. Değerlendirme başladı...")
    y_pred_prob = model.predict(X_test) # Tahmin olasılıklarını al
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("Değerlendirme metrikleri:")
    print(classification_report(y_test, y_pred, labels=[0,1], target_names=['bad', 'good']))

    return model, tokenizer


