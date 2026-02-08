import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore (importing Sequential model)
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D # type: ignore (importing layers)
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore (importing Tokenizer)
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore (importing pad_sequences)
from sklearn.model_selection import train_test_split # Test ve eğitim verilerini ayırmak için
from sklearn.metrics import  classification_report # Model değerlendirme metrikleri

# --- Metrikler (Constants) ---
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
    tokenizer = Tokenizer(
    char_level=True, # char_level=True ile karakter bazlı tokenization
    lower=True, # Tüm karakterleri küçük harfe çevir (Büyük/küçük harf ayrımını kaldırır)
    oov_token="<OOV>" # Bilinmeyen karakterler için özel bir token
) 
    tokenizer.fit_on_texts(urls) # Tokenizer'ı URL'lere göre eğit
    sequences = tokenizer.texts_to_sequences(urls) # URL'leri sayısal diz

    # Padding (Dizileri Sabitleme)
    # Her URL farklı boyda olamayacağı için 75 karaktere sabitliyoruz.
    X = pad_sequences(
    sequences, # Tokenizer tarafından oluşturulan sayısal diziler 
    maxlen=MAX_LEN, # Her URL'i 75 karaktere sabitle
    padding='post', # Kısa URL'leri sonuna sıfır ekleyerek doldur
    truncating='post' # Uzun URL'leri sonundan keserek 75 karaktere indir
) 

    # Eğitim ve test verilerini ayırma
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
      X,
      y,
      df.index,
      test_size=0.2,
      stratify=y,
      random_state=42
  )

    print(f"Eğitim verisi boyutu: {X_train.shape}, Test verisi boyutu: {X_test.shape}")

    # Model Mimarisi
    model = Sequential()

    # Embedding katmanı: Her karakteri 32 boyutlu vektörle temsil et
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=EMBEDDING_DIM, input_length=MAX_LEN))

    # Convolutional katmanı: Özellik çıkarımı için
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    
    # Pooling katmanı: Boyut indirgeme
    model.add(GlobalMaxPooling1D())

    # Dense katmanlar: Sınıflandırma için
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.6)) # Aşırı öğrenmeyi önlemek için Dropout

    # Çıktı katmanı 0 veya 1 için
    model.add(Dense(1, activation='sigmoid')) # İkili sınıflandırma için sigmoid aktivasyonu

    # Modeli derleme
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05), metrics=['accuracy']) # type: ignore # BinaryCrossentropy kaybı, ikili sınıflandırma için uygundur. label_smoothing=0.05 ile etiketleri biraz yumuşatıyoruz, bu da modelin aşırı güvenli tahminler yapmasını engeller ve genelleme yeteneğini artırır. 

    print("Model eğitimi başladı...")
    # Epochs = 10 yani 10 kez tüm eğitim verisi üzerinde geçiş yap
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test)) # Eğitim sırasında doğrulama verisi kullan

    # Accuracy ve Loss grafiklerini çiz
    plot_accuracy_loss(history)

    print("Model eğitimi tamamlandı. Değerlendirme başladı...")
    y_pred_prob = model.predict(X_test) # Tahmin olasılıklarını al
    # --- Dinamik Threshold (Kısa domain bias'ını azaltmak için) ---
    def dynamic_threshold(prob, url):
        # Domain kısmını al (path ve parametreleri at)
        domain = url.split("/")[0]
        domain_len = len(domain)

        # Kısa domainler için daha yüksek eşik
        if domain_len <= 15:
            return int(prob > 0.85)
        else:
            return int(prob > 0.65)

    # Dinamik threshold ile tahmin üret
    test_urls = df.loc[idx_test, 'url'].astype(str).tolist()
    y_pred = np.array([
        dynamic_threshold(p[0], u) for p, u in zip(y_pred_prob, test_urls)
    ])

    print("Değerlendirme metrikleri:")
    print(classification_report(y_test, y_pred, labels=[0,1], target_names=['good', 'bad']))

    return model, tokenizer


def plot_accuracy_loss(history):
    """
    1D-CNN modelinin eğitim sürecindeki Accuracy ve Loss grafiklerini çizer.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Grafiği
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
    axes[0].set_title('1D-CNN Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Loss Grafiği
    axes[1].plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    axes[1].set_title('1D-CNN Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Grafik kaydetme
    save_path = 'visualization/Feature_Extraction/1D_CNN_Accuracy_Loss.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nGrafik kaydedildi: {save_path}")
    
    plt.show()

    # Son epoch sonuçlarını yazdır
    print(f"\nSon Epoch Sonuçları:")
    print(f"  Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"  Validation Loss: {history.history['val_loss'][-1]:.4f}")
