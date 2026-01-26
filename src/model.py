import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_evaluate(df):
    """
    Random Forest modelini eğitir ve detaylı performans metriklerini (Matrix, F1, Recall) gösterir.
    """
    print("Model eğitimi ve değerlendirmesi başlatılıyor...")

    # Veriyi Hazırla (X : özellikler ve y : hedef değişken)
    # Model sadece sayıları anlar. Metin sütunlarını atıyoruz.
    X = df.drop(['URL', 'Label', 'target'], axis = 1)
    y = df['target'] # 1: Bad (Phishing), 0 : Good (Güvenli)

    # Eğitim ve Test Olarak BÖL (%80 Eğitim - % 20 Test)
    print(f"     - Toplam Veri: {len(df)} satır")
    print("      - Veri seti bölünüyor (%20 Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    # Modeli Kur (Random Forest)
    # n_estimators : Ağaç sayısı 100 olsun
    model = RandomForestClassifier(n_estimators = 100, random_state = 42, class_weight='balanced') # dengesiz veri için class_weight ekledik
    print("      - Model eğitiliyor...")
    model.fit(X_train, y_train) # Modeli eğit

    # Test Et (Tahmin Yap)
    print("      - Model test ediliyor...")
    y_pred = model.predict(X_test)

    # Performans Metrikleri

    # Genel Başarı (Accuracy)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n Genel Başarı (Accuracy) : %{acc * 100:.2f}")

    # Detaylı Rapor (Precision, Recall, F1-Score)
    print("\n Detaylı Sınıflandırma Raporu:")
    print("-" * 60)
    # target_names = ['Good', 'Bad'] diyerek 0 ve 1'in ne olduğunu belirtiyoruz.
    print(classification_report(y_test, y_pred, target_names= ['Good', 'Bad']))
    print("-" * 60)

    # Confusion Matrix (Karmaşıklık Matrisi) Çizimi
    plot_confusion_matrix(y_test, y_pred)

    # Özellik Önem Düzeyleri (Feature Importances)
    plot_feature_importance(model, X.columns)

    return model, acc

def plot_confusion_matrix(y_test, y_pred):
    """
    Modelin nerede hata yaptığını gösteren renkli tabloyu çizer.
    """

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt= 'd', cmap = 'Blues', cbar = False,
                xticklabels= ['Tahmin: Güvenli', 'Tahmin: Phishing'],
                yticklabels= ['Gerçek: Güvenli', 'Gerçek: Phishing'])
    
    plt.title('Confusion Matrix (Hata Tablosu)')
    plt.ylabel('Gerçek Durum')
    plt.xlabel('Modelin Tahmini')
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Modelin karar verirken en çok hangi özelliğe baktığını çizer.
    """

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1] # Büyükten küçüğe sıralama için

    plt.figure(figsize= (10, 6))
    plt.title("Model Karar Verirken Neye Bakıyor? (Feature Importances)")
    sns.barplot(x = importances[indices],
                y = [feature_names[i] for i in indices],
                hue = [feature_names[i] for i in indices],
                legend = False,                palette = 'viridis')
    plt.xlabel("Önem Derecesi (Skor)")
    plt.ylabel("Özellikler")
    plt.grid(axis = 'x', linestyle = '--', alpha = 0.3)
    plt.show()


