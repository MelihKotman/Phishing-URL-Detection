import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from collections import Counter

palette_good_bad = {
    'bad' : '#D32F2F',
    'good' : '#388E3C'
}

def plot_class_distribution(df):
    """
    İyi ve kötü URL sayılarını sütun grafiği ile gösterir.
    """
    plt.figure(figsize = (7, 5)) # Grafik boyutunu ayarla
    sns.countplot(x = 'Label', data = df, palette = palette_good_bad) # 'Label' sütunundaki sınıf dağılımını göster
    plt.title("URL Sınıf Dağılımı") # Başlık ekle
    plt.xlabel("Etiket")
    plt.ylabel("Sayı")
    plt.savefig("visualization/class_distribution.png") # Grafiği kaydet
    plt.show() # Grafiği göster

def plot_url_length_distribution(df):
    """
    URL uzunluklarının histogramını çizer.
    Phishing ve Güvenli sitelerin uzunluk farkını gösterir.
    """
    # Önce URL uzunluğunu hesaplayan  yeni bir sütun ekle
    df['URL_length'] = df['URL'].apply(lambda x: len(str(x)))

    plt.figure(figsize = (12, 6))
    #Seaborn ile histogram çizimi
    sns.histplot(data = df,
                x = 'URL_length', # "URL_length" sütununu kullan
                hue = "Label", #  Renkleri "Label" sütununa göre ayarla
                bins = 100, # Kaç adet çubuk (bin) olacağı
                kde = True, #kde: Kernel Density Estimate yani yoğunluk tahmini
                palette= palette_good_bad, # Renk paleti
                element= 'bars' # Çubuk stili (bars, step, polygon)
                ) 
    
    plt.xlim(0, 150) # Çok uzun URL'ler bozmasın diye sınırla
    plt.title("URL Uzunluk Dağılımı")
    plt.xlabel("URL Uzunluğu")
    plt.ylabel("Frekans")
    plt.savefig("visualization/url_length_distribution.png") # Grafiği kaydet
    plt.show()

def plot_special_char_breakdown(df):
    """
    Hangi karakterler tehlikeli?
    Güvenli ve Zaararlı sitelerde ortalama kaç tane nokta, tire, slash var?
    """
    #Geçici bir frame kopyası alalım
    analiz_df = df.copy()

    #İncelenecek karakter
    chars_to_check = ['.', ',', '-', '/', '@','=','%']
    
    #İncelenecek karakterlerin hepsinin sayısını alıp ayrı ayrı sütun oluşturalım
    for char in chars_to_check:
        analiz_df[f"'{char}' Sayısı"] = analiz_df['URL'].apply(lambda x : str(x).count(char)) # Her char'ı aldık ve her URL için ölçtük
    
    #Veriyi "Label"a göre gruplayıp ortalamasını alalım
    char_means = analiz_df.groupby('Label')[[f"'{c}' Sayısı" for c in chars_to_check]].mean().T

    
    char_means.plot(kind = 'bar', # Çubuk grafik
                    color = [palette_good_bad['bad'], palette_good_bad['good']], # Renkler
                    figsize = (10, 6))  
    
    plt.title("URL İçindeki Özel Karakterlerin Ortalaması")
    plt.ylabel("Ortalama Adet")
    plt.xlabel("Karakter")
    plt.xticks(rotation = 0) # X ekseni etiketlerini yatay yap
    plt.legend(["Bad (Phishing)", "Good (Güvenli)"]) # Legend etiketleri
    plt.grid(axis = 'y', alpha = 0.5, color = 'gray') # Y ekseni için grid ekle
    plt.savefig("visualization/special_char_breakdown.png") # Grafiği kaydet
    plt.show()

def plot_top_tlds(df):
    """
    Domain Uzantıları Belirleme (.com vea .xyz)
    Hangi Uzantılar Tehlikeli
    """

    #URL'in sonundaki (TLD) alalım
    #1. Noktadan böl, son parçayı al
    df['TLD'] = df['URL'].apply(lambda x : str(x).lower().split('.')[-1].split('/')[0]) # . ile / arasındaki kısmı al

    # Sadece en çok geçen ilk 10 uzantıyı alıyoruz
    top_10_tlds = df['TLD'].value_counts().head(10).index # İlk 10 TLD'yi al
    filtered_df = df[df['TLD'].isin(top_10_tlds)] # Sadece bu TLD'leri içeren satırları al

    plt.figure(figsize = (12, 6))
    sns.countplot(x = 'TLD', # X ekseninde TLD'ler
                  hue = 'Label', # Renkleri Label'a göre ayarla
                  data = filtered_df, # Filtrelenmiş dataframe
                  palette = palette_good_bad, # Renk paleti
                  order = filtered_df['TLD'].value_counts().index # TLD'leri sayısına göre sırala
                  )
    plt.title("En Yaygın 10 Domain Uzantısının Güvenlik Dağılımı")
    plt.xlabel("Domain Uzantısı (TLD)")
    plt.ylabel("URL Sayısı")
    plt.legend(title = "Durum") # Legend başlığı
    plt.savefig("visualization/top_tlds.png") # Grafiği kaydet
    plt.show()

def plot_word_frequency(df):
    """
    Phishing sitelerinde en çok hangi kelimeler geçiyor?
    """
    good_df = df[df['Label'] == 'good'].copy()
    bad_df = df[df['Label'] == 'bad'].copy()

    stop_words = {'com', 'www', 'http', 'https', 'net', 'org', 'html', 'htm', 'php', 'index'} # Çok fazlaca var ve işe yaramaz

    word_good = []
    for url in good_df['URL']:
        tokens = re.split(r'[./\-_?=&%]', str(url).lower()) # URL'i parçalara ayır regex ile 
        for token in tokens:
            if token and token not in stop_words and not token.isdigit(): # Boş değilse, stop word değilse ve rakam değilse
                word_good.append(token)

    word_bad = []
    for url in bad_df['URL']:
        tokens = re.split(r'[./\-_?=&%]', str(url).lower()) # URL'i parçalara ayır regex ile 
        for token in tokens:
            if token and token not in stop_words and not token.isdigit(): # Boş değilse, stop word değilse ve rakam değilse
                word_bad.append(token)

    

    word_good_count = Counter(word_good)
    word_bad_count = Counter(word_bad)

    top_20_word_good = word_good_count.most_common(20)
    top_20_word_bad = word_bad_count.most_common(20)

    _, axes = plt.subplots(1, 2, figsize=(16, 8)) # 1 satır, 2 sütunlu grafik alanı oluştur

    # Kötü Kelimeler Grafiği (Sol Taraf - axes[0])
    df_bad = pd.DataFrame(top_20_word_bad, columns=['Kelime','Frekans']) 

    sns.barplot(x = 'Frekans', # X ekseni frekans
                y = 'Kelime', # Y ekseni kelime
                data = df_bad, # Veri kaynağı
                ax = axes[0], # Sol grafik alanı
                palette = 'Reds_r' # Kırmızı tonları
                )
    axes[0].set_title("Phishing Sitelerinde En Sık Geçen 20 Kelime")
    axes[0].grid(axis = 'x', linestyle = '--', alpha = 0.5) # X ekseni için grid ekle

    df_good = pd.DataFrame(top_20_word_good, columns=['Kelime','Frekans'])

    sns.barplot(x = 'Frekans', # X ekseni frekans
                y = 'Kelime', # Y ekseni kelime
                data = df_good, # Veri kaynağı
                ax = axes[1], # Sağ grafik alanı
                palette = 'Greens_r' # Yeşil tonları
                ) 
    axes[1].set_title("Güvenli Sitelerde En Sık Geçen 20 Kelime")
    axes[1].grid(axis = 'x', linestyle = '--', alpha = 0.5) # X ekseni için grid ekle

    plt.tight_layout()
    plt.savefig("visualization/word_frequency.png") # Grafiği kaydet
    plt.show()
    

