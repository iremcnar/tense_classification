# 🧠 İngilizce Zaman Tahmin Modeli | English Tense Prediction Model

Bu proje, verilen İngilizce cümlelerin hangi zaman diliminde yazıldığını sınıflandırmak için derin öğrenme tabanlı bir model kullanmaktadır. Model, cümleyi analiz ederek, cümlenin **şimdiki zaman**, **geçmiş zaman** veya **gelecek zaman** olduğunu tahmin eder. Ayrıca, kullanıcıların cümleleri girdikçe, modelin güven seviyesi ile birlikte tahminler sunulur.

## 🚀 Proje Özeti

- **Model**: Bidirectional LSTM (Long Short-Term Memory) ve Embedding katmanları kullanarak cümlelerin zaman dilimlerini sınıflandırır.
- **Kullanıcı Arayüzü**: PyWebIO ile geliştirilen interaktif bir web uygulaması, kullanıcıların cümle girmesini ve anında tahmin almasını sağlar.
- **Fonksiyonellik**:
  - Cümlelerin zaman dilimi sınıflandırması (Present, Past, Future)
  - Her zaman dilimi için güven seviyesi gösterimi
  - Kullanıcı dostu arayüz

---

## 📥 Başlangıç

Bu projeyi çalıştırmak için aşağıdaki adımları takip edebilirsiniz:

### 1. Gereksinimler

Proje için gerekli olan Python kütüphanelerini yüklemek için aşağıdaki komutları kullanabilirsiniz:

```bash
pip install tensorflow pywebio scikit-learn pandas numpy matplotlib
