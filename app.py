from pywebio.input import input
from pywebio.output import put_text, put_markdown, put_scope, use_scope, put_html
from pywebio import start_server
import tensorflow as tf
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Model ve tokenizer yükleme
model = tf.keras.models.load_model("tense_classifier.h5")
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

max_length = 50

# Temizleme
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

# Tahmin
def combined_predict(sentence):
    processed = preprocess_text(sentence)
    seq = tokenizer.texts_to_sequences([processed])
    pad = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    pred = model.predict(pad)[0]
    predicted_class = np.argmax(pred)
    return predicted_class + 1, pred

# Ana fonksiyon
def main():
    # Arka plan rengini ve tasarımını CSS ile belirleme
    put_html("""
    <style>
        body {
            background: linear-gradient(to right, #FFB6C1, #ADD8E6, #98FB98);
            font-family: 'Arial', sans-serif;
            color: #333;
            text-align: center;
            padding: 20px;
        }
        h1 {
            color: #FF6347;
        }
        h3 {
            color: #32CD32;
        }
        .result-container {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        .input-container {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        .result-container h3 {
            font-size: 18px;
            color: #FF6347;
        }
    </style>
    """)

    put_markdown("# 🧠 İngilizce Cümle Zaman Tahmini")

    # Sabit cümle girme kısmıe
    put_markdown("### ✏️ Bir İngilizce cümle girin (çıkmak için boş bırakın):")

    # Tahminler için scope (sonuçlar her yeni tahminle alt kısmı günceller)
    put_scope("results")

    while True:
        # Kullanıcının cümle girmesi
        sentence = input("", placeholder="Bir İngilizce cümle girin...", type="text")
        if not sentence:
            break

        tense_index, confidence = combined_predict(sentence)
        tense_map = {1: "Present Tense", 2: "Past Tense", 3: "Future Tense"}

        # Sonuçları eklemek için scope kullanıyoruz
        with use_scope("results", clear=False):  # Öncekileri silmeden yeni tahmini ekle
            put_markdown(f"---")
            put_html(f"""
            <div class="result-container">
                <h3>✏️ Girilen Cümle:</h3>
                <p>{sentence}</p>
                <h3>🎯 Tahmin:</h3>
                <p><strong>{tense_map[tense_index]}</strong></p>
                <h3>🔎 Güven Seviyesi:</h3>
                <p>Present: <strong>{confidence[0]*100:.1f}%</strong></p>
                <p>Past: <strong>{confidence[1]*100:.1f}%</strong></p>
                <p>Future: <strong>{confidence[2]*100:.1f}%</strong></p>
            </div>
            """)

# Çalıştır
if __name__ == '__main__':
    start_server(main, port=8080, debug=True)
