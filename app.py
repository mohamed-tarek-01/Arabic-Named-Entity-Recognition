import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import re
import random
import matplotlib.pyplot as plt 
import base64
import os
from PIL import Image

@st.cache_resource
def load_files():    
    model = load_model("ner_model.keras")
    with open("word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    with open("tag2idx.pkl", "rb") as f:
        tag2idx = pickle.load(f)
    with open("tags.pkl", "rb") as f:
        tags = pickle.load(f)
    with open("punctuations.pkl", "rb") as f:
        punctuations = pickle.load(f)
    with open("stop_words.pkl", "rb") as f:
        stop_words = pickle.load(f)

    idx2tag = {i: t for t, i in tag2idx.items()}
    return model, word2idx, tag2idx, tags, idx2tag, punctuations, stop_words

model, word2idx, tag2idx, tags, idx2tag, punctuations, stop_words = load_files()

def predict(text, model, word2idx, tags, stop_words, punctuations):
    processed_words = []
    removed_words = []

    for word in text.split():
        original_word = word

        word = re.sub(rf"[{re.escape(''.join(punctuations))}]", "", word)

        non_arabic_parts = re.findall(r'[^\u0600-\u06FF]+', word)
        if non_arabic_parts:
            for part in non_arabic_parts:
                removed_words.append((part, "Non-Arabic"))
            word = re.sub(r'[^\u0600-\u06FF]', '', word)

        if word in stop_words:
            removed_words.append((original_word, "Stopword"))
            continue

        word = re.sub(r'\d+', '', word)

        if len(word) == 1 and word != '.':
            removed_words.append((original_word, "Too Short"))
            continue

        if word.startswith('ال') and len(word) > 2:
            word = word[2:]

        word = re.sub(r'[إأآا]', 'ا', word)
        word = re.sub(r'ى', 'ي', word)
        word = re.sub(r'ؤ', 'و', word)
        word = re.sub(r'ئ', 'ي', word)
        word = re.sub(r'ة', 'ه', word)
        word = re.sub(r'[ًٌٍَُِّْ]', '', word)

        if word.strip():
            processed_words.append(word)
        else:
            continue

    if not processed_words:
        return [], processed_words, removed_words

    word_indices = [word2idx.get(w, word2idx.get("ENDPAD", 0)) for w in processed_words]
    word_indices = pad_sequences([word_indices], maxlen=100, padding='post', value=word2idx.get("ENDPAD", 0))
    
    pred = model.predict(np.array(word_indices))
    pred_tags = np.argmax(pred, axis=-1)[0]
    
    result = [(word, tags[tag_idx]) for word, tag_idx in zip(processed_words, pred_tags[:len(processed_words)])]
    
    return result, processed_words, removed_words

    word_indices = [word2idx.get(w, word2idx.get("ENDPAD", 0)) for w in processed_words]
    word_indices = pad_sequences([word_indices], maxlen=100, padding='post', value=word2idx.get("ENDPAD", 0))
    
    pred = model.predict(np.array(word_indices))
    pred_tags = np.argmax(pred, axis=-1)[0]

    result = [(word, tags[tag_idx]) for word, tag_idx in zip(processed_words, pred_tags[:len(processed_words)])]
    
    return result, processed_words, removed_words

st.title(" Arabic Named Entity Recognition (NER)")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analysis", "📊 Statistics", "🧰 Preprocessing"])

with tab1:
    st.markdown("## ✍️ Enter Text or Use Sample")

    political_samples = [
    "الرئيس عبد الفتاح السيسي زار دولة الإمارات لتعزيز العلاقات الثنائية.",
    "الرئيس الامريكي أكد التزام الولايات المتحدة بدعم حلفائها في حلف الناتو خلال قمة بروكسل.",
    "مؤتمر القمة العربية في الجزائر شهد حضور العديد من الزعماء العرب لمناقشة قضايا المنطقة.",
    "الأمم المتحدة دعت إلى زيادة المساعدات الإنسانية لشعب فلسطين في غزة بعد تصاعد الأعمال العدائية.",
    "المستشارة الألمانيه أنجيلا ميركل أعلنت عن خطة جديدة لمكافحة أزمة الهجرة في ألمانيا  و أوروبا."
    ]

    sports_samples = [
        "منتخب مصر لكرة القدم تأهل إلى كأس العالم بعد فوزه على منتخب تونس.",
        "اللاعب محمد صلاح سجل هدفًا رائعًا في مباراة ليفربول ضد مانشستر سيتي.",
        "فريق الزمالك حقق انتصارًا ساحقًا على الاهلي في المباراة النهائية.",
        "لاعب كرة القدم ليونيل ميسي انتقل إلى نادي باريس سان جيرمان بعد فترة طويلة مع برشلونة.",
        "منتخب مصر لكرة اليد تأهل إلى الدور نصف النهائي في البطولة الإفريقية."
    ]

    education_samples = [
    "جامعة القاهرة أعلنت عن فتح باب التقديم للطلاب الجدد في مصر للعام الدراسي المقبل.",
    "الأكاديمية العربية للعلوم والتكنولوجيا في الإسكندرية توفر منح دراسية للطلاب الدوليين في مجال الهندسة.",
    "جامعة الملك سعود في الرياض أطلقت مركزًا جديدًا للبحث والتطوير في مجالات الطاقة المتجددة." ,
    "الجامعة الأردنية في عمان بدأت في تقديم برامج تعليمية مشتركة مع جامعة كاليفورنيا في الولايات المتحدة الأمريكية لتطوير مهارات الطلاب في الهندسة.",
    "جامعة محمد بن زايد في أبوظبي قدمت منحًا دراسية للطلاب من جميع أنحاء الشرق الأوسط في مجال دراسات الأمن السيبراني."
    ]

    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📰 Political Sample"):
            selected_sample = random.choice(political_samples)
            st.session_state["user_input"] = selected_sample

    with col2:
        if st.button("🏟️ Sports Sample"):
            selected_sample = random.choice(sports_samples)
            st.session_state["user_input"] = selected_sample

    with col3:
        if st.button("🏫 Education Sample"):
            selected_sample = random.choice(education_samples)
            st.session_state["user_input"] = selected_sample

    
    user_input = st.text_area("📝Enter Arabic Sentences:", height=150, key="user_input")

    if st.button("🔍 Analyze Entities"):
        if user_input.strip():
            st.subheader("📄 Results:")
            entities,_,_ = predict(user_input, model, word2idx, tags, stop_words, punctuations)
            df_results = pd.DataFrame(entities, columns=["Word", "Tag"])
            st.markdown("### 🧾 Prediction Table")
            st.table(df_results)
            csv = df_results.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results as CSV", csv, "ner_results.csv", "text/csv")
            st.session_state["entities"] = entities
        else:
            st.warning("⚠️ Please enter some text first.")

with tab2:
    st.markdown("## 📊 Named Entity Statistics")

    if "entities" in st.session_state:
        entities = st.session_state["entities"]
        tags_only = [tag for _, tag in entities if tag != "O"]
        tag_counts = Counter(tags_only)
        total_entities = len(tags_only)
        st.subheader(f"Total Named Entities: {total_entities}")
        if tag_counts:
            chart_data = pd.DataFrame(
                tag_counts.values(),
                index=tag_counts.keys(),
                columns=["Count"]
            )
            st.markdown("### 📊 Entity Tag Distribution (Bar Chart)")
            st.bar_chart(chart_data)
        
            st.markdown("### 🥧 Entity Tag Distribution (Pie Chart)")
            pie_data = pd.Series(tag_counts).plot.pie(autopct="%.1f%%", colors=["#FF9999","#66B3FF","#99FF99","#FFCC99"], startangle=90, figsize=(7, 7))
            st.pyplot(pie_data.figure)

            st.markdown("### 📊 Entity Tag Counts")
            st.write(tag_counts)

        else:
            st.info("❕ No named entities found to display.")
    else:
        st.info("ℹ️ Run an analysis from the first tab.")

with tab3:
    st.markdown("## 🧰 Preprocessing Preview")

    user_input = st.session_state.get("user_input", "").strip()

    if user_input:
        original_text = st.session_state["user_input"]
        st.markdown("### 🔹 Original Text")
        st.info(original_text)
        _, cleaned_words, removed_words = predict(
            original_text, model, word2idx, tags, stop_words, punctuations
        )
        st.markdown("### ✅ Cleaned Text After Preprocessing")
        st.success(" ".join(cleaned_words) if cleaned_words else "No words left after cleaning.")

        if removed_words:
            st.markdown("### ❌ Removed / Filtered Words")
            df_removed = pd.DataFrame(removed_words, columns=["Word", "Reason"])
            st.table(df_removed)
        else:
            st.info("No words were removed during preprocessing.")
    else:
        st.info("ℹ️ Enter some text from Tab 1 to preview preprocessing.")
