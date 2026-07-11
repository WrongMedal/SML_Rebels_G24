"""
02c_preprocessing_test_holdout.py

Applica al test set esterno (test_holdout_raw.csv) la STESSA pipeline di
preprocessing usata per il training (02_preprocessing_EDA.ipynb):
troncamento al 25% delle frasi, filtro 30 parole minime, NER masking
(PERSON/NORP -> ENT), POS tagging. Nessun fold assignment: e' un test
set fisso, non serve la cross-validation.

Requisiti: spaCy con 'en_core_web_sm' gia' installato (lo stesso ambiente
usato per generare dataset_processed_quantile1_sentences.csv).

Output: test_holdout_processed.csv, stesso formato/colonne del dataset
di training (article_id, topic_id, original_orientation, binary_label,
full_text, word_count, text_bert, raw_text, pos_text) tranne 'fold'.
"""

import re
import nltk
import spacy
import pandas as pd

MIN_WORDS_VALID = 30

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nlp = spacy.load('en_core_web_sm')


def truncate_to_25pct(text):
    if not isinstance(text, str):
        return ""
    sentences = nltk.sent_tokenize(text)
    return " ".join(sentences[:max(1, len(sentences) // 4)])


def mask_entities(text):
    doc = nlp(text)
    masked = text
    for ent in reversed(doc.ents):  # reversed per non rompere gli offset mentre sostituisco
        if ent.label_ in ('PERSON', 'NORP'):
            masked = masked[:ent.start_char] + 'ENT' + masked[ent.end_char:]
    return masked


def clean_for_pos(text):
    text = re.sub(r'[^\w\s.,!?]', '', str(text))
    return re.sub(r'\s+', ' ', text).strip()


def pos_tags(text):
    return " ".join(t for _, t in nltk.pos_tag(nltk.word_tokenize(text)))


def process_test_holdout(input_csv='test_holdout_raw.csv',
                          output_csv='test_holdout_processed.csv'):
    df = pd.read_csv(input_csv)
    print(f"Input: {df.shape[0]} righe, {df['topic_id'].nunique()} topic")

    # troncamento al 25%
    df['full_text'] = df['full_text'].apply(truncate_to_25pct)
    df['full_text'] = df['full_text'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # filtro 30 parole + completezza tripletta (stessa logica del training)
    df['word_count'] = df['full_text'].str.split().str.len()
    invalid_topics = df[df['word_count'] < MIN_WORDS_VALID]['topic_id'].unique()
    n_dropped = len(invalid_topics)
    df = df[~df['topic_id'].isin(invalid_topics)].copy()

    counts = df['topic_id'].value_counts()
    df = df[df['topic_id'].isin(counts[counts == 3].index)].copy()
    df = df.drop(columns=['date'], errors='ignore')

    print(f"Dopo troncamento 25% + filtro 30 parole: {n_dropped} topic scartati")
    print(f"Test set finale: {df.shape[0]} righe, {df['topic_id'].nunique()} topic")

    # NER masking (PERSON/NORP -> ENT)
    print("NER masking in corso (puo' richiedere qualche minuto)...")
    df['raw_text'] = df['full_text']
    df['text_bert'] = df['full_text'].apply(mask_entities)

    # pulizia + POS tagging sul testo mascherato
    df['full_text'] = df['text_bert'].apply(clean_for_pos)
    df['pos_text'] = df['full_text'].apply(pos_tags)

    df = df[['article_id', 'topic_id', 'original_orientation', 'binary_label',
             'full_text', 'word_count', 'text_bert', 'raw_text', 'pos_text']]

    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\nSalvato in '{output_csv}'")
    n_ent = df['text_bert'].str.contains(r'\bENT\b', regex=True).sum()
    print(f"Righe con almeno un ENT: {n_ent} / {len(df)}")
    return df


if __name__ == '__main__':
    process_test_holdout()
