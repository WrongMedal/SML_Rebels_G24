"""
02b_cleaning_test_holdout.py

Pulisce il jsonl scrapato dalla pagina 14 (test set esterno, mai visto da
nessun modello) e applica lo STESSO controllo qualita' a livello di
tripletta usato in 02_cleaning_merge.py -- ma qui NON c'e' merge con
nessun altro dataset: l'output e' uno stand-alone, pensato solo per
testare i modelli gia' allenati sul training set.

Output: test_holdout_raw.csv (stesse 6 colonne del dataset di training,
cosi' puo' passare dritto per lo stesso preprocessing -- troncamento al
25%, NER masking, POS tagging -- prima della valutazione finale).
"""

import json
import re
import uuid

import pandas as pd

MIN_WORDS_VALID = 60

INVISIBLE_CHARS_RE = re.compile(r'[\u200b\u200c\u200d\u2060\ufeff\xad]')

COOKIE_PATTERNS = [
    r'we (use|and our partners use) cookies',
    r'this (site|website) uses cookies',
    r'by (clicking|continuing)[^.]{0,40}(accept|consent|agree)',
    r'accept all cookies',
    r'cookie (policy|settings|preferences)',
    r'manage (your )?(cookie|privacy) preferences',
    r'privacy preference center',
    r'we value your privacy',
    r'necessary cookies',
    r'third[- ]party cookies',
    r'consent to the (use|processing) of',
    r'gdpr',
    r'to provide (you with )?the best (possible )?experience',
    r'we and our (\d+ )?partners (store|process)',
]
COOKIE_RE = re.compile('|'.join(COOKIE_PATTERNS), re.IGNORECASE)


def strip_cookie_boilerplate(text):
    if not isinstance(text, str) or not text:
        return text
    parts = re.split(r'(?<=[.!?])\s+', text)
    kept = [p for p in parts if not COOKIE_RE.search(p)]
    return ' '.join(kept).strip()


def looks_like_cookie_wall(original_text, stripped_text):
    orig_words = len(original_text.split()) if original_text else 0
    stripped_words = len(stripped_text.split()) if stripped_text else 0
    if orig_words == 0:
        return True
    removed_ratio = 1 - (stripped_words / orig_words)
    return removed_ratio > 0.5


def clean_text(text):
    if not isinstance(text, str):
        return text
    text = INVISIBLE_CHARS_RE.sub('', text)
    text = strip_cookie_boilerplate(text)
    replacements = {
        '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-',
        '\u2026': '...',
        '\u00a0': ' ',
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_triplets(jsonl_path):
    triplets = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            triplets.append(json.loads(line))
    return triplets


def triplet_passes_quality(triplet):
    for bias in ['left', 'center', 'right']:
        text = triplet.get(bias)
        if not text:
            return False, f"manca il testo '{bias}'"
        stripped = strip_cookie_boilerplate(text)
        if looks_like_cookie_wall(text, stripped):
            return False, f"'{bias}' sembra un cookie wall"
        cleaned = clean_text(text)
        if len(cleaned.split()) < MIN_WORDS_VALID:
            return False, f"'{bias}' troppo corto dopo pulizia ({len(cleaned.split())} parole)"
    return True, None


def triplets_to_rows(triplets):
    rows = []
    rejected_log = []
    for triplet in triplets:
        topic = triplet.get('topic', 'unknown_topic')
        ok, reason = triplet_passes_quality(triplet)
        if not ok:
            rejected_log.append((topic, reason))
            continue
        for orientation in ['left', 'center', 'right']:
            full_text = clean_text(triplet[orientation])
            binary_label = 1 if orientation in ['left', 'right'] else 0
            rows.append({
                'article_id': str(uuid.uuid4()),
                'topic_id': topic,
                'date': float('nan'),
                'original_orientation': orientation,
                'binary_label': binary_label,
                'full_text': full_text,
            })
    df = pd.DataFrame(rows, columns=['article_id', 'topic_id', 'date',
                                      'original_orientation', 'binary_label',
                                      'full_text'])
    return df, rejected_log


def build_test_holdout(jsonl_path, output_csv_path='test_holdout_raw.csv'):
    triplets = load_triplets(jsonl_path)
    df, rejected = triplets_to_rows(triplets)

    print(f"Triplette totali scrapate: {len(triplets)}")
    print(f"Scartate per qualita': {len(rejected)}")
    print(f"Topic puliti nel test set: {df['topic_id'].nunique() if len(df) else 0}")
    if rejected:
        print("\nTriplette scartate (topic, motivo):")
        for topic, reason in rejected:
            print(f"  - {topic}: {reason}")

    # sanity check: ogni topic deve avere esattamente 3 righe
    counts = df.groupby('topic_id').size()
    incomplete = counts[counts != 3].index.tolist()
    if incomplete:
        print(f"\nATTENZIONE: {len(incomplete)} topic incompleti, li scarto:")
        for t in incomplete:
            print(f"  - {t} ({counts[t]} righe)")
        df = df[~df['topic_id'].isin(incomplete)]

    # controllo di sicurezza: nessun overlap con topic gia' visti in training
    # (opzionale ma utile -- richiede il csv di training a portata di mano)
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"\nTest set esterno: {df.shape[0]} righe, {df['topic_id'].nunique()} topic. "
          f"Salvato in '{output_csv_path}'.")
    return df


if __name__ == '__main__':
    build_test_holdout(
        jsonl_path='allsides_test_holdout.jsonl',
        output_csv_path='test_holdout_raw.csv',
    )
