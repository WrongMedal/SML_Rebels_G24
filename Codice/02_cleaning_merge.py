"""
02_cleaning_merge.py

Pulisce il nuovo jsonl scrapato, applica il controllo qualita' a livello
di TRIPLETTA (non riga), poi unisce il risultato con il vecchio CSV
(articles_clean.csv) secondo questa logica:

- topic in entrambi          -> tieni la versione NUOVA
- topic solo nel vecchio,
  ed era pulito (non tra
  quelli corrotti)           -> tieni il vecchio
- topic solo nel nuovo,
  e passa il controllo
  qualita'                   -> aggiungilo
- topic corrotto nel vecchio
  e non ripescato dal nuovo  -> droppato, non compare nel dataset finale

Output: dataset_finale_merged.csv
"""

import json
import re
import uuid

import pandas as pd

# --- stessa soglia usata nell'EDA sul vecchio dataset ---
MIN_WORDS_VALID = 60

# gli article_id (dal vecchio CSV) identificati come corrotti in EDA
OLD_CORRUPTED_ARTICLE_IDS = set()  # <- incolla qui la lista dei 10 article_id se li hai a portata di mano
                                    #    (altrimenti il controllo qualita' punto 3 li becca comunque di nuovo)


INVISIBLE_CHARS_RE = re.compile(r'[\u200b\u200c\u200d\u2060\ufeff\xad]')

# Frasi tipiche di cookie banner / GDPR consent wall che a volte finiscono
# nel testo estratto al posto del vero corpo articolo. Superano facilmente
# la soglia minima di parole, quindi vanno intercettate a parte.
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
    """Rimuove frasi di cookie/consenso dal testo, spezzando su punti
    fermi in modo semplice (senza dipendenze pesanti tipo nltk qui)."""
    if not isinstance(text, str) or not text:
        return text
    # split leggero in "frasi" su '. ' mantenendo il punto
    parts = re.split(r'(?<=[.!?])\s+', text)
    kept = [p for p in parts if not COOKIE_RE.search(p)]
    return ' '.join(kept).strip()


def looks_like_cookie_wall(original_text, stripped_text):
    """Se dopo aver tolto le frasi di cookie il testo si e' ridotto
    drasticamente (o e' rimasto vuoto), l'intero blocco estratto era
    probabilmente un cookie wall e non un vero articolo."""
    orig_words = len(original_text.split()) if original_text else 0
    stripped_words = len(stripped_text.split()) if stripped_text else 0
    if orig_words == 0:
        return True
    removed_ratio = 1 - (stripped_words / orig_words)
    return removed_ratio > 0.5  # piu' della meta' del testo era cookie boilerplate


def clean_text(text):
    if not isinstance(text, str):
        return text
    # rimuove caratteri invisibili/anti-scraping
    text = INVISIBLE_CHARS_RE.sub('', text)
    # rimuove frasi di cookie/consenso GDPR
    text = strip_cookie_boilerplate(text)
    # normalizza smart quotes e simboli tipografici agli equivalenti ASCII
    replacements = {
        '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-',
        '\u2026': '...',
        '\u00a0': ' ',
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    # rimuove URL residui
    text = re.sub(r'http\S+', '', text)
    # normalizza whitespace multipli
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_new_triplets(jsonl_path):
    """Legge il jsonl e ritorna una lista di dict {topic, left, center, right, sources}."""
    triplets = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            triplets.append(json.loads(line))
    return triplets


def triplet_passes_quality(triplet):
    """Una tripletta e' valida solo se TUTTE e tre le orientazioni superano
    la soglia minima di parole dopo la pulizia E non sono dominate da
    testo di cookie/consenso (es. articolo bloccato da un cookie wall
    che lo scraper ha catturato al posto del vero corpo)."""
    for bias in ['left', 'center', 'right']:
        text = triplet.get(bias)
        if not text:
            return False, f"manca il testo '{bias}'"

        stripped = strip_cookie_boilerplate(text)
        if looks_like_cookie_wall(text, stripped):
            return False, f"'{bias}' sembra un cookie wall (oltre il 50% del testo era boilerplate)"

        cleaned = clean_text(text)
        if len(cleaned.split()) < MIN_WORDS_VALID:
            return False, f"'{bias}' troppo corto dopo pulizia ({len(cleaned.split())} parole)"
    return True, None


def new_triplets_to_rows(triplets):
    """Converte le triplette valide in righe nello STESSO formato a 6 colonne
    del CSV originale: article_id, topic_id, date, original_orientation,
    binary_label, full_text. (source_name e lead_sentences non ci sono nel
    vecchio dataset, quindi non li aggiungiamo per restare coerenti)."""
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


def merge_datasets(old_csv_path, new_jsonl_path, output_csv_path='dataset_finale_merged.csv'):
    old_df = pd.read_csv(old_csv_path)
    old_df['full_text'] = old_df['full_text'].apply(clean_text)

    # applica lo stesso controllo qualita' anche al vecchio dataset,
    # topic per topic, cosi' scartiamo qualsiasi tripletta ancora
    # corrotta indipendentemente dal fatto che fosse gia' stata segnalata
    old_df['word_count'] = old_df['full_text'].str.split().str.len()
    bad_topics_old = set(
        old_df.loc[old_df['word_count'] < MIN_WORDS_VALID, 'topic_id']
    )
    if OLD_CORRUPTED_ARTICLE_IDS:
        bad_topics_old |= set(
            old_df.loc[old_df['article_id'].isin(OLD_CORRUPTED_ARTICLE_IDS), 'topic_id']
        )

    old_df_clean = old_df[~old_df['topic_id'].isin(bad_topics_old)].drop(columns=['word_count'])
    print(f"Vecchio dataset: {old_df['topic_id'].nunique()} topic totali, "
          f"{len(bad_topics_old)} scartati per qualita', "
          f"{old_df_clean['topic_id'].nunique()} topic puliti mantenuti.")

    new_triplets = load_new_triplets(new_jsonl_path)
    new_df, rejected = new_triplets_to_rows(new_triplets)
    print(f"Nuovo scraping: {len(new_triplets)} triplette totali, "
          f"{len(rejected)} scartate per qualita', "
          f"{new_df['topic_id'].nunique() if len(new_df) else 0} topic puliti disponibili.")
    if rejected:
        print("\nTriplette nuove scartate (topic, motivo):")
        for topic, reason in rejected:
            print(f"  - {topic}: {reason}")

    # topic in comune: la versione nuova sostituisce la vecchia
    common_topics = set(old_df_clean['topic_id']) & set(new_df['topic_id']) if len(new_df) else set()
    if common_topics:
        print(f"\n{len(common_topics)} topic presenti in entrambi: tengo la versione nuova per questi.")
    old_df_final = old_df_clean[~old_df_clean['topic_id'].isin(common_topics)]

    merged = pd.concat([old_df_final, new_df], ignore_index=True)

    # sanity check finale: ogni topic deve avere esattamente 3 righe
    # (left, center, right); se non e' cosi' qualcosa e' andato storto
    # nel merge e va scartato per sicurezza
    counts = merged.groupby('topic_id').size()
    incomplete_topics = counts[counts != 3].index.tolist()
    if incomplete_topics:
        print(f"\nATTENZIONE: {len(incomplete_topics)} topic con != 3 righe dopo il merge, li scarto:")
        for t in incomplete_topics:
            print(f"  - {t} ({counts[t]} righe)")
        merged = merged[~merged['topic_id'].isin(incomplete_topics)]

    merged.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"\nDataset finale: {merged.shape[0]} righe, {merged['topic_id'].nunique()} topic. "
          f"Salvato in '{output_csv_path}'.")
    return merged


if __name__ == '__main__':
    merge_datasets(
        old_csv_path='dataset_finale_merged.csv',
        new_jsonl_path='allsides_triplets_fixed.jsonl',
        output_csv_path='dataset_finale_merged_v2.csv',
    )
