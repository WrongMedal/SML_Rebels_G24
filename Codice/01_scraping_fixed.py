"""
AllSides scraper - versione corretta.

COSA E' CAMBIATO rispetto all'originale:
- extract_article_text() non prende piu' TUTTI i <p> della pagina.
  Prima cerca un contenitore "vero" dell'articolo (tag <article>,
  o classi/attributi tipici come 'article-body', 'entry-content',
  'story-body', 'itemprop=articleBody' ecc). Solo se non trova
  nessun contenitore prova il fallback sui <p> globali (comportamento
  vecchio), ma filtrando via i paragrafi "boilerplate" (bio autore,
  "Send tips to...", copyright, call-to-action per iscrizioni, ecc).
- Se il testo estratto e' comunque troppo corto (< 60 parole) la
  funzione ritorna None invece di salvare comunque un testo spazzatura:
  meglio saltare la tripletta e riprovare dopo, che inquinare il csv.
- Aggiunto un log (scraping_issues.log) che elenca i casi in cui
  l'estrazione e' fallita o e' stata scartata, cosi' sai subito quali
  articoli controllare a mano se servisse.

Il resto della logica (navigazione, ricerca triplette left/center/right,
salvataggio incrementale in jsonl) e' invariato rispetto al notebook
originale.
"""

import time
import json
import random
import re
import uuid
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize


# --- Paragrafi "boilerplate" da scartare anche quando compaiono
# dentro al contenitore giusto (bio, call to action, disclaimer ecc.) ---
BOILERPLATE_PATTERNS = [
    r'send tips to',
    r'correspondent (focused on|covering)',
    r'^\s*©',
    r'all rights reserved',
    r'subscribe (to|now)',
    r'sign up for our newsletter',
    r'follow (us|him|her) on (twitter|x|facebook|instagram)',
    r'read (the )?full story',
    r'related:?\s',
    r'^\s*photo:?\s',
    r'getty images',
    r'reuters/',
    r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b',   # email
]
BOILERPLATE_RE = re.compile('|'.join(BOILERPLATE_PATTERNS), re.IGNORECASE)

# Selettori, in ordine di priorita', usati per trovare il VERO corpo
# articolo invece di prendere tutti i <p> della pagina.
CONTENT_SELECTORS = [
    ('article', {}),
    ('div', {'itemprop': 'articleBody'}),
    ('div', {'class': re.compile(r'article[-_]?body', re.I)}),
    ('div', {'class': re.compile(r'entry[-_]?content', re.I)}),
    ('div', {'class': re.compile(r'story[-_]?body', re.I)}),
    ('div', {'class': re.compile(r'post[-_]?content', re.I)}),
    ('div', {'id': re.compile(r'article[-_]?body', re.I)}),
    ('main', {}),
]

MIN_PARAGRAPH_LEN = 40      # come nell'originale
MIN_WORDS_VALID = 60        # sotto questa soglia, scartiamo invece di salvare junk


def clean_paragraph_list(paragraphs):
    """Filtra via i paragrafi troppo corti o che sembrano boilerplate."""
    out = []
    for p in paragraphs:
        text = p.get_text(separator=' ', strip=True)
        if len(text) <= MIN_PARAGRAPH_LEN:
            continue
        if BOILERPLATE_RE.search(text):
            continue
        out.append(text)
    return out


class AllSidesScraper:
    def __init__(self, output_file='allsides_triplets_fixed.jsonl',
                 issues_log='scraping_issues.log', max_stories=100,
                 only_topics=None):
        """
        only_topics: se fornito (set/list di titoli), lo scraper salva
        SOLO le triplette il cui 'topic' e' in questa lista. Utile per
        ri-scrapare solo i topic corrotti invece di tutto da capo.
        """
        self.output_file = output_file
        self.issues_log = issues_log
        self.base_url = 'https://www.allsides.com'
        self.max_stories = max_stories
        self.only_topics = set(only_topics) if only_topics else None

        chrome_options = Options()
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-gpu")
        self.driver = uc.Chrome(options=chrome_options, version_main=150)
        # Senza questo, una singola pagina che non risponde blocca lo
        # script all'infinito. Con questo, dopo 15s Selenium alza
        # TimeoutException invece di restare appeso.
        self.driver.set_page_load_timeout(15)
        # Il client Selenium di default aspetta fino a 120s la risposta
        # di QUALSIASI comando (non solo il caricamento pagina) prima di
        # alzare un errore di connessione. Se Chrome si blocca per un
        # attimo (es. il Mac va in standby), ogni comando bloccato costa
        # 2 minuti pieni. Abbassando a 20s, un blocco costa molto meno e
        # il ciclo di retry riparte prima.
        self.driver.command_executor.set_timeout(20)

    def _smart_delay(self, min_s=2.0, max_s=4.0):
        time.sleep(random.uniform(min_s, max_s))

    def _log_issue(self, topic, bias, reason):
        with open(self.issues_log, 'a', encoding='utf-8') as f:
            f.write(f"{topic} | {bias} | {reason}\n")

    def _safe_page_source(self):
        """Legge il page_source proteggendo dal timeout del renderer:
        se Chrome e' bloccato a rendere qualcosa, non restiamo appesi."""
        try:
            return self.driver.page_source
        except TimeoutException:
            try:
                self.driver.execute_script("window.stop();")
                return self.driver.page_source
            except Exception:
                return ""
        except Exception:
            return ""

    def get_external_url(self, allsides_news_url):
        try:
            self.driver.get(allsides_news_url)
        except TimeoutException:
            try:
                self.driver.execute_script("window.stop();")
            except Exception:
                pass
        self._smart_delay()
        soup = BeautifulSoup(self._safe_page_source(), 'html.parser')
        button = soup.find('a', id=lambda x: x and str(x).startswith('Read-Full-Story'))
        if not button:
            button = soup.find('a', string=lambda text: text and 'Read Full Story' in text)
        return button['href'] if button and button.get('href') else None

    def extract_article_text(self, url):
        try:
            self.driver.get(url)
        except TimeoutException:
            # la pagina non ha finito di caricare in tempo: fermiamo il
            # caricamento e proviamo comunque a leggere quello che c'e'
            # gia' nel DOM, invece di restare bloccati
            try:
                self.driver.execute_script("window.stop();")
            except Exception:
                pass
        self._smart_delay()
        soup = BeautifulSoup(self._safe_page_source(), 'html.parser')
        for tag, attrs in CONTENT_SELECTORS:
            container = soup.find(tag, attrs=attrs)
            if container:
                paragraphs = container.find_all('p')
                blocks = clean_paragraph_list(paragraphs)
                if len(blocks) > 0:
                    text = ' '.join(blocks)
                    if len(text.split()) >= MIN_WORDS_VALID:
                        return text

        # 2) fallback: tutti i <p> della pagina, ma filtrati
        paragraphs = soup.find_all('p')
        blocks = clean_paragraph_list(paragraphs)
        if len(blocks) == 0:
            return None
        text = ' '.join(blocks)

        # 3) sanity check finale: se troppo corto, meglio scartare
        # che salvare un testo spazzatura
        if len(text.split()) < MIN_WORDS_VALID:
            return None

        return text

    def get_triplet_blocks(self, current_blocks, seen_topics):
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        soup = BeautifulSoup(self._safe_page_source(), 'html.parser')

        containers = soup.find_all(
            'div',
            class_='flex flex-col flex-wrap col-span-3 gap-0 p-0 mt-0 w-full '
                   'smmd:mt-2 md:w-3/4 float-end md:pl-4 smmd:gap-4 smmd:flex-row'
        )

        for container in containers:
            if len(current_blocks) >= self.max_stories:
                break

            parent = container.find_parent()
            title_elem = parent.find(['h2', 'h3'], class_=re.compile(r'text-lg', re.I))
            topic_title = title_elem.get_text(strip=True) if title_elem else "Unknown Topic"

            if topic_title in seen_topics:
                continue

            # se stiamo ri-scrapando solo alcuni topic, salta gli altri
            if self.only_topics is not None and topic_title not in self.only_topics:
                continue

            links = {}
            for bias in ['left', 'center', 'right']:
                news_item = container.find('div', class_=f'news-item flex-1 {bias}')
                if news_item:
                    a_tag = news_item.find('a', href=True)
                    source_tag = news_item.find('p', class_='news-source-title')
                    if a_tag:
                        links[bias] = {
                            'internal_url': urljoin(self.base_url, a_tag['href']),
                            'source_name': source_tag.get_text(strip=True) if source_tag else "Unknown"
                        }

            if 'left' in links and 'center' in links and 'right' in links:
                current_blocks.append({'topic': topic_title, 'links': links})
                seen_topics.add(topic_title)
        return current_blocks

    def _driver_alive(self):
        try:
            _ = self.driver.current_url
            return True
        except Exception:
            return False

    def _restart_driver(self):
        try:
            self.driver.quit()
        except Exception:
            pass
        chrome_options = Options()
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-gpu")
        self.driver = uc.Chrome(options=chrome_options, version_main=150)
        self.driver.set_page_load_timeout(15)
        self.driver.command_executor.set_timeout(20)

    def _scrape_blocks(self, blocks):
        """Logica condivisa: dato un elenco di blocchi (topic + link),
        scarica i 3 articoli di ciascuno e salva le triplette valide.
        Ritorna il numero di triplette salvate con successo."""
        valid_count = 0
        for block in tqdm(blocks, desc="Scraping"):
            topic = block['topic']
            try:
                if not self._driver_alive():
                    self._log_issue(topic, 'ALL', "driver crashed, restarting")
                    self._restart_driver()

                triplet = {'topic': topic, 'sources': {}}
                success = True
                for bias in ['left', 'center', 'right']:
                    ext = self.get_external_url(block['links'][bias]['internal_url'])
                    if not ext:
                        self._log_issue(topic, bias, "no external url found")
                        success = False
                        break

                    text = self.extract_article_text(ext)
                    if text:
                        triplet[bias] = text
                        triplet['sources'][bias] = block['links'][bias]['source_name']
                    else:
                        self._log_issue(topic, bias, f"extraction failed/too short ({ext})")
                        success = False
                        break

                if success:
                    with open(self.output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(triplet, ensure_ascii=False) + '\n')
                    valid_count += 1
            except Exception as e:
                self._log_issue(topic, 'ALL', f"exception: {e}")
                if not self._driver_alive():
                    try:
                        self._restart_driver()
                    except Exception:
                        pass
                continue
        return valid_count

    def _dismiss_cookie_banner(self):
        """Chiude il banner cookie OneTrust se presente, cliccando
        'Accept'. Senza questo, il banner puo' sovrapporsi ai link della
        paginazione e far fallire i click (visto nel log: 'element click
        intercepted... onetrust-policy-text')."""
        selectors = [
            "#onetrust-accept-btn-handler",
            "button#onetrust-accept-btn-handler",
            ".onetrust-close-btn-handler",
        ]
        for sel in selectors:
            try:
                btn = self.driver.find_element(By.CSS_SELECTOR, sel)
                btn.click()
                time.sleep(1)
                return True
            except Exception:
                continue
        return False

    def _click_next_page(self, next_page_number):
        """Cerca in pagina un link il cui HREF contiene 'page=N' (N = pagina
        successiva) e lo clicca. Se non lo trova, stampa diagnostica
        dettagliata e salva la pagina su disco per ispezione manuale,
        invece di fallire silenziosamente."""
        page_pattern = re.compile(rf'[?&]page={next_page_number}(&|$)')
        try:
            links = self.driver.find_elements(By.TAG_NAME, 'a')
        except Exception as e:
            print(f"DEBUG: find_elements fallito: {e}")
            return False

        page_like_hrefs = []
        for link in links:
            try:
                href = link.get_attribute('href') or ''
            except Exception:
                continue
            if page_pattern.search(href):
                try:
                    link.click()
                    return True
                except Exception as e:
                    print(f"DEBUG: trovato href giusto ma click fallito: {e}")
                    continue
            if 'page=' in href.lower() or 'page/' in href.lower():
                page_like_hrefs.append(href)

        # nessun link trovato: stampa diagnostica e salva la pagina per ispezione
        print(f"DEBUG: {len(links)} tag <a> totali trovati in pagina.")
        if page_like_hrefs:
            print(f"DEBUG: nessuno con page={next_page_number} esatto, ma questi contengono 'page':")
            for h in page_like_hrefs[:15]:
                print(f"   - {h}")
        else:
            print("DEBUG: NESSUN link contiene 'page' nell'href, neanche parzialmente.")
        try:
            debug_path = f"debug_page_source_before_page_{next_page_number}.html"
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(self._safe_page_source())
            print(f"DEBUG: pagina salvata in '{debug_path}' per ispezione manuale "
                  f"(aprila con un browser o cerca 'page=' col tuo editor).")
        except Exception as e:
            print(f"DEBUG: impossibile salvare la pagina di debug: {e}")
        return False

    def run(self, start_url):
        seen_topics = set()
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        seen_topics.add(json.loads(line).get('topic'))
        except FileNotFoundError:
            pass

        if not self._driver_alive():
            self._restart_driver()

        try:
            self.driver.get(start_url)
        except TimeoutException:
            try:
                self.driver.execute_script("window.stop();")
            except Exception:
                pass
        all_blocks = self.get_triplet_blocks([], seen_topics)

        valid_count = self._scrape_blocks(all_blocks)
        print(f"Sessione terminata: {valid_count} nuove triplette salvate.")
        print(f"Controlla '{self.issues_log}' per eventuali problemi da rivedere a mano.")

    def run_multi_page(self, start_url, max_pages=4, start_page_number=1):
        """Come run(), ma dopo aver scrapato la pagina corrente CLICCA
        il link della pagina successiva (invece di navigare a un URL
        ?page=N costruito a mano) e ripete, fino a max_pages pagine o
        finche' non trova piu' il link 'pagina successiva'.

        IMPORTANTE: durante lo scraping dei singoli articoli il browser
        naviga via su siti esterni (i giornali) e li' resta. Prima di
        cercare il link della pagina successiva bisogna quindi tornare
        esplicitamente sulla pagina di listing di AllSides, altrimenti
        si cerca il link nella pagina sbagliata (un articolo qualsiasi)."""
        seen_topics = set()
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        seen_topics.add(json.loads(line).get('topic'))
        except FileNotFoundError:
            pass

        if not self._driver_alive():
            self._restart_driver()

        listing_url = start_url
        total_valid = 0
        current_page = start_page_number
        for i in range(max_pages):
            print(f"\n--- Pagina {current_page} ---")
            try:
                self.driver.get(listing_url)
            except TimeoutException:
                try:
                    self.driver.execute_script("window.stop();")
                except Exception:
                    pass
            self._dismiss_cookie_banner()

            blocks = self.get_triplet_blocks([], seen_topics)
            if len(blocks) == 0:
                print(f"Nessun nuovo topic trovato in pagina {current_page} "
                      f"(o sono gia' tutti in seen_topics).")
            valid_count = self._scrape_blocks(blocks)
            total_valid += valid_count
            print(f"Pagina {current_page}: {valid_count} nuove triplette salvate.")

            if i == max_pages - 1:
                break

            # il driver ora e' fermo sull'ultimo sito esterno scrapato:
            # torniamo alla pagina di listing prima di cercare il link
            try:
                self.driver.get(listing_url)
            except TimeoutException:
                try:
                    self.driver.execute_script("window.stop();")
                except Exception:
                    pass
            self._dismiss_cookie_banner()

            next_page = current_page + 1
            clicked = self._click_next_page(next_page)
            if not clicked:
                # il link non esiste come <a> cliccabile (visto nel debug:
                # la paginazione a volte salta il numero immediatamente
                # successivo). Lo schema URL e' confermato valido, quindi
                # navighiamo direttamente invece di arrenderci.
                fallback_url = f'https://www.allsides.com/recent-headline-roundups?page={next_page}'
                print(f"Link per pagina {next_page} non trovato via click, "
                      f"provo navigazione diretta a {fallback_url}")
                try:
                    self.driver.get(fallback_url)
                    clicked = True
                except TimeoutException:
                    try:
                        self.driver.execute_script("window.stop();")
                        clicked = True
                    except Exception:
                        clicked = False
                except Exception as e:
                    print(f"DEBUG: navigazione diretta fallita: {e}")
                    clicked = False

            if not clicked:
                print(f"Anche la navigazione diretta a pagina {next_page} e' fallita, mi fermo qui.")
                break
            time.sleep(3)
            listing_url = self.driver.current_url  # url reale dopo il passaggio
            current_page = next_page

        print(f"\nSessione multi-pagina terminata: {total_valid} nuove triplette salvate in totale.")
        print(f"Controlla '{self.issues_log}' per eventuali problemi da rivedere a mano.")


def process_triplets_to_dataframe(jsonl_file_path, num_lead_sentences=3):
    rows = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            triplet = json.loads(line)
            topic_id = triplet.get('topic', triplet.get('story_url', 'unknown_topic'))

            for orientation in ['left', 'center', 'right']:
                if orientation in triplet and triplet[orientation]:
                    binary_label = 1 if orientation in ['left', 'right'] else 0
                    full_text = triplet[orientation]

                    source_name = 'Unknown'
                    if 'sources' in triplet and orientation in triplet['sources']:
                        source_name = triplet['sources'][orientation]

                    sentences = sent_tokenize(full_text)
                    lead_sentences = sentences[:num_lead_sentences]

                    rows.append({
                        'article_id': str(uuid.uuid4()),
                        'topic_id': topic_id,
                        'source_name': source_name,
                        'date': None,
                        'original_orientation': orientation,
                        'binary_label': binary_label,
                        'full_text': full_text,
                        'lead_sentences': lead_sentences
                    })

    return pd.DataFrame(rows)


if __name__ == '__main__':
    # --- RUN COMPLETO DA ZERO (come richiesto) ---
    # Rilancia lo scraping su tutte le pagine dell'archivio, con la
    # extract_article_text corretta. Ricorda di usare un output_file
    # NUOVO (nome diverso) per non mescolare con l'jsonl vecchio che
    # contiene i testi corrotti.

    scraper = AllSidesScraper(output_file='allsides_triplets_fixed.jsonl')
    scraper.run_multi_page(
        start_url='https://www.allsides.com/recent-headline-roundups?page=10',
        max_pages=4,
        start_page_number=10,
    )

    # se dopo queste 7 pagine non hai ancora abbastanza topic, rilancia
    # lo script con start_page_number=8 (e magari partendo direttamente
    # da start_url='https://www.allsides.com/recent-headline-roundups?page=8'
    # visto che quell'URL diretto SEMBRA funzionare per il primo caricamento,
    # e' solo il click "successiva" ad essere piu' affidabile del salto diretto)

    scraper.driver.quit()

    df_dataset = process_triplets_to_dataframe('allsides_triplets_fixed.jsonl',
                                                num_lead_sentences=5)
    df_dataset.to_csv('dataset_strutturato_allsides_fixed.csv', index=False,
                       encoding='utf-8')
    print(df_dataset.shape)
