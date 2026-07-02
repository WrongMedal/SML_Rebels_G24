# SML_Rebels_G24
Se avete dubbi sulla struttura e/o l'utilizzo di questa repo chiedete sul gruppo. Non modificate il "main" ma usate uno dei due branch. Una volta che la modifica viene finalizzata e ci assicuriamo che non ci sono errori la introduciamo nel main. Sono già stati creati due branch.

Di seguito sono riportati la struttura e a che punto del lavoro stiamo secondo una prima bozza di scaletta.

## Struttura della repo

```
SML_Rebels_G24/
│
├── Dati/
│   ├── Raw/          # dati originali scaricati, puliti
│   └── Processed/    # dati vettorizzati e pronti per il modello
│
├── Codice/
│   ├── 01_scraper.ipynb            # raccolta dei dati
│   ├── 02_preprocessing_EDA.ipybn  # pulizia, preprocessing e grafici --> libreria per grafici finali, vedi commenti
│   ├── 03_model_1.py               # implementazione del modello 1
│   ├── 04_model_2.py               # implementazione del modello 2
│   └── 05_eval_comparison.py       # risultati e confronto con i paper e tra modelli
│
├── Papers/           #papers selezionati con motivazioni associate
│
├── Risultati/        # grafici, immagini e risultati numerici, parziali e finali 
│
├── .gitignore        #ignorate
├── requirements.txt  #ignorate
└── README.md         #questo file che state leggendo
```


Nota: ai requirements manca la parte dello scraper... probebilmente non funziona su tutti i pc e su tutti i browser, comunque il dataset originale è a disposizione in \Dati\Raw\dataset_strutturato_allsides.csv

---

## Scaletta
Per ora è abbastanza generale, mano mano che ci lavoriamo inseriamo dettagli, note e scadenze.

- ~~**Analisi dei paper** — lettura dei paper di riferimento e formalizzazione dell'obiettivo del progetto~~
- ~~**Scraping** — raccolta dei dati testuali~~
- ~~**Pulizia dei dati** — preprocessing, normalizzazione...~~
- ~~**Analisi esplorativa** — visualizzazione e comprensione della distribuzione dei dati~~
- **Primo modello** — implementazione del modello 1:
- **Risultati** — valutazione delle performance e confronto con i paper di riferimento
- **Secondo modello** — implementazione del modello 2
- **Risultati** — valutazione delle performance e confronto con i paper di riferimento
- **Confronto** — confronto tra i due modelli
- **Presentazione** — preparazione delle slide e dei materiali finali
