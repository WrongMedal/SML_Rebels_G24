# SML_Rebels_G24
Se avete dubbi sulla struttura e/o l'utilizzo di questa repo chiedete sul gruppo. Non modificate il "main" ma usate uno dei due branch. Una volta che la modifica viene finalizzata e ci assicuriamo che non ci sono errori la introduciamo nel main. Sono già stati creati due branch.

Di seguito sono riportati la struttura e a che punto del lavoro stiamo secondo una prima bozza di scaletta.

## Struttura della repo

```
SML_Rebels_G24/
│
├── Dati/
│   ├── Raw/          # dati originali scaricati
│   └── Processed/    # dati puliti e pronti per il modello
│
├── Codice/
│   ├── 01_scraping.py        # raccolta dei dati
│   ├── 02_cleaning.py        # pulizia e preprocessing
│   ├── 03_vis_an.py          # visualizzazioni e analisi dati 
│   ├── 04_features.py        # feature exatraction (eventuale)
│   ├── 05_model.py           # implementazione del modello
│   └── 06_evaluations.py     # risultati e confronto con i paper
│
├── Papers/           #papers selezionati con motivazioni associate
│
├── Risultati/        # grafici, immagini e risultati numerici, parziali e finali 
│
├── .gitignore        #ignorate
├── requirements.txt  #ignorate
└── README.md         #questo file che state leggendo
```

---

## Scaletta
Per ora è abbastanza generale, mano mano che ci lavoriamo inseriamo dettagli, note e scadenze.

- **Analisi dei paper** — lettura dei paper di riferimento e formalizzazione dell'obiettivo del progetto
- **Scraping** — raccolta dei dati testuali
- **Pulizia dei dati** — preprocessing, normalizzazione...
- **Analisi esplorativa** — visualizzazione e comprensione della distribuzione dei dati
- **Primo modello** — implementazione del modello principale di machine learning
- **Risultati** — valutazione delle performance e confronto con i paper di riferimento
- **Secondo modello** — implementazione o descrizione teorica di un approccio alternativo
- **Presentazione** — preparazione delle slide e dei materiali finali
