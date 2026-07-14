# SML_Rebels_G24

## Struttura della repo

```
SML_Rebels_G24/
|
|-- Dati/
|   |-- Raw/                     # dati originali scaricati, puliti
|   |-- Processed/               # dati vettorizzati e pronti per il modello
|
|-- Codice/
|   |-- 01_scraper.ipynb                    # raccolta dei dati
|   |-- 01_scraping_fixed.py                # script corretto/alternativo per lo scraping
|   |-- 02_cleaning_merge.py                # script per la pulizia e l'unione dei dati
|   |-- 02_preprocessing_EDA.ipynb          # pulizia, preprocessing e grafici
|   |-- 02b_cleaning_test_holdout.py        # script per la pulizia del set di test/holdout
|   |-- 02c_preprocessing_test_holdout.py   # script per il preprocessing del set di test/holdout
|   |-- 03_model_logreg.ipynb               # implementazione modello Logistic Regression
|   |-- 03_prova_logreg_elmo.ipynb          # test Logistic Regression con embeddings ELMo
|   |-- 04_model_SVC.ipynb                  # implementazione modello SVC
|   |-- 05_BERT_finale.ipynb                # implementazione definitiva modello BERT
|   |-- 05_model_BERT_meanpooling.ipynb     # implementazione modello BERT con mean pooling
|
|-- Papers/                      # papers selezionati con motivazioni associate
|
|-- Risultati/                   # report e presentazione
|
|-- .gitignore                   # file/cartelle da ignorare su git
|-- requirements.txt             # librerie necessarie per il progetto
|-- README.md                    # questo file che state leggendo
```
