# SML_Rebels_G24
Se avete dubbi sulla struttura e/o l'utilizzo di questa repo chiedete sul gruppo. Non modificate il "main" ma usate uno dei due branch. Una volta che la modifica viene finalizzata e ci assicuriamo che non ci sono errori la introduciamo nel main. Sono già stati creati due branch.

Di seguito sono riportati la struttura e a che punto del lavoro stiamo secondo una prima bozza di scaletta.

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
|-- Risultati/                   # grafici, immagini e risultati numerici, parziali e finali
|
|-- .gitignore                   # file/cartelle da ignorare su git
|-- requirements.txt             # librerie necessarie per il progetto
|-- README.md                    # questo file che state leggendo
```
