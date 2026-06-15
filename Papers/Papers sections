# Paper Notes — SML_Rebels_G24

Papers read and annotated for the project. For each one: original excerpt from the relevant section, what it actually says, and how it connects to what we're doing.

---

## 1. Kiesel et al. (2019) — SemEval-2019 Task 4: Hyperpartisan News Detection

**Full reference:** Kiesel, J., Mestre, M., Shukla, R., Vincent, E., Adineh, P., Corney, D., Stein, B., & Potthast, M. (2019). SemEval-2019 Task 4: Hyperpartisan News Detection. *Proceedings of the 13th International Workshop on Semantic Evaluation*, 829–839. https://aclanthology.org/S19-2145/

---

### Relevant excerpt

> "Hyperpartisan news is news that takes an extreme left-wing or right-wing standpoint. If one is able to reliably compute this meta information, news articles may be automatically tagged, this way encouraging or discouraging readers to consume the text. [...] We developed new resources for this purpose, including a manually labeled dataset with 1,273 articles, and a second dataset with 754,000 articles, labeled via distant supervision. [...] The best team achieved an accuracy of 0.822 on a balanced sample (yes : no hyperpartisan) drawn from the manually tagged corpus."

---

### What this section says

The paper formalizes hyperpartisan news detection as an NLP task and runs it as a shared competition (SemEval 2019). They build two datasets: one annotated by hand (1,273 articles) and a much larger one labeled automatically based on the publisher's known political orientation — a technique called distant supervision. 42 teams submitted systems. The best one hit 82.2% accuracy on the manually labeled test set. Among the competitive systems, an SVM using TF-IDF features combined with structural features (number of paragraphs, number of hyperlinks) performed well without using deep learning.

---

### Relation to our project

**Differences:**
- Their task is binary classification (hyperpartisan / not hyperpartisan). Ours is regression on a continuous score (cosine distance from the Center article).
- Their labels come either from manual annotation or from publisher identity (distant supervision). Our target is computed geometrically from the text itself, which avoids the noise of assuming every article from a given outlet has the same level of extremism.
- Their dataset is not organized in triples — they don't compare articles covering the same event across political orientations.

**What we can borrow:**
- The structural features that worked well in their top systems: number of paragraphs, number of external links, number of quotes. These are easy to extract and validated by the competition results.
- The baseline setup: SVM + TF-IDF is competitive and interpretable. We can use SVR (the regression version) with a similar feature setup.
- The evaluation framing: they note that models trained on publisher-labeled data generalize poorly to manually annotated data. This supports our choice of a content-based target rather than a label derived from the source.

**Their results:**
- Best system: 82.2% accuracy (manually labeled test set)
- Ensemble of all submitted systems: 87.0%
- SVM with TF-IDF + structural features: competitive without neural methods

**Features used:**
- TF-IDF on article text
- LIWC (Linguistic Inquiry and Word Count) psychological word categories
- Number of paragraphs
- Number of external hyperlinks
- Number of quoted words

**How we replicate this:**
```python
# structural features — straightforward to extract during scraping/cleaning
n_paragraphs = article_text.count('\n\n')
n_external_links = len(re.findall(r'https?://', article_html))
n_quoted_words = len(re.findall(r'"[^"]*"', article_text).split())
```

---

## 2. Potthast et al. (2018) — A Stylometric Inquiry into Hyperpartisan and Fake News

**Full reference:** Potthast, M., Kiesel, J., Reinartz, K., Bevendorff, J., & Stein, B. (2018). A Stylometric Inquiry into Hyperpartisan and Fake News. *Proceedings of the 56th Annual Meeting of the ACL (Volume 1: Long Papers)*, 231–240. https://aclanthology.org/P18-1022/

---

### Relevant excerpt — Section 4.1 (Style Features)

> "Our writing style model incorporates common features as well as ones specific to the news domain. The former are n-grams, n in [1, 3], of characters, stop words, and parts-of-speech. Further, we employ 10 readability scores and dictionary features, each indicating the frequency of words from a tailor-made dictionary in a document, using the General Inquirer Dictionaries as a basis. The domain-specific features include ratios of quoted words and external links, the number of paragraphs, and their average length."

### Relevant excerpt — Section 5.1 (Key Finding)

> "Our first experiment addressing the first question uncovered an odd behavior of our classifier: it would often misjudge left-wing for right-wing news, while being much better at distinguishing both combined from the mainstream. To explain this behavior, we hypothesized that maybe the writing style of the hyperpartisan left and right are more similar to one another than to the mainstream. [...] 66% of misclassifications of left-wing articles are falsely classified as right-wing articles, whereas 60% of all misclassified right-wing articles are classified as mainstream articles."

### Relevant excerpt — Table 1 (Corpus statistics)

| Orientation | Paragraphs (avg) | Words (avg) | Quoted words (avg) |
|---|---|---|---|
| Mainstream | 20.1 | 692.0 | 18.1 |
| Left-wing | 14.6 | 423.2 | 28.6 |
| Right-wing | 14.1 | 397.4 | 24.6 |

---

### What this section says

The paper runs a stylometric analysis on 1,627 articles from 9 publishers (3 mainstream, 3 hyperpartisan left, 3 hyperpartisan right), fact-checked by BuzzFeed journalists during the 2016 US election period. The key finding is that left-wing and right-wing hyperpartisan articles are much more similar to each other in writing style than either is to mainstream articles — validated by three independent methods including a novel application of Unmasking. They also show that style-based classification can detect hyperpartisanship in general (F1=0.78) but fails at distinguishing left from right, and completely fails at fake news detection (F1=0.46).

Looking at the corpus statistics, hyperpartisan articles (both left and right) are significantly shorter than mainstream ones (~400 words vs ~692), have fewer paragraphs, and quote proportionally more words per article.

---

### Relation to our project

**Differences:**
- They do binary and ternary classification; we do regression.
- Their corpus is fixed (7 days, 9 publishers, 2016 US elections). Ours covers AllSides data across many topics and time periods — likely more diverse.
- They don't use a control group for the same news event: articles from different publishers are about different stories, so topic effects can leak into the style signal.

**What we can borrow:**
- The core theoretical justification for our project: since left and right share style more than either does with the mainstream, the direction of bias is not the meaningful signal — distance from the neutral center is. This is exactly what our cosine distance target measures.
- The full feature set from Section 4.1. In particular: character n-grams (1-3), POS n-grams, readability scores, ratio of quoted words, ratio of external links, number of paragraphs, average paragraph length.
- The finding that basic structural features alone (number of paragraphs, quotations, links) don't discriminate orientation, but do help when combined with lexical features.

**Their results:**
- Hyperpartisan vs. mainstream: F1 = 0.78 (style features)
- Satire vs. both: F1 = 0.81
- Fake news detection: F1 = 0.46 (style is not enough)
- Accuracy on ternary task (left / right / mainstream): 0.60

**Features used:**
- Character n-grams (n = 1, 2, 3)
- Stop word n-grams
- POS tag n-grams
- 10 readability scores: Automated Readability Index, Coleman Liau, Flesch-Kincaid Grade Level, Flesch Reading Ease, Gunning Fog, LIX, McAlpine EFLAW, RIX, SMOG Grade, Strain Index
- General Inquirer dictionary word frequencies
- Ratio of quoted words per article
- Ratio of external links per article
- Number of paragraphs
- Average paragraph length

**How we replicate this (Python):**
```python
import spacy
import textstat

nlp = spacy.load("en_core_web_sm")

def extract_dense_features(text):
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_space]
    
    features = {
        "n_tokens": len(tokens),
        "n_sentences": len(list(doc.sents)),
        "avg_sentence_length": len(tokens) / max(len(list(doc.sents)), 1),
        "n_paragraphs": text.count('\n\n'),
        "ratio_adjectives": sum(1 for t in tokens if t.pos_ == "ADJ") / max(len(tokens), 1),
        "ratio_adverbs": sum(1 for t in tokens if t.pos_ == "ADV") / max(len(tokens), 1),
        "ratio_nouns": sum(1 for t in tokens if t.pos_ == "NOUN") / max(len(tokens), 1),
        "exclamation_density": text.count('!') / max(len(text), 1),
        "question_density": text.count('?') / max(len(text), 1),
        "quote_density": text.count('"') / max(len(text), 1),
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "gunning_fog": textstat.gunning_fog(text),
        "ttr": len(set(t.lower_ for t in tokens)) / max(len(tokens), 1),
    }
    return features
```

Note: they use TreeTagger (Java), we use spaCy — same POS categories, different implementation. Readability scores via `textstat` library (`pip install textstat`).

---

## 3. Spinde et al. (2021) — Neural Media Bias Detection Using Distant Supervision With BABE

**Full reference:** Spinde, T., Plank, M., Krieger, J.-D., Ruas, T., Gipp, B., & Aizawa, A. (2021). Neural Media Bias Detection Using Distant Supervision With BABE — Bias Annotations By Experts. *Findings of ACL: EMNLP 2021*, 1166–1177. https://aclanthology.org/2021.findings-emnlp.101/

---

### Relevant excerpt

> "Media coverage has a substantial effect on the public perception of events. Nevertheless, media outlets are often biased. One way to bias news articles is by altering the word choice. The automatic identification of bias by word choice is challenging, primarily due to the lack of a gold standard data set and high context dependencies. This paper presents BABE, a robust and diverse data set created by trained experts, for media bias research. [...] With noisy but scalable training data, distant supervision has become a promising method for producing weak labels from partisan sources (like AllSides)."

---

### What this section says

The paper introduces BABE, a dataset of sentences annotated for bias at word level by trained experts (not crowdsourced). The key motivation is that existing datasets — including those derived from AllSides via distant supervision — are noisy: not every article from a right-wing outlet is equally biased, and not every sentence in a biased article contains bias markers. They use AllSides as a source of weak labels for pre-training, then refine with expert annotation. Their main model is BERT fine-tuned on BABE, which outperforms previous approaches significantly.

---

### Relation to our project

**Differences:**
- Their task is sentence-level word choice bias detection. Ours is article-level radicalization scoring.
- They use BERT (transformer). We use SVR, Ridge, XGBoost — models we can fully explain if asked.
- Their BABE dataset is annotated by experts; our target is computed automatically from cosine distance.

**What we can borrow:**
- Validation that AllSides is the standard academic source for distant supervision in media bias research. Citing this paper when we justify our data source is legitimate.
- The explicit acknowledgment that distant supervision from AllSides produces noisy labels. Our cosine distance target is a direct improvement on this: instead of assuming all articles from a "Right" outlet have the same bias level, we measure the actual linguistic distance from the neutral version of the same story. We should mention this explicitly in the report.
- The observation that bias is largely about word choice — which means TF-IDF (word frequency weighting) is a theoretically grounded choice of feature representation, not just a practical convenience.

**Their results:**
- BERT fine-tuned on BABE: F1 ≈ 0.79 on sentence-level bias detection
- Traditional models (logistic regression, random forest with manual features): significantly lower, but still above chance
- Distant supervision alone (from AllSides labels): worse than expert-annotated fine-tuning

**Features used (traditional baseline):**
- Bias lexicon word counts
- Sentiment scores
- Linguistic features (POS-based)

**How this affects our pipeline:**
The gap between distant supervision and expert annotation in this paper is one motivation for computing our cosine distance target rather than using AllSides labels directly as the training signal. Our target is continuous and event-specific, which should reduce the noise that Spinde et al. identify as the main problem with publisher-level labeling.

---

## 4. Horne & Adali (2017) — This Just In: Fake News Packs a Lot in Title

**Full reference:** Horne, B. D., & Adali, S. (2017). This Just In: Fake News Packs a Lot in Title, Uses Simpler, Repetitive Content in Text Body, More Similar to Satire than Real News. *Proceedings of the International AAAI Conference on Web and Social Media*, 11(1), 759–766. https://doi.org/10.1609/icwsm.v11i1.14976

---

### Relevant excerpt — Section "Features"

> "Stylistic Features: The stylistic features are based on natural language processing to understand the syntax, text style, and grammatical elements of each article content and title. To test for differences in syntax, we use the Python Natural Language Toolkit (Bird 2006) part of speech (POS) tagger and keep a count of how many times each tag appears in the article. Along with this, we keep track of the number of stopwords, punctuation, quotes, negations (no, never, not), informal/swear words, interrogatives (how, when, what, why), and words that appear in all capital letters."

> "Complexity Features: [...] we compute the readability of each document using three different grade level readability indexes: Gunning Fog, SMOG Grade, and Flesh-Kincaid grade level index. [...] we compute what is called the Type-Token Ratio (TTR) of a document as the number of unique words divided by the total number of words in the document. TTR is meant to capture the lexical diversity of the vocabulary in a document. A low TTR means a document has more word redundancy."

### Relevant excerpt — Abstract

> "We show overall title structure and the use of proper nouns in titles are very significant in differentiating fake from real. This leads us to conclude that fake news is targeted for audiences who are not likely to read beyond titles and is aimed at creating mental associations between entities and claims."

---

### What this section says

The paper studies three datasets (Buzzfeed election data, their own political news dataset, and a satire dataset) and computes a large set of features on both the title and body of articles separately. Their main finding is that fake news and satire are more similar to each other than to real news. More specifically: fake news titles are longer, use fewer stop words, fewer nouns but more proper nouns; fake news bodies are shorter, use simpler and more repetitive language (lower TTR), and less punctuation. The title/body distinction is the key methodological contribution — treating them as separate signals.

---

### Relation to our project

**Differences:**
- Their ground truth is fake/real/satire. Ours is a continuous radicalization score.
- Their datasets are small (35–75 articles per category in datasets 1 and 2). Ours from AllSides should be much larger.
- They study fake news; we study rhetorical extremism, which overlaps but is not the same thing.

**What we can borrow:**
- The idea of extracting features from the title separately from the body. This is easy to implement and validated by their results — title features alone are strongly predictive.
- The full feature list in Table 3 of the paper: it's one of the most systematic enumerations of NLP stylometric features in this area. POS counts, readability scores, TTR, negation counts, exclamation marks, quotes, proper noun counts, interrogatives — all straightforward to implement with NLTK or spaCy.
- The finding that proper noun density in titles is particularly significant. Easy to add as a feature.

**Their results:**
- Linear SVM on style features: competitive on all three datasets
- Title features alone outperform body features alone on fake vs. real
- TTR, proper noun count, and structural title features are among the most discriminative

**Features used (full list from Table 3):**

Stylistic: word count (WC), words per sentence (WPS), noun count (NN), proper noun count (NNP), personal pronouns (PRP), possessive pronouns (PRP$), adverbs (RB), adjectives (JJ), past/present/future tense verb counts, exclamation marks, negations (no/never/not), interrogatives (how/what/why), stopword percentage, total punctuation count, number of quotes.

Complexity: Gunning Fog, SMOG Grade, Flesch-Kincaid Grade Level, syntax tree depth (median), noun phrase tree depth, verb phrase tree depth, TTR, word fluency (frequency in COCA corpus), average word length.

Psychological (LIWC-based): analytic, insight, causal, discrepancy, tentative, certainty, differentiation, power, reward, risk, emotional tone, affect (anger, sadness, etc.), positive/negative sentiment (SentiStrength).

**How we replicate this (without LIWC which is paid):**
```python
import nltk
from collections import Counter

def extract_title_features(title):
    tokens = nltk.word_tokenize(title)
    pos_tags = nltk.pos_tag(tokens)
    pos_counts = Counter(tag for _, tag in pos_tags)
    
    return {
        "title_length": len(tokens),
        "title_proper_nouns": pos_counts.get("NNP", 0) + pos_counts.get("NNPS", 0),
        "title_stop_ratio": sum(1 for t in tokens if t.lower() in nltk.corpus.stopwords.words("english")) / max(len(tokens), 1),
        "title_exclamations": title.count("!"),
        "title_questions": title.count("?"),
        "title_all_caps": sum(1 for t in tokens if t.isupper() and len(t) > 1),
    }
```

For the psychological features, LIWC is commercial. Reasonable open-source alternatives: VADER for sentiment, Empath for psychological categories, or building a small custom dictionary of certainty/tentative/negation words.

---

## 5. McInnes et al. (2018) — UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction

**Full reference:** McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*. https://arxiv.org/abs/1802.03426

---

### Relevant excerpt

> "UMAP (Uniform Manifold Approximation and Projection) is a novel manifold learning technique for dimension reduction. UMAP is constructed from a theoretical framework based in Riemannian geometry and algebraic topology. The result is a practical scalable algorithm that applies to real world data. The UMAP algorithm is competitive with t-SNE for visualization quality, and arguably preserves more of the global structure with superior run time performance. Furthermore, UMAP has no computational restrictions on embedding dimension, making it viable as a general purpose dimension reduction technique for machine learning."

---

### What this section says

UMAP is a dimensionality reduction method. It works by building a fuzzy topological representation of the high-dimensional data and finding a low-dimensional projection that preserves that structure. The main advantage over t-SNE (the most common alternative) is that UMAP preserves more of the global structure of the data — clusters that are far apart in the original space tend to stay far apart in the 2D projection — and it is faster. It also supports transforming new data points without retraining.

---

### Relation to our project

We use UMAP exclusively for visualization (step 03_vis_an.py). The goal is to produce a 2D scatter plot of all articles in our dataset, where each point is colored by its radicalization score (cosine distance from the Center article). If the feature engineering works, we expect to see a gradient structure: articles near the neutral center in one region, and more radicalized articles spreading outward.

**Why UMAP over t-SNE:**
- With TF-IDF vectors, the input space is very high-dimensional and sparse. UMAP handles this well.
- We care about the global structure (are radicalized articles systematically far from neutral ones?) — t-SNE would distort this.
- Faster on larger datasets.

**No meaningful "results" to compare:** this paper is a methods paper, not a study of hyperpartisanship. We cite it only to properly reference the visualization algorithm.

**How to use it:**
```python
import umap
import matplotlib.pyplot as plt

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_features)  # X_features: our full feature matrix

scatter = plt.scatter(
    embedding[:, 0], embedding[:, 1],
    c=y_scores,           # cosine distance target [0.0, 1.0]
    cmap="YlOrRd",
    alpha=0.6, s=8
)
plt.colorbar(scatter, label="Radicalization score")
plt.title("UMAP projection of AllSides articles")
plt.savefig("../Risultati/umap_articles.png", dpi=150, bbox_inches="tight")
```

Key hyperparameters to try: `n_neighbors` (15–50), `min_dist` (0.05–0.3). Lower `min_dist` makes clusters more compact; higher `n_neighbors` captures more global structure.

---

*Notes compiled by Gabriele — June 2026*
