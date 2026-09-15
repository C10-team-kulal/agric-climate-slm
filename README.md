# 🌾 Domain-Specific SLM for Agriculture & Climate Advisory

A retrieval-augmented, LoRA-fine-tuned small language model that answers
agricultural and climate-adaptation questions for smallholder farmers in sub-Saharan Africa.
---
## 📊 Dataset
**Sources :** 

Corpus merges the competition-provided, synthetic documents with real
sources: the [CGIAR GARDIAN document corpus](https://huggingface.co/datasets/CGIAR/gardian-cigi-ai-documents)
(agricultural research publications) and the FAO's open knowledge repository
(harvested via OAI-PMH from CGSpace/openknowledge.fao.org)

**Key fields :**

- `document_id`, `title`, `topic`, `crop`, `agro_zone`, `text`, `origin`, `source_url`, `license`
- `QuestionId`, `question`, `topic`, `crop`, `agro_zone`, `document_id`, `reference_answer`

---
## ⚙️ Training Pipeline
**Data Processing Highlights :**

- ***Institutional-text cleaning:*** Removed citations, page furniture, DOIs/ISBNs, and
  licensing boilerplate from FAO/CGIAR source text using sentence-level filtering
  (a single inline regex proved unreliable against boilerplate spanning version numbers
  like "3.0 IGO.").
- ***Document ID normalization:*** Renamed sourced documents to match the competition's
  `doc_{topic-prefix}_{number}` convention, continuing numbering from the competition's
  own existing IDs to avoid collisions.
- ***LLM-generated Q&A validation:*** Every LLM-generated question/answer pair is checked
  for placeholder leakage (`<question>`, `SKIP`), minimum length, and — for restyled
  answers — exact preservation of any numeric figures in the original text, with an
  automatic fallback to the original answer if a rewrite fails validation.
- ***Unicode normalization:*** Typographic characters (non-breaking hyphens, curly quotes) normalized to plain ASCII equivalents, since evaluation is character-level.

**Retrieval :**

Metadata filtering (`crop`, `agro_zone`, `topic`) narrows candidates before semantic ranking via `sentence-transformers` (`all-MiniLM-L6-v2`) and cosine similarity - giving rise to embeddings (`data/data_06_kulal_embeddings.npy`) of the slm corpus (`data/data_01_documents.csv`).

**Hyperparameter Search :** 

Tested epoch counts (3–6), few-shot prompt-example counts (2–10), `max_new_tokens` (10–128), and generation strategy (greedy vs.beam search with `num_beams=4`, `length_penalty=0.8`). Six few-shot examples with heavy in-context repetition caused a metadata-leakage failure — the
model began hallucinating new `Crop:`/`Question:` headers after each answer. 
Training loss was monitored per run; optimum result came from 4 well-chosen, terse examples embedded into prompt function.

**Model/Design Choice :** 

Base model `google/gemma-2-2b-it`, loaded in 4-bit (`bitsandbytes` NF4), fine-tuned with **LoRA**.

---
## 🤖 Evaluation

Evaluated using **mean Levenshtein distance** against reference answers, on a
held-out 15% validation split (`random_state=42`), never used in training. Another test done using test questions from `data/data_07_test.csv` and **mean Levenshtein distance** was evaluated. Below are a few highlight results from sequential, improvement experiments :

| Configuration | Mean Levenshtein (val_split) | Mean Levenshtein (test_questions)
|---|---|---|
| Initial retrieval-only baseline | 172.65 | 181.0 |
| Retrieval-only baseline + tuned extractive answer length  | 112.13 | 77.4 |
| LoRA, early config | 105 | 98.0 |
| LoRA, tuned generation config | 90.0 | 54.8 |
| LoRA, few-shot-prompt-examples + data improvements + tuned generation config | 84.29 | **50.0** |
| LoRA, few-shot-prompt-examples + data improvements + tuned generation config | 83.83 | **47.8** |

Retrieval was separately validated with Recall@1 (~90%) and Recall@3 (~93%)
on the same validation split, confirming retrieval was not the primary
bottleneck once generation was tuned. A retrieval-only vs. fine-tuned
comparison is included in `scripts/scr_04_lora_finetuning.ipynb` to isolate the
contribution of fine-tuning from retrieval and prompt-engineering alone.

---
## 🛠️ Reproduction

### Steps To Reproducing Results :
1. **Run the notebooks in order:**
   
	-  `scripts/src_01_data_curation.ipynb` :

	Expands synthetic (baseline-format) corpus (**'data/data_01_documents.csv'**). This expansion (FAO and CGIAR sources) serves as source for **'data/data_03_kulal_documents.csv'**.

   -  `scripts/src_02_generate_qa.ipynb` :
	
	Preprocesses and cleans data obtained from `scripts/src_01_data_curation.ipynb`, further. Here, using an LLM (Groq), additional training Q&A pairs are generated and validated from **'data/data_03_kulal_documents.csv'** for training our model with resultant columns, following baseline format (**'data/data_02_train_qa.csv'**). This notebook produces **'data/data_04_fao_cgiar_checkpoint.jsonl'**. 

	- `scripts/src_03_data_preparation.ipynb` :

	Converts the json file (**data/data_04_fao_cgiar_checkpoint.jsonl**) into a dataframe and clean our 'train' data further by taking care of expressions and aligning and validating columns and data points to match baseline format (**'data/data_02_train_qa.csv'**). Resultant data is found in **'data/data_05_kulal_train_qa.csv'**.

	- `scripts/src_04_lora_finetuning.ipynb` :

	Builds the retrieval pipeline, validates Recall@1/@3, produces the retrieval-only baseline submission (**'data/data_08_baseline_submission.csv'**), fine-tunes the LoRA adapter, evaluates against the held-out validation set, and produces the final submission (**'data/data_09_lora_submission.csv'**).

2. **Installations:**
	- `scripts/src_05_requiremeants.txt` :	
	
	Lists needed dependencies. 
   
   Requires a CUDA GPU (developed on Kaggle T4 x2) for the fine-tuning step; retrieval and the extractive baseline run on CPU. `GROQ_API_KEY` is required only for generating additional training Q&A pairs.

---
## 📜 Appendix

This project was developed for the ***triAI Saturdays*** 'Agriculture & Climate' SLM Challenge, applying RAG and LoRA fine-tuning to a real-world agricultural advisory problem.

**Team :** Kulal

**Contributor(s) :**

- [Ekaette Samuel](www.linkedin.com/in/ini-ekaette) 
- [Adepitan Oluwatosin](https://github.com/Mosimilolu)

**Mentor(s) :** [Taiwo Samuel](linkedin.com/in/samuel-taiwo-oladipupo)

