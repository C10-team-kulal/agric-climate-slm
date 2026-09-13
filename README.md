# agric-climate-slm

A retrieval-augmented, LoRA-fine-tuned small language model that answers
agricultural and climate-adaptation questions for smallholder farmers in sub-Saharan Africa.

## Dataset

Our corpus merges the competition-provided synthetic documents with real
sources: the [CGIAR GARDIAN document corpus](https://huggingface.co/datasets/CGIAR/gardian-cigi-ai-documents)
(agricultural research publications) and the FAO's open knowledge repository
(harvested via OAI-PMH from CGSpace/openknowledge.fao.org)

Sourced documents were cleaned to remove institutional-document noise —
citations, page furniture, DOIs/ISBNs, and licensing boilerplate — using
sentence-level filtering rather than a single inline regex, since boilerplate
often spans version numbers (e.g. "3.0 IGO.") that break naive pattern
matching. New `document_id`s were normalized to match the competition's
`doc_{topic-prefix}_{number}` convention, continuing numbering from the
competition's own IDs to avoid collisions.

Training Q&A pairs for sourced documents were generated via LLM (Groq,
`openai/gpt-oss-120b`), with every generated pair validated for placeholder
leakage, minimum length, and — where answers were later restyled for
consistency — exact preservation of any numeric figures, falling back to the
original text if a rewrite failed validation.

Final corpus: `data/documents.csv` (merged, deduplicated) and
`data/train_qa.csv` (merged training Q&A).

## Training Pipeline

**Retrieval:** Metadata filtering (`crop`, `agro_zone`, `topic`) narrows
candidates before semantic ranking via `sentence-transformers`
(`all-MiniLM-L6-v2`) and cosine similarity.

**Generation:** Base model `google/gemma-2-2b-it`, loaded in 4-bit
(`bitsandbytes` NF4). Fine-tuned with **LoRA** (rank 16, targeting attention
projection layers `q_proj/k_proj/v_proj/o_proj`) rather than full
fine-tuning, given compute constraints. Training examples use **retrieved**
(not gold) document context, so training conditions match test-time
conditions, since the test set provides no `document_id` to retrieve against.

**Hyperparameter search:** Tested epoch counts (3–6), few-shot example
counts (2–10), `max_new_tokens` (10–128), and generation strategy (greedy vs.
beam search with `num_beams=4`, `length_penalty=0.8`). Six few-shot examples
with heavy in-context repetition caused a metadata-leakage failure — the
model began hallucinating new `Crop:`/`Question:` headers after each answer. 
Training loss was monitored per run; a 787-row/10-epoch run
was manually interrupted after loss collapsed toward zero by step 60
(overfitting), motivating a return to a plateauing configuration.
Reducing to 4 well-chosen, terse examples and adding explicit stop conditions
resolved this. 

## Evaluation

Evaluated using **mean Levenshtein distance** against reference answers, on a
held-out 15% validation split (`random_state=42`), never used in training.

| Configuration | Mean Levenshtein |
|---|---|
| Initial retrieval-only baseline | 172.65 |
| Tuned extractive answer length | 98.94 |
| Retrieval-only baseline | 77.4 |
| LoRA, early config | 77.0 |
| LoRA, tuned generation config | 54.8 |
| LoRA, few-shot + data improvements | **47.8** |

Retrieval was separately validated with Recall@1 (~90%) and Recall@3 (~93%)
on the same validation split, confirming retrieval was not the primary
bottleneck once generation was tuned. A retrieval-only vs. fine-tuned
comparison is included in `scripts/lora_finetuning.ipynb` to isolate the
contribution of fine-tuning from retrieval and prompt engineering alone.

## Reproduction

Run in order:

1. `scripts/data_preparation.ipynb` — sources, cleans, and merges the corpus;
   generates and validates additional training Q&A pairs.
2. `scripts/retrieval_and_baseline.ipynb` — builds the retrieval pipeline,
   validates Recall@1/@3, and produces the retrieval-only baseline submission.
3. `scripts/lora_finetuning.ipynb` — fine-tunes the LoRA adapter, evaluates
   against the held-out validation set, and produces the final submission.

Dependencies: `pip install -r requirements.txt`. Requires a CUDA GPU
(developed on Kaggle T4 x2) for the fine-tuning step; retrieval and the
extractive baseline run on CPU. `GROQ_API_KEY` is required only for
regenerating training Q&A pairs from scratch — not required to reproduce the
final submission from the provided `data/train_qa.csv`.

Fine-tuned adapter available at:
[Hugging Face](https://huggingface.co/ini-ekaette/agric-climate-lora-adapter)

## Appendix

**Team:** Kulal

**Contributors:**
- Ekaette Samuel 
- Adepitan Oluwatosin

**Mentors:** Taiwo Samuel

**Program:** TRI AI Saturdays Lagos, Cohort 10
