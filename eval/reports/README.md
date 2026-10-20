# Evaluation runs

The runs quoted in the main README, kept as they came out of `docurag eval` and `maintagent eval`
on 2026-10-20 (`qwen3.5:9b` with thinking off, `bge-m3`, Ollama 0.40.6, RTX 5060 Ti 16 GB,
PostgreSQL 17.11 + pgvector 0.8.6, no reranker, 36 questions of `examples/eval/questions.jsonl`).

| File | Command |
|---|---|
| `2026-10-20_vector.json` | `docurag eval --dataset examples/eval/questions.jsonl --groups maintenance --label vector --mode vector --no-rerank` |
| `2026-10-20_hybrid.json` | `docurag eval --dataset examples/eval/questions.jsonl --groups maintenance --label hybrid --no-rerank --baseline <vector run>` |
| `2026-10-20_agent.json` | `maintagent eval --scenarios examples/agent/scenarios.jsonl --approve-as sup-7 --label qwen3.5-9b` |

Each JSON file holds every question, the answer as written, its citations, retrieval scores,
tokens and timings.

## Answers graded by hand

Each answer was read against the reference answer of the question: right when it gives the
reference's facts (wording aside) with citations to the right documents.

| Question | Category | vector | hybrid | Note |
|---|---|:---:|:---:|---|
| q01–q08, q10 | version, code, procedure, safety, kb, multi-doc | ✓ | ✓ | Revision C's values every time (49 N·m, 2 000 h, 12 g) |
| q09 | records | ✓ | ✓ | vector: right cause and value, plus a claim that thermal expansion made it worse, which the record does not say |
| q13–q26, q28, q29 | table, code, procedure, safety, version, records, multi-doc, kb | ✓ | ✓ | |
| q27 | multi-doc | ✗ | ✗ | Quotes 18,400 h and 16,000 h correctly, answers "no" |
| q30 | acl (methods user) | ✓ | ✓ | Answered from the restricted note |
| q31 | acl (maintenance user) | ✓ | ✓ | Refused: the only source is not readable by this user |
| q32 | scanned | ✗ | ✗ | Refused: the answer is on the scanned page, not indexed without OCR |
| q11, q12, q33–q36 | unanswerable | ✓ | ✓ | All refused, including q35 (maximum discharge pressure) |

Answerable questions: 27 of 29 right in both runs. Questions to refuse: 7 of 7 refused.
