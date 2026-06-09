# Entity-Relation Guided Focused Retrieval for Vietnamese Medical RAG

This repository accompanies a research manuscript on Vietnamese medical question answering with graph-based retrieval-augmented generation (RAG). The code is built on top of [LightRAG](https://github.com/HKUDS/LightRAG) and adds a focused retrieval pipeline for Vietnamese medical content.

The repository contains:

- A Vietnamese medical corpus and derived LightRAG knowledge-graph artifacts.
- A focused retrieval mode that combines lexical entity anchoring, semantic reranking, 1-hop relation scoring, and chunk-level reranking.
- Scripts for querying, evaluating, and reproducing the paper experiments.
- API and optional WebUI components inherited from LightRAG.

## Paper Status

Paper metadata will be updated here after the manuscript is published or assigned an arXiv/DOI link.

Current citation placeholder:

```bibtex
@misc{nguyentien2026focusedmedicalrag,
  title  = {Entity-Relation Guided Focused Retrieval for Vietnamese Medical Question Answering},
  author = {Nguyen-Tien, Ha and Vu-Duy, Linh and Nguyen-Thi, Minh-Huyen},
  year   = {2026},
  note   = {Manuscript in preparation},
  url    = {https://github.com/Jmhzbmcn2/Graduation-Thesis}
}
```

## Method Overview

The proposed method extends LightRAG with a `focused` retrieval mode designed for Vietnamese medical questions.

Key components:

- Semantic Markdown chunking that preserves article title, section heading, and source URL context.
- BM25-based lexical recall over entity names for robust matching of Vietnamese drug names, diseases, symptoms, anatomy terms, and herbal remedies.
- Name-only semantic reranking for final anchor-node selection.
- 1-hop relation scoring with a joint query-anchor and query-relation similarity score.
- Vietnamese-domain reranking over the final candidate chunks before answer generation.

The main comparison modes are:

- `focused`: proposed retrieval pipeline.
- `hybrid`: LightRAG graph-based hybrid retrieval baseline.
- `mix`: LightRAG mixed graph and vector retrieval baseline.
- `naive`: dense vector retrieval baseline.

## Repository Layout

```text
lightrag/              Core LightRAG package and modified retrieval logic
lightrag/api/          FastAPI server and query endpoints
lightrag_webui/        Optional React + Vite WebUI
medical_rag_v6/        Main Vietnamese medical LightRAG index artifacts
vietmed_crawled/       Crawled Vietnamese medical text corpus
reproduce/             Reproduction and batch-evaluation helpers
testcase/              Evaluation scripts and result workbooks
paper/                 Paper-related drafts and experiment text
scripts/               Utility scripts for corpus and quality checks
docs/                  LightRAG deployment and implementation notes
```

## Data and Artifacts

The current indexed corpus is derived from Vietnamese medical articles collected from YouMed. The local index in `medical_rag_v6/` includes:

- `graph_chunk_entity_relation.graphml`: knowledge graph exported by LightRAG.
- `kv_store_doc_status.json`: document processing status.
- `kv_store_text_chunks.json`: chunk store.
- `vdb_chunks.json`, `vdb_entities.json`, `vdb_relationships.json`: vector stores.
- `bm25_chunks.pkl`, `bm25_entities.pkl`: BM25 indexes used by focused retrieval.
- `name_list.json`, `name_to_idx.json`, `name_matrix_normed.npy`: name-only entity matching artifacts.

Current local corpus/index snapshot:

- Crawled text files: 1,932.
- Processed documents in `medical_rag_v6`: 1,931.
- Pending documents: 5.
- Failed documents: 1.
- Knowledge-graph nodes reported in the paper draft: 49,274.
- Knowledge-graph edges reported in the paper draft: 106,666.

Before public release, verify redistribution rights for raw text and derived artifacts that may contain source article content.

## Installation

Python 3.10 or newer is required.

```powershell
git clone https://github.com/Jmhzbmcn2/Graduation-Thesis.git
cd Graduation-Thesis

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[api]"
```

Install the embedding model used by the current index:

```powershell
ollama pull embeddinggemma:300m
```

For the optional WebUI:

```powershell
cd lightrag_webui
bun install
bun run dev
```

## Configuration

Create a local environment file from the example:

```powershell
Copy-Item env.example .env
```

Typical services used in the experiments:

- Embedding model: `embeddinggemma:300m` via Ollama.
- Answer generator: `Qwen2.5-14B-Instruct-AWQ` via a local OpenAI-compatible vLLM endpoint.
- Evaluation judge: `qwen/qwen3-30b-a3b-instruct-2507` via OpenRouter or an equivalent OpenAI-compatible endpoint.
- Reranker: `AITeamVN/Vietnamese_Reranker` served as a local reranking endpoint.

Do not commit `.env`, API keys, local logs, or private connection strings.

## Quickstart

Start Ollama:

```powershell
ollama serve
```

Start the LightRAG API against the main index:

```powershell
lightrag-server --working-dir ./medical_rag_v6
```

The API documentation is available at:

```text
http://localhost:9621/docs
```

Example focused query:

```powershell
curl -X POST http://localhost:9621/query `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"Paracetamol co tac dung phu gi?\",\"mode\":\"focused\"}"
```

## Reproducing Experiments

The evaluation scripts assume the LightRAG API is running at `http://localhost:9621` and the required LLM, embedding, and reranker endpoints are available.

Representative scripts:

```powershell
python testcase/eval_ragas_hybrid_mix_beam.py
python testcase/eval_lightrag_paper.py
python testcase/calculate_icc.py
```

The paper draft compares retrieval and answer quality using RAGAS and pairwise LLM-as-judge evaluation. Evaluation workbooks and intermediate outputs are stored under `testcase/`, `outputs/`, and `tmp/` in the local workspace.

## Authors

**Ha Nguyen-Tien**

- Affiliation: Faculty of Engineering Technology, Hung Vuong University, Phu Tho Province, Viet Nam.
- ORCID: [0009-0001-6190-9635](https://orcid.org/0009-0001-6190-9635).
- Email: nguyentienha@hvu.edu.vn.
- Contributions: Conceptualization, Methodology, Software, Investigation, Writing - original draft, Writing - review and editing.

**Linh Vu-Duy**

- Affiliation: Faculty of Mathematics, Mechanics and Informatics, University of Science, Vietnam National University, Hanoi, Viet Nam.
- Email: vuduylinh_t67@hus.edu.vn.
- Contributions: Methodology, Software, Data curation, Formal analysis, Validation, Writing - review and editing.

**Minh-Huyen Nguyen-Thi**

- Affiliation: Faculty of Mathematics, Mechanics and Informatics, University of Science, Vietnam National University, Hanoi, Viet Nam.
- Email: huyenntm@hus.edu.vn.
- Contributions: Supervision, Methodology, Validation, Writing - review and editing.

Corresponding author: Ha Nguyen-Tien.

## License

The codebase inherits the MIT license from the LightRAG project. Dataset redistribution and derived medical-content artifacts may be subject to the terms of the original data sources; verify these terms before public release.

## Medical Disclaimer

This repository is intended for research on retrieval-augmented generation. It is not a medical device and must not be used as a substitute for professional medical advice, diagnosis, or treatment.
