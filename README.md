# Adaptive Directed Graph Retrieval (ADGR) for Graph-Based Retrieval-Augmented Generation

This repository accompanies the paper:

**Adaptive Directed Graph Retrieval for Graph-Based Retrieval-Augmented Generation**

The repository contains the implementation of **Adaptive Directed Graph Retrieval (ADGR)**, a query-aware GraphRAG framework designed to improve retrieval quality through semantically guided graph exploration. The codebase is built on top of [LightRAG](https://github.com/HKUDS/LightRAG) and extends its graph, vector, API, and evaluation workflows for Vietnamese medical retrieval-augmented generation.

The publicly released **ViMedGraph** dataset is available separately through Hugging Face:

https://huggingface.co/datasets/vuduylinh150804/vietmed-rag-dataset

## Repository Contents

This repository contains:

- The implementation of the proposed ADGR framework.
- Experimental scripts and evaluation pipelines.
- GraphRAG indexing and retrieval artifacts.
- Scripts for retrieval evaluation, RAGAS evaluation, LLM-as-Judge evaluation, and reproducibility.
- API and optional WebUI components inherited from LightRAG.

## Paper

Paper metadata will be updated here after the manuscript is published or assigned an arXiv/DOI link.

### Citation

```bibtex
@article{nguyentien2026adgr,
  title   = {Adaptive Directed Graph Retrieval for Graph-Based Retrieval-Augmented Generation},
  author  = {Ha Nguyen-Tien and Linh Vu-Duy and Minh-Huyen Nguyen-Thi and Toan Le-Ngoc and Dinh-Thai Kim},
  year    = {2026},
  note    = {Submitted manuscript}
}
```

## Method Overview

ADGR extends GraphRAG retrieval through query-aware and semantically guided graph exploration.

The framework consists of five major components:

- Overlap-aware hybrid anchor retrieval combining BM25 lexical retrieval and dense semantic retrieval.
- Query-aware semantic edge scoring for estimating relation relevance with respect to the query.
- Semantic edge filtering and controlled graph exploration to suppress noisy evidence propagation.
- Multi-source context fusion integrating graph-guided and semantic retrieval evidence.
- Cross-encoder reranking for final retrieval context construction.

The primary retrieval modes evaluated in the paper are:

- `ADGR`: the proposed retrieval framework. In this repository, the ADGR implementation is exposed through the API query mode `focused`.
- `hybrid`: LightRAG Hybrid baseline.
- `mix`: LightRAG Mix baseline.
- `naive`: dense vector retrieval baseline used in some evaluation scripts.

## Data Availability

### ViMedGraph Dataset

The ViMedGraph dataset used in this study is publicly available at:

https://huggingface.co/datasets/vuduylinh150804/vietmed-rag-dataset

### Source Code

The implementation of ADGR is publicly available at:

https://github.com/Jmhzbmcn2/Graduation-Thesis

## ViMedGraph Dataset Statistics

ViMedGraph is a Vietnamese medical GraphRAG resource released together with the ADGR framework.

### Corpus Statistics

- Source documents: 1,932
- Retrieval chunks: 17,236
- Entities: 49,274
- Relations: 106,666

### Graph Statistics

- Average node degree (undirected): 4.33
- Maximum node degree: 726
- Domain-Specific Entity Ratio (DSER): 93.63%
- Schema Correctness Ratio (SCR): 92.77%
- Largest Component Ratio (LCR): 90.59%
- Average Clustering Coefficient: 0.029

## Repository Layout

```text
lightrag/              Core LightRAG package and modified retrieval logic
lightrag/api/          FastAPI server and query endpoints
lightrag_webui/        Optional React + Vite WebUI
medical_rag_v6/        Main Vietnamese medical GraphRAG index artifacts
vietmed_crawled/       Crawled Vietnamese medical corpus
reproduce/             Reproduction and batch-evaluation helpers
testcase/              Evaluation scripts and result workbooks
paper/                 Paper drafts and experiment materials
scripts/               Utility scripts and reranker server helpers
docs/                  Deployment and implementation notes
```

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

Main models used in the paper configuration:

- Generation model: `Qwen2.5-14B-Instruct-AWQ`
- Knowledge graph extraction model: `Qwen2.5-14B-Instruct-AWQ`
- Embedding model: `embeddinggemma:300m`
- Reranker: `Qwen3-Reranker-0.6B`
- Evaluation judge: configured through `EVAL_LLM_MODEL` or the paper evaluation environment

Do not commit `.env`, API keys, local logs, or private connection strings.

## Quickstart

Start Ollama:

```powershell
ollama serve
```

Start the GraphRAG API against the main index:

```powershell
lightrag-server --working-dir ./medical_rag_v6
```

The API documentation is available at:

```text
http://localhost:9621/docs
```

Example ADGR query:

```powershell
curl -X POST http://localhost:9621/query `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"Paracetamol co tac dung phu gi?\",\"mode\":\"focused\"}"
```

`focused` is the current API mode name for the ADGR retrieval implementation in this repository.

## Reproducing Experiments

The evaluation scripts assume the LightRAG API is running at `http://localhost:9621` and the required LLM, embedding, and reranker endpoints are available.

Representative evaluation scripts:

```powershell
python testcase/eval_ragas_hybrid_mix_beam.py
python testcase/eval_lightrag_paper.py
python testcase/calculate_icc.py
```

The paper evaluates:

- Retrieval effectiveness with metrics such as Hit Rate, MRR, and nDCG.
- RAGAS answer and context quality.
- LLM-as-Judge answer preference and quality assessment.
- Retrieval efficiency and latency.

Evaluation workbooks and intermediate outputs are stored under `testcase/`, `outputs/`, and `tmp/` in the local workspace.

## Authors

### Ha Nguyen-Tien (Corresponding Author)

Faculty of Engineering Technology, Hung Vuong University, Phu Tho Province, Viet Nam

ORCID: https://orcid.org/0009-0001-6190-9635

Email: [nguyentienha@hvu.edu.vn](mailto:nguyentienha@hvu.edu.vn)

### Linh Vu-Duy

Faculty of Mathematics, Mechanics and Informatics, University of Science, Vietnam National University, Hanoi, Viet Nam

Email: [vuduylinh_t67@hus.edu.vn](mailto:vuduylinh_t67@hus.edu.vn)

### Minh-Huyen Nguyen-Thi

Faculty of Mathematics, Mechanics and Informatics, University of Science, Vietnam National University, Hanoi, Viet Nam

Email: [huyenntm@hus.edu.vn](mailto:huyenntm@hus.edu.vn)

### Toan Le-Ngoc

Faculty of Mathematics, Mechanics and Informatics, University of Science, Vietnam National University, Hanoi, Viet Nam

Email: [lengoctoan@hus.edu.vn](mailto:lengoctoan@hus.edu.vn)

### Dinh-Thai Kim

International School, Vietnam National University, Hanoi, Viet Nam

Email: [thaikd@vnu.edu.vn](mailto:thaikd@vnu.edu.vn)

## License

The codebase inherits the MIT license from the LightRAG project. Dataset redistribution and derived medical-content artifacts may be subject to the terms of the original data sources; verify these terms before public release.

## Medical Disclaimer

This repository is intended solely for research on retrieval-augmented generation and GraphRAG systems. It is not a medical device and must not be used as a substitute for professional medical advice, diagnosis, or treatment.
