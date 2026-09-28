# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Project scaffold: FastAPI app, health endpoint, JSON logging, CI, Docker.
- Official BOE XML of the XIX consultancy/IT and IV metal agreements, with a catalog.
- Ingestion: split agreements into citable chunks (article, provision or annex) and store
  them in PostgreSQL (`agreements`, `chunks`); `scripts/ingest.py`, idempotent.
- Hybrid search: Spanish full-text + local multilingual embeddings in pgvector, fused with
  RRF; `GET /search`. Search tests run against PostgreSQL in CI.
- `POST /ask`: answer from the retrieved articles with verified citations and official BOE
  links, or an honest "not found".
- Evaluation (`make eval`): 30 handwritten questions + 5 trick questions; retrieval hit@5 and
  MRR per search mode, answer accuracy, wrong answers and refusals; results in `evals/results/`.
- Demo page at `/` and the README GIF, recorded with `scripts/record_demo.py`.
- `docker compose up` works from scratch: the image ships the BOE data and loads it on the
  first start (`scripts.ingest --if-empty`); the embedding model is kept in a volume.
