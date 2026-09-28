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
