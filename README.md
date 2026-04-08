# My LLM Harness

This is a personal project I built to make my local LLM workflow feel less scrappy and more reliable.

I wanted one place where I could:
- run different agents behind a simple API,
- keep outputs structured so they are actually reusable,
- save runs and observations to NocoDB,
- and plug in retrieval/memory pieces without rewriting everything each time.

The goal is not a huge framework. It is a practical harness I can evolve as my agent setup changes.

Right now this project is focused on:
- FastAPI endpoints for running agents,
- a shared output schema for consistent responses,
- NocoDB logging for runs, outputs, and observations,
- and modular components for chunking, embeddings, graph, memory, and RAG.

I made this mostly for myself, but also as a clean baseline I can reuse across future experiments and client-style workflows.

#### Docker Details
```YAML
mst-ag-harness:
  image: ghcr.io/blameturner/mst-harness:latest
  container_name: mst-ag-harness
  environment:
    - MODEL_PORT_START=8080
    - MODEL_PORT_END=8090
    - EMBEDDER_PORT=8083
    - RERANKER_PORT=8084
    - CHROMA_URL=http://chroma:8000
    - FALKORDB_HOST=falkordb
    - FALKORDB_PORT=6379
    - NOCODB_URL=http://nocodb:8080
    - NOCODB_TOKEN=your_token
    - NOCODB_BASE_ID=base_id
    - ENVIRONMENT=production
  ports:
    - "3800:3800"
  restart: unless-stopped
  networks:
    - mst-ag-01-network
```

