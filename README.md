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
  extra_hosts:
    - "host.docker.internal:host-gateway"
  environment:
    - MODEL_HOSTS=http://mst-ag-reasoner-gemma4-e4b:8080,http://mst-ag-fast-gemma4-e2b:8081,http://mst-ag-coder-qwen-14b:8082
    - EMBEDDER_URL=http://mst-ag-embedder-nomic-embed-v1.5:8083
    - RERANKER_URL=http://mst-ag-reranker-bge-reranker-v2-m3:8084
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
    - default
```

Model containers are addressed explicitly via `MODEL_HOSTS` (comma-separated
list of base URLs). On startup the harness calls `/v1/models` on each to
discover the model ID and registers it under a friendly name. If
`MODEL_HOSTS` is unset the harness falls back to scanning
`MODEL_PORT_START`–`MODEL_PORT_END` on the Docker gateway, which is handy
for running the harness on the host against containerised models.

