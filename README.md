# JeffGPT

This is a personal project to set up a series of local LLMs that allow controlled and reasonably secure autonomous agentic paths, and also standard coding and chat functionality.

It is called JeffGPT because Jeff is my cat and he can be slow at times (like local LLMs on CPU. Also its just a great name for a cat.

Having a harness build was around better understanding tooling and LLM architecture, but to also use a basic local set up and building some grounding and RAG tooling for my use cases.

The harness also handles (legal) scraping and pathfinding to search for information and data to automatically find new information, but also for manual review.

The underlying set up this is deployed on:
- Dedicated Server (CPU Compute - which is a minor limitation) - Intel Xeon W-2145
- 128gb RAM
- 512gb SSD
- Ubuntu 22.04

The entire build is deployed in Docker Containers. The Frontend is a React/Typescript App, with a Hono/NodeJS Gateway to handle requests and types.

The repo for the Frontend is here https://github.com/blameturner/mst-harness-ui

The Docker Compose config for this container is:
```YAML
    mst-ag-harness:
        image: ghcr.io/blameturner/mst-harness:latest
        container_name: mst-ag-harness
        extra_hosts:
          - "host.docker.internal:host-gateway"
        environment:
          - MODEL_REASONER_URL=http://host.docker.internal:8080
          - MODEL_FAST_URL=http://host.docker.internal:8081
          - MODEL_CODER_URL=http://host.docker.internal:8082
          - MODEL_TOOL_URL=http://host.docker.internal:8085
          - EMBEDDER_URL=http://host.docker.internal:8083
          - RERANKER_URL=http://host.docker.internal:8084
          - CHROMA_URL=http://host.docker.internal:XXXX
          - FALKORDB_HOST=falkordb
          - FALKORDB_PORT=XXXX
          - NOCODB_URL=http://host.docker.internal:XXXX
          - NOCODB_TOKEN=XXXX
          - NOCODB_BASE_ID=XXXX
          - ENVIRONMENT=production
          - SEARXNG_URL=http://host.docker.internal:XXXX
          - ENRICHMENT_TOKEN_BUDGET=50000
          - ENRICHMENT_LOG_RETENTION_DAYS=30
        ports:
          - "{TAILSCALEIP}:XXXX:XXXX"
        restart: unless-stopped
        networks:
            - default
```

As with anything, the docker config is important. If you need more info just contact me.

The rest of the stack:
- LLMs: Uses Llama.cpp. Hardware determines the correct LCPP configuration. Read the Llama.cpp documentation and the .gguf file source for best configuration for your selected models
  - You can use any LLM, just configure in the docker-compose file, and download the image in Docker. You can add more, or less models, just check config.py
- RAG: ChromaDB for vector storage, harness runs through the embedder model to create vectors and retrieve before running the main model selected
- Reranker: Second pass for RAG results to improve relevance. 
- FalkorDB is the graph database storing entities and relationships. Harness calls the tool to extract these and writes as nodes and edges.
- NocoDB - operational DB. Handles bare multi-tenancy (in connection with Gateway Better-Auth), but handles tables that are used for the outputs, and the UI.
- SearXNG and Playwright are used for scraping and enrichment. UI configurable. Runs autonomously. Chat and Code can web search and display inline.

If you are setting this up, it is critical you protect the endpoints and server. I use a combination of Tailscale, and other security measures on the server itself plus optimistic monitoring and reporting. 