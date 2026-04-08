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
