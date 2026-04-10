# MST Harness

This is a personal project.

I built it because I wanted my local agent setup to feel less hacked together and more like an actual working system. I kept ending up with half-reusable scripts, random experiments, and too much glue code, so this became the place where I pulled it all into one harness.

The main idea is pretty simple:
- run different agents behind one API,
- keep outputs structured,
- store useful history and traces,
- and make it easy to bolt on memory, search, RAG, and enrichment without rebuilding everything every time.

It is not meant to be a giant framework or a polished product. It is mostly me trying to build a practical base I can keep extending as my workflows change.

At the moment this repo is the part where I experiment with:
- agent orchestration,
- web search and enrichment,
- retrieval and memory,
- graph + vector storage,
- and generally making local models more useful in day-to-day work.

I made it for myself first, but I like having it public because it doubles as a record of how I am thinking about agent systems when I am actually building them instead of just talking about them.

