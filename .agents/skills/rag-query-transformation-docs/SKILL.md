---
name: rag-query-transformation-docs
description: >-
  Standardized guide and procedure for creating and maintaining educational documentation
  on RAG Query Transformation techniques with intuitive explanations, before-and-after examples,
  and Mermaid flow diagrams.
---

# RAG Query Transformation Documentation Standard

This skill defines the pedagogical structure and formatting standards for explaining Advanced RAG Query Transformation techniques.

## Document Structure Standards

Each documentation module must include:
1. **Concept at a Glance**: 2-3 sentence intuitive explanation using everyday analogies.
2. **The Problem it Solves**: Contrast Naive RAG failure modes with this approach.
3. **Visual Architecture Diagram**: A Mermaid diagram illustrating the step-by-step query flow.
4. **Concrete Examples**:
   - Real-world user input
   - Transformed output
   - Why the transformed query yields superior vector embeddings or search hits.
5. **Implementation Blueprint**: Clean, modern Python code using LangChain / OpenAI SDK / LangGraph.
6. **Key Trade-offs & Production Tips**: Latency vs. recall, token costs, caching strategies.

## Directory Layout in `docs/`
- `00_problem_with_naive_rag.md`
- `01_query_rewriting.md`
- `02_step_back_prompting.md`
- `03_sub_query_decomposition.md`
- `04_multi_query_generation.md`
- `05_query_routing.md`
- `06_query_compression.md`
- `07_contextual_query_enhancement.md`
- `08_query_translation_to_filters.md`
- `09_agentic_rag_and_frameworks.md`
- `README.md` (Master index and learning path)
