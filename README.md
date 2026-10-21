# Query Transformations for Advanced RAG Systems

> 🎮 **Interactive Demo:** Experience how query transformation techniques bridge the gap between user intent and vector search in this live visualization:  
> 👉 [**Query Transformation in RAG — Interactive Demo**](https://claude.ai/artifact/JJP6NTiKECWCe23o7L57P9)

---

## 💡 What is Query Transformation? (Simple Explainer)

Imagine walking into a massive university library and asking the clerk:
> *"Tell me about the big issue."*

If the clerk searches their computer database for the literal words *"big issue"*, you will get articles about oversized magazine publications. The clerk completely missed your intent because humans ask questions casually, make typos, use slang, or combine multiple thoughts into one sentence.

**This is the exact flaw of Naive RAG (Retrieval-Augmented Generation):**  
It blindly takes whatever messy text a user types, converts it into a vector embedding, and hopes the vector database finds the right documents. Most of the time, it retrieves irrelevant noise or partial answers.

**Query Transformation is the intelligent middle layer:**  
Before any database is queried, an LLM acts as an expert search librarian. It inspects your original question and reformulates it:
* **Rewrites** vague phrases into specific, technical vocabulary.
* **Decomposes** complicated multi-part questions into simpler sub-questions.
* **Steps back** to retrieve foundational background principles.
* **Extracts** hard metadata filters (like year, price, and category).
* **Removes** conversational fluff and filler words.

By transforming the question first, the retrieval engine receives queries precision-tuned for high-accuracy semantic and structured search.

---

## 📖 Complete Documentation in `docs/`

Each technique has a dedicated, easy-to-understand guide featuring analogies, real-world examples, and Mermaid diagrams:

* [**00 - The Problems with Naive RAG**](./docs/00_problem_with_naive_rag.md): Why raw vector similarity fails on ambiguous, conversational, or multifaceted questions.
* [**01 - Query Rewriting**](./docs/01_query_rewriting.md): Reformulates brief or vague queries into specific, detailed search queries.
* [**02 - Step-Back Prompting**](./docs/02_step_back_prompting.md): Generates broader high-level conceptual questions to retrieve foundational background context.
* [**03 - Sub-Query Decomposition**](./docs/03_sub_query_decomposition.md): Breaks complex multifaceted questions into 2-4 parallel sub-questions.
* [**04 - Multi-Query Generation**](./docs/04_multi_query_generation.md): Produces diverse phrasing variations and fuses results with Reciprocal Rank Fusion (RRF).
* [**05 - Query Routing**](./docs/05_query_routing.md): Directs queries to the most relevant specialized database or retriever.
* [**06 - Query Compression**](./docs/06_query_compression.md): Condenses verbose chat queries down to dense semantic keywords.
* [**07 - Contextual Query Enhancement**](./docs/07_contextual_query_enhancement.md): Personalizes queries using user role metadata and conversation history.
* [**08 - Query Translation to Filters**](./docs/08_query_translation_to_filters.md): Extracts semantic queries and structured metadata filters (dates, numbers, categories).
* [**09 - Agentic RAG & Frameworks**](./docs/09_agentic_rag_and_frameworks.md): Builds an autonomous ReAct Query Transformation Agent with LangGraph.

---

## 📂 Repository Structure

```text
├── 04_query_transformations.ipynb   # Executable Jupyter notebook with all 8 techniques
├── docs/                            # Educational documentation & diagrams for each technique
│   ├── 00_problem_with_naive_rag.md
│   ├── 01_query_rewriting.md
│   ├── 02_step_back_prompting.md
│   ├── 03_sub_query_decomposition.md
│   ├── 04_multi_query_generation.md
│   ├── 05_query_routing.md
│   ├── 06_query_compression.md
│   ├── 07_contextual_query_enhancement.md
│   ├── 08_query_translation_to_filters.md
│   ├── 09_agentic_rag_and_frameworks.md
│   └── README.md
├── requirements.txt                 # Project dependencies
├── .env.example                     # Example environment variables template
└── .gitignore                       # Ensures secrets and venvs are never committed
```

---

## ⚡ Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/d-hackmt/query-transformations.git
cd query-transformations
```

### 2. Set up virtual environment & install dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the notebook
Launch Jupyter or open `04_query_transformations.ipynb` in your IDE:
```bash
jupyter notebook 04_query_transformations.ipynb
```
