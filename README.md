# The Inbox Conductor

**Semantic Email Search with Faceted Navigation** - INFO 202 Final Project

A web-based semantic retrieval system for personal email archives that combines RAG (Retrieval-Augmented Generation) with faceted metadata filtering. Built using the Enron QA dataset.

## Features

- **Semantic Search**: Solve the vocabulary problem with embeddings
- **Faceted Navigation**: Filter emails by sender, folder, and date
- **RAG Q&A**: Ask questions and get AI-synthesized answers with source citations
- **RAG Evaluation**: Test system performance with ground truth Q&A pairs, using Claude as a judge for answer quality

## Tech Stack

- **Vector DB**: ChromaDB for semantic search
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Claude 3.5 Haiku via Anthropic API
- **UI**: Streamlit
- **Dataset**: Enron QA (15k emails, 500 Q&A pairs)

## Setup Instructions

### 1. Clone and Navigate

```bash
cd /Users/rishbhajain/Documents/202
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

Or for Streamlit deployment, create `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "your_api_key_here"
```

### 5. Prepare Data (Run Once)

```bash
python scripts/01_prepare_data.py
```

This downloads the Enron dataset, parses emails, and creates a 15k subset.

### 6. Build Index (Run Once)

```bash
python scripts/02_build_index.py
```

This generates embeddings and builds the ChromaDB vector database (~25 min on CPU).

### 7. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Project Structure

```
202/
├── data/
│   ├── processed/           # Parsed emails and Q&A pairs
│   └── chroma_db/          # Vector database
├── src/
│   ├── config.py           # Centralized settings
│   ├── data_loader.py      # Dataset loading
│   ├── email_parser.py     # Email parsing logic
│   ├── embedder.py         # Embedding generation
│   ├── indexer.py          # ChromaDB indexing
│   ├── search.py           # Search logic
│   ├── rag.py              # RAG implementation
│   └── evaluator.py        # Evaluation metrics
├── scripts/
│   ├── 01_prepare_data.py  # Data preparation
│   ├── 02_build_index.py   # Index building
│   └── 03_evaluate.py      # Evaluation metrics
├── app.py                  # Main Streamlit app
└── requirements.txt
```

## Usage

### Basic Search

1. Enter a natural language query in the search box
2. Use faceted filters to narrow results by sender, folder, or date
3. View relevance scores (cosine similarity) for each result

### RAG Q&A

1. Switch to the "Q&A" tab
2. Enter a question in natural language
3. The system retrieves relevant emails and generates an answer
4. View source emails with relevance scores

### Evaluation

Navigate to the "Evaluation" tab to test RAG performance:
- **Retrieval Metrics**: Recall@K, MRR, Precision@1 on ground truth Q&A pairs
- **Answer Quality**: Claude-as-Judge scoring (0-10) comparing RAG answers to ground truth
- **Interactive**: Adjust sample sizes and view score distributions
- **Detailed**: See top and bottom scoring answers with explanations

## INFO 202 Concepts Demonstrated

- **Week 4-5**: Faceted navigation and controlled vocabulary
- **Week 7-8**: Vocabulary problem, semantic similarity, word embeddings
- **Week 10**: Information seeking and foraging
- **Week 13**: RAG techniques, search evaluation (Precision@K, MRR)
- **Week 15**: Bias discussion (Enron dataset limitations)

## Development

To run evaluation metrics:

```bash
python scripts/03_evaluate.py
```

## Deployment

To deploy to Streamlit Cloud:

1. Push to GitHub (ChromaDB directory included, ~50MB)
2. Connect repository to Streamlit Cloud
3. Add `ANTHROPIC_API_KEY` in Streamlit Cloud secrets
4. Deploy!

## License

Educational project for INFO 202 - UC Berkeley School of Information
