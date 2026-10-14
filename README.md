# Screenshotmaxxing

**AI-powered semantic search for personal screenshots.**

Aims to make screenshots easier to find using natural-language queries instead of manually browsing through folders.

## How It Works

- **CLIP:** Converts screenshots and text queries into compatible embeddings.
- **FAISS:** Performs similarity search to retrieve relevant screenshots efficiently.
- **OCR (Tesseract):** Extracts text from screenshots for future text-aware retrieval.

## Current Features

- Image embedding using a pretrained CLIP model.
- Screenshot indexing with FAISS.
- Natural-language query-based semantic search.
- Ranked search results with similarity scores.
- Command-line interface for indexing and searching.

## Tech Stack

Python · PyTorch · Hugging Face Transformers · CLIP · FAISS · Tesseract OCR

## Roadmap

- [ ] Integrate OCR text into screenshot retrieval and ranking.
- [ ] Combine visual similarity with text relevance.
- [ ] Evaluate retrieval quality using Recall@K, Precision@K and MRR.
- [ ] Improve search accuracy and performance.
- [ ] Develop a Windows desktop interface or executable.

## Current Status

Command-line proof of concept. Visual-semantic indexing and retrieval are implemented; OCR-aware ranking and desktop integration are planned.

## Running Locally

1. Install the dependencies.
2. Place screenshots in the `screenshots/` directory.
3. Run `indexer.py` to build the search index.
4. Run `search.py` to search using natural-language queries.

*Note: The project uses pretrained models and currently runs locally. Keep personal screenshots and sensitive information out of public repositories.*
