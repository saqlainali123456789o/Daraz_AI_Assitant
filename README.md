# Daraz Support Operations Assistant

A professional Streamlit RAG chat application for answering Daraz customer-support and operations questions from a pre-built FAISS knowledge base.

## Architecture

```text
Pre-built PDFs
      ↓
Existing ingestion pipeline
      ↓
FAISS index + metadata
      ↓
Streamlit application
      ↓
Semantic retrieval
      ↓
Department filter
      ↓
Groq GPT-OSS 120B
      ↓
Grounded support answer
```

## Important

This Streamlit application does **not** process PDFs and does **not** rebuild or re-embed the knowledge base.

The deployed project must already contain:

```text
faiss_index/
├── daraz.index
├── metadata.json
└── manifest.json
```

The embedding model is loaded only to embed the user's query at search time. The stored document vectors are never regenerated.

## Required project structure

```text
daraz-support-assistant/
│
├── app.py
├── requirements.txt
├── README.md
│
└── faiss_index/
    ├── daraz.index
    ├── metadata.json
    └── manifest.json
```

## Streamlit Secret

In Streamlit Community Cloud, open:

**App → Settings → Secrets**

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

The application reads the key from `st.secrets`.

There is no API-key input box in the user interface.

## Model

The application uses:

```text
openai/gpt-oss-120b
```

through Groq.

## Knowledge sections

The sidebar provides:

- Returns
- Delivery
- Refunds
- Sellers
- Payments
- Customer Support
- All sections

Selecting a section restricts semantic retrieval to chunks belonging to that department.

## Retrieval

The application:

1. Embeds the user's question using the same embedding model used during ingestion.
2. Searches the existing FAISS index.
3. Retrieves a larger candidate pool.
4. Applies the selected department filter.
5. Keeps the top relevant chunks.
6. Sends only those retrieved chunks to the Groq model.
7. Generates a grounded response.

## Metadata

Each chunk is expected to contain fields such as:

```json
{
  "id": 0,
  "department": "returns",
  "source_file": "return_policy.pdf",
  "page": 1,
  "text": "..."
}
```

## Local testing

Install:

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

For local secrets, create:

```text
.streamlit/secrets.toml
```

with:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Do not commit `.streamlit/secrets.toml` to GitHub.

## GitHub deployment

Upload:

```text
app.py
requirements.txt
README.md
faiss_index/daraz.index
faiss_index/metadata.json
faiss_index/manifest.json
```

Then deploy the repository using Streamlit Community Cloud and add the Groq key through the Streamlit Secrets interface.

## Design principles

- No PDF upload in the production chat UI
- No visible API-key input
- Pre-built FAISS knowledge base
- Department-level retrieval filtering
- Grounded generation
- Source visibility
- Cached FAISS and embedding model
- Clean support-agent interface
- New-conversation control
- Helpful quick-start questions
