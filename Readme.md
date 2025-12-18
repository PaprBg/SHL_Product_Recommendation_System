# SHL Assessment Recommendation System

An end-to-end NLP-powered semantic recommendation system that recommends suitable SHL assessments based on job descriptions or natural language queries using Retrieval-Augmented Generation (RAG) architecture.

---

## Project Objective

To build an intelligent system that:
- Accepts unstructured job descriptions or natural language queries
- Understands semantic intent using NLP
- Retrieves the most relevant SHL assessments using vector similarity
- Returns ranked recommendations with comprehensive metadata

---

## Key Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **Scrapy** | Web scraping framework |
| **Pandas / NumPy** | Data cleaning and manipulation |
| **Sentence Transformers** | Text embeddings (all-MiniLM-L6-v2) |
| **FAISS** | Vector similarity search |
| **Flask** | Backend REST API |
| **Streamlit** | Interactive frontend UI |
| **Hugging Face API** | Embedding inference (deployment-ready) |

---

## System Architecture

```
┌─────────────────┐
│  User Query     │
│ "Entry-level    │
│  accounting"    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         Streamlit Frontend UI               │
│  - Query Input                              │
│  - Results Display                          │
└────────┬────────────────────────────────────┘
         │ HTTP POST
         ▼
┌─────────────────────────────────────────────┐
│           Flask Backend API                 │
│  - Query Processing                         │
│  - Embedding Generation                     │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│      Sentence Transformer (MiniLM)          │
│  - Convert query to 384-dim vector          │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         FAISS Vector Index                  │
│  - Cosine similarity search                 │
│  - Retrieve top-k matches                   │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│      Ranked Assessment Results              │
│  - Product name, score, metadata            │
└─────────────────────────────────────────────┘
```

---

## Step-by-Step Workflow

### 1️⃣ Data Collection (Web Scraping)

**Objective:** Collect SHL assessment data from their product catalog

**Tools:** Scrapy web scraping framework

**Process:**
- Scraped SHL product catalog (https://www.shl.com/products/product-catalog/) using custom Scrapy spider
- Extracted structured data for each assessment

**Fields Extracted:**
- Product Name
- Product URL
- Test Type
- Job Levels
- Remote Testing Availability
- Product Description

📁 **Output:** `products.csv`

---

### 2️⃣ Data Cleaning & Normalization

**Objective:** Clean and standardize raw scraped data

**Tools:** Pandas, Python

**Cleaning Steps:**
- ✅ Removed HTML tags and formatting artifacts
- ✅ Normalized `test_type` values (standardized naming)
- ✅ Normalized `job_levels` (consistent format)
- ✅ Standardized `remote_testing_available` (Yes/No)
- ✅ Cleaned long descriptions (removed special characters)
- ✅ Handled missing values
- ✅ Removed duplicate entries

📁 **Output:** `products_cleaned.csv`and `products_cleaned.json`

---

### 3️⃣ Text → Embeddings (NLP)

**Objective:** Convert text to semantic vector representations

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- Lightweight and fast
- 384-dimensional embeddings
- Optimized for semantic similarity

**Process:**
- Generated embeddings for each assessment
- Normalized vectors for cosine similarity

📁 **Output:** Embedding vectors (384-dim) for each product

**Script:**
```python
python scripts/generate_embeddings.py
```

---

### 4️⃣ Vector Store Creation (FAISS)

**Objective:** Build efficient similarity search index

**Tools:** FAISS (Facebook AI Similarity Search)

**Process:**
- Created FAISS index with all product embeddings
- Used Flat L2 index for exact nearest neighbor search
- Maintained index-to-product mapping (pickle file)

📁 **Outputs:**
- `vector.index` - FAISS index file
- `vector_texts.pkl` - Index-to-metadata mapping

---

### 5️⃣ Semantic Search / Retrieval Layer

**Objective:** Enable semantic query-based retrieval

**File:** `search_assessments.py`

**Functions:**

1. **`embed_query(query)`**
   - Converts user query to vector embedding
   - Uses same model as indexing

2. **`find_assessments(query, k=5)`**
   - Searches FAISS index for similar vectors
   - Returns top-k ranked results
   - Converts L2 distance to similarity score:
     ```python
     similarity = 1 / (1 + distance)
     ```
---

### 6️⃣ Evaluation using Labeled Dataset

**Objective:** Quantify system performance

**File:** `evaluation.py`

**Dataset:**
- Labeled queries with relevant assessment URLs
- Format: `{query, [relevant_urls]}`

**Metrics Computed:**

| Metric | Description | Formula |
|--------|-------------|---------|
| **Precision@K** | % of retrieved items that are relevant | TP / (TP + FP) |
| **Recall@K** | % of relevant items that are retrieved | TP / (TP + FN) |
| **MRR** | Mean Reciprocal Rank | 1 / rank of first relevant item |

**Results:**

Precision@5: 0.82
Recall@5: 0.76
MRR: 0.89

---

### 7️⃣ Flask Backend API

**Objective:** Expose recommendation engine as REST API

**File:** `app.py`

**Endpoints:**

#### `POST /api/recommend`

**Request:**
```json
{
  "query": "Entry-level accounting assessment with remote testing",
  "k": 5
}
```

**Response:**
```json
{
    "query": "Entry level accounting assessment with remote testing",
    "results": [
        {
            "Job_levels": "['Entry-Level', 'Graduate', 'Manager', 'Mid-Professional', 'Professional Individual Contributor', 'Supervisor']",
            "Score": 0.566,
            "product_description": "Multi-choice test that measures the ability to post journal entries, classify items into assets and liabilities, analyze financial statements and calculate financial ratios.",
            "product_name": "Financial Accounting (New)",
            "product_url": "https://www.shl.com/products/product-catalog/view/financial-accounting-new/",
            "remote_testing_available": "Yes",
            "test_type": "['Knowledge & Skills']"
        },
        {
            "Job_levels": "['Entry-Level', 'Graduate', 'Mid-Professional', 'Professional Individual Contributor']",
            "Score": 0.51,
            "product_description": "Multiple-choice test that measures the knowledge of processing payables and vendor invoices, and the posting of journal entries.",
            "product_name": "Accounts Payable (New)",
            "product_url": "https://www.shl.com/products/product-catalog/view/accounts-payable-new/",
            "remote_testing_available": "Yes",
            "test_type": "['Knowledge & Skills']"
        },
        {
            "Job_levels": "['Entry-Level', 'Graduate', 'Mid-Professional', 'Professional Individual Contributor']",
            "Score": 0.5093,
            "product_description": "Multiple-choice test that measures the knowledge of processing receivables and invoices.",
            "product_name": "Accounts Receivable (New)",
            "product_url": "https://www.shl.com/products/product-catalog/view/accounts-receivable-new/",
            "remote_testing_available": "Yes",
            "test_type": "['Knowledge & Skills']"
        },
        {
            "Job_levels": "['Entry-Level']",
            "Score": 0.5034,
            "product_description": "This assessment measures ability to efficiently compare information and detect errors. The test taker is required to examine four pairs of numbers and select the set of numbers that are notidentical.",
            "product_name": "Visual Comparison - UK",
            "product_url": "https://www.shl.com/products/product-catalog/view/visual-comparison-uk/",
            "remote_testing_available": "Yes",
            "test_type": "['Knowledge & Skills']"
        },
        {
            "Job_levels": "['General Population', 'Graduate', 'Manager', 'Mid-Professional', 'Professional Individual Contributor', 'Supervisor', 'Director', 'Entry-Level', 'Executive', 'Front Line Manager']",
            "Score": 0.5006,
            "product_description": "Using the Apta™ Architecture to focus on the relevant competency behaviors in the Universal Competency Framework, SHL developed the RemoteWorkQ to measure self-reported behavioral tendencies in competency areas that are important to performing effectively in remote work environments across three competency areas: o Work Relationships o Work Habits o Self-Development & Well-Being The RemoteWorkQ is intended for use across job families and levels for which working in a remote environment is important for the role. This report is designed to help you be more successful in a remote working environment by providing: o Insights into your identified strengths and potential risks for working remotely o Individualized coaching tips on how you can use your identified strengths to overcome risks",
            "product_name": "RemoteWorkQ Participant Report",
            "product_url": "https://www.shl.com/products/product-catalog/view/remoteworkq-participant-report/",
            "remote_testing_available": "Yes",
            "test_type": "['Competencies']"
        }
    ]
}
```

---

### 8️⃣ Streamlit Frontend

**Objective:** User-friendly web interface

**File:** `streamlit_app.py`

**Features:**
- 🔍 Text area for job descriptions or queries
- 📊 Slider to adjust number of results (1-10)
- 📋 Detailed assessment cards with metadata
- 🔗 Direct links to assessment pages
- ⚡ Real-time search with loading indicator

---

## 📈 Future Enhancements

- [ ] **Hybrid Search**: Combine semantic + keyword search
- [ ] **Query Expansion**: Use LLM to expand user queries
- [ ] **User Feedback Loop**: Learn from user selections
- [ ] **Multi-language Support**: Embeddings for multiple languages
- [ ] **Advanced Filters**: Industry, duration, difficulty
- [ ] **A/B Testing**: Compare different embedding models
- [ ] **Caching Layer**: Redis for faster repeated queries
- [ ] **Analytics Dashboard**: Usage statistics and insights

---
