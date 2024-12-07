# CRAG Task 1: Retrieval-Augmented Generation (RAG)

This repository contains code to run and evaluate models for CRAG Task 1, focusing on retrieval-augmented generation with various enhancements such as modified retrieval, chain-of-thought reasoning, and time-aware processing.

## Setup

1. Clone the repository and navigate to the course code folder:
   ```bash
   git clone https://github.com/HARRY1708/CRAG_Meta_KDD_cup.git
   cd course_code
   pip install -r requirements.txt


## Setup the HuggingFace Token and Open AI Key in the respective model file and evaluation model file

# Generation
   python generate.py --model_name <model_name> --llm_name "gpt-4o-mini" --vllm_server "https://api.openai.com/v1"

# Evaluation
   python evaluate.py --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" --model_name <model_name> --llm_name "gpt-4o-mini" --max_retries 10 --vllm_server "https://api.openai.com/v1" --is_server

<model_name> : "vanilla_baseline","rag_baseline","rag_baseline_chunked", "rag_baseline_embedding", "modified_rag","modified_rag_classification"

# Explanation of Workflow

The provided Python workflow is designed for a Retrieval-Augmented Generation (RAG) system that processes user queries, extracts relevant document chunks, retrieves and reranks information, and generates responses using a large language model (LLM). Below, we explain the flow and its components.

## Key Components

### 1. **Configuration Parameters**
   - Several configurable parameters are set to manage context lengths, chunk sizes, GPU utilization, and batch sizes:
     - **`NUM_CONTEXT_SENTENCES`**: Number of sentences to include in context for generating an answer.
     - **`CHUNK_SIZE`** and **`CHUNK_OVERLAP`**: Defines the size and overlap of document chunks to preserve semantic coherence.
     - **`VLLM_GPU_MEMORY_UTILIZATION`**: Sets the memory utilization limit for running the LLM on GPUs.

---

### 2. **Chunk Extraction**
   - **Purpose**: Extracts meaningful document chunks from raw HTML content.
   - **Implementation**:
     - Uses `BeautifulSoup` to parse HTML and extract clean text.
     - Splits text into chunks using the `RecursiveCharacterTextSplitter` from LangChain, ensuring overlapping for semantic continuity.
     - Tags chunks with metadata like "Page Last Modified" for time-sensitive queries.
   - **Parallelization**:
     - The process is distributed using `ray` to handle large datasets efficiently.

---

### 3. **Time Difference Calculation**
   - **Purpose**: Determines the difference between query timestamps and document modification times to prioritize temporally relevant information.
   - **Implementation**:
     - Parses timestamps in different formats using `dateutil.parser`.
     - Converts timestamps into specific time zones (e.g., Pacific Time, UTC).
     - Returns the difference in hours for ranking relevance.

---

### 4. **Retrieval and Reranking**
   - **Retrieval**:
     - Uses the `multi-qa-distilbert-cos-v1` model to encode queries and documents into dense vectors for semantic matching.
   - **Reranking**:
     - Applies the `FlagReranker` model (`BAAI/bge-reranker-v2-m3`) to refine the relevance of retrieved documents, ensuring high-quality results for ambiguous queries.

---

### 5. **Query Classification**
   - **Purpose**: Categorizes queries based on temporal dynamics (e.g., real-time, fast-changing) to guide downstream retrieval and reasoning.
   - **Implementation**:
     - Uses a predefined prompt to classify queries into categories such as `Real-time`, `Fast-changing`, `Slow-changing`, and `Stable`.

---

### 6. **Prompt Formatting**
   - **Purpose**: Prepares inputs for the LLM by combining query text with retrieved and reranked references.
   - **Structure**:
     - Includes instructions for question categorization and step-by-step reasoning.
     - Incorporates metadata like "Page Last Modified" and the current query time for time-aware responses.

---

### 7. **Response Generation**
   - **Model**:
     - Uses `vLLM` with GPT-4o-mini, a long-context LLM capable of processing up to 128,000 tokens.
   - **Reasoning**:
     - Incorporates Chain-of-Thought (CoT) reasoning to handle complex queries requiring intermediate steps.

---

### 8. **Batch Processing**
   - Handles multiple queries in parallel:
     - Extracts chunks, computes embeddings, retrieves results, reranks them, and generates responses for each query in the batch.
     - Ensures adherence to time constraints by adjusting batch sizes dynamically.

---

### Summary of Workflow
1. Parse and chunk raw HTML using `BeautifulSoup` and `RecursiveCharacterTextSplitter`.
2. Classify queries to tailor retrieval strategies based on temporal dynamics.
3. Retrieve and rerank documents for relevance.
4. Format prompts with references and instructions for reasoning.
5. Generate responses using GPT-4o-mini with CoT prompting.
6. Handle batches efficiently with parallel processing using `ray`.

This workflow is designed for scalability, efficiency, and accuracy, making it suitable for tasks like the CRAG benchmark.
