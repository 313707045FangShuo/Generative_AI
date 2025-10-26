import json
import os
import time, random
from tqdm import tqdm
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from sklearn.metrics.pairwise import cosine_similarity
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_models import ChatOllama
import gc

DATASET_PATH = "./data/private_dataset.json"
OUTPUT_PATH = "./data/sample_submission.json"
RETRIEVE_TOP_K = 32
MODEL_NAME = "gemma2-9b-it"
GROQ_API_KEY = "your_key"
SYSTEM_PROMPT = str = """Answer the question based on the context below. If the question cannot be answered using the information provided answer with "I don't know"."""

with open(DATASET_PATH, "r") as f:
    dataset = json.load(f)

results = []

llm = ChatGroq(
  # model="llama-3.3-70b-versatile",
  # model = "meta-llama/llama-4-maverick-17b-128e-instruct",
  model="meta-llama/llama-4-scout-17b-16e-instruct",
  # model="llama-3.3-70b-specdec",
  # model="llama-3.1-8b-instant",
  # model = "gemma2-9b-it",
  api_key= "your_key"
  max_tokens=80,
  # model_kwargs={"frequency_penalty": 0.8},
)

llm_reranker = ChatOllama(
    model="gemma3",  # 根據你下載的模型名
    temperature=0,
    max_tokens=16
)

# PAIRWISE_SLEEP_TIME = 1.2
PAIRWISE_SLEEP_TIME = random.uniform(1.5, 3.0)

json_path = "private_final_submit.jsonl"
results = []
seen_titles = set()

# 讀取已存在的 json 結果
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            results.append(data)
            seen_titles.add(data["title"])

# === Pointwise rerank ===
def rank_by_pointwise(query, docs, embed_model, top_m):
    query_emb = embed_model.embed_query(query)
    doc_texts = [d.page_content for d in docs]
    doc_embs = embed_model.embed_documents(doc_texts)
    scores = cosine_similarity([query_emb], doc_embs)[0]
    scored_docs = sorted(zip(docs, scores), key=lambda x: -x[1])
    return [d for d, _ in scored_docs[:top_m]]

# === Pairwise rerank ===
def rerank_by_pairwise(query, docs, llm_reranker, top_n):
    scores = defaultdict(int)
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            prompt = f"""
Given a query:
{query}

Which of the following two passages is more relevant to the query?

Passage A:
{docs[i].page_content}

Passage B:
{docs[j].page_content}

Output Passage A or Passage B:
"""
            time.sleep(PAIRWISE_SLEEP_TIME)
            result = llm_reranker.invoke(prompt)
            text = result.content if hasattr(result, 'content') else str(result)
            text = text.strip().lower()
            if "a" in text:
                scores[i] += 1
            elif "b" in text:
                scores[j] += 1
    sorted_indices = sorted(scores, key=lambda x: -scores[x])
    return [docs[i] for i in sorted_indices[:top_n]]

# === 主流程 ===
for i, data in tqdm(enumerate(dataset), total=len(dataset), desc=f"Running {DATASET_PATH} QA pipeline"):
    title = data["title"]
    if title in seen_titles:
        continue

    question = data["question"]
    full_text = data["full_text"]

    # Step 1: 文件分段
    documents = full_text.split("\n\n\n")[:-1]
    docs = [Document(page_content=doc) for doc in documents]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=128,
        length_function=len,
        add_start_index=True,
    )
    docs_splits = text_splitter.split_documents(docs)

    model_name = "BAAI/bge-m3" #sentence-transformers/all-mpnet-base-v2"
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs
    )

    # Step 2: Retrieve
    vector_store = InMemoryVectorStore.from_documents(docs_splits, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K})
    retrieved_docs = retriever.get_relevant_documents(question)

    # Step 3: rerank
    top_m_docs = rank_by_pointwise(question, retrieved_docs, embeddings, top_m=16) 
    reranked_docs = rerank_by_pairwise(question, top_m_docs, llm_reranker, top_n=10)  

    # Step 4: RAG 推論
    CHAT_TEMPLATE_RAG = (
    """Give the answer to the question ```{input}``` using the information given in context delimited by triple backticks ```{context}```.

If there is no relevant information in the provided context, try to answer yourself, but tell user that you did not have any relevant context to base your answer on.

Be concise and output the answer of size less than 80 tokens.
assistant:
""")

    #prompt = PromptTemplate.from_template(CHAT_TEMPLATE_RAG)
    prompt = PromptTemplate(
        input_variables=["input", "context"],
        template=CHAT_TEMPLATE_RAG
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(
        retriever=RunnableLambda(lambda _: reranked_docs),
        combine_docs_chain=combine_docs_chain
    )

    response = rag_chain.invoke({
        "input": question,
        "context": "\n".join([doc.page_content for doc in reranked_docs]),
    })
    pred_answer = response["answer"]

    # Step 5: 存答案與證據
    evidence_list = [doc.page_content for doc in reranked_docs]
    result = {
        "title": title,
        "answer": pred_answer,
        "evidence": evidence_list
    }
    results.append(result)

    with open(json_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()

    # === 清除單輪暫存變數，釋放記憶體 ===
    del docs, docs_splits, retriever, vector_store
    del top_m_docs, reranked_docs, response, pred_answer, evidence_list
    gc.collect()

with open("private_final_submit.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Completed! Total {len(results)} answers saved to {json_path}") 