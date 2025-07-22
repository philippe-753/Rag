import os
from tqdm import tqdm
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from unstructured.partition.pdf import partition_pdf

# Load environment (for OpenAI API key)
load_dotenv()

# Embedder
embeddings = OpenAIEmbeddings()

# PDF Text Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # larger chunks
    chunk_overlap=50
)

# PDF Text Extractor
def extract_text_from_pdf(path):
    try:
        elements = partition_pdf(path)
        text_blocks = [str(el) for el in elements if str(el).strip() and len(str(el)) > 30]
        return "\n".join(text_blocks)
    except Exception as e:
        print(f"❌ Error extracting {path}: {e}")
        return ""

# Directory of PDFs
pdf_folder = "./pdfs"
all_docs = []

# Loop over all PDFs
for file_name in tqdm(os.listdir(pdf_folder)):
    if not file_name.endswith(".pdf"):
        continue

    path = os.path.join(pdf_folder, file_name)
    text = extract_text_from_pdf(path)

    if len(text.strip()) < 100:
        print(f"Skipping {file_name}: too little extractable text.")
        continue

    chunks = splitter.split_text(text)
    for chunk in chunks:
        if len(chunk.strip()) > 50:
            all_docs.append(Document(page_content=chunk, metadata={"source": file_name}))

print(f"\n📄 Total valid chunks: {len(all_docs)}")
print("🔍 Embedding and saving to FAISS index...")

# Split into manageable batches
BATCH_SIZE = 100
batched_docs = [all_docs[i:i+BATCH_SIZE] for i in range(0, len(all_docs), BATCH_SIZE)]

index = None
for batch in tqdm(batched_docs, desc="Embedding in batches"):
    if index is None:
        index = FAISS.from_documents(batch, embeddings)
    else:
        index.add_documents(batch)

# Save index
if index is not None:
    index.save_local("faiss_index")
    print("✅ FAISS corpus built and saved.")
else:
    print("No valid documents were embedded. Index not saved.")
