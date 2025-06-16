from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnableMap
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader, UnstructuredPDFLoader, PyMuPDFLoader
from langchain_core.documents import Document
from tqdm import tqdm
import os
import pickle
import re

INDEX_PATH = "./faiss_index"
PROCESSED_DOCS_FILE = os.path.join(INDEX_PATH, "processed_docs.pkl")

chapter_separator_regex = r"^(#{1,6} .+|\*\*.+\*\*)$"


def split_llm_output_into_chapters(llm_text):
    """
    Splits the LLM output text into chapters, assuming the LLM separates chapters using markdown-style headings like '# Title'
    """
    chapters = []
    current_chapter = []
    lines = llm_text.splitlines()

    for line in lines:
        if re.match(chapter_separator_regex, line):
            if current_chapter:
                chapters.append("\n".join(current_chapter).strip())
            current_chapter = [line]
        else:
            current_chapter.append(line)

    if current_chapter:
        chapters.append("\n".join(current_chapter).strip())

    return chapters

def semantic_chunking(text):
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.1-405b-instruct",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.compartment.oc1..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        auth_profile="DEFAULT",
        model_kwargs={"temperature": 0.1, "top_p": 0.75, "max_tokens": 4000}
    )

    prompt = f"""
    You received the following text extracted via OCR:

    {text}

    Your task:
    1. Identify headings (short uppercase or bold lines, no period at the end)
    2. Separate paragraphs by heading
    3. Indicate columns with [COLUMN 1], [COLUMN 2] if present
    4. Indicate tables with [TABLE] in markdown format
    """

    response = llm.invoke(prompt)
    return response

def read_pdfs(pdf_path):
    if "-ocr" in pdf_path:
        doc_pages = PyMuPDFLoader(str(pdf_path)).load()
    else:
        doc_pages = UnstructuredPDFLoader(str(pdf_path)).load()
    full_text = "\n".join([page.page_content for page in doc_pages])
    return full_text

def smart_split_text(text, max_chunk_size=10_000):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + max_chunk_size, text_length)

        # Try to find the last sentence end before the limit (., ?, !, \n\n)
        split_point = max(
            text.rfind('.', start, end),
            text.rfind('!', start, end),
            text.rfind('?', start, end),
            text.rfind('\n\n', start, end)
        )

        # If not found, make a hard cut
        if split_point == -1 or split_point <= start:
            split_point = end
        else:
            split_point += 1  # Include the ending character

        chunk = text[start:split_point].strip()
        if chunk:
            chunks.append(chunk)

        start = split_point

    return chunks

def load_previously_indexed_docs():
    if os.path.exists(PROCESSED_DOCS_FILE):
        with open(PROCESSED_DOCS_FILE, "rb") as f:
            return pickle.load(f)
    return set()

def save_indexed_docs(docs):
    with open(PROCESSED_DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)

def append_text_to_file(file_path, text):
    """
    Appends text to the end of a file.
    If the file doesn't exist, it will be created.

    Args:
        file_path (str): Path to the file where the text will be saved.
        text (str): Text to append.
    """
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

class SemanticParagraphSplitter:
    def __init__(self, embedding_model=None, max_title_words=20):
        self.embedding_model = embedding_model
        self.max_title_words = max_title_words
        self.invalid_title_tokens = [":", "-"]

    def split(self, document: Document):
        text = document.page_content
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        chunks = []
        current_title = None
        current_content = []

        def is_title(line):
            if len(line.split()) > self.max_title_words:
                return False
            if any(token in line for token in self.invalid_title_tokens):
                return False
            return True

        for line in lines:
            if is_title(line):
                if current_title and current_content:
                    chunk_text = "# " + current_title + "\n\n" + "\n".join(current_content)
                    chunks.append(Document(page_content=chunk_text.strip(), metadata=document.metadata))
                    append_text_to_file('chunks.txt', chunk_text.strip())
                current_title = line
                current_content = []
            else:
                current_content.append(line)

        # Add the last chunk
        if current_title and current_content:
            chunk_text = "# " + current_title + "\n\n" + "\n".join(current_content)
            chunks.append(Document(page_content=chunk_text.strip(), metadata=document.metadata))
            append_text_to_file('chunks.txt', chunk_text.strip())

        print(f"[✓] Generated {len(chunks)} chunks based on titles and paragraphs.")
        return chunks

def chat():
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.1-405b-instruct",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.compartment.oc1..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        auth_profile="DEFAULT",  # Replace with your profile name,
        model_kwargs={"temperature": 0.1, "top_p": 0.75, "max_tokens": 4000},
    )

    embeddings = OCIGenAIEmbeddings(
        model_id="cohere.embed-multilingual-v3.0",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.compartment.oc1..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        auth_profile="DEFAULT",  # Replace with your profile name,
    )

    pdf_paths = [
        './Manuals/using-integrations-oracle-integration-3-ocr.pdf',
        './Manuals/SOASUITE.pdf',
        './Manuals/SOASUITEHL7.pdf'
    ]

    semantic_splitter = SemanticParagraphSplitter(embedding_model=embeddings)

    already_indexed_docs = load_previously_indexed_docs()
    updated_docs = set()

    # Try loading existing FAISS index
    try:
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("✔️ FAISS index loaded.")
    except Exception:
        print("⚠️ FAISS index not found, creating a new one.")
        vectorstore = None

    new_chunks = []

    pages = []
    for pdf_path in tqdm(pdf_paths, desc=f"📄 Processing PDFs"):
        print(f" {os.path.basename(pdf_path)}")
        if pdf_path in already_indexed_docs:
            print(f"✅ Document already indexed: {pdf_path}")
            continue
        full_text = read_pdfs(pdf_path=pdf_path)

        # Split the text into ~10 KB chunks (~10,000 characters)
        text_chunks = smart_split_text(full_text, max_chunk_size=10_000)
        overflow_buffer = ""  # Remainder from the previous chapter, if any

        for chunk in tqdm(text_chunks, desc=f"📄 Processing text chunks", dynamic_ncols=True, leave=False):
            # Join with leftover from previous chunk
            current_text = overflow_buffer + chunk

            # Send text to LLM for semantic splitting
            treated_text = semantic_chunking(current_text)

            if hasattr(treated_text, "content"):
                chapters = split_llm_output_into_chapters(treated_text.content)

                # Check if the last chapter seems incomplete
                last_chapter = chapters[-1] if chapters else ""

                # Simple criteria: if text ends without punctuation (like . ! ?) or is too short
                if last_chapter and not last_chapter.strip().endswith((".", "!", "?", "\n\n")):
                    print("📌 Last chapter seems incomplete, saving for the next cycle")
                    overflow_buffer = last_chapter
                    chapters = chapters[:-1]  # Don't index the last incomplete chapter yet
                else:
                    overflow_buffer = ""  # Nothing left over

                # Save complete chapters as document chunks
                for chapter_text in chapters:
                    doc = Document(page_content=chapter_text, metadata={"source": pdf_path})
                    new_chunks.append(doc)
                    print(f"✅ New chapter indexed:\n{chapter_text}...\n")

            else:
                print(f"[ERROR] semantic_chunking returned unexpected type: {type(treated_text)}")

        updated_docs.add(str(pdf_path))

    # If there are new documents, index them
    if new_chunks:
        if vectorstore:
            vectorstore.add_documents(new_chunks)
        else:
            vectorstore = FAISS.from_documents(new_chunks, embedding=embeddings)

        vectorstore.save_local(INDEX_PATH)
        save_indexed_docs(already_indexed_docs.union(updated_docs))
        print(f"💾 {len(new_chunks)} chunks added to FAISS index.")
    else:
        print("📁 No new documents to index.")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20, "fetch_k": 50})

    template = """ 
        Document context:
        {context}
        
        Question:
        {input}
        
        Interpretation rules:
        Rule 1: SOA SUITE documents: `SOASUITE.pdf` and `SOASUITEHL7.pdf`
        Rule 2: Oracle Integration (known as OIC) document: `using-integrations-oracle-integration-3-ocr.pdf`
        Rule 3: If the query is not a comparison between SOA SUITE and Oracle Integration (OIC), only consider documents relevant to the product.
        Rule 4: If the question is a comparison between SOA SUITE and OIC, consider all documents and compare between them.
        Mention at the beginning which tool is being addressed: {input}
    """
    prompt = PromptTemplate.from_template(template)

    chain = (
            RunnableMap({
                "context": lambda x: retriever.invoke(x),
                "input": lambda x: x if isinstance(x, str) else x.get("input", "")
            })
            | prompt
            | llm
            | StrOutputParser()
    )

    print("READY")

    while True:
        query = input()
        if query == "quit":
            break
        response = chain.invoke(query)
        print(type(response))  # <class 'str'>
        print(response)

chat()