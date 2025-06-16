
# Tutorial: Indexação Semântica de PDFs com LangChain, LLM e FAISS na OCI Generative AI

## 📌 Objetivo

Este tutorial apresenta um pipeline de processamento de documentos PDF com separação semântica de capítulos usando LLM (Large Language Model) da Oracle Cloud Infrastructure (OCI Generative AI), armazenamento vetorial com FAISS e consulta com LangChain.

O código pode ser usado para:

- Indexar manuais técnicos ou PDFs longos em chunks semanticamente relevantes.
- Realizar perguntas e obter respostas baseadas no conteúdo indexado.
- Trabalhar com OCR, PDF não estruturado e separação por títulos detectados por LLM.

## 📚 Tecnologias Utilizadas

- **LangChain**
- **OCI Generative AI (LLM + Embeddings)**
- **FAISS (Facebook AI Similarity Search)**
- **Python (bibliotecas: `tqdm`, `re`, `pickle`, `os`)**
- **Unstructured.io / PyMuPDF para leitura de PDF**

## ✅ Pré-requisitos

- Conta na **Oracle Cloud** com acesso ao **OCI Generative AI Service**.
- Perfil OCI configurado localmente (`~/.oci/config`) com o profile usado no código (exemplo: `LATINOAMERICA`).
- Python 3.8+ com as bibliotecas:

```bash
pip install langchain oci tqdm unstructured pymupdf faiss-cpu
```

## 🏗️ Estrutura Geral do Código

Este projeto cria um **chatbot baseado em documentos PDF técnicos** (como manuais, guias, etc), usando a seguinte sequência de etapas:

1. **Leitura e Extração de Texto dos PDFs**
2. **Divisão Inteligente por Tamanho (Smart Split Text)**
3. **Segmentação Semântica via LLM (Semantic Chunking)**
4. **Divisão Final em Capítulos (Split LLM Output Into Chapters)**
5. **Indexação Vetorial (FAISS)**
6. **Chatbot com Busca RAG (Retrieval-Augmented Generation)**

---

## 📌 1. Leitura e Extração de Texto dos PDFs

### Estratégia:

- O código detecta se o PDF tem sufixo "-ocr" no nome. Isso indica que ele foi tratado com OCR e o carregamento deve usar um método diferente.
- Dois métodos principais de leitura:
  - **PyMuPDFLoader:** Para PDFs OCRizados.
  - **UnstructuredPDFLoader:** Para PDFs com texto digital "nativo".

### Código:

```python
def read_pdfs(pdf_path):
    if "-ocr" in pdf_path:
        doc_pages = PyMuPDFLoader(str(pdf_path)).load()
    else:
        doc_pages = UnstructuredPDFLoader(str(pdf_path)).load()
    full_text = "\n".join([page.page_content for page in doc_pages])
    return full_text
```

---

## 📌 2. Smart Split Text (Divisão por Tamanho Máximo)

### Por que fazer isso?

LLMs têm limites de tamanho para o texto de entrada. A solução é quebrar o texto em pedaços de até 20.000 caracteres.

### Estratégia:

- Tentar cortar no final de frases, procurando por `.`, `!`, `?`, ou `\n\n`.
- Se não houver um ponto de quebra, faz um corte forçado.
- No loop principal, ainda existe uma estratégia para validar se o capitulo não foi quebrado entre um split e outro. Neste caso, se houver quebra, junta-se os splits para gerar um texto conciso.

### Código:

```python
def smart_split_text(text, max_chunk_size=20_000):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + max_chunk_size, text_length)
        split_point = max(
            text.rfind('.', start, end),
            text.rfind('!', start, end),
            text.rfind('?', start, end),
            text.rfind('\n\n', start, end)
        )

        if split_point == -1 or split_point <= start:
            split_point = end
        else:
            split_point += 1

        chunk = text[start:split_point].strip()
        if chunk:
            chunks.append(chunk)

        start = split_point

    return chunks
```

```python
# Simple criteria: if text ends without punctuation (like . ! ?) or is too short
if last_chapter and not last_chapter.strip().endswith((".", "!", "?", "\n\n")):
    print("📌 Last chapter seems incomplete, saving for the next cycle")
    overflow_buffer = last_chapter
    chapters = chapters[:-1]  # Don't index the last incomplete chapter yet
else:
    overflow_buffer = ""  # Nothing left over

```

---

## 📌 3. Semantic Chunking (Separação Semântica via LLM)

### O problema:

Mesmo depois de dividir o texto por tamanho, os chunks ainda podem ser desestruturados (muito longos, sem seções claras).

### Solução:

Enviar cada chunk ao LLM com um **prompt especializado** para que o LLM identifique **títulos**, **seções**, **colunas** e **tabelas**, e devolva em formato Markdown.

### Exemplo de Prompt:

```python
prompt = f"""
Você recebeu o seguinte texto extraído via OCR:

{text}

Sua tarefa:
1. Identificar títulos (linhas curtas em maiúsculas ou negrito, sem ponto final)
2. Separar os parágrafos por título
3. Indicar colunas com [COLUMN 1], [COLUMN 2] se houver
4. Indicar tabelas com [TABLE] em formato markdown
"""
response = llm.invoke(prompt)
```

---

## 📌 4. Split LLM Output Into Chapters (Divisão Final em Capítulos)

### Por que essa etapa é necessária?

O LLM devolve um bloco de texto estruturado em Markdown, mas agora precisamos dividir esse texto em **capítulos individuais** para indexar cada um separadamente.

### Estratégia:

Usamos uma regex que identifica qualquer heading Markdown (`#`, `##`, etc).

### Código:

```python
def split_llm_output_into_chapters(llm_text):
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
```

### Exemplo de Regex usada:

```python
chapter_separator_regex = r"^#{1,6} .+"
```

---

## 📌 5. Controle de PDFs Já Processados (Evitar Reindexação)

Armazenamos os nomes dos arquivos já processados em um arquivo `.pkl`. Assim, o código não reprocessa PDFs antigos.

```python
def load_previously_indexed_docs():
    if os.path.exists(PROCESSED_DOCS_FILE):
        with open(PROCESSED_DOCS_FILE, "rb") as f:
            return pickle.load(f)
    return set()
```

---

## 📌 6. Indexação FAISS (Armazenamento Vetorial)

Transformamos os capítulos em embeddings usando o modelo de embeddings da OCI (`cohere.embed-multilingual-v3.0`), e armazenamos tudo no FAISS.

```python
vectorstore = FAISS.from_documents(new_chunks, embedding=embeddings)
vectorstore.save_local(INDEX_PATH)
```

---

## 📌 7. Chatbot com RAG (Retrieval + LLM)

A consulta funciona assim:

- O FAISS encontra os capítulos mais relevantes.
- Passamos o contexto + a pergunta para o LLM.
- O LLM monta a resposta final.

```python
def get_context(x):
    query = x.get("input") if isinstance(x, dict) else x
    return retriever.invoke(query)

chain = (
    RunnableMap({
        "context": RunnableLambda(get_context),
        "input": lambda x: x.get("input") if isinstance(x, dict) else x
    })
    | prompt
    | llm
    | StrOutputParser()
)
```
O prompt aplica regras específicas, como:

- Distinguir perguntas sobre **SOA Suite** ou **Oracle Integration (OIC)**.
- Usar só os documentos relevantes por produto.
- Fazer comparações se a pergunta exigir.


```python
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
```

---

## ⚙️ Configuração

1. **Configurar acesso ao OCI:**

Arquivo `~/.oci/config` com o profile usado (`DEFAULT`).

2. **Criar as pastas:**

```bash
mkdir ./faiss_index
mkdir ./Manuals
```

3. **Colocar os PDFs dentro de `./Manuals`.**

Você pode encontrar os arquivos PDF aqui:

- [SOASE.pdf](https://docs.oracle.com/middleware/12211/soasuite/develop/SOASE.pdf)
- [SOASUITEHL7.pdf](https://docs.oracle.com/en/learn/oci-genai-pdf/files/SOASUITEHL7.pdf)
- [using-integrations-oracle-integration-3.pdf](https://docs.oracle.com/en/cloud/paas/application-integration/integrations-user/using-integrations-oracle-integration-3.pdf)

4. **Configurar OCI Generative AI**

Configure os parâmetros de:

    model_id: Modelo atualizado que se pretende trabalhar com OCI Gen AI
    service_endpoint: Endpoint da região do OCI Gen AI
    compartment_id: Seu compartment ID
    auth_profile: Profile de sua OCI SDK/CLI configurada, normamlmente DEFAULT

Em:

```python
def semantic_chunking(text):
    llm = ChatOCIGenAI(
        model_id="meta.llama-3.1-405b-instruct",
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id="ocid1.compartment.oc1..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        auth_profile="DEFAULT",
    )
```
    
E:

```python
def chat():
  llm = ChatOCIGenAI(
    model_id="meta.llama-3.1-405b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    auth_profile="DEFAULT",  # Replace with your profile name,
    model_kwargs={"temperature": 0.7, "top_p": 0.75, "max_tokens": 1000},
  )

  embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.compartment.oc1..aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    auth_profile="DEFAULT",  # Replace with your profile name,
  )
```

## 🚀 Como Executar

```bash
python main.py
```

O script vai:

- Ler os PDFs
- Processar em chunks
- Fazer a separação semântica
- Salvar o FAISS index
- Iniciar o chat interativo

Digite perguntas no terminal e veja as respostas.

Para encerrar o chat:
```bash
quit
```

## ✅ Teste Rápido

Exemplos de perguntas para testar:

- "Quais as diferenças entre SOA Suite e OIC?"
- "Como configurar integrações no OIC?"
- "Quais protocolos o SOA Suite suporta?"

## 🔎 Verificando os Resultados

- **FAISS index:** Verifique a pasta `./faiss_index/`
- **Chunks gerados:** Veja o arquivo `chunks.txt`
- **Documentos já indexados:** Verifique `processed_docs.pkl`

## 🧱 Possíveis Extensões Futuras

- Indexar PowerPoints ou outros formatos.
- Persistência de logs de consultas.
- Deploy como API para consulta via web.

---

## ✅ Conclusão

Com este fluxo:

- Extraímos, organizamos e indexamos documentos PDF de forma semântica.
- Garantimos uma recuperação de informações mais precisa, com capítulos relevantes.
- O chatbot pode responder consultas complexas baseadas em documentos longos.

Este é um exemplo completo de **pipeline RAG com FAISS + LangChain + LLM OCI Generative AI**.
