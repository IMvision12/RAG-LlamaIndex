# RAG-LlamaIndex

RAG-LlamaIndex is a project aimed at leveraging RAG (Retriever, Reader, Generator) architecture along with Llama-2 and sentence transformers to create an efficient search and summarization tool for PDF documents. This tool allows users to query information from PDF files using natural language and obtain relevant answers or summaries.


# Setup 💻

1. Clone Github Repo: 

```bash
$ git clone https://github.com/IMvision12/RAG-LlamaIndex.git
$ cd RAG-LlamaIndex
```

2. Install Libraries

```bash
$ pip install -r requirements.txt
```

3. Get PDF data

The provided links will download pdf files, which will then be stored in a folder named "data". If you have your own PDF files, please relocate them to the "data" folder.
```bash
$ python utils.py --links https://arxiv.org/pdf/2302.13971 https://arxiv.org/pdf/2403.08295
```

4. Run Main.py

```bash
$python main.py --data-directory "/content/RAG-LlamaIndex/dat" \
                --llm-model "meta-llama/Llama-2-7b-chat-hf" \
                --embed-model "sentence-transformers/all-mpnet-base-v2" \
                --hf-api "Your HuggingFace Access Token" \
                --query "Enter your Query!"
```
