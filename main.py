import argparse
import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_data(data_directory):
    try:
        reader = SimpleDirectoryReader(input_dir=data_directory)
        return reader.load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return []


def setup_llm(llm_model, hf_api):
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=False,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "token": hf_api,
            "quantization_config": bnb_config,
        }

        tokenizer_kwargs = {"token": hf_api}

        generate_kwargs = {
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.9,
        }

        llm = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=256,
            generate_kwargs=generate_kwargs,
            tokenizer_name=llm_model,
            model_name=llm_model,
            device_map="auto",
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )

        return llm
    except Exception as e:
        print(f"Error setting up LLM: {e}")
        return None


def setup_embedding_model(embed_model):
    try:
        return HuggingFaceEmbeddings(model_name=embed_model)
    except Exception as e:
        print(f"Error setting up embedding model: {e}")
        return None


def main(args):
    try:
        data_directory = args.data_directory
        llm_model = args.llm_model
        embed_model = args.embed_model
        hf_api = args.hf_api
        query = args.query

        # Check if GPU is available
        if not torch.cuda.is_available():
            print("Error: This code requires a GPU for execution.")
            return

        documents = load_data(data_directory)
        if not documents:
            print("No documents loaded. Exiting.")
            return

        llm = setup_llm(llm_model, hf_api)
        if not llm:
            print("LLM setup failed. Exiting.")
            return

        embedding_model = setup_embedding_model(embed_model)
        if not embedding_model:
            print("Embedding model setup failed. Exiting.")
            return

        Settings.llm = llm
        Settings.embed_model = embedding_model
        Settings.chunk_size = 1024

        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

        response = query_engine.query(query)
        print(f"LLM Output: {response}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--data-directory", type=str, help="Path to the data directory")
    parser.add_argument(
        "--llm-model", type=str, help="Name of the Hugging Face model to use"
    )
    parser.add_argument("--embed-model", type=str, help="Name of the Embedding Model")
    parser.add_argument("--hf-api", type=str, help="Hugging Face API token")
    parser.add_argument("--query", type=str, help="Input the query")
    args = parser.parse_args()
    main(args)
