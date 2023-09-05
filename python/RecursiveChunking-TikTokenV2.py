from pathlib import Path
import math, os

from dotenv import load_dotenv

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from huggingface_hub import InferenceClient
from transformers import BartTokenizer


load_dotenv(Path(".env"))
API_TOKEN = os.getenv("API_TOKEN")
headers = {"Authorization": f"Bearer {API_TOKEN}"}

client = InferenceClient()
tok = BartTokenizer.from_pretrained("facebook/bart-large")


import tiktoken

tok = tiktoken.get_encoding("cl100k_base")
tok_len_of = lambda x: len(tok.encode(x))

MAX_INPUT_SIZE = 1024
MAX_OUTPUT_SIZE = 100
MIN_OUTPUT_SIZE = 50


def API_call(text):
    print("API call ->>>>>>>>>>>>>>> input length:", tok_len_of(text))
    summary = client.summarization(
        text,
        parameters={
            "min_length": min(tok_len_of(text), MIN_OUTPUT_SIZE),
            "max_length": MAX_OUTPUT_SIZE,
        },
    )
    # print("API response <<<<<<<<<<<<<-", summary)
    return summary


def adaptive_chunkify_bart(text):
    if tok_len_of(text) <= MAX_OUTPUT_SIZE:
        return [text]
    n_chunks = math.ceil(tok_len_of(text) / MAX_INPUT_SIZE)
    chunk_size = math.ceil(tok_len_of(text) / n_chunks) + 200
    print(f"{n_chunks=}, {chunk_size=}")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=50,
        is_separator_regex=False,
    )
    chunks = list(
        map(lambda page: page.page_content, text_splitter.create_documents([text]))
    )
    print("Chunks sizes are:", [tok_len_of(c) for c in chunks])
    return chunks


def summarize(comprehension):
    chunks = adaptive_chunkify_bart(comprehension)
    if len(chunks) == 1:
        return API_call(chunks[0])
    chunk_summaries = [API_call(chunk) for chunk in chunks]
    return summarize(" ".join(chunk_summaries))


print(summarize(open("text.txt").read()))
