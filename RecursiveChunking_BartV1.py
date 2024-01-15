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
API_TOKEN = os.getenv("HF_API_TOKEN")
headers = {"Authorization": f"Bearer {API_TOKEN}"}

client = InferenceClient()
tok = BartTokenizer.from_pretrained("facebook/bart-large")
bart_tok_len = lambda x: len(tok(x)["input_ids"])
MAX_INPUT_SIZE = 1024
MAX_OUTPUT_SIZE = 100
MIN_OUTPUT_SIZE = 50


def API_call(text):
    print("API call ->>>>>>>>>>>>>>> input length:", bart_tok_len(text))
    summary = client.summarization(
        text,
        parameters={
            "min_length": min(bart_tok_len(text), MIN_OUTPUT_SIZE),
            "max_length": MAX_OUTPUT_SIZE,
        },
    )
    # print("API response <<<<<<<<<<<<<-", summary)
    return summary


def chunkify_bart(text):
    if bart_tok_len(text) <= MAX_OUTPUT_SIZE:
        return [text]
    n_chunks = math.ceil(bart_tok_len(text) / MAX_INPUT_SIZE)
    chunk_size = math.ceil(bart_tok_len(text) / n_chunks) + 200
    print(f"{n_chunks=}, {chunk_size=}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
        length_function=bart_tok_len,
        is_separator_regex=False,
    )
    chunks = list(
        map(lambda page: page.page_content, text_splitter.create_documents([text]))
    )
    print("Chunks sizes are:", [bart_tok_len(c) for c in chunks])
    return chunks


def summarize(comprehension):
    chunks = chunkify_bart(comprehension)
    if len(chunks) == 1:
        return API_call(chunks[0])
    chunk_summaries = [API_call(chunk) for chunk in chunks]
    return summarize(" ".join(chunk_summaries))


print(summarize(open("text.txt").read()))
