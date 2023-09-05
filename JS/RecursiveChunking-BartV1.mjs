import { HfInference } from "@huggingface/inference";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import * as fs from "fs";
import tokenizer from "wink-tokenizer";
import "dotenv/config";

const HF_ACCESS_TOKEN = process.env.API_TOKEN;
const inference = new HfInference(HF_ACCESS_TOKEN);

var Tokenizer = tokenizer();

const MAX_INPUT_SIZE = 1024;
const MAX_OUTPUT_SIZE = 100;
const MIN_OUTPUT_SIZE = 50;

function tokenizer_len(x) {
  return Tokenizer.tokenize(x).length;
}

async function chunkify(text) {
  if (tokenizer_len(text) <= MAX_OUTPUT_SIZE) {
    return [text];
  }
  let n_chunks = Math.ceil(tokenizer_len(text) / MAX_INPUT_SIZE);
  let chunk_size = Math.ceil(tokenizer_len(text) / n_chunks) + 200;
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: chunk_size,
    chunkOverlap: 50,
    lengthFunction: tokenizer_len,
  });
  let chunks = await splitter.createDocuments([text]);
  return chunks.map((key) => key.pageContent);
}
async function API_call(input) {
  console.log("API Call >>>>>>>>>>>", tokenizer_len(input), "tokens");
  return inference
    .summarization({
      inputs: input,
      parameters: {
        min_length: Math.min(100, tokenizer_len(input)),
        max_length: MAX_OUTPUT_SIZE,
      },
    })
    .then((data) => {
      return data["summary_text"];
    });
}

async function summarize(text, token) {
  if (text === null) {
    return null;
  }
  text = await text;
  let chunks = await chunkify(text);
  if (chunks.length === 1) {
    let ret = await API_call(chunks[0], token);
    return ret;
  }
  let new_summary = "";
  let responses = [];
  for (let i = 0; i < chunks.length; i++) {
    let ret = await API_call(chunks[i], token);
    if (!ret) {
      return null;
    }
    responses.push(ret);
  }

  for (let j = 0; j < chunks.length; j++) {
    let ret = await responses[j];
    new_summary += ret;
  }

  return summarize(new_summary);
}

summarize(fs.readFileSync("text.txt", "utf8").toString()).then((summary) =>
  console.log("auybsa")
);

// import { getEncoding, encodingForModel } from "js-tiktoken";
// const enc = getEncoding("gpt2");

// console.log(enc.encode("hello world").length);
