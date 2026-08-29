import os
import shutil
import time

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

GGUF_REPO = os.environ.get("GGUF_REPO", "himalaya-ai/himalaya-gemma-4-e2b-it-gguf")
GGUF_FILE = os.environ.get("GGUF_FILE", "himalaya_gemma_4_q4_k_m.gguf")
MODEL_DIR = "/home/user/app/models"
MODEL_PATH = os.path.join(MODEL_DIR, GGUF_FILE)

os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(MODEL_PATH):
    downloaded = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE)
    shutil.copy(downloaded, MODEL_PATH)

llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=os.cpu_count(), verbose=False)


def reply(message, history):
    messages = [{"role": "user", "content": message}]
    result = llm.create_chat_completion(messages=messages, temperature=0.4, max_tokens=256)
    return result["choices"][0]["message"]["content"]


demo = gr.ChatInterface(reply, title="Himalaya Gemma")
demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)

app = demo.app


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    messages = request.get("messages", [])
    max_tokens = request.get("max_tokens", 256)
    temperature = request.get("temperature", 0.4)
    result = llm.create_chat_completion(messages=messages, temperature=temperature, max_tokens=max_tokens)
    return result


while True:
    time.sleep(3600)
