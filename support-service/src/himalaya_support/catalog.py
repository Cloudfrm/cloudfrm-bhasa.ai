"""Canonical Himalaya AI assets used by this support module.

Source org: https://huggingface.co/himalaya-ai
Site: https://himalayaai.org/
"""

from __future__ import annotations

ORG = "himalaya-ai"
ORG_URL = "https://huggingface.co/himalaya-ai"
SITE_URL = "https://himalayaai.org/"

# Chat / generation models (instruction-tuned first).
CHAT_MODELS = {
    "primary": {
        "id": "himalaya-ai/himalaya-gemma-4-e2b-it",
        "kind": "chat",
        "params": "5B",
        "languages": ["ne", "en"],
        "notes": "Full-parameter SFT on nepali-sft-dataset + OpenHermes-2.5. Use chat template.",
        "trust_remote_code": False,
    },
    "fallback": {
        "id": "himalaya-ai/himalayagpt-0.5b-it",
        "kind": "chat",
        "params": "0.5B",
        "languages": ["ne", "en", "hi"],
        "notes": "HimalayaGPT instruction checkpoint. Requires trust_remote_code.",
        "trust_remote_code": True,
    },
    "base": {
        "id": "himalaya-ai/himalayagpt-0.5b",
        "kind": "base",
        "params": "0.5B",
        "languages": ["ne"],
        "notes": "Base HimalayaGPT. Prefer the -it chat checkpoint for support.",
        "trust_remote_code": True,
    },
    "adapter": {
        "id": "himalaya-ai/gemma4-e2b-it-nepali",
        "kind": "lora-adapter",
        "base": "google/gemma-4-E2B-it",
        "languages": ["ne", "en"],
        "notes": "LoRA adapter; primary full SFT weights are preferred.",
        "trust_remote_code": False,
    },
}

LOCAL_GGUF = {
    "primary": "himalaya-ai/himalaya-gemma-4-e2b-it-gguf",
    "fallback": "himalaya-ai/himalayagpt-0.5b-it-gguf",
}

VISION_MODELS = {
    "ocr": {
        "id": "himalaya-ai/glm-ocr-devanagari-finetuned",
        "kind": "ocr",
        "notes": "Devanagari document / handwriting OCR for ticket attachments.",
    }
}

TOKENIZERS = {
    "bpe_128k": "himalaya-ai/nepali-tokenizer-bpe-128k-5b",
    "bpe_100k": "himalaya-ai/nepali-tokenizer-bpe-100k-5b",
    "bpe_64k": "himalaya-ai/nepali-tokenizer-bpe-64k-5b",
}

DATASETS = {
    "sft_compile": {
        "id": "himalaya-ai/nepali-sft-compile",
        "role": "rag_style",
        "notes": "1.67M SFT mix (Aya, Bactrian-X, Indic RAG). Context only — never copy as the reply.",
        "slices": [
            "aya_human_nepali",
            "aya_safe_translated_nepali",
            "bactrian_x_nepali",
            "indic_rag_nepali",
        ],
    },
    "sft_dataset": {
        "id": "himalaya-ai/nepali-sft-dataset",
        "role": "chat_training_mix",
        "notes": "Dataset used to SFT himalaya-gemma-4-e2b-it.",
    },
    "honorific": {
        "id": "himalaya-ai/nepali-honorific-bench",
        "role": "register",
        "file": "nepali_honorific_alignment_devanagari.jsonl",
        "notes": "तपाईं / तिमी / त alignment. Support uses तपाईं by default.",
    },
    "function_calling": {
        "id": "himalaya-ai/nepali-hermes-function-calling-v1",
        "role": "tools",
        "notes": "Hermes <tool_call> format for ticket actions.",
    },
    "json_mode": {
        "id": "himalaya-ai/nepali-json-mode-singleturn",
        "role": "structured_output",
        "notes": "JSON-schema extraction for ticket fields and intent.",
    },
    "proofreader": {
        "id": "himalaya-ai/nepali-proofreader",
        "role": "text_repair",
        "notes": "Corrupted→clean pairs. Used to repair OCR / messy Devanagari before generation.",
    },
    "ocr_eval": {
        "id": "himalaya-ai/ocr-document-processing-eval",
        "role": "document_ocr",
        "notes": "Document / KYC-style OCR eval used to shape attachment handling.",
    },
    "stt": {
        "id": "himalaya-ai/nepali-stt-annotations",
        "role": "voice_schema",
        "notes": "Voice annotation schema for later speech tickets.",
    },
    "wikipedia": {
        "id": "himalaya-ai/wikipedia_nepali_deva",
        "role": "optional_rag",
        "notes": "Optional Nepali Wikipedia slice for general-knowledge grounding.",
    },
}


def public_catalog() -> dict:
    return {
        "org": ORG,
        "org_url": ORG_URL,
        "site": SITE_URL,
        "chat_models": CHAT_MODELS,
        "gguf": LOCAL_GGUF,
        "vision_models": VISION_MODELS,
        "tokenizers": TOKENIZERS,
        "datasets": DATASETS,
        "generation_policy": (
            "Datasets are retrieved as context and style hints. "
            "The chat model always generates a new reply; it must not paste dataset rows."
        ),
    }
