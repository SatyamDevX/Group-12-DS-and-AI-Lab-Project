"""
Hindi to Haryanvi translation with env-selected model backend.

Supported backends:
  LLM_BACKEND=gguf    -> llama-cpp loads a GGUF from local disk or Hugging Face.
  LLM_BACKEND=hf_lora -> Transformers loads a HF/local base model plus LoRA.
"""
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from app.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class LoadedTranslator:
    key: str
    label: str
    backend: str
    model: Any
    tokenizer: Any = None
    prompt_template: str | None = None
    system_prompt: str | None = None


TRANSLATION_MODEL_LABELS = {
    "gemma_gguf": "Gemma 4 GGUF",
    "llama_lora": "LLaMA 3.1 QLoRA",
}

TRANSLATION_MODEL_ALIASES = {
    "gemma": "gemma_gguf",
    "gguf": "gemma_gguf",
    "gemma_gguf": "gemma_gguf",
    "llama": "llama_lora",
    "lora": "llama_lora",
    "llama_lora": "llama_lora",
    "llama3_lora": "llama_lora",
    "satyam_lora": "llama_lora",
}


SYSTEM_PROMPT = """तुम एक अत्यंत सटीक हरियाणवी भाषा विशेषज्ञ अनुवादक हो। तुम्हारा एकमात्र कार्य मानक हिंदी को ठेठ हरियाणवी बोली में बदलना है। तुम्हें नीचे दिए गए व्याकरणिक, ध्वन्यात्मक और शाब्दिक नियमों का 100% पालन करना है।

## 1. ध्वन्यात्मक नियम (Phonetic Shifts) - सर्वोच्च प्राथमिकता
- 'न' का 'ण' में परिवर्तन: अधिकांश शब्दों के बीच या अंत में 'न' को 'ण' कर दो (जैसे: पानी -> पाणी, खाना -> खाणा, आना -> आणा, अपना -> अपणा)।
- आदर सूचक पतन (Honorific Collapse): हरियाणवी में 'आप' का प्रयोग नहीं होता। सभी 'आप' को संदर्भ अनुसार 'थाम' (तुम) या 'तू' में बदल दो।

## 2. कारक और अव्यय (Postpositions & Conjunctions)
- में -> म्हैं / मा
- से -> तै / तैं
- के लिए -> खातर
- और -> अर
- लेकिन / पर -> पण / पर
- अगर -> जे

## 3. सर्वनाम (Pronouns)
- मेरा/मेरी -> म्हारा/म्हारी
- तेरा/तेरी -> थारा/थारी
- हम/हमारा -> आपां / म्हारा
- मुझे / मुझको -> म्हने / मन्नै
- तुम्हें / तुमको -> थाम्हने / तन्नै
- वह / यह -> ओ / यो

## 4. क्रिया और काल (Verbs & Tenses)
- है -> सै
- हैं -> सैं
- था/थी/थे -> था/थी/थे
- रहा है / रही है -> रह्या सै / लाग रह्या सै (जैसे: जा रहा है -> जाण लाग रह्या सै)
- क्रिया का मूल रूप: करना -> करणा, जाना -> जाणा, देखना -> देखणा, बोलना -> बोलणा।

## 5. नकारात्मक और प्रश्नवाचक (Negation & Questions)
- नहीं -> कोन्या (सर्वाधिक प्रयुक्त), नी, ना
- मत (मनाही) -> ना / ना कर
- क्या -> के
- क्यों -> क्यूं
- कहाँ -> कड़े
- कैसा -> किसा
- कब -> कद
- कौन -> कौण

## 6. मुख्य शब्दावली (Core Vocabulary)
- बहुत -> घणा
- लड़का / लड़की -> छोरा / छोरी
- पता (Knowledge) -> बेरा (जैसे: मुझे पता नहीं -> मन्नै बेरा कोन्या)
- यहाँ / वहाँ -> उरै / उड़ै
- जल्दी -> तावला / जल्दी
- आ जा -> आज्या
- ले लो -> ले-ले

## 7. वाक्य संरचना और आउटपुट नियम (STRICT CONSTRAINTS)
1. वाक्य संरचना (SOV) हिंदी जैसी ही रहेगी।
2. केवल और केवल हरियाणवी अनुवाद आउटपुट में दो।
3. कोई स्पष्टीकरण (explanation), नोट्स, या भूमिका (intro) बिल्कुल मत लिखो।
4. लिपि 100% देवनागरी होनी चाहिए।
5. व्यक्तिवाचक संज्ञाओं (Proper Nouns - नाम, स्थान) को मत बदलो।"""


FEW_SHOT_HISTORY = [
    {"role": "user", "content": "अपना खाना जल्दी खा लो।"},
    {"role": "assistant", "content": "अपणा खाणा तावला खा-ले।"},
    {"role": "user", "content": "आप यह काम मत करो, मैं कर लूँगा।"},
    {"role": "assistant", "content": "थाम यो काम ना कर, मैं कर लूंगा।"},
    {"role": "user", "content": "वह लड़का पानी पी रहा है।"},
    {"role": "assistant", "content": "ओ छोरा पाणी पीण लाग रह्या सै।"},
    {"role": "user", "content": "मुझे पता था लेकिन मैंने बात नहीं की।"},
    {"role": "assistant", "content": "म्हने बेरा था पण मन्नै बात कोन्या करी।"},
    {"role": "user", "content": "तुमने उस घर में क्या देखा?"},
    {"role": "assistant", "content": "तन्नै उस घर म्हैं के देख्या?"},
    {"role": "user", "content": "रोहित और अमित दिल्ली में काम कर रहे हैं।"},
    {"role": "assistant", "content": "रोहित अर अमित दिल्ली म्हैं काम करण लाग रहै सैं।"},
]


def _backend() -> str:
    backend = ModelConfig.LLM_BACKEND.lower()
    if backend not in {"gguf", "hf_lora"}:
        raise ValueError("LLM_BACKEND must be one of: gguf, hf_lora")
    return backend


def normalize_translation_model(model_key: str | None = None) -> str:
    key = (model_key or ModelConfig.DEFAULT_TRANSLATION_MODEL).strip().lower()
    normalized = TRANSLATION_MODEL_ALIASES.get(key)
    if not normalized:
        raise ValueError(
            "translation_model must be one of: "
            + ", ".join(sorted(TRANSLATION_MODEL_LABELS))
        )
    return normalized


def available_translation_models() -> list[dict[str, str]]:
    keys = []
    for raw_key in ModelConfig.TRANSLATION_MODELS.split(","):
        raw_key = raw_key.strip()
        if not raw_key:
            continue
        key = normalize_translation_model(raw_key)
        if key not in keys:
            keys.append(key)

    if not keys:
        keys = [normalize_translation_model(ModelConfig.DEFAULT_TRANSLATION_MODEL)]

    return [{"key": key, "label": TRANSLATION_MODEL_LABELS[key]} for key in keys]


def default_translation_model() -> str:
    default = normalize_translation_model(ModelConfig.DEFAULT_TRANSLATION_MODEL)
    configured = {item["key"] for item in available_translation_models()}
    if default not in configured:
        raise ValueError(
            f"DEFAULT_TRANSLATION_MODEL={default} is not present in TRANSLATION_MODELS."
        )
    return default


def load_translator(model_key: str | None = None) -> LoadedTranslator:
    key = normalize_translation_model(model_key)
    configured = {item["key"] for item in available_translation_models()}
    if key not in configured:
        raise ValueError(
            f"Translation model '{key}' is not enabled. Set TRANSLATION_MODELS to include it."
        )

    if key == "gemma_gguf":
        model = _load_gguf_llm()
        return LoadedTranslator(
            key=key,
            label=TRANSLATION_MODEL_LABELS[key],
            backend="gguf",
            model=model,
        )

    tokenizer = _load_hf_tokenizer(ModelConfig.LLM_LORA_BASE_MODEL_ID, padding_side="right")
    model = _load_hf_lora_model(
        base_model_id=ModelConfig.LLM_LORA_BASE_MODEL_ID,
        adapter_id=ModelConfig.LLM_LORA_ADAPTER_ID,
    )
    return LoadedTranslator(
        key=key,
        label=TRANSLATION_MODEL_LABELS[key],
        backend="hf_lora",
        model=model,
        tokenizer=tokenizer,
        system_prompt=ModelConfig.LLM_LORA_SYSTEM_PROMPT,
    )


def _resolve_gguf_path() -> Path:
    if ModelConfig.LLM_GGUF_MODEL_PATH:
        path = ModelConfig.LLM_GGUF_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"LLM_GGUF_MODEL_PATH does not exist: {path}")
        return path

    if ModelConfig.LLM_GGUF_REPO_ID:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=ModelConfig.LLM_GGUF_REPO_ID,
            filename=ModelConfig.LLM_GGUF_FILENAME,
        )
        return Path(downloaded)

    preferred = ModelConfig.LLM_GGUF_DIR / ModelConfig.LLM_GGUF_FILENAME
    if preferred.exists():
        return preferred

    gguf_files = sorted(ModelConfig.LLM_GGUF_DIR.glob("*.gguf"))
    if gguf_files:
        return gguf_files[0]

    raise FileNotFoundError(
        "No GGUF model found. Set LLM_GGUF_MODEL_PATH, or set "
        "LLM_GGUF_REPO_ID and LLM_GGUF_FILENAME, or place a .gguf file in "
        f"{ModelConfig.LLM_GGUF_DIR}."
    )


def _has_cuda() -> bool:
    import torch

    if ModelConfig.DEVICE == "cpu":
        return False
    has_gpu = torch.cuda.is_available()
    if ModelConfig.DEVICE == "cuda" and not has_gpu:
        raise RuntimeError("DEVICE=cuda was requested, but CUDA is not available.")
    return has_gpu


def load_tokenizer() -> Any:
    if _backend() == "gguf":
        return None

    return _load_hf_tokenizer(ModelConfig.LLM_HF_BASE_MODEL_ID)


def _load_hf_tokenizer(base_model_id: str, padding_side: str = "left") -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_llm() -> Any:
    if _backend() == "gguf":
        return _load_gguf_llm()
    return _load_hf_lora_llm()


def _load_gguf_llm() -> Any:
    from llama_cpp import Llama

    gguf_path = _resolve_gguf_path()
    logger.info("Loading GGUF LLM from %s", gguf_path)
    model = Llama(
        model_path=str(gguf_path),
        n_ctx=ModelConfig.LLM_GGUF_N_CTX,
        n_threads=ModelConfig.LLM_GGUF_N_THREADS,
        n_gpu_layers=ModelConfig.LLM_GGUF_N_GPU_LAYERS,
        chat_format=ModelConfig.LLM_GGUF_CHAT_FORMAT,
        verbose=False,
    )
    logger.info("GGUF LLM loaded")
    return model


def _load_hf_lora_llm() -> Any:
    return _load_hf_lora_model(
        base_model_id=ModelConfig.LLM_HF_BASE_MODEL_ID,
        adapter_id=_resolve_adapter_id(),
    )


def _load_hf_lora_model(
    base_model_id: str,
    adapter_id: str | None,
) -> Any:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    has_gpu = _has_cuda()
    logger.info("Loading HF base LLM: %s", base_model_id)

    if ModelConfig.LLM_LOAD_IN_4BIT and has_gpu:
        kwargs = dict(
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
        logger.info("CUDA detected; loading HF model with 4-bit quantization")
    elif has_gpu:
        kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)
        logger.info("CUDA detected; loading HF model in bfloat16")
    else:
        kwargs = dict(device_map="cpu", torch_dtype=torch.float32)
        logger.warning("Loading HF model on CPU; inference may be very slow.")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        **kwargs,
    )

    if adapter_id:
        logger.info("Applying LoRA adapter: %s", adapter_id)
        model = PeftModel.from_pretrained(model, adapter_id)
    else:
        logger.warning("No LoRA adapter configured; using base model only.")

    model.eval()
    return model


def _resolve_adapter_id() -> str | None:
    if ModelConfig.LLM_HF_ADAPTER_ID:
        return ModelConfig.LLM_HF_ADAPTER_ID
    if ModelConfig.LLM_USE_LOCAL_ADAPTER and ModelConfig.LLM_ADAPTER_DIR.exists():
        return str(ModelConfig.LLM_ADAPTER_DIR)
    return None


def translate(text: str, model: Any, tokenizer: Any) -> str:
    if model is None:
        raise RuntimeError("LLM not loaded")
    if _backend() == "gguf":
        return _translate_gguf(text, model)
    return _translate_hf_lora(text, model, tokenizer)


def translate_with_translator(text: str, translator: LoadedTranslator) -> str:
    if translator.backend == "gguf":
        return _translate_gguf(text, translator.model)
    return _translate_hf_lora(
        text,
        translator.model,
        translator.tokenizer,
        prompt_template=translator.prompt_template,
        system_prompt=translator.system_prompt,
    )


def validate_devanagari(text: str) -> bool:
    """Return true when most letters in the output are Devanagari."""
    if not text or not text.strip():
        return False

    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False

    devanagari = [
        char for char in letters if "\u0900" <= char <= "\u097f"
    ]
    return (len(devanagari) / len(letters)) >= 0.5


def _translate_gguf(text: str, model: Any) -> str:
    base_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    base_messages.extend(FEW_SHOT_HISTORY)
    base_messages.append({"role": "user", "content": text.strip()})

    response = model.create_chat_completion(
        messages=base_messages,
        max_tokens=ModelConfig.LLM_MAX_NEW_TOKENS,
        temperature=ModelConfig.LLM_TEMPERATURE,
        top_p=0.95,
        top_k=64,
        repeat_penalty=1.05,
    )
    result = _clean_output(response["choices"][0]["message"]["content"])
    if validate_devanagari(result):
        return result

    logger.warning(
        "Gemma output failed Devanagari validation. Raw output: %s", result
    )
    retry_messages = [
        *base_messages,
        {"role": "assistant", "content": "[अमान्य आउटपुट / Invalid Output]"},
        {
            "role": "user",
            "content": (
                "त्रुटि। कोई स्पष्टीकरण न दें। इस वाक्य का अनुवाद केवल "
                f"देवनागरी लिपि में करें:\n{text.strip()}"
            ),
        },
    ]
    retry_response = model.create_chat_completion(
        messages=retry_messages,
        max_tokens=ModelConfig.LLM_MAX_NEW_TOKENS,
        temperature=max(ModelConfig.LLM_TEMPERATURE, 0.3),
        top_p=0.95,
        top_k=64,
        repeat_penalty=1.05,
    )
    retry_result = _clean_output(retry_response["choices"][0]["message"]["content"])
    if validate_devanagari(retry_result):
        return retry_result

    logger.error("Gemma retry failed Devanagari validation. Falling back to source.")
    return text.strip()


def _translate_hf_lora(
    text: str,
    model: Any,
    tokenizer: Any,
    prompt_template: str | None = None,
    system_prompt: str | None = None,
) -> str:
    if tokenizer is None:
        raise RuntimeError("Tokenizer is required for LLM_BACKEND=hf_lora")

    import torch

    if system_prompt:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text.strip()},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = (prompt_template or ModelConfig.LLM_PROMPT_TEMPLATE).format(text=text.strip())
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kwargs = dict(
        max_new_tokens=ModelConfig.LLM_MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if ModelConfig.LLM_TEMPERATURE > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = ModelConfig.LLM_TEMPERATURE
    else:
        gen_kwargs["do_sample"] = False

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    return _clean_output(tokenizer.decode(new_tokens, skip_special_tokens=True))


def _clean_output(text: str) -> str:
    return (
        text.replace("<end_of_turn>", "")
        .replace("<start_of_turn>", "")
        .replace("Haryanvi:", "")
        .replace("हरियाणवी:", "")
        .replace("Hindi:", "")
        .replace("हिंदी:", "")
        .strip()
    )
