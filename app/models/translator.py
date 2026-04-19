"""
Hindi to Haryanvi translation with env-selected model backend.

Supported backends:
  LLM_BACKEND=gguf    -> llama-cpp loads a GGUF from local disk or Hugging Face.
  LLM_BACKEND=hf_lora -> Transformers loads a HF/local base model plus LoRA.
"""
import logging
from pathlib import Path
from typing import Any

from app.config import ModelConfig

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """तुम एक हरियाणवी भाषा विशेषज्ञ अनुवादक हो। तुम्हारा काम है हिंदी वाक्यों को असली हरियाणवी बोली में बदलना।

## हरियाणवी भाषा की विशेषताएं

### सर्वनाम (Pronouns)
- मेरा/मेरी → म्हारा/म्हारी
- तेरा/तेरी → थारा/थारी
- हम/हमारा → आपां/म्हारा
- मुझे → म्हने
- तुम्हें → थम्हने

### नकारात्मक शब्द (Negation)
- नहीं → कोन्या (सबसे आम), नी, ना
- मत → मत / ना कर

### क्रिया रूप (Verb Forms)
- है → सै
- हैं → सैं
- करना → करणा
- खाना → खाणा
- जाना → जाणा
- आना → आणा
- देखना → देखणा

### प्रश्नवाचक शब्द (Question Words)
- क्या → के
- क्यों → क्यूं
- कहाँ → कड़े
- कैसा → किसा / कियां
- कब → कद
- कौन → कौण

### सामान्य शब्द (Common Words)
- बहुत → घणा
- लड़का → छोरा
- लड़की → छोरी
- पता → बेरा
- पानी → पाणी
- खाना → खाणा
- आ जा → आज्या
- ले लो → ले-ले

## Few-shot Examples

Hindi: मुझे पता नहीं कि वह कहाँ गया
Haryanvi: म्हने बेरा कोन्या के ओ कड़े ग्या

Hindi: तुम क्या कर रहे हो?
Haryanvi: थम कै करै सो?

Hindi: आज बहुत गर्मी है
Haryanvi: आज घणी गर्मी सै

Hindi: वह लड़की बहुत अच्छी है
Haryanvi: ओ छोरी घणी अच्छी सै

Hindi: मैं खाना खाने जा रहा हूँ
Haryanvi: मैं खाणा खाण जारा सूं

## नियम
1. केवल हरियाणवी अनुवाद दो, कोई explanation नहीं
2. अनुवाद natural और authentic हो
3. Script हमेशा Devanagari हो
4. proper nouns unchanged रखो"""


def _backend() -> str:
    backend = ModelConfig.LLM_BACKEND.lower()
    if backend not in {"gguf", "hf_lora"}:
        raise ValueError("LLM_BACKEND must be one of: gguf, hf_lora")
    return backend


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

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.LLM_HF_BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
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
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    has_gpu = _has_cuda()
    logger.info("Loading HF base LLM: %s", ModelConfig.LLM_HF_BASE_MODEL_ID)

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
        ModelConfig.LLM_HF_BASE_MODEL_ID,
        **kwargs,
    )

    adapter_id = _resolve_adapter_id()
    if adapter_id:
        logger.info("Applying LoRA adapter: %s", adapter_id)
        model = PeftModel.from_pretrained(model, adapter_id)
        if has_gpu:
            model = model.merge_and_unload()
            logger.info("Merged LoRA adapter into base model")
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


def _translate_gguf(text: str, model: Any) -> str:
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"इस हिंदी वाक्य को हरियाणवी में अनुवाद करो:\n\n{text}",
            },
        ],
        max_tokens=ModelConfig.LLM_MAX_NEW_TOKENS,
        temperature=ModelConfig.LLM_TEMPERATURE,
        top_p=0.95,
        top_k=64,
        repeat_penalty=1.1,
        stop=["<end_of_turn>", "<start_of_turn>", "\n\n"],
    )
    result = response["choices"][0]["message"]["content"]
    return _clean_output(result)


def _translate_hf_lora(text: str, model: Any, tokenizer: Any) -> str:
    if tokenizer is None:
        raise RuntimeError("Tokenizer is required for LLM_BACKEND=hf_lora")

    import torch

    prompt = ModelConfig.LLM_PROMPT_TEMPLATE.format(text=text.strip())
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
        .strip()
    )
