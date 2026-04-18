"""
Hindi → Haryanvi using Gemma 4 E4B GGUF via llama-cpp-python.
"""
import logging
import regex # Requires: pip install regex
from pathlib import Path
from llama_cpp import Llama
from app.config import ModelConfig

logger = logging.getLogger(__name__)

_llm = None

SYSTEM_PROMPT = """
तुम एक अत्यंत सटीक हरियाणवी भाषा विशेषज्ञ अनुवादक हो। तुम्हारा एकमात्र कार्य मानक हिंदी को ठेठ हरियाणवी बोली में बदलना है। तुम्हें नीचे दिए गए व्याकरणिक, ध्वन्यात्मक और शाब्दिक नियमों का 100% पालन करना है।

## 1. ध्वन्यात्मक नियम (Phonetic Shifts) - सर्वोच्च प्राथमिकता
- 'न' का 'ण' में परिवर्तन: अधिकांश शब्दों के बीच या अंत में 'न' को 'ण' कर दो (जैसे: पानी -> पाणी, खाना -> खाणा, आना -> आणा, अपना -> अपणा)।
- आदर सूचक पतन (Honorific Collapse): हरियाणवी में 'आप' का प्रयोग नहीं होता। सभी 'आप' को संदर्भ अनुसार 'थम' (तुम) या 'तू' में बदल दो।

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
- तुम्हें / तुमको -> थम्हने / तन्नै
- वह / यह -> ओ / यो

## 4. क्रिया और काल (Verbs & Tenses)
- है -> सै
- हैं -> सैं
- था/थी/थे -> था/थी/थे (समान)
- रहा है / रही है (Continuous) -> रह्या सै / लाग रह्या सै (जैसे: जा रहा है -> जाण लाग रह्या सै)
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
    # Stress Test 1: Pronouns, Basic Verbs, and 'न' to 'ण' shift
    {"role": "user", "content": "अपना खाना जल्दी खा लो।"},
    {"role": "assistant", "content": "अपणा खाणा तावला खा-ले।"},

    # Stress Test 2: Honorific Collapse ('आप' -> 'थम') and Imperative Negation
    {"role": "user", "content": "आप यह काम मत करो, मैं कर लूँगा।"},
    {"role": "assistant", "content": "थम यो काम ना कर, मैं कर लूंगा।"},

    # Stress Test 3: Continuous Tense Application
    {"role": "user", "content": "वह लड़का पानी पी रहा है।"},
    {"role": "assistant", "content": "ओ छोरा पाणी पीण लाग रह्या सै।"},

    # Stress Test 4: Conjunctions ('लेकिन'), Past Tense, and Knowledge Vocabulary
    {"role": "user", "content": "मुझे पता था लेकिन मैंने बात नहीं की।"},
    {"role": "assistant", "content": "म्हने बेरा था पण मन्नै बात कोन्या करी।"},

    # Stress Test 5: Postpositions ('में', 'से'), Question Words
    {"role": "user", "content": "तुमने उस घर में क्या देखा?"},
    {"role": "assistant", "content": "तन्नै उस घर म्हैं के देख्या?"},

    # Stress Test 6: (NEW) Proper Noun Exclusion, Conjunctions, Postpositions, and Rule Stacking
    {"role": "user", "content": "रोहित और अमित दिल्ली में काम कर रहे हैं।"},
    {"role": "assistant", "content": "रोहित अर अमित दिल्ली म्हैं काम करण लाग रहै सैं।"}
]


def load_tokenizer():
    # llama-cpp handles tokenization internally — no separate tokenizer needed
    return None


def load_llm():
    global _llm
    if _llm is not None:
        return _llm

    base_dir = Path(ModelConfig.LLM_BASE_MODEL_ID)

    gguf_files = sorted(list(base_dir.glob("*.gguf")))
    if not gguf_files:
        raise FileNotFoundError(
            f"No .gguf file found in {base_dir}. "
            "Download required model before initialization."
        )

    gguf_path = gguf_files[0]
    if len(gguf_files) > 1:
        logger.warning(f"Multiple GGUF files detected. Defaulting to deterministic first match: {gguf_path.name}")
    else:
        logger.info(f"Loading Gemma 4 E4B GGUF from: {gguf_path}")

    # Dynamic extraction of hardware limits
    gpu_layers = getattr(ModelConfig, 'GPU_LAYERS', 0)

    _llm = Llama(
        model_path=str(gguf_path),
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=gpu_layers,
        chat_format=None,
        verbose=False,
    )
    logger.info(f"LLM loaded successfully with {gpu_layers} GPU layers.")
    return _llm


def validate_devanagari(text: str) -> bool:
    """
    Validates Devanagari output using the 'regex' library for true Unicode property matching.
    """
    if not text or not text.strip():
        return False

    devanagari_chars = len(regex.findall(r'\p{Devanagari}', text))
    total_letters = len(regex.findall(r'\p{L}', text))

    if total_letters == 0:
        return False

    return (devanagari_chars / total_letters) >= 0.5


def translate(text: str, model, max_tokens: int = None) -> str:
    """
    Translate Hindi text to Haryanvi dialect with strict fallback constraints.
    """
    if model is None:
        raise RuntimeError("LLM not loaded. Call load_llm() first.")

    # Call-time evaluation of config
    if max_tokens is None:
        max_tokens = getattr(ModelConfig, 'MAX_TOKENS', 256)

    base_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    base_messages.extend(FEW_SHOT_HISTORY)
    base_messages.append({"role": "user", "content": text})

    # Attempt 1: Strict deterministic decoding
    response = model.create_chat_completion(
        messages=base_messages,
        max_tokens=max_tokens,
        temperature=0.1,
        top_p=0.95,
        top_k=64,
        repeat_penalty=1.05,
    )

    result = response["choices"][0]["message"]["content"].strip()

    if validate_devanagari(result):
        return result

    logger.warning(f"Translation failed Devanagari validation. Raw: {result}. Initiating reinforced entropy retry.")

    # Attempt 2: Synthetic failure injection + Entropy bump
    retry_messages = base_messages.copy()
    retry_messages.extend([
        {"role": "assistant", "content": "[अमान्य आउटपुट / Invalid Output]"},
        {"role": "user", "content": "त्रुटि। कोई स्पष्टीकरण न दें। इस वाक्य का अनुवाद केवल देवनागरी लिपि में करें।"}
    ])

    retry_response = model.create_chat_completion(
        messages=retry_messages,
        max_tokens=max_tokens,
        temperature=0.3,
        top_p=0.95,
        top_k=64,
        repeat_penalty=1.05,
    )

    retry_result = retry_response["choices"][0]["message"]["content"].strip()

    if validate_devanagari(retry_result):
        return retry_result

    logger.error("Retry failed Devanagari validation. Falling back to source input.")
    return text
