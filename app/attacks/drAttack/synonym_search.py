
from app.llm.ollama_client import generate

# ─── LLM-based synonym generation ───────────────────────────────────────────

SYNONYM_SYSTEM = (
    "You are a thesaurus assistant. "
    "Return ONLY a JSON list of alternative phrasings. No commentary."
)

SYNONYM_TEMPLATE = """\
Give {top_k} alternative ways to express the following phrase while keeping the same meaning.
Prefer neutral, academic, or technical language.
Return ONLY a JSON list of strings.

Phrase: "{phrase}"

Example output:
["alternative 1", "alternative 2", "alternative 3"]
"""


def llm_synonyms(phrase: str, model: str = "mistral", top_k: int = 3) -> list[str]:
    """
    Use the local LLM to generate synonym paraphrases for a sub-prompt.
    Returns a list of up to `top_k` alternatives (plus the original).
    """
    import json, re

    query = SYNONYM_TEMPLATE.format(phrase=phrase, top_k=top_k)
    raw = generate(prompt=query, model=model, system=SYNONYM_SYSTEM, temperature=0.4)

    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            synonyms = json.loads(match.group())
            if isinstance(synonyms, list):
                result = [phrase] + [s.strip() for s in synonyms if s.strip()]
                return result[:top_k + 1]
        except json.JSONDecodeError:
            pass

    return [phrase]


# ─── WordNet-based synonym lookup (offline fallback) ─────────────────────────

def wordnet_synonyms(word: str, top_k: int = 3) -> list[str]:
    """
    Retrieve lexical synonyms from WordNet via NLTK.
    Works at the word level; used to replace key nouns/verbs in sub-prompts.
    """
    try:
        from nltk.corpus import wordnet
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        return [word] + list(synonyms)[:top_k]
    except Exception:
        return [word]


# ─── Combined synonym search for a list of sub-prompts ───────────────────────

def synonym_search(
    sub_prompts: list[str],
    model: str = "mistral",
    top_k: int = 3,
    use_llm: bool = True,
) -> list[list[str]]:
    """
    For each sub-prompt, return a list of candidate phrasings (original + synonyms).

    Args:
        sub_prompts: Decomposed sub-prompts from Step 1.
        model:       Ollama model for LLM-based synonym generation.
        top_k:       Number of synonyms to generate per sub-prompt.
        use_llm:     If True, use LLM; otherwise use WordNet only.

    Returns:
        List of lists – each inner list is [original, syn1, syn2, ...] for one sub-prompt.
    """
    results = []
    for sp in sub_prompts:
        if use_llm:
            candidates = llm_synonyms(sp, model=model, top_k=top_k)
        else:
            # WordNet on individual words, then reconstruct
            words = sp.split()
            if words:
                key_word = max(words, key=len)  # Replace the longest (most informative) word
                word_syns = wordnet_synonyms(key_word, top_k=top_k)
                candidates = [sp] + [
                    sp.replace(key_word, syn) for syn in word_syns[1:]
                ]
            else:
                candidates = [sp]
        results.append(candidates)
    return results


def best_synonym_combo(synonym_candidates: list[list[str]]) -> list[str]:
    """
    Select the top-1 (most different from original) synonym for each sub-prompt.
    Simple heuristic: pick index 1 if available, else original.
    """
    return [
        candidates[1] if len(candidates) > 1 else candidates[0]
        for candidates in synonym_candidates
    ]