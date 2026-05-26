
from app.llm.ollama_client import generate
import spacy
from typing import Optional


DECOMPOSE_SYSTEM = (
    "You are a linguistic analysis assistant. "
    "Your task is to decompose sentences into their semantic components. "
    "Output ONLY a JSON list of sub-phrases, nothing else."
)

DECOMPOSE_TEMPLATE = """\
Analyze the following sentence and split it into its core semantic sub-phrases.
Each sub-phrase should capture one semantic role (e.g. action, object, qualifier, purpose).
Return a JSON list of strings – each string is one sub-phrase.
Do NOT add commentary. Do NOT include the word "JSON".

Sentence: "{sentence}"

Example output format:
["sub-phrase 1", "sub-phrase 2", "sub-phrase 3"]
"""


def decompose_prompt(prompt: str, model: str = "mistral") -> list[str]:
    query = DECOMPOSE_TEMPLATE.format(sentence=prompt)
    raw = generate(prompt=query, model=model, system=DECOMPOSE_SYSTEM, temperature=0.2)

    # Parse the JSON list out of the response
    import json, re

    # Find the first [...] block in the response
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            sub_prompts = json.loads(match.group())
            if isinstance(sub_prompts, list) and all(
                isinstance(s, str) for s in sub_prompts
            ):
                return [s.strip() for s in sub_prompts if s.strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: split on commas / newlines if JSON parse fails
    fallback = [
        line.strip().strip('"').strip("'").strip(",")
        for line in raw.replace("[", "").replace("]", "").split("\n")
        if line.strip()
    ]
    return [f for f in fallback if f]





_nlp: Optional[object] = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            return None
    return _nlp


def decompose_spacy(prompt: str) -> list[str]:
    nlp = _get_nlp()
    if nlp is None:
        # Very simple split on 'to', 'for', 'that', commas
        import re
        parts = re.split(r"\s+(?:to|for|that|which|by|with|and)\s+|,\s*", prompt)
        return [p.strip() for p in parts if p.strip()]

    doc = nlp(prompt)
    sub_prompts = []

    # Collect subject span
    subj_tokens = [t for t in doc if t.dep_ in ("nsubj", "nsubjpass")]
    if subj_tokens:
        subj = subj_tokens[0]
        sub_prompts.append(" ".join(t.text for t in subj.subtree))

    # Collect root verb + its direct children (verb phrase)
    roots = [t for t in doc if t.dep_ == "ROOT"]
    if roots:
        root = roots[0]
        vp_tokens = [root] + [
            t for t in root.children if t.dep_ in ("aux", "auxpass", "neg", "advmod")
        ]
        vp_tokens.sort(key=lambda t: t.i)
        sub_prompts.append(" ".join(t.text for t in vp_tokens))

    # Collect object span
    obj_tokens = [t for t in doc if t.dep_ in ("dobj", "pobj", "attr")]
    for obj in obj_tokens:
        sub_prompts.append(" ".join(t.text for t in obj.subtree))

    # Collect prepositional phrases not already captured
    for token in doc:
        if token.dep_ == "prep":
            phrase = " ".join(t.text for t in token.subtree)
            if phrase not in " ".join(sub_prompts):
                sub_prompts.append(phrase)

    return [s.strip() for s in sub_prompts if s.strip()] or [prompt]