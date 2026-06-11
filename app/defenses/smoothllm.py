import random
import string
from app.llm.victim_model import VictimModel
 
# ── Perturbation functions ────────────────────────────────────────────────────
 
def _swap_chars(word: str) -> str:
    if len(word) < 2:
        return word
    i = random.randint(0, len(word) - 2)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)
 
 
def _insert_char(word: str) -> str:
    i = random.randint(0, len(word))
    c = random.choice(string.ascii_lowercase)
    return word[:i] + c + word[i:]
 
 
def _delete_char(word: str) -> str:
    if len(word) < 2:
        return word
    i = random.randint(0, len(word) - 1)
    return word[:i] + word[i + 1:]
 
 
PERTURBATION_FNS = {
    "swap":   _swap_chars,
    "insert": _insert_char,
    "delete": _delete_char,
}
 
 
def perturb_prompt(
    prompt: str,
    perturbation_type: str = "swap",
    perturbation_pct: float = 0.1,
) -> str:
    
    fn   = PERTURBATION_FNS.get(perturbation_type, _swap_chars)
    words = prompt.split()
    n_perturb = max(1, int(len(words) * perturbation_pct))
    indices   = random.sample(range(len(words)), min(n_perturb, len(words)))
 
    for i in indices:
        words[i] = fn(words[i])
 
    return " ".join(words)
 
 
# ── SmoothLLM defense ─────────────────────────────────────────────────────────
 
def smoothllm_defense(
    prompt: str,
    victim: VictimModel,
    n_copies: int = 10,
    perturbation_type: str = "swap",
    perturbation_pct: float = 0.1,
) -> dict:
   
    copies      = []
    refused_cnt = 0
 
    for _ in range(n_copies):
        perturbed  = perturb_prompt(prompt, perturbation_type, perturbation_pct)
        response   = victim.query(perturbed)
        is_refused = victim.is_refusal(response)
        if is_refused:
            refused_cnt += 1
        copies.append({
            "perturbed_prompt":  perturbed[:200],
            "response_snippet":  response[:200],
            "refused":           is_refused,
        })
 
    answered_cnt = n_copies - refused_cnt
    majority     = "refused" if refused_cnt > answered_cnt else "answered"
    blocked      = majority == "refused"
 
    return {
        "defense":          "smoothllm",
        "blocked":          blocked,
        "refused_count":    refused_cnt,
        "answered_count":   answered_cnt,
        "n_copies":         n_copies,
        "majority":         majority,
        "perturbation_type": perturbation_type,
        "copies":           copies,
    }
 