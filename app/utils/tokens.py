"""
TWO TOKEN SOURCES IN THIS CODEBASE:

  1. REAL tokens — from Ollama's prompt_eval_count + eval_count fields.
     Available whenever a call goes through generate_with_usage() or
     query_with_usage(). All defense functions return a "tokens" key with
     the real count from their LLM calls.

  2. ESTIMATED tokens — used for attack prompts and responses that were
     produced by the attack pipelines (DrAttack / PAIR / TAP). Those
     pipelines call victim.query() which returns only the response text,
     not token counts. For these we estimate: ~4 characters per token,
     the standard approximation for English text used by OpenAI, Anthropic,
     and most tokenizer-agnostic tooling.

The eval files use estimate_tokens() for attack_tokens (the attack
prompt + response pair) and real token counts for defense_tokens (the
defense's own LLM calls). Both are recorded in DefenseResult and shown
in the metrics table so the distinction is always visible.

FUNCTIONS:
  estimate_tokens(text)                     → rough int estimate
  estimate_tokens_detailed(text)            → dict with char/word/token breakdown
  tokens_from_ollama_response(data)         → extract real counts from Ollama JSON
  summarise_token_usage(results)            → aggregate token stats across an eval run
  compare_real_vs_estimated(real, text)     → accuracy check for the estimator
  format_token_report(defense_name, n, ...) → one-line formatted token summary
"""

from __future__ import annotations
from typing import Optional


# ── Constants ──────────────────────────────────────────────────────────────────

CHARS_PER_TOKEN = 4          # standard English approximation
WORDS_PER_TOKEN = 0.75       # ~0.75 words per token for English prose


# ── Core estimator ─────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:

    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_tokens_detailed(text: str) -> dict:
    if not text:
        return {
            "char_count": 0, "word_count": 0,
            "token_est": 0, "token_est_alt": 0,
            "method": "estimated",
        }

    chars = len(text)
    words = len(text.split())

    char_est = max(1, chars // CHARS_PER_TOKEN)
    word_est = max(1, int(words / WORDS_PER_TOKEN))

    return {
        "char_count":    chars,
        "word_count":    words,
        "token_est":     char_est,
        "token_est_alt": word_est,
        "method":        "estimated",
    }


# ── Real token extraction from Ollama responses ────────────────────────────────

def tokens_from_ollama_response(data: dict) -> dict:
    prompt_tokens   = int(data.get("prompt_eval_count", 0))
    response_tokens = int(data.get("eval_count",        0))

    return {
        "prompt_tokens":   prompt_tokens,
        "response_tokens": response_tokens,
        "total_tokens":    prompt_tokens + response_tokens,
        "method":          "real",
    }


def tokens_from_ollama_chat_response(data: dict) -> dict:
    return tokens_from_ollama_response(data)


# ── Comparison: real vs estimated ─────────────────────────────────────────────

def compare_real_vs_estimated(
    real_tokens: int,
    text: str,
    label: str = "",
) -> dict:

    estimated    = estimate_tokens(text)
    error        = real_tokens - estimated
    char_count   = len(text)
    error_pct    = abs(error) / real_tokens * 100 if real_tokens > 0 else 0.0
    chars_per_tk = char_count / real_tokens        if real_tokens > 0 else 0.0

    return {
        "label":           label,
        "real":            real_tokens,
        "estimated":       estimated,
        "error":           error,
        "error_pct":       round(error_pct, 1),
        "accuracy_pct":    round(100 - error_pct, 1),
        "char_count":      char_count,
        "chars_per_token": round(chars_per_tk, 2),
    }


# ── Aggregate token usage across an evaluation run ───────────────────────────

def summarise_token_usage(
    defense_results: list,          # list of DefenseResult objects from metrics.py
    attack_type: str = "",
    defense_name: str = "",
) -> dict:

    if not defense_results:
        return {}

    attack_tok_list  = [r.attack_tokens  for r in defense_results]
    defense_tok_list = [r.defense_tokens for r in defense_results]
    n = len(defense_results)

    total_attack  = sum(attack_tok_list)
    total_defense = sum(defense_tok_list)
    total_combined = total_attack + total_defense

    avg_attack  = total_attack  / n
    avg_defense = total_defense / n
    avg_combined = total_combined / n

    max_defense = max(defense_tok_list)
    min_defense = min(t for t in defense_tok_list if t > 0) if any(t > 0 for t in defense_tok_list) else 0

    return {
        "attack_type":        attack_type,
        "defense_name":       defense_name,
        "n_prompts":          n,
        "total_attack_tokens":   total_attack,
        "total_defense_tokens":  total_defense,
        "total_combined_tokens": total_combined,
        "avg_attack_tokens":     round(avg_attack,   1),
        "avg_defense_tokens":    round(avg_defense,  1),
        "avg_combined_tokens":   round(avg_combined, 1),
        "max_defense_tokens":    max_defense,
        "min_defense_tokens_nonzero": min_defense,
        "note": (
            "attack_tokens are estimated (char/4); "
            "defense_tokens are real from Ollama where available"
        ),
    }


# ── Formatted one-liner for display ──────────────────────────────────────────

def format_token_report(
    defense_name: str,
    n_prompts: int,
    avg_attack_tokens: float,
    avg_defense_tokens: float,
    dsr_pct: float,
    fpr_pct: float,
) -> str:

    return (
        f"{defense_name:<28}"
        f"DSR={dsr_pct:5.1f}%  "
        f"FPR={fpr_pct:5.1f}%  "
        f"AtkTok≈{avg_attack_tokens:5.0f}  "
        f"DefTok={'real' if avg_defense_tokens > 0 else '0 (structural)':>6}"
        if avg_defense_tokens == 0 else
        f"{defense_name:<28}"
        f"DSR={dsr_pct:5.1f}%  "
        f"FPR={fpr_pct:5.1f}%  "
        f"AtkTok≈{avg_attack_tokens:5.0f}  "
        f"DefTok={avg_defense_tokens:5.0f}"
    )


# ── Token cost classification ─────────────────────────────────────────────────

def classify_token_cost(avg_defense_tokens: float) -> str:
    """
    Tiers (tuned for local Mistral with ~512-token responses):
      free     →   0  tokens  (structural/regex, no LLM call)
      cheap    →   1- 150     (very short LLM call, e.g. binary classifier)
      moderate → 151- 400     (one medium LLM call, e.g. detailed classifier)
      expensive → 401-1000    (one long LLM call, e.g. paraphrase + re-query)
      heavy    → 1001+        (multiple LLM calls, e.g. SmoothLLM N>=3)
    """
    t = avg_defense_tokens
    if t == 0:
        return "free (structural/regex)"
    elif t <= 150:
        return "cheap (<150 tokens)"
    elif t <= 400:
        return "moderate (150-400 tokens)"
    elif t <= 1000:
        return "expensive (400-1000 tokens)"
    else:
        return f"heavy (>{t:.0f} tokens — consider reducing N or disabling)"


# ── Print token breakdown for one defense inspection ─────────────────────────

def print_token_breakdown(
    label: str,
    attack_prompt: str,
    attack_response: str,
    defense_tokens_real: Optional[int] = None,
    defense_name: str = "",
) -> None:

    atk_prompt_est  = estimate_tokens(attack_prompt)
    atk_resp_est    = estimate_tokens(attack_response)
    atk_total       = atk_prompt_est + atk_resp_est

    def_tok_display = (
        f"{defense_tokens_real} (real)"
        if defense_tokens_real is not None
        else "0 (free — no LLM call)"
    )
    def_tok_val = defense_tokens_real if defense_tokens_real is not None else 0
    grand_total = atk_total + def_tok_val

    W = 52
    print(f"\n  {'─'*W}")
    print(f"  TOKEN BREAKDOWN  —  {label}")
    print(f"  {'─'*W}")
    print(f"  Attack prompt    : ~{atk_prompt_est:>5} tokens (est.)")
    print(f"  Attack response  : ~{atk_resp_est:>5} tokens (est.)")
    print(f"  Attack subtotal  : ~{atk_total:>5} tokens")
    if defense_name:
        print(f"  {defense_name:<17}: {def_tok_display:>14}")
    print(f"  {'─'*W}")
    print(f"  GRAND TOTAL      : ~{grand_total:>5} tokens  "
          f"({'real+est.' if defense_tokens_real else 'est.'})")
    print(f"  {'─'*W}")