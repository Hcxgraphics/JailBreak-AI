
import time
from app.llm.victim_model           import VictimModel
from app.defenses.perplexity_filter   import perplexity_filter
from app.defenses.paraphrase_defense  import paraphrase_defense
from app.defenses.intent_classifier   import intent_classifier_defense
from app.defenses.smoothllm           import smoothllm_defense
from app.defenses.system_prompt_guard import system_prompt_guard, GUARD_LEVELS
from app.defenses.response_filter     import keyword_filter, llm_output_judge

W    = 65
SEP  = "=" * W
THIN = "─" * W
DASHED = "· " * 32


def _wrap(text: str, indent: int = 4, width: int = 60, max_chars: int = 600) -> None:
    pad  = " " * indent
    body = text[:max_chars]
    if len(text) > max_chars:
        body += f"\n{pad}[... {len(text)-max_chars} more chars]"
    for line in body.splitlines():
        # wrap long lines
        while len(line) > width:
            print(pad + line[:width])
            line = line[width:]
        print(pad + line)


def _section(title: str) -> None:
    print(f"\n  {THIN}")
    print(f"  {title}")
    print(f"  {THIN}")


def _verdict(blocked: bool, elapsed: float, tokens: int = 0) -> None:
    icon   = "🚫 BLOCKED" if blocked else "✅ PASSED THROUGH"
    result = "Defense stopped the attack." if blocked else "Defense did NOT stop the attack."
    print(f"\n  {'═'*55}")
    print(f"  VERDICT : {icon}  ({elapsed:.1f}s, {tokens} tokens used by this defense)")
    print(f"  {result}")
    print(f"  {'═'*55}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual defense display functions
# ─────────────────────────────────────────────────────────────────────────────

def _show_perplexity(attack_prompt: str) -> dict:
    _section("HOW IT WORKS")
    print("  Two-stage check:")
    print("    Stage 1 (free, deterministic): regex-detect known attack")
    print("            scaffolding (Fragment N:, word-game framing, roleplay")
    print("            openers, forced affirmative responses, etc.)")
    print("            → if matched, score forced to 1-3, 0 tokens used.")
    print("    Stage 2 (LLM, fallback only): if NO scaffolding detected,")
    print("            ask the LLM to rate naturalness 1-10.")
    print("  If score ≤ threshold (default 5) → BLOCKED.")

    _section("PROMPT BEING EVALUATED")
    _wrap(attack_prompt, max_chars=500)

    t  = time.time()
    pf = perplexity_filter(attack_prompt)
    elapsed = round(time.time() - t, 1)

    _section("RESULT")
    print(f"  Detection method  : {pf['method'].upper()}")
    if pf["method"] == "structural":
        print(f"  Matched patterns  :")
        for p in pf["matched_patterns"]:
            print(f"    - {p}")
    print(f"\n  Score             : {pf['score']}/10")
    print(f"  Threshold         : {pf['threshold']}")
    print(f"  Decision          : {'BLOCK (score ≤ threshold)' if pf['blocked'] else 'PASS (score > threshold)'}")

    print(f"\n  Tokens used       : {pf['tokens']}  "
          f"({'0 — structural match, no LLM call' if pf['method']=='structural' else 'LLM naturalness call'})")
    _verdict(pf["blocked"], elapsed, pf["tokens"])
    return pf


def _show_paraphrase(attack_prompt: str, victim: VictimModel) -> dict:
    _section("HOW IT WORKS")
    print("  Stage A (free, deterministic): physically extract the REAL request")
    print("          — Task Set fragments, or Phrases: list, or full prompt —")
    print("          removing any Demo Set / benign example via string slicing.")
    print("  Stage B (one LLM call): paraphrase the EXTRACTED text into plain")
    print("          language. The reframed version is sent to the victim.")
    print("  If the victim refuses the plain version → defense succeeded.")

    _section("ORIGINAL ATTACK PROMPT")
    _wrap(attack_prompt, max_chars=600)

    t  = time.time()
    pd = paraphrase_defense(attack_prompt)

    _section(f"STAGE A : EXTRACTED REAL REQUEST  (method: {pd['extraction_method']})")
    if pd["extraction_method"] == "task_set":
        print("  Found '--- Task Set ---' marker — Demo Set physically removed.")
    elif pd["extraction_method"] == "phrases_list":
        print("  Found 'Phrases:' list (word_game strategy) — extracted that line.")
    else:
        print("  No fragment/demo structure found — passing full prompt to paraphraser.")
    print()
    _wrap(pd["extracted_text"], max_chars=400)

    _section("STAGE B : REFRAMED (PLAIN) VERSION")
    print("  ↓ This is what the victim now receives:")
    print()
    _wrap(pd["paraphrased_prompt"], max_chars=600)

    print(f"\n  Changed from original: {'Yes' if pd['changed'] else 'No — paraphrase matched original'}")

    # Query victim with the reframed version
    para_usage    = victim.query_with_usage(pd["paraphrased_prompt"])
    para_response = para_usage["text"]
    para_blocked  = victim.is_refusal(para_response)
    elapsed = round(time.time() - t, 1)
    total_tokens  = pd["tokens"] + para_usage["total_tokens"]

    _section("VICTIM'S RESPONSE TO REFRAMED PROMPT")
    _wrap(para_response, max_chars=500)

    print(f"\n  Victim refused the reframed prompt: {'Yes — defense worked' if para_blocked else 'No — harmful content still returned'}")
    print(f"  Tokens used (paraphrase + re-query): {total_tokens}  "
          f"({pd['tokens']} paraphrase + {para_usage['total_tokens']} victim re-query)")

    _verdict(para_blocked, elapsed, total_tokens)
    return {**pd, "paraphrased_response": para_response, "blocked": para_blocked, "tokens": total_tokens}


def _show_intent_classifier(attack_prompt: str) -> dict:
    _section("HOW IT WORKS")
    print("  An LLM reads through the framing to classify the REAL intent.")
    print("  Sees past roleplay, academic, fictional, and fragment wrappers.")
    print("  Returns: SAFE / HARMFUL / SUSPICIOUS + underlying request + reasoning.")

    _section("PROMPT BEING CLASSIFIED")
    _wrap(attack_prompt, max_chars=500)

    t  = time.time()
    ic = intent_classifier_defense(attack_prompt, mode="detailed")
    elapsed = round(time.time() - t, 1)

    _section("CLASSIFIER OUTPUT  (extracted in this order: request → reasoning → label)")
    if ic.get("underlying_request"):
        print(f"  1. Underlying request : {ic['underlying_request']}")
        print("     (What the classifier thinks the prompt is REALLY asking for)")
    if ic.get("reasoning"):
        print(f"\n  2. Reasoning          : {ic['reasoning']}")
    print(f"\n  3. Classification     : {ic['label']}")
    if ic.get("consistency_override"):
        print(f"     ⚠ Consistency override applied: raw label was HARMFUL but")
        print(f"       reasoning described the request as benign — overridden to SAFE.")
    print(f"\n  Blocks on HARMFUL     : {'Yes → BLOCKED' if ic['blocked'] else 'No → PASSED'}")
    print(f"  (SUSPICIOUS prompts are flagged but not blocked by default)")
    print(f"\n  Tokens used           : {ic['tokens']}")

    _verdict(ic["blocked"], elapsed, ic["tokens"])
    return ic


def _show_smoothllm(attack_prompt: str, victim: VictimModel, n_copies: int) -> dict:
    _section("HOW IT WORKS")
    print(f"  Creates {n_copies} randomly perturbed copies of the attack prompt (N={n_copies}).")
    print("  Queries the victim with each copy separately.")
    print("  Majority vote: if >50% refused → BLOCKED.")
    print("  Attack prompts are brittle — small changes break them.")
    print("  Legitimate prompts are robust — small changes don't change meaning.")

    _section("ORIGINAL ATTACK PROMPT")
    _wrap(attack_prompt, max_chars=400)

    t  = time.time()
    sl = smoothllm_defense(attack_prompt, victim, n_copies=n_copies)
    elapsed = round(time.time() - t, 1)

    _section(f"PERTURBED COPIES & VICTIM RESPONSES  ({n_copies} copies, type: {sl['perturbation_type']})")
    for i, copy in enumerate(sl["copies"], 1):
        verdict = "REFUSED ✓" if copy["refused"] else "Answered ✗"
        print(f"\n  Copy {i} [{verdict}]  ({copy.get('tokens', 0)} tokens)")
        print(f"    Perturbed prompt : {copy['perturbed_prompt'][:200]}")
        print(f"    Victim response  : {copy['response_snippet'][:200]}")

    _section("MAJORITY VOTE")
    print(f"  Refused  : {sl['refused_count']}/{sl['n_copies']}")
    print(f"  Answered : {sl['answered_count']}/{sl['n_copies']}")
    print(f"  Majority : {sl['majority'].upper()}")
    print(f"  Decision : {'BLOCK (majority refused)' if sl['blocked'] else 'PASS (majority answered)'}")
    print(f"\n  Total tokens (sum of {n_copies} queries) : {sl['tokens']}")

    _verdict(sl["blocked"], elapsed, sl["tokens"])
    return sl


def _show_system_prompt_guard(
    attack_prompt: str,
    attack_response: str,
    victim: VictimModel,
    guard_level: str,
) -> dict:
    _section("HOW IT WORKS")
    print(f"  Injects a hardened system prompt (level: {guard_level.upper()}) into the victim.")
    print("  The system prompt explicitly names known attack patterns.")
    print("  The victim receives the attack prompt BUT also the guard instructions.")
    print("  Compares victim's response WITH guard vs the original response WITHOUT guard.")

    _section(f"SYSTEM PROMPT INJECTED  (level: {guard_level.upper()})")
    guard_text = GUARD_LEVELS[guard_level]
    _wrap(guard_text, max_chars=800)

    _section("ORIGINAL RESPONSE (no guard)")
    _wrap(attack_response, max_chars=400)

    t  = time.time()
    sg = system_prompt_guard(attack_prompt, victim, guard_level=guard_level)
    elapsed = round(time.time() - t, 1)

    _section(f"RESPONSE WITH {guard_level.upper()} GUARD")
    _wrap(sg["response"], max_chars=500)

    changed = sg["response"].strip() != attack_response.strip()
    print(f"\n  Response changed under guard : {'Yes' if changed else 'No — same response despite guard'}")
    print(f"  Victim refused under guard   : {'Yes' if sg['blocked'] else 'No'}")
    print(f"  Tokens used (guarded query)  : {sg['tokens']}")

    _verdict(sg["blocked"], elapsed, sg["tokens"])
    return sg


def _show_response_filter(attack_prompt: str, attack_response: str) -> dict:
    _section("HOW IT WORKS")
    print("  Scans the RESPONSE (not the prompt) for harmful content after generation.")
    print("  Two stages:")
    print("    Stage 1 — Keyword filter: regex patterns matching step-by-step instructions")
    print("    Stage 2 — LLM judge: semantic check for actionable harmful content")
    print("  Blocked if EITHER stage fires. Stage 2 only runs if Stage 1 passes.")

    _section("RESPONSE BEING EVALUATED")
    _wrap(attack_response, max_chars=500)

    t = time.time()

    # Stage 1
    _section("STAGE 1 : KEYWORD FILTER")
    kw = keyword_filter(attack_response)
    if kw["blocked"]:
        print(f"  Status  : BLOCKED")
        print(f"  Matched : {kw['matched_patterns']}")
        print("  These regex patterns matched — step-by-step harmful instructions detected.")
        print("  Stage 2 skipped (Stage 1 already blocked).")
        elapsed = round(time.time() - t, 1)
        _verdict(True, elapsed, kw["tokens"])
        return kw

    elif kw["reason"] == "model_refused":
        print("  Status  : PASSED (model already refused — response is safe)")
        print("  Stage 2 skipped (response already confirmed safe).")
        elapsed = round(time.time() - t, 1)
        _verdict(False, elapsed, kw["tokens"])
        return kw

    else:
        print("  Status  : PASSED (no harmful patterns matched)")
        print("  Moving to Stage 2 for semantic check...")

    # Stage 2 (only runs if Stage 1 passed without a refusal match)
    _section("STAGE 2 : LLM JUDGE")
    lj = llm_output_judge(attack_prompt, attack_response)
    elapsed = round(time.time() - t, 1)
    print(f"  Judge verdict : {lj['raw_judgment']}")
    print(f"  Blocked       : {'Yes' if lj['blocked'] else 'No'}")
    print("  (Judge distinguishes general discussion from actionable instructions)")

    combined_tokens = kw["tokens"] + lj["tokens"]
    print(f"\n  Tokens used (keyword=0 + LLM judge={lj['tokens']}) : {combined_tokens}")

    lj["tokens"] = combined_tokens
    _verdict(lj["blocked"], elapsed, combined_tokens)
    return lj



VALID_DEFENSES = [
    "perplexity_filter",
    "paraphrase_defense",
    "intent_classifier",
    "smoothllm",
    "system_prompt_soft",
    "system_prompt_medium",
    "system_prompt_hard",
    "response_filter",
]


def show_defense_detail(
    attack_name:     str,
    attack_prompt:   str,
    attack_response: str,
    defense_name:    str,
    n_smoothllm:     int = 5,
) -> dict:
    
    if defense_name not in VALID_DEFENSES:
        print(f"  Unknown defense '{defense_name}'.")
        print(f"  Valid options: {VALID_DEFENSES}")
        return {}

    victim = VictimModel(model_name="mistral")

    print(f"\n{SEP}")
    print(f"  SINGLE DEFENSE INSPECTION")
    print(SEP)
    print(f"  Attack   : {attack_name.upper()}")
    print(f"  Defense  : {defense_name.upper()}")
    print(f"{SEP}")

    if defense_name == "perplexity_filter":
        return _show_perplexity(attack_prompt)

    elif defense_name == "paraphrase_defense":
        return _show_paraphrase(attack_prompt, victim)

    elif defense_name == "intent_classifier":
        return _show_intent_classifier(attack_prompt)

    elif defense_name == "smoothllm":
        return _show_smoothllm(attack_prompt, victim, n_smoothllm)

    elif defense_name.startswith("system_prompt_"):
        level = defense_name.replace("system_prompt_", "")   # soft / medium / hard
        return _show_system_prompt_guard(attack_prompt, attack_response, victim, level)

    elif defense_name == "response_filter":
        return _show_response_filter(attack_prompt, attack_response)

    return {}