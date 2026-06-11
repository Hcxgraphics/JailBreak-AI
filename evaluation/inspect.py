
import time
from app.llm.victim_model           import VictimModel
from app.defenses.perplexity_filter   import perplexity_filter
from app.defenses.paraphrase_defense  import paraphrase_defense
from app.defenses.intent_classifier   import intent_classifier_defense
from app.defenses.smoothllm           import smoothllm_defense
from app.defenses.system_prompt_guard import system_prompt_guard, GUARD_LEVELS
from app.defenses.response_filter     import response_filter, keyword_filter, llm_output_judge

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


def _verdict(blocked: bool, elapsed: float) -> None:
    icon   = "🚫 BLOCKED" if blocked else "✅ PASSED THROUGH"
    result = "Defense stopped the attack." if blocked else "Defense did NOT stop the attack."
    print(f"\n  {'═'*55}")
    print(f"  VERDICT : {icon}  ({elapsed:.1f}s)")
    print(f"  {result}")
    print(f"  {'═'*55}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual defense display functions
# ─────────────────────────────────────────────────────────────────────────────

def _show_perplexity(attack_prompt: str) -> dict:
    # _section("HOW IT WORKS")
    # print("  Scores how 'natural' the prompt reads (1-10).")
    # print("  Attack prompts score low — they are convoluted and fragmented.")
    # print("  If score ≤ threshold (default 4) → BLOCKED.")

    _section("PROMPT BEING EVALUATED")
    _wrap(attack_prompt, max_chars=500)

    t  = time.time()
    pf = perplexity_filter(attack_prompt)
    elapsed = round(time.time() - t, 1)

    _section("RESULT")
    print(f"  Naturalness score : {pf['score']}/10")
    print(f"  Threshold         : {pf['threshold']}")
    score = pf['score']
    if score <= 2:
        label = "Very unnatural — highly fragmented or convoluted"
    elif score <= 4:
        label = "Unusual structure — indirect framing or scaffolding detected"
    elif score <= 6:
        label = "Somewhat indirect but readable"
    else:
        label = "Natural, direct language"
    print(f"  Score meaning     : {label}")
    print(f"  Decision          : {'BLOCK (score ≤ threshold)' if pf['blocked'] else 'PASS (score > threshold)'}")

    _verdict(pf["blocked"], elapsed)
    return pf


def _show_paraphrase(attack_prompt: str, victim: VictimModel) -> dict:
    # _section("HOW IT WORKS")
    # print("  Strips attack scaffolding by rewriting the prompt in plain language.")
    # print("  The reframed (clean) version is then sent to the victim instead.")
    # print("  If the victim refuses the plain version → defense succeeded.")

    _section("ORIGINAL ATTACK PROMPT")
    _wrap(attack_prompt, max_chars=600)

    t  = time.time()
    pd = paraphrase_defense(attack_prompt)

    _section("REFRAMED (PLAIN) VERSION")
    print("  ↓ Scaffolding stripped — this is what the victim now receives:")
    print()
    _wrap(pd["paraphrased_prompt"], max_chars=600)

    print(f"\n  Changed significantly: {'Yes' if pd['changed'] else 'No — prompt was already direct'}")

    # Query victim with the reframed version
    para_response = victim.query(pd["paraphrased_prompt"])
    para_blocked  = victim.is_refusal(para_response)
    elapsed = round(time.time() - t, 1)

    _section("VICTIM'S RESPONSE TO REFRAMED PROMPT")
    _wrap(para_response, max_chars=500)

    print(f"\n  Victim refused the reframed prompt: {'Yes — defense worked' if para_blocked else 'No — harmful content still returned'}")

    _verdict(para_blocked, elapsed)
    return {**pd, "paraphrased_response": para_response, "blocked": para_blocked}


def _show_intent_classifier(attack_prompt: str) -> dict:
    # _section("HOW IT WORKS")
    # print("  An LLM reads through the framing to classify the REAL intent.")
    # print("  Sees past roleplay, academic, fictional, and fragment wrappers.")
    # print("  Returns: SAFE / HARMFUL / SUSPICIOUS + underlying request + reasoning.")

    _section("PROMPT BEING CLASSIFIED")
    _wrap(attack_prompt, max_chars=500)

    t  = time.time()
    ic = intent_classifier_defense(attack_prompt, mode="detailed")
    elapsed = round(time.time() - t, 1)

    _section("CLASSIFIER OUTPUT")
    print(f"  Classification     : {ic['label']}")
    if ic.get("underlying_request"):
        print(f"\n  Underlying request : {ic['underlying_request']}")
        print("  (What the classifier thinks the prompt is REALLY asking for)")
    if ic.get("reasoning"):
        print(f"\n  Reasoning          : {ic['reasoning']}")
    print(f"\n  Blocks on HARMFUL  : {'Yes → BLOCKED' if ic['blocked'] else 'No → PASSED'}")
    print(f"  (SUSPICIOUS prompts are flagged but not blocked by default)")

    _verdict(ic["blocked"], elapsed)
    return ic


def _show_smoothllm(attack_prompt: str, victim: VictimModel, n_copies: int) -> dict:
    # _section("HOW IT WORKS")
    # print(f"  Creates {n_copies} randomly perturbed copies of the attack prompt.")
    # print("  Queries the victim with each copy separately.")
    # print("  Majority vote: if >50% refused → BLOCKED.")
    # print("  Attack prompts are brittle — small changes break them.")
    # print("  Legitimate prompts are robust — small changes don't change meaning.")

    _section("ORIGINAL ATTACK PROMPT")
    _wrap(attack_prompt, max_chars=400)

    t  = time.time()
    sl = smoothllm_defense(attack_prompt, victim, n_copies=n_copies)
    elapsed = round(time.time() - t, 1)

    _section(f"PERTURBED COPIES & VICTIM RESPONSES  ({n_copies} copies, type: {sl['perturbation_type']})")
    for i, copy in enumerate(sl["copies"], 1):
        verdict = "REFUSED ✓" if copy["refused"] else "Answered ✗"
        print(f"\n  Copy {i} [{verdict}]")
        print(f"    Perturbed prompt : {copy['perturbed_prompt'][:200]}")
        print(f"    Victim response  : {copy['response_snippet'][:200]}")

    _section("MAJORITY VOTE")
    print(f"  Refused  : {sl['refused_count']}/{sl['n_copies']}")
    print(f"  Answered : {sl['answered_count']}/{sl['n_copies']}")
    print(f"  Majority : {sl['majority'].upper()}")
    print(f"  Decision : {'BLOCK (majority refused)' if sl['blocked'] else 'PASS (majority answered)'}")

    _verdict(sl["blocked"], elapsed)
    return sl


def _show_system_prompt_guard(
    attack_prompt: str,
    attack_response: str,
    victim: VictimModel,
    guard_level: str,
) -> dict:
    # _section("HOW IT WORKS")
    # print(f"  Injects a hardened system prompt (level: {guard_level.upper()}) into the victim.")
    # print("  The system prompt explicitly names known attack patterns.")
    # print("  The victim receives the attack prompt BUT also the guard instructions.")
    # print("  Compares victim's response WITH guard vs the original response WITHOUT guard.")

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

    _verdict(sg["blocked"], elapsed)
    return sg


def _show_response_filter(attack_prompt: str, attack_response: str) -> dict:
    # _section("HOW IT WORKS")
    # print("  Scans the RESPONSE (not the prompt) for harmful content after generation.")
    # print("  Two stages:")
    # print("    Stage 1 — Keyword filter: regex patterns matching step-by-step instructions")
    # print("    Stage 2 — LLM judge: semantic check for actionable harmful content")
    # print("  Blocked if EITHER stage fires.")

    _section("RESPONSE BEING EVALUATED")
    _wrap(attack_response, max_chars=500)

    # Show keyword stage separately
    _section("STAGE 1 — KEYWORD FILTER")
    kw = keyword_filter(attack_response)
    if kw["blocked"]:
        print(f"  Status  : BLOCKED ✗")
        print(f"  Matched : {kw.get('matched_patterns', [])}")
        print("  These regex patterns matched — step-by-step harmful instructions detected.")
    elif kw.get("reason") == "model_refused":
        print("  Status  : PASSED (model already refused — response is safe)")
    else:
        print("  Status  : PASSED (no harmful patterns matched)")
        print("  Moving to LLM judge for semantic check...")

    # Show LLM judge stage
    _section("STAGE 2 — LLM JUDGE")
    t  = time.time()
    lj = llm_output_judge(attack_prompt, attack_response)
    elapsed = round(time.time() - t, 1)
    print(f"  Judge verdict : {lj['raw_judgment']}")
    print(f"  Blocked       : {'Yes' if lj['blocked'] else 'No'}")
    print("  (Judge distinguishes general discussion from actionable instructions)")

    # Combined result
    combined_blocked = kw["blocked"] or lj["blocked"]
    t2 = time.time()
    rf = response_filter(attack_prompt, attack_response, use_llm_judge=True)

    _verdict(combined_blocked, elapsed)
    return rf


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

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
    n_smoothllm:     int = 3,
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