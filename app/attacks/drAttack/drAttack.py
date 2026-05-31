
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from app.attacks.drAttack.decompose import decompose_prompt, decompose_spacy
from app.attacks.drAttack.synonym_search import synonym_search, best_synonym_combo
from app.attacks.drAttack.reconstruct import reconstruct_prompt, RECONSTRUCTION_STRATEGIES
from app.llm.victim_model import VictimModel


# ─── Per-strategy result ──────────────────────────────────────────────────────

@dataclass
class StrategyResult:
    """Result of running ONE reconstruction strategy."""
    strategy:     str
    attack_prompt: str
    response:     str
    success:      bool
    elapsed_s:    float


# ─── Full run result ──────────────────────────────────────────────────────────

@dataclass
class DrAttackResult:
    original_prompt:    str
    sub_prompts:        list[str]
    synonym_candidates: list[list[str]]
    selected_synonyms:  list[str]

    # Best strategy result (highest impact / first success)
    strategy:       str
    attack_prompt:  str
    response:       str
    success:        bool

    num_queries:       int
    elapsed_seconds:   float

    # All per-strategy results (for display / comparison)
    strategy_results:  list[StrategyResult] = field(default_factory=list)

    # Baseline (no attack) for comparison
    baseline_response: Optional[str]  = None
    baseline_success:  Optional[bool] = None

    attack_type: str = "dr_attack"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["strategy_results"] = [asdict(sr) for sr in self.strategy_results]
        return d


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class DrAttackPipeline:

    def __init__(
        self,
        victim_model_name:  str   = "mistral",
        helper_model_name:  str   = "mistral",
        temperature:        float = 0.7,
        top_k_synonyms:     int   = 3,
        use_llm_decompose:  bool  = True,
        use_llm_synonyms:   bool  = True,
    ):
        self.victim           = VictimModel(model_name=victim_model_name, temperature=temperature)
        self.helper_model     = helper_model_name
        self.top_k_synonyms   = top_k_synonyms
        self.use_llm_decompose = use_llm_decompose
        self.use_llm_synonyms  = use_llm_synonyms

    # ── Step 0: Baseline ──────────────────────────────────────────────────────

    def baseline_query(self, prompt: str) -> tuple[str, bool]:
        response = self.victim.query(prompt)
        return response, self.victim.attack_success(response)

    # ── Step 1: Decompose ─────────────────────────────────────────────────────

    def decompose(self, prompt: str) -> list[str]:
        if self.use_llm_decompose:
            sub_prompts = decompose_prompt(prompt, model=self.helper_model)
            if len(sub_prompts) < 2:
                sub_prompts = decompose_spacy(prompt)
        else:
            sub_prompts = decompose_spacy(prompt)
        return sub_prompts

    # ── Step 2: Synonym Search ────────────────────────────────────────────────

    def search_synonyms(self, sub_prompts: list[str]) -> list[list[str]]:
        return synonym_search(
            sub_prompts,
            model=self.helper_model,
            top_k=self.top_k_synonyms,
            use_llm=self.use_llm_synonyms,
        )

    # ── Step 3: Single strategy attack ───────────────────────────────────────

    def attack_with_strategy(
        self,
        sub_prompts: list[str],
        strategy: str,
    ) -> StrategyResult:
        t0 = time.time()
        attack_prompt = reconstruct_prompt(sub_prompts, strategy=strategy)
        response      = self.victim.query(attack_prompt)
        success       = self.victim.attack_success(response)
        return StrategyResult(
            strategy=strategy,
            attack_prompt=attack_prompt,
            response=response,
            success=success,
            elapsed_s=round(time.time() - t0, 2),
        )

    # ── Full single-prompt run ────────────────────────────────────────────────

    def run(
        self,
        prompt:           str,
        strategies:       Optional[list[str]] = None,
        run_baseline:     bool = True,
        stop_on_success:  bool = False,   # default False → always run all strategies
    ) -> DrAttackResult:
        
        if strategies is None:
            strategies = list(RECONSTRUCTION_STRATEGIES.keys())

        t_start     = time.time()
        num_queries = 0

        # Step 0: Baseline
        baseline_response = None
        baseline_success  = None
        if run_baseline:
            baseline_response, baseline_success = self.baseline_query(prompt)
            num_queries += 1

        # Step 1 & 2
        sub_prompts        = self.decompose(prompt)
        synonym_candidates = self.search_synonyms(sub_prompts)
        selected_synonyms  = best_synonym_combo(synonym_candidates)

        # Step 3: Try every strategy, store all results
        strategy_results: list[StrategyResult] = []
        best_result: Optional[StrategyResult]  = None

        for strategy in strategies:
            sr = self.attack_with_strategy(selected_synonyms, strategy)
            strategy_results.append(sr)
            num_queries += 1

            # Track best: first success, or if none then last attempted
            if best_result is None or (sr.success and not best_result.success):
                best_result = sr

            if sr.success and stop_on_success:
                break

        # If nothing succeeded, best = fastest among all (tie-break by time)
        if best_result is None or not best_result.success:
            best_result = min(strategy_results, key=lambda r: r.elapsed_s) if strategy_results else None

        br = best_result or StrategyResult("none", "", "", False, 0.0)

        return DrAttackResult(
            original_prompt    = prompt,
            sub_prompts        = sub_prompts,
            synonym_candidates = synonym_candidates,
            selected_synonyms  = selected_synonyms,
            strategy           = br.strategy,
            attack_prompt      = br.attack_prompt,
            response           = br.response,
            success            = br.success,
            num_queries        = num_queries,
            elapsed_seconds    = round(time.time() - t_start, 2),
            strategy_results   = strategy_results,
            baseline_response  = baseline_response,
            baseline_success   = baseline_success,
        )

    # ── Batch ─────────────────────────────────────────────────────────────────

    def run_batch(
        self,
        prompts:          list[str],
        strategies:       Optional[list[str]] = None,
        run_baseline:     bool = True,
        stop_on_success:  bool = False,
    ) -> list[DrAttackResult]:
        results = []
        for i, prompt in enumerate(prompts):
            print(f"[DrAttack] Prompt {i+1}/{len(prompts)}: {prompt[:60]}...")
            result = self.run(prompt, strategies=strategies,
                              run_baseline=run_baseline, stop_on_success=stop_on_success)
            results.append(result)
            print(f"  → Success: {result.success} | Best strategy: {result.strategy} "
                  f"| Time: {result.elapsed_seconds}s")
        return results

    # ── Metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def compute_metrics(results: list[DrAttackResult]) -> dict:
        n = len(results)
        if n == 0:
            return {}

        attack_successes   = sum(1 for r in results if r.success)
        baseline_successes = sum(1 for r in results if r.baseline_success is True)
        avg_queries        = sum(r.num_queries for r in results) / n
        avg_time           = sum(r.elapsed_seconds for r in results) / n

        # Per-strategy success counts across all results
        strat_breakdown: dict[str, int] = {s: 0 for s in RECONSTRUCTION_STRATEGIES}
        for r in results:
            for sr in r.strategy_results:
                if sr.success:
                    strat_breakdown[sr.strategy] = strat_breakdown.get(sr.strategy, 0) + 1

        return {
            "total_prompts":         n,
            "attack_asr":            round(attack_successes / n, 4),
            "attack_asr_percent":    round(attack_successes / n * 100, 2),
            "baseline_asr":          round(baseline_successes / n, 4) if baseline_successes else None,
            "baseline_asr_percent":  round(baseline_successes / n * 100, 2) if baseline_successes else None,
            "avg_queries_per_prompt": round(avg_queries, 2),
            "avg_time_seconds":       round(avg_time, 2),
            "strategy_breakdown":     strat_breakdown,
        }