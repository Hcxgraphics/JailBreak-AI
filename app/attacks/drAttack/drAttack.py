
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from app.attacks.drAttack.decompose import decompose_prompt, decompose_spacy
from app.attacks.drAttack.synonym_search import synonym_search, best_synonym_combo
from app.attacks.drAttack.reconstruct import reconstruct_prompt, RECONSTRUCTION_STRATEGIES
from app.llm.victim_model import VictimModel



@dataclass
class DrAttackResult:
    original_prompt: str
    sub_prompts: list[str]
    synonym_candidates: list[list[str]]
    selected_synonyms: list[str]
    strategy: str
    attack_prompt: str
    response: str
    success: bool
    num_queries: int
    elapsed_seconds: float
    baseline_response: Optional[str] = None
    baseline_success: Optional[bool] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)



class DrAttackPipeline:


    def __init__(
        self,
        victim_model_name: str = "mistral",
        helper_model_name: str = "mistral",
        temperature: float = 0.7,
        top_k_synonyms: int = 3,
        use_llm_decompose: bool = True,
        use_llm_synonyms: bool = True,
    ):
        self.victim = VictimModel(model_name=victim_model_name, temperature=temperature)
        self.helper_model = helper_model_name
        self.top_k_synonyms = top_k_synonyms
        self.use_llm_decompose = use_llm_decompose
        self.use_llm_synonyms = use_llm_synonyms

    # ── Step 0: Baseline (no attack) ─────────────────────────────────────────

    def baseline_query(self, prompt: str) -> tuple[str, bool]:
        response = self.victim.query(prompt)
        return response, self.victim.attack_success(response)

    # ── Step 1: Decompose ─────────────────────────────────────────────────────

    def decompose(self, prompt: str) -> list[str]:
        if self.use_llm_decompose:
            sub_prompts = decompose_prompt(prompt, model=self.helper_model)
            # Fallback to spaCy if LLM returns garbage
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

    # ── Step 3: Reconstruct & Attack ──────────────────────────────────────────

    def attack_with_strategy(
        self,
        sub_prompts: list[str],
        strategy: str,
    ) -> tuple[str, bool]:
        attack_prompt = reconstruct_prompt(sub_prompts, strategy=strategy)
        response = self.victim.query(attack_prompt)
        success = self.victim.attack_success(response)
        return attack_prompt, response, success

    # ── Full single-prompt attack ─────────────────────────────────────────────

    def run(
        self,
        prompt: str,
        strategies: Optional[list[str]] = None,
        run_baseline: bool = True,
        stop_on_success: bool = True,
    ) -> DrAttackResult:
        
        if strategies is None:
            strategies = list(RECONSTRUCTION_STRATEGIES.keys())

        t_start = time.time()
        num_queries = 0

        # Step 0: Baseline
        baseline_response = None
        baseline_success = None
        if run_baseline:
            baseline_response, baseline_success = self.baseline_query(prompt)
            num_queries += 1

        # Step 1: Decompose
        sub_prompts = self.decompose(prompt)

        # Step 2: Synonym search
        synonym_candidates = self.search_synonyms(sub_prompts)
        selected_synonyms = best_synonym_combo(synonym_candidates)

        # Step 3: Try reconstruction strategies until success (or exhausted)
        final_attack_prompt = ""
        final_response = ""
        final_success = False
        final_strategy = strategies[0]

        for strategy in strategies:
            attack_prompt, response, success = self.attack_with_strategy(
                selected_synonyms, strategy
            )
            num_queries += 1
            final_attack_prompt = attack_prompt
            final_response = response
            final_success = success
            final_strategy = strategy

            if success and stop_on_success:
                break

        elapsed = time.time() - t_start

        return DrAttackResult(
            original_prompt=prompt,
            sub_prompts=sub_prompts,
            synonym_candidates=synonym_candidates,
            selected_synonyms=selected_synonyms,
            strategy=final_strategy,
            attack_prompt=final_attack_prompt,
            response=final_response,
            success=final_success,
            num_queries=num_queries,
            elapsed_seconds=round(elapsed, 2),
            baseline_response=baseline_response,
            baseline_success=baseline_success,
        )

    # ── Batch evaluation ──────────────────────────────────────────────────────

    def run_batch(
        self,
        prompts: list[str],
        strategies: Optional[list[str]] = None,
        run_baseline: bool = True,
        stop_on_success: bool = True,
    ) -> list[DrAttackResult]:
        """Run the attack on a list of prompts and return all results."""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"[DrAttack] Running prompt {i+1}/{len(prompts)}: {prompt[:60]}...")
            result = self.run(
                prompt,
                strategies=strategies,
                run_baseline=run_baseline,
                stop_on_success=stop_on_success,
            )
            results.append(result)
            print(
                f"  → Success: {result.success} | Queries: {result.num_queries} | "
                f"Strategy: {result.strategy} | Time: {result.elapsed_seconds}s"
            )
        return results


    @staticmethod
    def compute_metrics(results: list[DrAttackResult]) -> dict:
        """
        Compute Attack Success Rate (ASR) and efficiency metrics.
        """
        n = len(results)
        if n == 0:
            return {}

        attack_successes = sum(1 for r in results if r.success)
        baseline_successes = sum(
            1 for r in results if r.baseline_success is True
        )
        avg_queries = sum(r.num_queries for r in results) / n
        avg_time = sum(r.elapsed_seconds for r in results) / n

        return {
            "total_prompts": n,
            "attack_asr": round(attack_successes / n, 4),
            "attack_asr_percent": round(attack_successes / n * 100, 2),
            "baseline_asr": round(baseline_successes / n, 4) if baseline_successes else None,
            "baseline_asr_percent": round(baseline_successes / n * 100, 2) if baseline_successes else None,
            "avg_queries_per_prompt": round(avg_queries, 2),
            "avg_time_seconds": round(avg_time, 2),
            "strategy_breakdown": {
                strategy: sum(1 for r in results if r.strategy == strategy and r.success)
                for strategy in RECONSTRUCTION_STRATEGIES
            },
        }