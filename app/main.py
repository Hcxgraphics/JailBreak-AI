
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from app.attacks.drAttack.drAttack import DrAttackPipeline, DrAttackResult
from app.attacks.drAttack.decompose import decompose_prompt, decompose_spacy
from app.attacks.drAttack.synonym_search import synonym_search
from app.attacks.drAttack.reconstruct import reconstruct_prompt, RECONSTRUCTION_STRATEGIES
from app.llm.ollama_client import is_ollama_running, list_models
from app.llm.victim_model import VictimModel
from app.logging_system.logger import log_attack, get_all_logs, get_summary_stats, clear_logs

app = FastAPI(
    title="JailBreak-AI – DrAttack PoC",
    description=(
        "Proof-of-concept demonstration of the Jailbreak attacks "
        "(Prompt Decomposition & Reconstruction) against a local Mistral LLM via Ollama. "
        "For research and educational purposes only."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


pipeline = DrAttackPipeline(
    victim_model_name="mistral",
    helper_model_name="mistral",
    temperature=0.7,
    top_k_synonyms=3,
)



class AttackRequest(BaseModel):
    prompt: str
    strategies: Optional[list[str]] = None
    run_baseline: bool = True
    stop_on_success: bool = True
    victim_model: str = "mistral"
    helper_model: str = "mistral"


class BatchAttackRequest(BaseModel):
    prompts: list[str]
    strategies: Optional[list[str]] = None
    run_baseline: bool = True
    stop_on_success: bool = True


class DecomposeRequest(BaseModel):
    prompt: str
    use_llm: bool = True
    model: str = "mistral"


class SynonymRequest(BaseModel):
    sub_prompts: list[str]
    top_k: int = 3
    use_llm: bool = True
    model: str = "mistral"


class ReconstructRequest(BaseModel):
    sub_prompts: list[str]
    strategy: str = "icl_structured"


class DirectQueryRequest(BaseModel):
    prompt: str
    model: str = "mistral"



@app.get("/")
def root():
    return {
        "service": "JailBreak-AI DrAttack PoC",
        "ollama_running": is_ollama_running(),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    running = is_ollama_running()
    models = []
    if running:
        try:
            models = list_models()
        except Exception:
            pass
    return {
        "ollama": "up" if running else "down",
        "available_models": models,
        "reconstruction_strategies": list(RECONSTRUCTION_STRATEGIES.keys()),
    }



@app.post("/attack/decompose")
def step_decompose(req: DecomposeRequest):
    if req.use_llm:
        sub_prompts = decompose_prompt(req.prompt, model=req.model)
        if len(sub_prompts) < 2:
            sub_prompts = decompose_spacy(req.prompt)
    else:
        sub_prompts = decompose_spacy(req.prompt)
    return {"prompt": req.prompt, "sub_prompts": sub_prompts, "count": len(sub_prompts)}


@app.post("/attack/synonyms")
def step_synonyms(req: SynonymRequest):
    candidates = synonym_search(
        req.sub_prompts,
        model=req.model,
        top_k=req.top_k,
        use_llm=req.use_llm,
    )
    return {
        "sub_prompts": req.sub_prompts,
        "synonym_candidates": candidates,
    }


@app.post("/attack/reconstruct")
def step_reconstruct(req: ReconstructRequest):
    if req.strategy not in RECONSTRUCTION_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy. Valid options: {list(RECONSTRUCTION_STRATEGIES.keys())}",
        )
    attack_prompt = reconstruct_prompt(req.sub_prompts, strategy=req.strategy)
    return {
        "sub_prompts": req.sub_prompts,
        "strategy": req.strategy,
        "attack_prompt": attack_prompt,
    }



@app.post("/attack/run")
def run_attack(req: AttackRequest):
    if not is_ollama_running():
        raise HTTPException(status_code=503, detail="Ollama is not running. Start it with: ollama serve")

    p = DrAttackPipeline(
        victim_model_name=req.victim_model,
        helper_model_name=req.helper_model,
    )

    result: DrAttackResult = p.run(
        prompt=req.prompt,
        strategies=req.strategies,
        run_baseline=req.run_baseline,
        stop_on_success=req.stop_on_success,
    )

    result_dict = result.to_dict()
    log_attack(result_dict)

    return result_dict


@app.post("/attack/batch")
def run_batch_attack(req: BatchAttackRequest):
    """Run DrAttack on multiple prompts and return aggregate metrics."""
    if not is_ollama_running():
        raise HTTPException(status_code=503, detail="Ollama is not running.")

    results = pipeline.run_batch(
        prompts=req.prompts,
        strategies=req.strategies,
        run_baseline=req.run_baseline,
        stop_on_success=req.stop_on_success,
    )

    results_dicts = [r.to_dict() for r in results]
    for rd in results_dicts:
        log_attack(rd)

    metrics = DrAttackPipeline.compute_metrics(results)
    return {"results": results_dicts, "metrics": metrics}


# ─── Baseline (no attack) endpoint ───────────────────────────────────────────

@app.post("/baseline/query")
def baseline_query(req: DirectQueryRequest):
    """Query the victim model directly with no attack (for comparison)."""
    if not is_ollama_running():
        raise HTTPException(status_code=503, detail="Ollama is not running.")

    victim = VictimModel(model_name=req.model)
    response = victim.query(req.prompt)
    success = victim.attack_success(response)

    return {
        "prompt": req.prompt,
        "response": response,
        "model": req.model,
        "is_refusal": not success,
        "attack_success": success,
    }


# ─── Logging endpoints ────────────────────────────────────────────────────────

@app.get("/logs")
def get_logs():
    return {"logs": get_all_logs()}


@app.get("/logs/summary")
def logs_summary():
    return get_summary_stats()


@app.delete("/logs")
def delete_logs():
    clear_logs()
    return {"message": "Logs cleared."}