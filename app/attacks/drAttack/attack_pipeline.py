from app.attacks.drAttack.decomposition import (
    decompose_prompt
)

from app.attacks.drAttack.semantic_rewriter import (
    semantic_rewrite
)

from app.attacks.drAttack.reconstruction import (
    build_reconstruction_prompt
)

def generate_attack_prompt(prompt):

    chunks = decompose_prompt(prompt)

    rewritten_chunks = []

    for chunk in chunks:

        rewritten = semantic_rewrite(chunk)

        rewritten_chunks.append(rewritten)

    final_prompt = build_reconstruction_prompt(
        rewritten_chunks
    )

    return final_prompt