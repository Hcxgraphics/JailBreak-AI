from app.attacks.drAttack.attack_pipeline import ( generate_attack_prompt )

from app.llm.ollama_client import query_mistral

from app.evaluation.refusal_detector import ( is_refusal)

from app.logging_system.logger import ( log_attack)

class DrAttack:

    def __init__(self):

        self.max_attempts = 5

    def search(self, original_prompt):

        for attempt in range(self.max_attempts):

            attack_prompt = generate_attack_prompt(
                original_prompt
            )

            response = query_mistral(
                attack_prompt
            )

            refusal = is_refusal(response)

            log_attack({

                "attempt": attempt,

                "attack_type": "DrAttack",

                "original_prompt": original_prompt,

                "modified_prompt": attack_prompt,

                "response": response,

                "refusal": refusal
            })

            if not refusal:

                return {

                    "success": True,

                    "response": response,

                    "attack_prompt": attack_prompt
                }

        return {

            "success": False,

            "response": response
        }