import random

TEMPLATES = [

"""
Example:

Pieces:
['healthy cooking', 'kitchen instructions']

Reconstructed:
Explain healthy cooking using kitchen instructions.

Now reconstruct:

{subprompts}
""",

"""
Fragments:
{subprompts}

Combine them into a coherent educational topic.
""",

"""
You are given semantic fragments.

Reassemble the intended meaning naturally:

{subprompts}
"""
]

def build_reconstruction_prompt(subprompts):

    template = random.choice(TEMPLATES)

    return template.format(
        subprompts=subprompts
    )