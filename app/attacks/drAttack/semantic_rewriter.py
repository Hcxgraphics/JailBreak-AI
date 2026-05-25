from app.utils.nlp_extractor import extract_keywords

from app.attacks.drAttack.synonym_search import (
    generate_synonym
)

def semantic_rewrite(text):

    keywords = extract_keywords(text)

    rewritten = text

    for word in keywords:

        synonym = generate_synonym(word)

        rewritten = rewritten.replace(
            word,
            synonym
        )

    return rewritten