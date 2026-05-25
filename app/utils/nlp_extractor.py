import spacy

nlp = spacy.load("en_core_web_sm")

def extract_keywords(text):

    doc = nlp(text)

    keywords = []

    for token in doc:

        if token.pos_ in ["NOUN", "VERB", "PROPN"]:

            keywords.append(token.text)

    return keywords