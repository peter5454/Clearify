import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

#loading the model
nlp = spacy.load("en_core_web_sm")

#Adding the sentiment analyzer
nlp.add_pipe("spacytextblob")

def extract_entities(text: str):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def analyze_entities(text: str):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def analyze_sentiment(text: str): 
    doc = nlp(text)
    return {
        "polarity": doc._.polarity,
        "subjectivity": doc._.subjectivity
    }

def full_sentiment(text: str):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    sentiment = {
        "polarity": doc._.polarity,
        "subjectivity": doc._.subjectivity
    }
    
def full_analysis(text: str):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ent]
    sentiment = {
        "polarity": doc._.polarity,
        "subjectivity": doc._.subjectivity
    }

    return {
        "text": text,
        "entities": entities,
        "sentiment": sentiment
    }