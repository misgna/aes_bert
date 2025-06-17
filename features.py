import spacy
import textstat

from spellchecker import SpellChecker # pip install pyspellchecker

nlp = spacy.load('en_core_web_sm')
spell = SpellChecker()

def get_features(essay):
    doc = nlp(essay)
    
    words = [token.text.lower() for token in doc if token.is_alpha]
    sentences = list(doc.sents)
    misspelled = spell.unknown(words)
    
    return {
        "num_words": len(words),
        "num_sentences": len(sentences),
        "avg_length_words": sum(len(word) for word in words) / len(words) if len(words) > 0 else 0,
        "avg_length_sentences": len(words) / len(sentences) if len(sentences) > 0 else 0,
        "num_lemmas": len(set([token.lemma_ for token in doc if not token.is_punct and not token.is_space])),
        "num_commas": sum(1 for token in doc if token.text == ','),
        "num_exclamation_marks": sum(1 for token in doc if token.text == '!'),
        "num_question_marks": sum(1 for token in doc if token.text == '?'),
        "num_nouns": sum(1 for token in doc if token.pos_ == 'NOUN'),
        "num_verbs": sum(1 for token in doc if token.pos_ == 'VERB'),
        "num_adverbs": sum(1 for token in doc if token.pos_ == 'ADV'),
        "num_adjectives": sum(1 for token in doc if token.pos_ == 'ADJ'),
        "num_conjunctions": sum(1 for token in doc if token.pos_ == 'CCONJ'),
        "num_stop_words": sum(1 for token in doc if token.is_stop),
        "num_spell_errors": len(misspelled),
        "syllable_count": textstat.syllable_count(essay),
        "flesch_reading_ease": textstat.flesch_reading_ease(essay),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(essay),
        "gunning_fog": textstat.gunning_fog(essay),
        "smog_index": textstat.smog_index(essay),
        "automated_readability_index": textstat.automated_readability_index(essay),
        "coleman_liau_index": textstat.coleman_liau_index(essay),
        "linsear_write_formula": textstat.linsear_write_formula(essay),
        "dale_chal_readability_score": textstat.dale_chall_readability_score(essay),
        "difficult_word_count": textstat.difficult_words(essay)
    }