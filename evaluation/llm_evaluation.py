# Evaluate response quality using BLEU, ROUGE, etc.
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer


def evaluate_response(reference, generated):
    # Compute metrics
    bleu = sentence_bleu([reference.split()], generated.split())
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge = scorer.score(reference, generated)
    return {'bleu': bleu, 'rouge': rouge}