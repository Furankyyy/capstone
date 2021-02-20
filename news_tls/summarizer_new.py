from transformers import pipeline
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

class Summarizer:

    def summarize(self, sents, k, vectorizer, filters=None):
        raise NotImplementedError


class BART(Summarizer):

    def __init__(self):
        if torch.cuda.is_available():
            device = 0
        else:
            device = -1
        self.model = pipeline('summarization',device = device)

    def summarize(self, sents, k=None, vectorizer=None, filters=None):


        raw_sents = [w for s in sents for w in word_tokenize(s.raw)]

        truncated = raw_sents[:512]
        summary_input = (' ').join(truncated)


        summary = self.model(summary_input, min_length = 5, max_length=20)[0]['summary_text']
        summary = sent_tokenize(summary)
        summary = [s.strip() for s in summary]

        return summary


