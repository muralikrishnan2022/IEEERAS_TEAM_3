import sys
import nltk
import numpy as np
from preprocess import text  # Importing text from preprocess.py
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Download necessary resources if not already downloaded
nltk.download("punkt")
nltk.download("stopwords")

class DocumentSummarizer:
    def __init__(self, text, summary_ratio=0.2):
        if not text.strip():
            raise ValueError("Error: The input text is empty.")
        
        self.text = text
        self.summary_ratio = summary_ratio
        self.sentences = sent_tokenize(text)
        self.num_sentences = max(1, int(len(self.sentences) * self.summary_ratio))

    def summarize_textrank(self):
        """Summarization using TextRank (sumy)"""
        parser = PlaintextParser.from_string(self.text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, self.num_sentences)
        return " ".join(str(sentence) for sentence in summary)

    def summarize_tfidf(self):
        """Summarization using TF-IDF"""
        vectorizer = TfidfVectorizer(stop_words="english")
        sentence_vectors = vectorizer.fit_transform(self.sentences)
        sentence_scores = np.array(sentence_vectors.sum(axis=1)).flatten()

        top_sentence_indices = np.argsort(sentence_scores)[-self.num_sentences:]
        top_sentences = [self.sentences[i] for i in sorted(top_sentence_indices)]
        return " ".join(top_sentences)

    def summarize_lsa(self):
        """Summarization using Latent Semantic Analysis (LSA)"""
        parser = PlaintextParser.from_string(self.text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, self.num_sentences)
        return " ".join(str(sentence) for sentence in summary)

# Main Execution
if __name__ == "__main__":
    try:
        print("\n **Summarizing Text from preprocess.py**...\n")
        summarizer = DocumentSummarizer(text)

        print("\n🔹 **TextRank Summary (sumy):**\n", summarizer.summarize_textrank())
        print("\n🔹 **TF-IDF Summary:**\n", summarizer.summarize_tfidf())
        print("\n🔹 **LSA Summary:**\n", summarizer.summarize_lsa())

    except Exception as e:
        print(f"\n Error: {e}")
        sys.exit(1)
