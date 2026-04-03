"""
BM25 keyword search implementation for memory recall.

Implements the BM25 (Best Matching 25) algorithm, which is the industry
standard for text retrieval used by search engines like Elasticsearch.
"""

import math
from typing import List, Dict
from collections import Counter
from .text_utils import normalize_text


class BM25Scorer:
    """
    BM25 (Best Matching 25) algorithm for keyword relevance scoring.

    BM25 is a probabilistic ranking function that scores documents based on
    query term frequency, document length, and term rarity across the corpus.

    Better than TF-IDF for short documents and queries.

    Formula:
        score(D, Q) = sum over terms in Q of:
            IDF(term) * (f(term, D) * (k1 + 1)) / (f(term, D) + k1 * (1 - b + b * |D| / avgdl))

    Where:
        - IDF(term) = log((N - df(term) + 0.5) / (df(term) + 0.5))
        - f(term, D) = frequency of term in document D
        - |D| = length of document D in words
        - avgdl = average document length in corpus
        - k1 = term frequency saturation parameter (default: 1.5)
        - b = length normalization parameter (default: 0.75)
    """

    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with document collection.

        Args:
            documents: Corpus of memory texts
            k1: Term frequency saturation parameter (1.2-2.0, default: 1.5)
            b: Length normalization parameter (0.0-1.0, default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.N = len(documents)  # Total number of documents

        # Precompute document statistics
        self.doc_lengths = []
        self.doc_term_freqs = []

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            self.doc_term_freqs.append(Counter(tokens))

        # Average document length
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0

        # Compute IDF values for all terms in corpus
        self.idf_cache = self._compute_idf()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens (lowercase words)
        """
        normalized = normalize_text(text)
        # Simple whitespace tokenization
        return normalized.split()

    def _compute_idf(self) -> Dict[str, float]:
        """
        Compute IDF (Inverse Document Frequency) for all terms.

        IDF measures how rare/important a term is across the corpus.
        Rare terms get higher IDF scores.

        Returns:
            Dict mapping term -> IDF score
        """
        idf = {}

        # Count document frequency for each term
        df = Counter()  # df[term] = number of documents containing term
        for term_freq in self.doc_term_freqs:
            for term in term_freq.keys():
                df[term] += 1

        # Compute IDF for each term
        for term, doc_freq in df.items():
            # BM25 IDF formula
            idf[term] = math.log((self.N - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

        return idf

    def score(self, query: List[str], document: str) -> float:
        """
        Calculate BM25 score for query against document.

        Args:
            query: List of query keywords (already extracted/tokenized)
            document: Document text to score

        Returns:
            BM25 score (higher is better, range ~0-10 for typical queries)
        """
        # Find document index
        try:
            doc_idx = self.documents.index(document)
        except ValueError:
            # Document not in corpus, score on the fly
            return self._score_new_document(query, document)

        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]

        score = 0.0

        for term in query:
            term_lower = term.lower()

            # Get term frequency in this document
            f = term_freqs.get(term_lower, 0)
            if f == 0:
                continue  # Term not in document

            # Get IDF for this term (default to 0 if not in corpus)
            idf = self.idf_cache.get(term_lower, 0)

            # BM25 formula components
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)

            score += idf * (numerator / denominator)

        return score

    def _score_new_document(self, query: List[str], document: str) -> float:
        """
        Score a document that's not in the original corpus.

        Uses corpus IDF values but computes term frequencies on the fly.

        Args:
            query: Query keywords
            document: Document text

        Returns:
            BM25 score
        """
        tokens = self._tokenize(document)
        doc_len = len(tokens)
        term_freqs = Counter(tokens)

        score = 0.0

        for term in query:
            term_lower = term.lower()
            f = term_freqs.get(term_lower, 0)
            if f == 0:
                continue

            # Use corpus IDF, or compute on-the-fly if term is new
            idf = self.idf_cache.get(term_lower, 0)
            if idf == 0:
                # New term not in corpus - assign moderate IDF
                idf = math.log(self.N + 1)

            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)

            score += idf * (numerator / denominator)

        return score

    def normalize_score(self, score: float, max_score: float) -> float:
        """
        Normalize a raw BM25 score to 0-1 given an explicit max.

        Args:
            score: Raw BM25 score
            max_score: The maximum raw score across all documents for this query

        Returns:
            Normalized score in range [0, 1]
        """
        if score == 0 or max_score <= 0:
            return 0.0
        return min(score / max_score, 1.0)

    def score_all_normalized(self, query: List[str]) -> List[float]:
        """
        Score all corpus documents against a query and return normalized scores.

        Computes all raw scores first, then normalizes by the true max — so
        results are order-independent and consistent regardless of iteration order.

        Args:
            query: List of query keywords

        Returns:
            List of normalized scores [0, 1] aligned with self.documents
        """
        raw_scores = [self.score(query, doc) for doc in self.documents]
        max_score = max(raw_scores) if raw_scores else 0.0
        if max_score <= 0:
            return [0.0] * len(raw_scores)
        return [min(s / max_score, 1.0) for s in raw_scores]

