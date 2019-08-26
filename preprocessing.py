import numpy as np
import os
import pandas as pd
import re
import spacy

from collections import defaultdict


def build_cooccurrence_matrix(corpus, window_size=5, scale_factor="scaled",
                              vocab_size=5000, unkown_vector=True):
    """
    Builds a co-occurrence matrix with `window_size` and `scale_factor` over `corpus`.

    Parameters
    ----------
    corpus : iterator
        Should be a generator where each document is returned as a list of words. This
        corpus should already be normalized.
    window_size : int
        The size of the symmetrical window. The words to be considered when doing the count
        will be the ones being `window_size` before and after the center word.
    scale_factor : str, one of {"flat", "scaled"}
        The factor to use in order to weight the importance of a word regarding the center
        word. Can be either "flat" (equal weights to every word), or "scaled" where the
        word is weighted by 1/d where d is the distance to the center word.
    vocab_size : int
        The maximum size of the vocabulary
    unknown_vector : bool
        Whether to use a vector "UNK" for the words outside of the vocabulary.

    Returns
    -------
    pd.DataFrame
        A tuple where the first element is the dictionary with the word_count of the
        top `vocab_size` words in the vocabulary, and the second element is the (scaled) matrix
        of cooccurrences for the top `vocab_size` words in the vocabulary.
    """

    assert scale_factor in {"flat", "scaled"}

    word_count = defaultdict(int)
    word_word = defaultdict(int)

    for document in corpus:
        for idx, word in enumerate(document):
            word_count[word] += 1
            lwindow = reversed(document[max(idx-window_size,0):idx])
            rwindow = document[idx+1:idx+1+window_size]

            for lidx, lword in enumerate(lwindow):
                word_word[(word, lword)] += 1/(lidx+1) if scale_factor == "scaled" else 1

            for ridx, rword in enumerate(rwindow):
                word_word[(word, rword)] += 1/(ridx+1) if scale_factor == "scaled" else 1

    vocab = [word for word in sorted(word_count, key=lambda w: word_count[w],
                                     reverse=True)[:vocab_size]]
    vocab = {word: idx for idx, word in enumerate(sorted(vocab))}

    if unkown_vector:
        vocab["UNK"] = len(vocab)

    word_matrix = np.zeros((len(vocab), len(vocab)))

    for (w1, w2), count in word_word.items():
        w1_idx = vocab.get(w1, vocab.get("UNK", None))
        w2_idx = vocab.get(w2, vocab.get("UNK", None))

        if w1_idx is not None and w2_idx is not None:
            word_matrix[w1_idx, w2_idx] = count

    vocab_ = [word for word in sorted(vocab, key=lambda x: vocab[x])]
    word_count = {word: count for word, count in word_count.items() if word in vocab}

    return word_count, pd.DataFrame(word_matrix, index=vocab_, columns=vocab_)


def corpus_processor(corpus_directory, language_model="es", remove_stopwords=True,
                     lowercase=True):
    """
    Generator to retrieve and process the files of a corpus.

    Parameters
    ----------
    corpus_directory : str
        Path to where the corpus is stored as text documents.
    language_model : str
        The SpaCy language model to use for the processing.
    remove_stopwords : bool
        Whether to remove stopwords from the processed corpus.
    lowercase : bool
        Whether to lowercase the words.

    Returns
    -------
        generator
        A generator object where each entry is a processed document.
    """

    nlp = spacy.load(language_model)

    for fname in os.listdir(corpus_directory):
        with open(os.path.join(corpus_directory, fname), "r") as fh:
            # Careful with this for very large docs
            document = re.sub(r"\s+", " ", fh.read())
        tokens = [
            token.text for token in nlp(document, disable=["tagger", "parser", "ner"])
            if not (remove_stopwords and token.is_stop)
        ]

        if lowercase:
            tokens = [token.lower() for token in tokens]

        yield tokens
