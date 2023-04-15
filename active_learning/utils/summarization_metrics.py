from nltk import ngrams
import re
import string


def split_into_words(text):
    return re.sub(
        "\s+",
        " ",
        text.translate(str.maketrans("", "", string.punctuation))
        .replace("\n", " ")
        .lower(),
    ).split()


def calculate_ngram_overlap(summary, text, n=1, use_modified=True):
    summary_ngrams = list(ngrams(summary, n))
    text_ngrams = list(ngrams(text, n))

    if len(summary_ngrams) > 0:
        ngrams_intersection = set(summary_ngrams).intersection(set(text_ngrams))
        if use_modified:
            word_is_part_of_ngram_copied = [
                any((x in ngram for ngram in ngrams_intersection)) for x in summary
            ]
            return 1 - sum(word_is_part_of_ngram_copied) / len(
                word_is_part_of_ngram_copied
            )
        else:
            return sum([x not in ngrams_intersection for x in summary_ngrams]) / len(
                summary_ngrams
            )
    return None
