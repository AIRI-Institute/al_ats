import os

os.environ["HF_DATASETS_OFFLINE"] = "1"

from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import numpy as np
from tqdm import tqdm
import logging
import multiprocessing as mp
from math import ceil
from pathlib import Path
import pickle

from rouge_score.tokenize import tokenize
from rouge_score.rouge_scorer import porter

logging.basicConfig(
    filename="log.txt",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)

log = logging.getLogger()


def batch_calculate_rouges(
    i_batch, batch_size, ngramm, texts, num_ngrams, path, i_proc
):
    ngrammer = CountVectorizer(
        ngram_range=(ngramm, ngramm), tokenizer=lambda x: x.split(), lowercase=False
    )
    if batch_size < len(texts):
        batch = texts[i_batch * batch_size : (i_batch + 1) * batch_size]
        ngrammer.fit(batch)
        batch_output = ngrammer.transform(texts)

    for i_text, vector in enumerate(batch, start=i_batch * batch_size):
        log.info(f"Processing text {i_text} on CPU {i_proc}")
        result_i_text = (
            sparse.vstack([batch_output[i_text] for _ in range(batch_output.shape[0])])
            - batch_output
        )
        result_i_text[result_i_text < 0] = 0

        ngrams_i_text = batch_output[i_text].sum()
        ngrams_diff_rec = result_i_text.sum(axis=1)

        ngrams_common = ngrams_i_text - ngrams_diff_rec
        recalls_i_text = (ngrams_common / ngrams_i_text).A
        precs_i_text = (ngrams_common / num_ngrams).A
        fscores_i_text = (
            2 * recalls_i_text * precs_i_text / (precs_i_text + recalls_i_text + 1e-15)
        )

        with open(path / f"recalls_{ngramm}_{i_text}", "wb") as f:
            pickle.dump(recalls_i_text.astype(np.float16), f)
        with open(path / f"precisions_{ngramm}_{i_text}", "wb") as f:
            pickle.dump(precs_i_text.astype(np.float16), f)
        with open(path / f"fscores_{ngramm}_{i_text}", "wb") as f:
            pickle.dump(fscores_i_text.astype(np.float16), f)


if __name__ == "__main__":

    data = load_dataset("aeslc", cache_dir="data/")
    train_instances = data["train"]

    texts = [
        " ".join(tokenize(text, porter.PorterStemmer()))
        for text in tqdm(train_instances["email_body"])
    ]
    path = Path("output")
    path.mkdir(exist_ok=True)

    batch_size = 100
    if batch_size is None:
        batch_size = len(texts)

    num_procs = 16
    total_procs = ceil(len(texts) / batch_size)
    num_procs_per_core = ceil(total_procs / num_procs)

    for ngramm in range(1, 3):

        num_ngrams = (
            CountVectorizer(
                ngram_range=(ngramm, ngramm),
                tokenizer=lambda x: x.split(),
                lowercase=False,
            )
            .fit_transform(texts)
            .sum(axis=1)
        )
        processes = []

        for i_core_proc in range(num_procs_per_core):
            for i in tqdm(range(num_procs)):
                idx_batch = i + i_core_proc * num_procs
                if idx_batch * batch_size >= len(texts):
                    continue

                proc = mp.Process(
                    target=batch_calculate_rouges,
                    args=(idx_batch, batch_size, ngramm, texts, num_ngrams, path, i),
                )
                proc.start()
                processes.append(proc)
            for proc in processes:
                proc.join()
