from collections import defaultdict
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Union, List

from active_learning.utils.general import pickle_load


class ActiveSelector:
    def __init__(
        self,
        scores_dir: Path or str,
        idxs: Union[int, List[int], np.ndarray[int]],
        ngramms_range: tuple = (1, 2),
        query_ids: list or tuple = (),
        strategy: str = "mmr_rec_prec",  # "mmr_f1" or "mmr_rec_prec"
        lamb: float = 0.5,
    ):
        self.scores_dir = Path(scores_dir)
        self.ngramms_range = ngramms_range
        self.query_ids = list(query_ids)

        if isinstance(idxs, int):
            self.idxs = np.arange(idxs)
        else:
            self.idxs = np.array(idxs)

        self.strategy = strategy
        self.lamb = lamb
        self.cache = defaultdict(list)

    def query(self, num_queries=1):
        for i_query in range(num_queries):
            self._query_one_sample()
        return self.query_ids

    def _query_one_sample(self):
        ids_in = self.query_ids
        ids_out = np.setdiff1d(self.idxs, ids_in)
        length_out_minus_one = len(ids_out) - 1
        cache = self.cache
        strategy = self.strategy

        if len(cache) == 0:
            for idx in tqdm(ids_out):
                for ngramm in self.ngramms_range:
                    if strategy == "mmr_f1":
                        scores_in_idx = scores_out_idx = pickle_load(
                            self.scores_dir / f"fscores_{ngramm}_{idx}"
                        )
                    elif strategy == "mmr_rec_prec":
                        # These are "how much info of each text is contained in text `idx`"
                        # i.e. [precision(text_0, text_idx), precision(text_1, text_idx), ...]
                        scores_in_idx = pickle_load(
                            self.scores_dir / f"recalls_{ngramm}_{idx}"
                        )
                        # These are "how much info of text `idx` is contained in each text "
                        # i.e. [recall(text_0, text_idx), recall(text_1, text_idx), ...]
                        scores_out_idx = pickle_load(
                            self.scores_dir / f"precisions_{ngramm}_{idx}"
                        )
                    else:
                        raise NotImplementedError

                    sim_in = scores_in_idx[ids_in].max() if len(ids_in) else 0
                    sim_out = (scores_out_idx[ids_out].sum() - 1) / length_out_minus_one
                    cache[f"idx_out_scores_{ngramm}"].append(sim_out)
                    cache[f"idx_in_scores_{ngramm}"].append(sim_in)

            for ngramm in self.ngramms_range:
                cache[f"idx_out_scores_{ngramm}"] = np.array(
                    cache[f"idx_out_scores_{ngramm}"], dtype=float
                )
                cache[f"idx_in_scores_{ngramm}"] = np.array(
                    cache[f"idx_in_scores_{ngramm}"], dtype=float
                )
        else:
            n = len(ids_out)
            for ngramm in self.ngramms_range:
                last_query_out_scores = cache[f"query_out_out_scores_{ngramm}"]
                last_query_in_scores = cache[f"query_out_in_scores_{ngramm}"]
                # n instead if (n+1) since we ecxlude the sim of each item with itself
                cache[f"idx_out_scores_{ngramm}"] = (
                    cache[f"idx_out_scores_{ngramm}"] * n - last_query_out_scores
                ) / (n - 1)
                cache[f"idx_in_scores_{ngramm}"] = np.maximum(
                    cache[f"idx_in_scores_{ngramm}"], last_query_in_scores
                )

        ngramm_scores = []
        for ngramm in self.ngramms_range:
            cache[f"idx_scores_{ngramm}"] = (
                self.lamb * cache[f"idx_out_scores_{ngramm}"]
                - (1 - self.lamb) * cache[f"idx_in_scores_{ngramm}"]
            )
            ngramm_scores.append(cache[f"idx_scores_{ngramm}"])

        ngramm_scores = cache["ngramm_scores"] = np.stack(ngramm_scores, axis=1)
        scores = np.prod(np.maximum(ngramm_scores, 0), axis=1) ** (
            1 / len(self.ngramms_range)
        )
        one_of_values_lower_zero = (ngramm_scores < 0).sum(axis=1) > 0
        modified_scores = (
            np.prod(np.maximum(ngramm_scores + 1, 0), axis=1)
            ** (1 / len(self.ngramms_range))
            - 1
        )
        cache["scores"] = final_scores = (
            scores * (~one_of_values_lower_zero)
            + modified_scores * one_of_values_lower_zero
        )

        argmax = np.argmax(final_scores)
        id_query = int(ids_out[argmax])  # to cast it to standard int type
        self.query_ids.append(id_query)
        # Cache query out scores
        ids_out = np.setdiff1d(ids_out, id_query)
        for ngramm in self.ngramms_range:
            if strategy == "mmr_f1":
                scores_in_idx = scores_out_idx = pickle_load(
                    self.scores_dir / f"fscores_{ngramm}_{id_query}"
                )
            elif strategy == "mmr_rec_prec":
                # Note that we use recall for `out` and precision for `in` (opposite to what was done earlier)
                # since precision(x1, x2) = recall(x2, x1)
                scores_in_idx = pickle_load(
                    self.scores_dir / f"precisions_{ngramm}_{id_query}"
                )
                scores_out_idx = pickle_load(
                    self.scores_dir / f"recalls_{ngramm}_{id_query}"
                )
            else:
                raise NotImplementedError
            # First "out" refers to the fact that these are scores with "out" samples; second - to the metric ("in" / "out")
            cache[f"query_out_out_scores_{ngramm}"] = scores_out_idx[ids_out].ravel()
            cache[f"query_out_in_scores_{ngramm}"] = scores_in_idx[ids_out].ravel()
            # Update cached in and out scores: remove the score of the query
            cache[f"idx_out_scores_{ngramm}"] = np.delete(
                cache[f"idx_out_scores_{ngramm}"], argmax
            )
            cache[f"idx_in_scores_{ngramm}"] = np.delete(
                cache[f"idx_in_scores_{ngramm}"], argmax
            )

        self.cache = cache
