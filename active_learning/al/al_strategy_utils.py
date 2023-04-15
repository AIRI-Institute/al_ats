import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple
from datasets.arrow_dataset import Dataset as ArrowDataset
from datasets import load_metric, concatenate_datasets
import logging
import time
from sklearn.neighbors import KNeighborsClassifier
from math import ceil
from pathlib import Path
from omegaconf.dictconfig import DictConfig
from requests.models import HTTPError
import pickle
import gc
from collections import Counter
import random

from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers.data.data_collator import torch_default_data_collator

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..utils.get_embeddings import get_embeddings
from ..utils.general import json_dump, json_load
from ..utils.transformers_dataset import TransformersDataset

from .strategy_utils import (
    compute_centroids,
    compute_inv_covariance,
    compute_inv_covariance_v2,
    mahalanobis_distance_with_known_centroids_sigma_inv,
    mahalanobis_distance_with_known_centroids_sigma_inv_v2,
)


log = logging.getLogger()


def get_query_idx_for_selecting_by_number_of_tokens(
    data: Union[
        List[Union[str, List[str]]],
        np.ndarray,
        ArrowDataset,
        TransformersDataset,
    ],
    sorted_idx: List[int] or np.ndarray[int],
    num_to_select: int,
    tokens_column_name: str or None = None,
    include_last: bool = True,
):
    """
    Function to get the id of the last selected sample `id_bound`, so that sorted_data[:id_bound] will be selected
    """
    sorted_data = (
        data[sorted_idx]
        if isinstance(data, np.ndarray)
        else [data[i] for i in sorted_idx]
        if isinstance(data, list)
        else data.select(sorted_idx)
    )

    if tokens_column_name is None:
        tokens_column_name = "tokens"

    if isinstance(sorted_data[0], str):
        sample_num_tokens = [len(x.split()) for x in sorted_data]
    elif isinstance(sorted_data[0], list):
        sample_num_tokens = [len(x) for x in sorted_data]
    else:
        sample_num_tokens = [len(x) for x in sorted_data[tokens_column_name]]

    cumsum_num_tokens = np.cumsum(sample_num_tokens)
    # get index of last id to select
    last_id = np.argwhere(cumsum_num_tokens > num_to_select).ravel().min() + int(
        include_last
    )
    query_idx = sorted_idx[:last_id]
    return query_idx


def get_ups_sampling_probas(argsort, gamma, T):
    ranks = argsort.argsort() / len(argsort)
    return np.exp(-np.maximum(0, ranks - gamma) / np.maximum(T, 1e-8))


def sample_idxs(sampling_probas):
    to_select = []
    for i in range(len(sampling_probas)):
        proba_to_choose_i = sampling_probas[i]
        if np.random.uniform() < proba_to_choose_i:
            to_select.append(i)
    log.info(f"\nTaken {len(to_select) / len(sampling_probas)} samples\n")
    return np.array(to_select)


def calculate_mnlp_score(probas) -> np.ndarray:
    return np.array([-np.sum(np.log(np.max(i, axis=1))) / len(i) for i in probas])


def probability_variance_ner(sampled_probabilities, mean_probabilities=None):
    if mean_probabilities is None:
        mean_probabilities = np.mean(sampled_probabilities, axis=1)

    mean_probabilities = np.expand_dims(mean_probabilities, axis=1)
    tmp = sampled_probabilities - mean_probabilities
    tmp = np.array([i**2 for i in tmp])
    tmp = np.array([np.mean(i) for i in tmp])
    return np.array([np.sum(i) for i in tmp])


def mean_entropy_ner(sampled_probabilities):
    sum_m = np.mean(sampled_probabilities, axis=1)
    H = np.array([-np.sum(i * np.log(np.clip(i, 1e-8, 1)), axis=-1) for i in sum_m])
    return np.array([np.mean(i) for i in H])


def calculate_bald_score_ner(probas_array) -> np.ndarray:
    # mean over M - number of forward passes
    sum_m = np.mean(probas_array, axis=1)
    H = np.array([-np.sum(i * np.log(np.clip(i, 1e-8, 1)), axis=-1) for i in sum_m])
    E_H = np.mean(
        [
            [-np.array(np.sum(j * np.log(np.clip(j, 1e-8, 1)), axis=-1)) for j in i]
            for i in probas_array
        ],
        axis=1,
    )
    bald = H - E_H

    return np.array([np.mean(i) for i in bald])


def entropy(x):
    return np.sum(-x * np.log(np.clip(x, 1e-8, 1)), axis=-1)


def mean_entropy(sampled_probabilities):
    return entropy(np.mean(sampled_probabilities, axis=1))


def calculate_bald_score_cls(sampled_probabilities) -> np.ndarray:
    # sampled_probabilities: batch_size (n) x num_samples (k) x num_labels (c)
    predictive_entropy = entropy(np.mean(sampled_probabilities, axis=1))
    expected_entropy = np.mean(entropy(sampled_probabilities), axis=1)

    return predictive_entropy - expected_entropy


def var_ratio(sampled_probabilities):
    top_classes = np.argmax(sampled_probabilities, axis=-1)
    # count how many time repeats the strongest class
    mode_count = lambda preds: np.max(np.bincount(preds))
    modes = [mode_count(point) for point in top_classes]
    ue = 1.0 - np.array(modes) / sampled_probabilities.shape[1]
    return ue


def sampled_max_prob(sampled_probabilities):
    mean_probabilities = np.mean(sampled_probabilities, axis=1)
    top_probabilities = np.max(mean_probabilities, axis=-1)
    return 1 - top_probabilities


def probability_variance(sampled_probabilities, mean_probabilities=None):
    if mean_probabilities is None:
        mean_probabilities = np.mean(sampled_probabilities, axis=1)

    mean_probabilities = np.expand_dims(mean_probabilities, axis=1)

    return ((sampled_probabilities - mean_probabilities) ** 2).mean(1).sum(-1)


def take_idx(X_pool, idx):
    return X_pool[idx] if isinstance(X_pool, np.ndarray) else X_pool.select(idx)


def concatenate_data(
    data_1: Union[ArrowDataset, TransformersDataset],
    data_2: Union[ArrowDataset, TransformersDataset],
):
    if isinstance(data_1, ArrowDataset):
        data_1 = concatenate_datasets([data_1, data_2], info=data_1.info)
    elif isinstance(data_1, TransformersDataset):
        data_1.add(data_2)
    else:
        raise NotImplementedError
    return data_1


def _get_embeddings_wrapper(
    model_wrapper, data, model=None, data_config=None, **get_embeddings_kwargs
):
    data_config = data_config if data_config is not None else model_wrapper.data_config
    model = model_wrapper.model if model is None else model
    if model._get_name().endswith("Model"):
        prepare_model = False
    else:
        prepare_model = True
    tokenizer = model_wrapper.tokenizer
    tokenization_kwargs = dict(
        task=model_wrapper.task,
        text_name=data_config["text_name"],
        label_name=data_config["label_name"],
    )
    get_embeddings_kwargs.update(tokenization_kwargs)
    torch.manual_seed(model_wrapper.seed)

    return get_embeddings(
        model,
        data,
        tokenizer=tokenizer,
        prepare_model=prepare_model,
        **get_embeddings_kwargs,
    )


def get_similarities(
    model_name,
    X_u,
    X_l,
    sims_func="scalar_product",
    average=False,
    text_name="text",
    device="cuda",
    cache_dir=None,
    return_embeddings=False,
    batch_size: int = 100,
) -> Tuple[torch.Tensor, List[int]]:

    if cache_dir is not None:
        model_cache_dir = Path(cache_dir) / "model"
        tokenizer_cache_dir = Path(cache_dir) / "tokenizer"
    else:
        model_cache_dir = None
        tokenizer_cache_dir = None

    if X_l is not None:
        all_data = concatenate_data(X_u, X_l)
    else:
        all_data = X_u
        X_l = []

    # init models and tokenizer
    model = AutoModel.from_pretrained(model_name, cache_dir=model_cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=tokenizer_cache_dir)
    if tokenizer.model_max_length > 5000:
        tokenizer.model_max_length = 512
    all_data = all_data.map(
        lambda x: tokenizer(
            x[text_name],
            truncation=True,
        ),
        batched=True,
        remove_columns=all_data.column_names,
        load_from_cache_file=False,
    )
    embeddings = get_embeddings(
        model,
        all_data,
        prepare_model=False,
        data_is_tokenized=True,
        tokenizer=tokenizer,
        use_averaging=average,
        use_automodel=True,
        batch_size=batch_size,
    )

    # TODO: seems to be a bug that must be fixed. Check whether is equal to selecting unique documents.
    embs_unique, counts = torch.unique(
        embeddings[: len(X_u)], return_inverse=True, dim=0
    )
    counts = counts.cpu().numpy().tolist()
    if len(X_l) > 0:
        embs_unique = torch.cat([embs_unique, embeddings[len(X_u) :]], dim=0)

    if sims_func == "scalar_product":
        similarities = torch.mm(embs_unique, embs_unique.T)
    elif sims_func == "cosine_similarity":
        embs_unique = embs_unique / embs_unique.norm(dim=1)[:, None]
        similarities = torch.mm(embs_unique, embs_unique.T)
    elif sims_func == "euclidean_distance":
        similarities = _get_euclidean_distances(embs_unique)
    elif sims_func == "euclidean_normalized_distance":
        embs_unique = embs_unique / embs_unique.norm(dim=1)[:, None]
        similarities = _get_euclidean_distances(embs_unique)
    elif sims_func == "mahalanobis_similarity":
        similarities = _get_mahalanobis_similarities(embs_unique)
    elif sims_func == "mahalanobis_distance":
        similarities = _get_mahalanobis_distances(embs_unique)
    else:
        raise NotImplementedError

    del model, tokenizer, embeddings
    torch.cuda.empty_cache()
    gc.collect()

    if return_embeddings:
        return similarities, counts, embs_unique

    del embs_unique
    torch.cuda.empty_cache()
    return similarities, counts


def _get_mahalanobis_similarities(embeddings):

    train_labels = np.repeat(0, len(embeddings))
    centroid, _ = compute_centroids(embeddings, train_labels, 1)
    sigma_inv, _ = compute_inv_covariance(centroid, embeddings, train_labels)
    return embeddings @ sigma_inv @ embeddings.T


def _get_mahalanobis_distances(embeddings):
    num_obj = len(embeddings)

    train_labels = np.repeat(0, num_obj)
    centroid, _ = compute_centroids(embeddings, train_labels, 1)
    sigma_inv, _ = compute_inv_covariance(centroid, embeddings, train_labels)

    # the smaller the value, the closer two objects are
    maha_scores = torch.empty(
        num_obj, num_obj, dtype=torch.float32, device=embeddings.device
    )
    batch_size = 100
    num_batches = num_obj // batch_size + int(num_obj % batch_size > 0)
    for i_batch in tqdm(range(num_batches)):
        iloc = slice(i_batch * batch_size, (i_batch + 1) * batch_size)
        batch_diff = embeddings[iloc, None, :] - embeddings[None, :, :]
        maha_scores[iloc].copy_((batch_diff @ sigma_inv * batch_diff).sum(-1))
    return -maha_scores


def _get_euclidean_distances(embeddings):
    num_obj = len(embeddings)

    train_labels = np.repeat(0, num_obj)
    centroid, _ = compute_centroids(embeddings, train_labels, 1)
    sigma_inv, _ = compute_inv_covariance(centroid, embeddings, train_labels)

    # the smaller the value, the closer two objects are
    dists = torch.empty(num_obj, num_obj, dtype=torch.float32, device=embeddings.device)
    batch_size = 100
    num_batches = num_obj // batch_size + int(num_obj % batch_size > 0)
    for i_batch in tqdm(range(num_batches)):
        iloc = slice(i_batch * batch_size, (i_batch + 1) * batch_size)
        batch_dists = torch.square(
            embeddings[iloc, None, :] - embeddings[None, :, :]
        ).sum(-1)
        dists[iloc].copy_(batch_dists)
    return -dists


def _get_similarities_from_cache_or_from_scratch(
    cache_dir, model_name, X_u, X_l, text_name, label_name, obj_id_name, device
):
    # check if we can get all stuff from cache
    if cache_dir is not None:
        similarities_path = Path(cache_dir) / "embeddings"
        indexes_path = Path(cache_dir) / "indexes"
        if similarities_path.is_file() and indexes_path.is_file():
            idx2obj = pickle.load(indexes_path.open("rb"))
            obj2idx = {obj: idx for idx, obj in idx2obj.items()}
            similarities = torch.load(similarities_path)
            return similarities, idx2obj, obj2idx

    all_data = concatenate_data(X_u, X_l)
    idx2obj = {i: all_data[i][obj_id_name] for i in range(len(all_data))}
    obj2idx = {obj: idx for idx, obj in idx2obj.items()}

    # init models and tokenizer
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    all_data = all_data.map(
        lambda x: tokenizer(
            x[text_name],
            truncation=True,
        ),
        batched=True,
        remove_columns=all_data.column_names,
    )
    embeddings = get_embeddings(
        model,
        all_data,
        prepare_model=False,
        data_is_tokenized=True,
        tokenizer=tokenizer,
        use_automodel=True,
    )
    # get similarities from embeddings
    # https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
    embs_norm = embeddings / embeddings.norm(dim=1)[:, None]
    similarities = torch.mm(embs_norm, embs_norm.T)

    # save to cache if path is provided
    if cache_dir is not None:
        similarities_path = Path(cache_dir) / "embeddings"
        indexes_path = Path(cache_dir) / "indexes"
        pickle.dump(idx2obj, indexes_path.open("wb"))
        torch.save(similarities, similarities_path)

    return similarities, idx2obj, obj2idx


def calculate_unicentroid_mahalanobis_distance(embs_unique, labeled_indices):
    train_embeddings = embs_unique[labeled_indices]
    train_labels = np.repeat(0, len(labeled_indices))

    centroids, centroids_mask = compute_centroids(train_embeddings, train_labels, 1)
    sigma_inv, _ = compute_inv_covariance(centroids, train_embeddings, train_labels)
    dists = mahalanobis_distance_with_known_centroids_sigma_inv(
        centroids,
        centroids_mask,
        sigma_inv,
        embs_unique,
    )
    closest_centroid_distance = dists.min(dim=1)[0]
    return closest_centroid_distance


def calculate_mahalanobis_distance(
    model_wrapper,
    train_data: Union[ArrowDataset, TransformersDataset] = None,
    unlabeled_data: Union[ArrowDataset, TransformersDataset] = None,
    embeddings: torch.Tensor = None,  # if provided, expected to be the emb-s of train followed by emb-s of unlabeled
    classwise: bool = True,  # use an own centroid for each class
    batched: bool = False,  # whether to recalculate centroids & sigma_inv after each iteration
    outlier_ids: Union[
        np.ndarray, List[int]
    ] = None,  # necessary when `batched=True` to filter outlers
    n_instances: Union[
        int, None
    ] = None,  # necessary when `batched=True` to know how many instances to select
    use_triplet: bool = False,  # whether calculate Mahalanobis Triplet scores
    unicentroid_dists: torch.Tensor = None,  # used when `use_triplet`, otherwise ignored
    triplet_lambda: float = 0.25,  # used when `use_triplet`, otherwise ignored
    use_v2: bool = False,
    use_activation: bool = False,
    use_spectralnorm: bool = True,
    model=None,  # if provided, is used instead of `model_wrapper.model`
    data_is_tokenized: bool = False,
    data_config=None,
    batch_size: int = 100,
    to_numpy: bool = True,
) -> Union[Union[torch.Tensor, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    func_for_inv_cov = (
        compute_inv_covariance if not use_v2 else compute_inv_covariance_v2
    )
    func_for_dists_calc = (
        mahalanobis_distance_with_known_centroids_sigma_inv
        if not use_v2
        else mahalanobis_distance_with_known_centroids_sigma_inv_v2
    )
    if use_triplet:
        assert (
            0 <= triplet_lambda <= 1
        ), f"Lambda (triplet_lambda) must be between 0 and 1 but received the value {triplet_lambda}"

    # Calculate the embeddings if not provided
    if embeddings is None:
        all_data = concatenate_data(train_data, unlabeled_data)
        embeddings = _get_embeddings_wrapper(
            model_wrapper,
            all_data,
            model,
            data_config,
            use_activation=use_activation,
            use_spectralnorm=use_spectralnorm,
            data_is_tokenized=data_is_tokenized,
            batch_size=batch_size,
            to_numpy=False,
        )

    if use_triplet and unicentroid_dists is None:
        unicentroid_dists = calculate_mahalanobis_distance(
            model_wrapper,
            train_data,
            unlabeled_data,
            embeddings=embeddings,
            classwise=False,
            batched=False,
            model=model,
            use_v2=use_v2,
            use_activation=use_activation,
            use_spectralnorm=use_spectralnorm,
            data_is_tokenized=data_is_tokenized,
            data_config=data_config,
            batch_size=batch_size,
            to_numpy=False,
        ).squeeze(1)

    train_embeddings = embeddings[: len(train_data)]
    eval_embeddings = embeddings[len(train_data) :]
    if classwise:
        label_name = (
            data_config["label_name"]
            if data_config is not None
            else model_wrapper.data_config["label_name"]
        )
        train_labels = np.array(train_data[label_name])
        num_labels = model_wrapper.num_labels
    else:
        train_labels = np.zeros(len(train_data))
        num_labels = 1

    if not batched:
        centroids, centroids_mask = compute_centroids(
            train_embeddings, train_labels, num_labels
        )
        sigma_inv, _ = func_for_inv_cov(centroids, train_embeddings, train_labels)
        dists = func_for_dists_calc(
            centroids,
            centroids_mask,
            sigma_inv,
            eval_embeddings,
        )
        if use_triplet:
            closest_centroid_distance = dists.min(dim=1)[0]
            dists = (
                closest_centroid_distance * triplet_lambda
                - (1 - triplet_lambda) * unicentroid_dists
            )
        if to_numpy:
            return dists.cpu().detach().numpy()
        return dists
    else:
        if not classwise:
            raise ValueError(
                "Non-classwise Mahalanobis does not need to be run batched! Check your configuration."
            )
        id_queries = []
        queries_uncertainty = []
        for i in tqdm(range(n_instances), desc="Queries done:"):
            centroids, centroids_mask = compute_centroids(
                train_embeddings, train_labels, num_labels
            )
            sigma_inv, _ = func_for_inv_cov(centroids, train_embeddings, train_labels)
            dists = func_for_dists_calc(
                centroids,
                centroids_mask,
                sigma_inv,
                eval_embeddings,
            )
            closest_centroid_distance = dists.min(dim=1)[
                0
            ]  # taking zero el. since torch.min returns (vals, ids)
            if use_triplet:
                scores = (
                    closest_centroid_distance * triplet_lambda
                    - (1 - triplet_lambda) * unicentroid_dists
                )
            else:
                scores = closest_centroid_distance
            scores[id_queries] = -float("inf")
            if outlier_ids is not None:
                scores[outlier_ids] = -float("inf")
            id_query = scores.argmax().item()
            id_queries.append(id_query)
            queries_uncertainty.append(scores[id_query].item())
            train_embeddings = torch.cat(
                [train_embeddings, eval_embeddings[id_query].unsqueeze(0)], dim=0
            )
            train_labels = np.append(train_labels, unlabeled_data[id_query][label_name])
        uncertainty_estimates = scores.cpu().detach().numpy()
        uncertainty_estimates[id_queries] = queries_uncertainty
        return np.array(id_queries), uncertainty_estimates


def calculate_mahalanobis_filtering_scores(
    model_wrapper,
    train_data: Union[ArrowDataset, TransformersDataset],
    unlabeled_data: Union[ArrowDataset, TransformersDataset],
    strategy: str = "lc",  # strategy to use for uncertainty calculation
    filtering_share: float = 0.01,
    use_finetuned: bool = True,
    **mahalanobis_kwargs,
):
    assert (
        0 <= filtering_share < 1
    ), f"Parameter filtering share must be between 0 and 1 but received the value {filtering_share}"

    if not use_finetuned:
        checkpoint_name = model_wrapper.model.name_or_path
        try:
            model = AutoModel.from_pretrained(checkpoint_name).cuda()
        except HTTPError:
            model = AutoModel.from_pretrained(
                checkpoint_name, local_files_only=True
            ).cuda()
    else:
        model = None

    batched = mahalanobis_kwargs.pop("batched")

    unicentroid_dists = calculate_mahalanobis_distance(
        model_wrapper,
        train_data,
        unlabeled_data,
        classwise=False,
        batched=False,
        model=model,
        to_numpy=True,
        **mahalanobis_kwargs,
    ).squeeze(1)

    num_to_filter = round(filtering_share * len(unicentroid_dists))
    outlier_ids = np.argsort(unicentroid_dists)[-num_to_filter:]

    if strategy == "lc":
        uncertainty_scores = 1 - model_wrapper.predict_proba(unlabeled_data).max(axis=1)
    elif strategy == "entropy":
        probas = model_wrapper.predict_proba(unlabeled_data)
        uncertainty_scores = np.sum(-probas * np.log(probas), axis=1)
    elif strategy == "margin":
        probas = model_wrapper.predict_proba(unlabeled_data)
        probas.sort(axis=1)
        max_probas = probas[:, -1]
        second_max_probas = probas[:, -2]
        uncertainty_scores = 1 + second_max_probas - max_probas
    elif strategy == "mahalanobis":
        maha_output = calculate_mahalanobis_distance(
            model_wrapper,
            train_data,
            unlabeled_data,
            classwise=True,  # use an own centroid for each class
            batched=batched,
            outlier_ids=outlier_ids,
            to_numpy=True,
            **mahalanobis_kwargs,
        )
        if not batched:
            uncertainty_scores = maha_output.min(axis=1)
        else:
            return maha_output
    else:
        raise NotImplementedError
    if not batched:
        uncertainty_scores[outlier_ids] = -float("inf")

    return uncertainty_scores


def _get_labels(data, data_config):
    if isinstance(data, DataLoader):
        data = data.dataset
    return np.array(data[data_config["label_name"]])


def probably_to_cpu_and_calculate_time(
    scores, to_numpy, start_time, time_dict_path, model_name
):
    if to_numpy:
        scores = scores.cpu().detach().numpy()

    time_work = time.time() - start_time
    time_dict = json_load(time_dict_path)
    time_dict[model_name + "_predict"].append(time_work)
    json_dump(time_dict, time_dict_path)
    return scores


def get_embeddings_(
    net,
    loader: torch.utils.data.DataLoader,
    num_dim: int,
    dtype,
    device,
    storage_device,
    batch_size=100,
):
    num_samples = len(loader.dataset)
    embeddings = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for batch in tqdm(loader):
            data = batch["image"].to(device)
            label = batch["labels"].to(device)

            if isinstance(net, nn.DataParallel):
                net.module(data, label)
                out = net.module.feature
            else:
                net(data, label)
                out = net.feature

            end = start + len(data)
            embeddings[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end

    return embeddings, labels


def get_X_pool_subsample(X_pool, subsample: Union[int, float], seed: int):
    subsample_length = round(len(X_pool) / subsample)
    np.random.seed(seed)
    subsample_indices = np.random.choice(range(len(X_pool)), subsample_length, False)
    X_pool_subsample = X_pool.select(subsample_indices)
    return X_pool_subsample, subsample_indices


def assign_ue_scores_for_unlabeled_data(
    len_unlabeled_data: int, subsample_indices: np.ndarray, subsample_scores: np.ndarray
):
    scores = np.zeros(shape=len_unlabeled_data, dtype=subsample_scores.dtype) + float(
        "inf"
    )
    scores[subsample_indices] = subsample_scores
    return scores


def filter_by_uncertainty(
    uncertainty_estimates: Union[
        np.ndarray
    ],  # the larger the uncertainty, the more valuable the example is
    uncertainty_threshold: Union[float, None],
    uncertainty_mode: str,
    n_instances: int,
    uncertainty_scores: Union[
        np.ndarray, None
    ] = None,  # probability scores obtained via - `.generate`.
    # If none, coincides with `uncertainty_estimates`.
):
    if uncertainty_scores is None:
        uncertainty_scores = uncertainty_estimates
    if uncertainty_mode == "absolute":
        cutoff_idx = np.sum(uncertainty_scores >= uncertainty_threshold)
    elif uncertainty_mode == "relative":
        cutoff_idx = ceil(len(uncertainty_estimates) * uncertainty_threshold)
    cutoff_idx = min(cutoff_idx, len(uncertainty_estimates) - n_instances)

    argsort = np.argsort(-uncertainty_estimates)
    uncertainty_estimates[argsort[:cutoff_idx]] = np.inf
    query_idx = argsort[cutoff_idx : n_instances + cutoff_idx]
    return query_idx, uncertainty_estimates


def filter_by_metric(
    uncertainty_threshold: Union[float, None],
    texts: Union[ArrowDataset, TransformersDataset],
    generated_sequences_ids: Union[List, np.ndarray],
    tokenizer,
    metric_cache_dir: Union[Path, str],
    uncertainty_mode: str = "absolute",
    uncertainty_estimates: np.ndarray = None,
    n_instances: int = None,
    metric_name: str = "sacrebleu",
    agg: str = "precision",
    modify_uncertainties: bool = True,
):
    sequences = tokenizer.batch_decode(
        generated_sequences_ids, skip_special_tokens=True
    )
    metric_name_to_load = metric_name
    kwargs = {}
    if metric_name.startswith("rouge"):
        metric_name_to_load = "rouge"
        kwargs = {"use_stemmer": True}
    elif metric_name == "sacrebleu":
        texts = [[x] for x in texts]
        metric_name = "score"

    metric = load_metric(metric_name_to_load, cache_dir=metric_cache_dir)
    metric_scores = metric.compute(
        predictions=sequences, references=texts, use_agregator=False, **kwargs
    )[metric_name]
    if metric_name_to_load == "rouge":
        metric_scores = np.array([getattr(x, agg) for x in metric_scores])

    if uncertainty_mode == "absolute":
        cutoff_idx = np.sum(metric_scores <= uncertainty_threshold)
    elif uncertainty_mode == "relative":
        cutoff_idx = ceil(len(generated_sequences_ids) * uncertainty_threshold)
    cutoff_idx = min(cutoff_idx, len(generated_sequences_ids) - n_instances)
    metric_scores_argsort = np.argsort(metric_scores)

    if not modify_uncertainties:
        return cutoff_idx

    # Make them "confident" to avoid querying them
    uncertainty_estimates[metric_scores_argsort[:cutoff_idx]] = -np.inf
    query_idx = np.argsort(-uncertainty_estimates)[:n_instances]
    # Return large scores (do we need it?)
    uncertainty_estimates[metric_scores_argsort[:cutoff_idx]] = np.inf
    return query_idx, uncertainty_estimates


def calculate_bleuvar_scores(
    summaries: List[List[str]],
    metric_name: str = "sacrebleu",
    cache_dir=None,
    tokenizer=None,
) -> np.ndarray:
    mc_iterations = len(summaries)
    len_sample = len(summaries[0])
    # sacrebleu is normally more robust than bleu
    sacrebleu_metric = load_metric(metric_name, cache_dir=cache_dir)
    key_name = "score" if metric_name != "bleu" else "bleu"
    if metric_name == "bleu":
        for i in range(len(summaries)):
            for idx in range(len(summaries[i])):
                summaries[i][idx] = tokenizer.tokenize(summaries[i][idx])
    sacrebleu_vars = []
    for idx in tqdm(range(len_sample)):
        sacrebleuvar_sum = 0.0
        for i in range(mc_iterations):
            for j in range(mc_iterations):
                if i == j:
                    continue

                sacrebleuvar_sum += (
                    1
                    - round(
                        sacrebleu_metric.compute(
                            predictions=[summaries[i][idx]],
                            references=[[summaries[j][idx]]],
                        )[key_name],
                        4,
                    )
                    / 100.0
                ) ** 2

        sacrebleu_vars.append(
            1 / (mc_iterations * (mc_iterations - 1)) * sacrebleuvar_sum
        )

    return np.array(sacrebleu_vars)


def choose_cm_samples(cluster_labels, query_idx, n_instances, random_query):
    # assume that we added cluster labels as coulmn in dataset
    # so we get samples, scores and clusters indices
    cluster_sizes = Counter(cluster_labels)
    new_query_idx = []
    samples_idx = []
    # split all query_idx to array by cluster label
    query_idx_by_clusters = {
        idx: list(query_idx[np.where(cluster_labels == idx)])
        for idx in list(cluster_sizes.keys())
    }

    # try faster approach - collect all indices that we will sample
    samples_per_cluster = []
    avg_sample_per_cluster = np.ceil(n_instances / len(cluster_sizes))

    sorted_cluster = [
        el[1] for el in sorted([(v, k) for k, v in cluster_sizes.items()])
    ]
    # else find smaller cluster, and sample its size, after subtract n_clusters * min_size and do same
    curr_idx = 0
    while sum(cluster_sizes.values()) > 0 and len(new_query_idx) < n_instances:
        # sample data from each cluster
        # sorted_cluster - array with cluster numbers in ascending order by size
        # cluster_labels - dict with cluster sizes as values
        curr_cluster = sorted_cluster[curr_idx]
        if cluster_sizes[curr_cluster] == 0:
            curr_idx = (curr_idx + 1) % len(sorted_cluster)
            continue
        # randomly sample from data with curr_cluster labels
        if random_query:
            sample_idx = random.choice(
                np.arange(len(query_idx_by_clusters[curr_cluster]))
            )
        else:
            sample_idx = 0
        samples_idx.append(sample_idx)
        new_query_idx.append(query_idx_by_clusters[curr_cluster][sample_idx])
        # and remove this sample from data
        query_idx_by_clusters[curr_cluster] = np.delete(
            query_idx_by_clusters[curr_cluster], sample_idx
        )
        # after subtract 1 from this cluster
        cluster_sizes[curr_cluster] -= 1
        curr_idx = (curr_idx + 1) % len(sorted_cluster)
    return np.array(new_query_idx), samples_idx


def get_num_query_instances_or_tokens(
    config: DictConfig,
    initial_data: Union[list, TransformersDataset, ArrowDataset],
    unlabeled_data: Union[list, TransformersDataset, ArrowDataset],
    framework: str = "transformers",
    tokens_column_name: str or None = None,
) -> int:
    """
    Function to get number of insatnces / tokens to query
    :param config: config.al from original config
    :param tokens_column_name: only used when `framework == "transformers"`
    """
    if isinstance(config.step_p_or_n, int):
        return config.step_p_or_n
    elif (
        "split_by_tokens" in config and config.split_by_tokens
    ):  # and config.al.step_p_or_n is of type float
        all_data = list(initial_data) + list(unlabeled_data)
        total_num_tokens = sum([len(x[tokens_column_name]) for x in all_data])
        return round(config.step_p_or_n * total_num_tokens)
    else:
        len_train = len(initial_data) + len(unlabeled_data)
        return round(config.step_p_or_n * len_train)
