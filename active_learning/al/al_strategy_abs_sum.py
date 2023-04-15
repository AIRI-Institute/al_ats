import numpy as np
from datasets.arrow_dataset import Dataset
from typing import Union
import torch
import logging


from .al_strategy_utils import (
    get_X_pool_subsample,
    get_similarities,
    filter_by_uncertainty,
    filter_by_metric,
    calculate_bleuvar_scores,
    assign_ue_scores_for_unlabeled_data,
    take_idx,
    calculate_unicentroid_mahalanobis_distance,
)

from ..utils.transformers_dataset import TransformersDataset


log = logging.getLogger()


### Abstractive summarization strategies


def sequence_score_sampling(
    model,
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    **kwargs,
):
    """
    Args:
        model: model fine-tuned on current labeled data
        X_pool: unlabeled data pool
        n_instances: number of instances to select for labeling
        **kwargs:

    Returns:
    n instances from X_pool according to
    the average of the cumulative production of probabilities
    of tokens predicted by the model
    """
    # Filtering part begin
    filtering_mode = kwargs.get("filtering_mode", None)
    uncertainty_threshold = kwargs.get("uncertainty_threshold", None)
    uncertainty_mode = kwargs.get(
        "uncertainty_mode", "absolute"
    )  # "relative" or "absolute"
    # Filtering part end
    generate_output = model.generate(X_pool, to_numpy=True)
    scores = generate_output["sequences_scores"]
    sequences_ids = generate_output["sequences"]
    # The larger the score, the more confident the model is
    uncertainty_estimates = -scores
    # Filtering part begin
    if filtering_mode == "uncertainty":
        query_idx, uncertainty_estimates = filter_by_uncertainty(
            uncertainty_estimates=uncertainty_estimates,
            uncertainty_threshold=uncertainty_threshold,
            uncertainty_mode=uncertainty_mode,
            n_instances=n_instances,
        )
    elif filtering_mode in ["rouge1", "rouge2", "rougeL", "sacrebleu"]:
        query_idx, uncertainty_estimates = filter_by_metric(
            uncertainty_estimates=uncertainty_estimates,
            uncertainty_threshold=uncertainty_threshold,
            uncertainty_mode=uncertainty_mode,
            n_instances=n_instances,
            texts=X_pool[model.data_config["text_name"]],
            generated_sequences_ids=sequences_ids,
            tokenizer=model.tokenizer,
            metric_cache_dir=model.cache_dir / "metrics",
            metric_name=filtering_mode,
            agg=kwargs.get("filtering_aggregation", "precision"),
        )
    else:
        argsort = np.argsort(-uncertainty_estimates)
        query_idx = argsort[:n_instances]
    # Filtering part end
    query = take_idx(X_pool, query_idx)

    return query_idx, query, uncertainty_estimates


def sequence_score_stochastic_sampling(
    model: "ModalTransformersWrapper",
    X_pool: Union[np.ndarray, Dataset, TransformersDataset],
    n_instances: int,
    **kwargs,
):
    """

    Args:
        model: model fine-tuned on current labeled data
        X_pool: unlabeled data pool
        n_instances: number of instances to select for labeling
        **kwargs:

    Returns:
    n instances from X_pool scored with Expected Sequence Probability
    or Expected Sequence Variance.
    """
    aggregation = kwargs.get("aggregation", "var")
    func_for_agg = getattr(np, aggregation)
    larger_is_more_uncertain = True
    if aggregation in ["mean", "median", "max", "min"]:
        larger_is_more_uncertain = False
    use_log = kwargs.get("use_log", False)

    generate_kwargs = dict(to_numpy=True, do_sample=False, to_eval_mode=False)
    if kwargs.get("enable_dropout", False):
        # Since BART exploits F.dropout instead of nn.Dropout, we can only turn it on via .train()
        model.model.train()
    else:
        model.model.eval()
        generate_kwargs["do_sample"] = True
        generate_kwargs["top_p"] = kwargs.get("generate_top_p", 0.95)

    mc_iterations = kwargs.get("mc_iterations", 5)
    X_pool_subsample, subsample_indices = get_X_pool_subsample(
        X_pool, mc_iterations, model.seed
    )

    log_scores = []
    for _ in range(mc_iterations):
        generate_output = model.generate(X_pool_subsample, **generate_kwargs)
        log_scores.append(generate_output["sequences_scores"])

    scores = np.r_[log_scores]
    if not use_log:
        scores = np.exp(scores)
    subsample_uncertainty_estimates = func_for_agg(scores, axis=0)
    if larger_is_more_uncertain:
        subsample_uncertainty_estimates = -subsample_uncertainty_estimates

    argsort = np.argsort(subsample_uncertainty_estimates)
    subsample_query_idx = argsort[:n_instances]
    query = take_idx(X_pool_subsample, subsample_query_idx)
    query_idx = subsample_indices[subsample_query_idx]

    uncertainty_estimates = assign_ue_scores_for_unlabeled_data(
        len(X_pool), subsample_indices, subsample_uncertainty_estimates
    )

    return query_idx, query, uncertainty_estimates


def bleuvar_sampling(
    model: "ModalTransformersWrapper",
    X_pool: Union[Dataset, TransformersDataset],
    n_instances: int,
    **kwargs,
):
    """

    Args:
        model: model fine-tuned on current labeled data
        X_pool: unlabeled data pool
        n_instances: number of instances to select for labeling
        **kwargs:

    Returns:
    n instances from X_pool scored with BLEUVar
    proposed in https://arxiv.org/pdf/2006.08344.pdf.
    """
    mc_iterations = kwargs.get("mc_iterations", 10)
    metric_name = kwargs.get("var_metric", "sacrebleu")

    filtering_mode = kwargs.get("filtering_mode", None)
    bleuvar_threshold = kwargs.get("uncertainty_threshold", 1.0)
    uncertainty_mode = kwargs.get(
        "uncertainty_mode", "absolute"
    )  # "relative" or "absolute"
    X_pool_subsample, subsample_indices = get_X_pool_subsample(
        X_pool, mc_iterations, model.seed
    )

    generate_kwargs = dict(
        return_decoded_preds=True, do_sample=False, to_eval_mode=False
    )
    if kwargs.get("enable_dropout", False):
        model.enable_dropout()  # model.model.train()
    else:
        model.model.eval()
        generate_kwargs["do_sample"] = True
        generate_kwargs["top_p"] = kwargs.get("generate_top_p", 0.95)

    summaries = []  # mc_iterations x len(X_pool_subsample) of str
    for _ in range(mc_iterations):
        generated_texts = model.generate(X_pool_subsample, **generate_kwargs)[
            "summaries"
        ]
        generated_texts = [
            text if len(text) > 0 else "CLS" for text in generated_texts
        ]  # fix empty texts
        summaries.append(generated_texts)

    # sacrebleu is normally more robust than bleu
    bleu_vars = calculate_bleuvar_scores(
        summaries,
        metric_name=metric_name,
        cache_dir=model.cache_dir / "metrics",
        tokenizer=model.tokenizer,
    )

    if filtering_mode == "uncertainty":
        subsample_query_idx, bleu_vars = filter_by_uncertainty(
            bleu_vars, bleuvar_threshold, uncertainty_mode, n_instances
        )
    else:
        subsample_query_idx = np.argsort(-bleu_vars)[:n_instances]

    query = take_idx(X_pool_subsample, subsample_query_idx)
    query_idx = subsample_indices[subsample_query_idx]

    uncertainty_estimates = assign_ue_scores_for_unlabeled_data(
        len(X_pool), subsample_indices, bleu_vars
    )

    return query_idx, query, uncertainty_estimates


def idds_sampling(
    model, X_pool, n_instances, X_train, seed=None, device=None, **kwargs
):
    cache_dir = kwargs.get("cache_dir")
    model_name = kwargs.get("embeddings_model_name", "bert-base-uncased")
    text_name = kwargs.get("text_name", "document")
    subsample_ratio = kwargs.get("subsample_ratio", 1)
    lamb = kwargs.get("lambda", 0.667)
    u_top = kwargs.get("u_top", None)
    l_top = kwargs.get("l_top", None)
    average = kwargs.get("average", False)
    sims_func = kwargs.get("sims_func", "scalar_product")
    filter_outliers = kwargs.get("filter_outliers", None)
    filtering_mode = kwargs.get("filtering_mode", None)
    batch_size = kwargs.get("embeddings_batch_size", 100)

    log.info(f"Used similarities function: {sims_func}; u-top: {u_top}; l-top: {l_top}")

    if filtering_mode is not None:
        uncertainty_threshold = kwargs.get("uncertainty_threshold", 0.0)
        uncertainty_mode = kwargs.get(
            "uncertainty_mode", "absolute"
        )  # "relative" or "absolute"
        generation_output = model.generate(X_pool, to_numpy=True)
        scores = generation_output["sequences_scores"]
        sequences_ids = generation_output["sequences"]

        # if filtering_mode == "uncertainty":
        #     query_idx, uncertainty_estimates = filter_by_uncertainty(
        #         uncertainty_estimates=-scores,
        #         uncertainty_threshold=uncertainty_threshold,
        #         uncertainty_mode=uncertainty_mode,
        #         n_instances=n_instances,
        #     )
        #
        # elif filtering_mode in ["rouge1", "rouge2", "rougeL", "sacrebleu"]:
        #     query_idx, uncertainty_estimates = filter_by_metric(
        #         uncertainty_threshold=uncertainty_threshold,
        #         uncertainty_mode=uncertainty_mode,
        #         texts=X_pool[model.data_config["text_name"]],
        #         generated_sequences_ids=sequences_ids,
        #         tokenizer=model.tokenizer,
        #         metric_cache_dir=model.cache_dir / "metrics",
        #         metric_name=filtering_mode,
        #         agg=kwargs.get("filtering_aggregation", "precision"),
        #         modify_uncertainties=False,
        #     )

    # subsample size = pool size / subsample_ratio
    if device is None:
        device = model.model.device
    if seed is None:
        seed = model.seed

    if subsample_ratio is not None:
        X_pool_subsample, subsample_indices = get_X_pool_subsample(
            X_pool, subsample_ratio, seed
        )  # `subsample_indices` indicated the indices of the subsample in the original data
    else:
        X_pool_subsample = X_pool

    similarities, counts, embeddings = get_similarities(
        model_name,
        X_pool_subsample,
        X_train,
        sims_func=sims_func,
        average=average,
        text_name=text_name,
        device=device,
        cache_dir=cache_dir,
        return_embeddings=True,
        batch_size=batch_size,
    )
    num_obs = len(similarities)
    if X_train is None:
        X_train = []

    labeled_indices = list(range(num_obs - len(X_train), num_obs))
    unlabeled_indices = list(range(num_obs - len(X_train)))

    unlabeled_indices_without_queries = list(unlabeled_indices)
    top_scores_indices = []
    top_scores = []

    if filter_outliers is not None:
        outliers_idx = []
        num_outliers = round(filter_outliers * num_obs)

    for i_query in range(n_instances):
        # Calculate similarities
        if u_top is None:
            similarities_with_unlabeled = (
                similarities[unlabeled_indices][
                    :, unlabeled_indices_without_queries
                ].sum(dim=1)
                - 1
            ) / (len(unlabeled_indices_without_queries) - 1)
        else:
            similarities_with_unlabeled = (
                similarities[unlabeled_indices][:, unlabeled_indices_without_queries]
                .topk(u_top + 1)[0]
                .sum(dim=1)
                - 1
            ) / u_top
        if len(labeled_indices) == 0:
            similarities_with_labeled = torch.zeros(len(unlabeled_indices)).to(
                similarities_with_unlabeled
            )
        elif l_top is None:
            similarities_with_labeled = similarities[unlabeled_indices][
                :, labeled_indices
            ].mean(dim=1)
        else:
            similarities_with_labeled = (
                similarities[unlabeled_indices][:, labeled_indices]
                .topk(min(len(labeled_indices), l_top))[0]
                .mean(dim=1)
            )
        scores = (
            (
                similarities_with_unlabeled * lamb
                - similarities_with_labeled * (1 - lamb)
            )
            .cpu()
            .detach()
            .numpy()
        )
        scores[top_scores_indices] = -np.inf
        if filter_outliers is not None and len(outliers_idx) > 0:
            scores[outliers_idx.cpu().numpy()] = -np.inf

        # TODO: BUG when subsample_ratio is not None
        most_similar_idx = np.argmax(scores)
        labeled_indices.append(most_similar_idx)
        if most_similar_idx in unlabeled_indices_without_queries:
            unlabeled_indices_without_queries.remove(most_similar_idx)
        top_scores_indices.append(most_similar_idx)
        top_scores.append(scores[most_similar_idx])

        if filter_outliers is not None and i_query > 0:
            outliers_idx = (
                calculate_unicentroid_mahalanobis_distance(embeddings, labeled_indices)
                .topk(num_outliers)
                .indices
            )

    scores[top_scores_indices] = top_scores
    top_scores_idx = [counts.index(i) for i in top_scores_indices]
    scores = scores[counts]

    if subsample_ratio is not None:
        query_idx = subsample_indices[top_scores_idx]
        uncertainty_estimates = assign_ue_scores_for_unlabeled_data(
            len(X_pool), subsample_indices, scores
        )
    else:
        query_idx = np.array(top_scores_idx)
        uncertainty_estimates = scores

    query = X_pool.select(query_idx)

    return query_idx, query, uncertainty_estimates

def idds_with_uncertainty_sampling(
    model, X_pool, n_instances, X_train, seed=None, device=None, **kwargs
):
    cache_dir = kwargs.get("cache_dir")
    model_name = kwargs.get("embeddings_model_name", "bert-base-uncased")
    text_name = kwargs.get("text_name", "document")
    subsample_ratio = kwargs.get("subsample_ratio", 1)
    lamb = kwargs.get("lambda", 0.667)
    u_top = kwargs.get("u_top", None)
    l_top = kwargs.get("l_top", None)
    average = kwargs.get("average", False)
    sims_func = kwargs.get("sims_func", "scalar_product")
    filter_outliers = kwargs.get("filter_outliers", None)
    batch_size = kwargs.get("embeddings_batch_size", 100)
    uncertainty_func = kwargs.get("uncertainty_function", "sequence_score")

    log.info(f"Used similarities function: {sims_func}; u-top: {u_top}; l-top: {l_top}")
    # subsample size = pool size / subsample_ratio
    if device is None:
        device = model.model.device
    if seed is None:
        seed = model.seed

    if subsample_ratio is not None:
        X_pool_subsample, subsample_indices = get_X_pool_subsample(
            X_pool, subsample_ratio, seed
        )  # `subsample_indices` indicated the indices of the subsample in the original data
    else:
        X_pool_subsample = X_pool

    similarities, counts, embeddings = get_similarities(
        model_name,
        X_pool_subsample,
        X_train,
        sims_func=sims_func,
        average=average,
        text_name=text_name,
        device=device,
        cache_dir=cache_dir,
        return_embeddings=True,
        batch_size=batch_size,
    )
    num_obs = len(similarities)
    if X_train is None:
        X_train = []

    labeled_indices = list(range(num_obs - len(X_train), num_obs))
    unlabeled_indices = list(range(num_obs - len(X_train)))

    unlabeled_indices_without_queries = list(unlabeled_indices)
    top_scores_indices = []
    top_scores = []

    if filter_outliers is not None:
        outliers_idx = []
        num_outliers = round(filter_outliers * num_obs)

    if uncertainty_func == "sequence_score":
        generate_output = model.generate(X_pool_subsample, to_numpy=False)
        uncertainty = -generate_output["sequences_scores"].to(similarities)
        uncertainty_unique = uncertainty[[counts.index(i) for i in unlabeled_indices]]
        if kwargs.get("normalize_uncertainty"):
            min_unc = uncertainty_unique.min()
            max_unc = uncertainty_unique.max()
            uncertainty_unique = (uncertainty_unique - min_unc) / (max_unc - min_unc)
    else:
        raise NotImplementedError

    for i_query in range(n_instances):
        # Calculate similarities
        if len(labeled_indices) == 0:
            similarities_with_labeled = torch.zeros(len(unlabeled_indices)).to(
                uncertainty_unique
            )
        elif l_top is None:
            similarities_with_labeled = similarities[unlabeled_indices][
                :, labeled_indices
            ].mean(dim=1)
        else:
            similarities_with_labeled = (
                similarities[unlabeled_indices][:, labeled_indices]
                .topk(min(len(labeled_indices), l_top))[0]
                .mean(dim=1)
            )
        scores = (
            (
                uncertainty_unique * lamb
                - similarities_with_labeled * (1 - lamb)
            )
            .cpu()
            .detach()
            .numpy()
        )
        scores[top_scores_indices] = -np.inf
        if filter_outliers is not None and len(outliers_idx) > 0:
            scores[outliers_idx.cpu().numpy()] = -np.inf

        # TODO: BUG when subsample_ratio is not None
        most_similar_idx = np.argmax(scores)
        labeled_indices.append(most_similar_idx)
        unlabeled_indices_without_queries.remove(most_similar_idx)
        top_scores_indices.append(most_similar_idx)
        top_scores.append(scores[most_similar_idx])

        if filter_outliers is not None and i_query > 0:
            outliers_idx = (
                calculate_unicentroid_mahalanobis_distance(embeddings, labeled_indices)
                .topk(num_outliers)
                .indices
            )

    scores[top_scores_indices] = top_scores
    top_scores_idx = [counts.index(i) for i in top_scores_indices]
    scores = scores[counts]

    if subsample_ratio is not None:
        query_idx = subsample_indices[top_scores_idx]
        uncertainty_estimates = assign_ue_scores_for_unlabeled_data(
            len(X_pool), subsample_indices, scores
        )
    else:
        query_idx = np.array(top_scores_idx)
        uncertainty_estimates = scores

    query = X_pool.select(query_idx)

    return query_idx, query, uncertainty_estimates
