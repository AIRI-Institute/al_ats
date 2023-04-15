from copy import deepcopy
from functools import partial
from pathlib import Path

from ..al.learner import ActiveLearner
from ..al.al_strategy import (
    random_sampling,
)

from ..al.al_strategy_abs_sum import (
    sequence_score_sampling,
    sequence_score_stochastic_sampling,
    bleuvar_sampling,
    idds_sampling,
    idds_with_uncertainty_sampling
)
from ..al.sampling_strategy import (
    ups_subsampling,
    random_subsampling,
    naive_subsampling,
)


QUERY_STRATEGIES = {
    # Abstractive summarization strategies
    "random": partial(random_sampling, select_by_number_of_tokens=False),
    "seq_score": sequence_score_sampling,
    "bleuvar": bleuvar_sampling,
    "seq_score_stochastic": sequence_score_stochastic_sampling,
    "idds": idds_sampling,
    "idds_unc": idds_with_uncertainty_sampling,
}

sampling_strategies = {
    "ups": ups_subsampling,
    "random": random_subsampling,
    "naive": naive_subsampling,
}


def construct_active_learner(
    model, config, initial_data, log_dir: str or Path, framework: str = "transformers"
):

    # TODO: rewrite using `split_by_tokens` as `strategy_kwargs`
    initial_data_copy = deepcopy(initial_data)
    use_ups = config.sampling_type is not None
    postfix = ""
    if ("split_by_tokens" in config) and (config.split_by_tokens):
        postfix += "_tokens"
    elif "split_by_tokens" in config:  # avoid adding "_samples" for classification
        postfix += "_samples"

    if config.strategy == "bald" and getattr(config, "head_only_dropout", False):
        postfix += "_head"

    query_strategy = QUERY_STRATEGIES[f"{config.strategy}{postfix}"]
    sampling_strategy = sampling_strategies[config.sampling_type] if use_ups else None
    sampling_kwargs = {
        "gamma_or_k_confident_to_save": config.gamma_or_k_confident_to_save,
        "T": config.T,
    }
    strategy_kwargs = config.strategy_kwargs

    learner = ActiveLearner(
        estimator=model,
        query_strategy=query_strategy,
        train_data=initial_data_copy,
        strategy_kwargs=strategy_kwargs,
        sampling_strategy=sampling_strategy,
        sampling_kwargs=sampling_kwargs,
        framework=framework,
        log_dir=log_dir,
    )

    return learner
