import numpy as np
import logging

from .al_strategy_utils import (
    get_query_idx_for_selecting_by_number_of_tokens,
    take_idx,
)


log = logging.getLogger()


def random_sampling(
    model,
    X_pool,
    n_instances,
    *args,
    select_by_number_of_tokens: bool = False,
    **kwargs,
):
    """emulates the case when an annotator labels randomly sampled instances"""
    if select_by_number_of_tokens:
        sorted_idx = np.arange(len(X_pool))
        np.random.shuffle(sorted_idx)
        query_idx = get_query_idx_for_selecting_by_number_of_tokens(
            X_pool, sorted_idx, n_instances
        )
    else:
        query_idx = np.random.choice(range(len(X_pool)), n_instances, replace=False)

    query = take_idx(X_pool, query_idx)
    # Uncertainty estimates are not defined with random sampling
    uncertainty_estimates = np.zeros(len(X_pool))
    return query_idx, query, uncertainty_estimates
