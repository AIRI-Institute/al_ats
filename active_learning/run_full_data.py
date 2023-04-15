import os
import hydra
from omegaconf import OmegaConf

import json
import logging
import os
from pathlib import Path

from torch import save

from active_learning.utils.general import get_time_dict_path_full_data, log_config
from active_learning.run_scripts.main_decorator import main_decorator
from active_learning.models.text_cnn import load_embeddings_with_text
from datasets import concatenate_datasets


log = logging.getLogger()

OmegaConf.register_new_resolver(
    "to_string", lambda x: x.replace("/", "_").replace("-", "_")
)
OmegaConf.register_new_resolver(
    "get_patience_value", lambda dev_size: 1000 if dev_size == 0 else 5
)


@main_decorator
def run_full_data(config, work_dir: Path or str):
    # Imports inside function to set environment variables before imports
    from active_learning.construct import construct_model
    from active_learning.utils.data.load_data import load_data

    # Log config so that it is visible from the console
    log_config(log, config)
    log.info("Loading data...")
    train_instances, dev_instances, test_instances, labels_or_id2label = load_data(
        config.data, config.model.type, config.framework.name
    )
    log.info(
        f"Training size: {len(train_instances)}, test size: {len(test_instances)}, dev size: {len(dev_instances)}"
    )

    embeddings, word2idx = None, None
    if config.model.name == "cnn":
        # load embeddings
        embeddings, word2idx = load_embeddings_with_text(
            concatenate_datasets([train_instances, dev_instances, test_instances]),
            config.model.embeddings_path,
            config.model.embeddings_cache_dir,
            text_name="text",
        )
    # Initialize time dict
    time_dict_path = get_time_dict_path_full_data(config)

    log.info("Fitting the model...")
    model = construct_model(
        config,
        config.model,
        dev_instances,
        config.framework.name,
        labels_or_id2label,
        "model",
        time_dict_path,
        embeddings=embeddings,
        word2idx=word2idx,
    )

    model.fit(train_instances, None)
    try:
        model.model.save_pretrained("tmp_full")
    except:
        import pdb

        pdb.set_trace()

    try:
        dev_metrics = model.evaluate(dev_instances)
    except:
        import pdb

        pdb.set_trace()
    log.info(f"Dev metrics: {dev_metrics}")

    try:
        test_metrics = model.evaluate(test_instances)
    except:
        import pdb

        pdb.set_trace()
    log.info(f"Test metrics: {test_metrics}")

    with open(work_dir / "dev_metrics.json", "w") as f:
        json.dump(dev_metrics, f)

    with open(work_dir / "metrics.json", "w") as f:
        json.dump(test_metrics, f)

    if config.dump_model:
        model.model.save_pretrained(work_dir / "model.pth")
    log.info("Done.")


@hydra.main(
    config_path=os.environ["HYDRA_CONFIG_PATH"],
    config_name=os.environ["HYDRA_CONFIG_NAME"],
)
def main(config):
    run_full_data(config)


if __name__ == "__main__":
    main()
