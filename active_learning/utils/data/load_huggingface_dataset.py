from datasets import load_dataset, Dataset
import logging
from pathlib import Path
from copy import deepcopy
import pandas as pd

from .preprocessing import (
    preprocess_img,
    _add_id_column_to_datasets,
    _use_test_subset,
    _use_train_subset,
    _filter_quantiles,
    _multiply_data,
)


log = logging.getLogger()


class HuggingFaceDatasetsReader:
    def __init__(self, *dataset_args, extra_key=None, **kwargs):
        self.dataset = load_dataset(*dataset_args, **kwargs)
        self.extra_key = extra_key

    def __call__(self, phase, text_name=None, label_name=None):
        dataset = self.dataset[phase]
        if self.extra_key is not None:
            dataset = Dataset.from_pandas(pd.DataFrame(dataset[self.extra_key]))
        if text_name is not None and label_name is not None:
            dataset.remove_columns_(
                [
                    x
                    for x in dataset.column_names
                    if x not in [text_name, label_name, "id"]
                ]
            )
        setattr(self, phase, dataset)
        return getattr(self, phase)


def load_huggingface_dataset(config, task, cache_dir=None):

    text_name = config.text_name
    label_name = config.label_name

    if task == "nmt":
        extra_key = config.get("extra_key", "translation")
    else:
        extra_key = config.get("extra_key")
    kwargs = {
        "cache_dir": Path(cache_dir) / "data" if cache_dir is not None else None,
        "extra_key": extra_key
    }
    if config.get("revision_master"):
        kwargs["revision"] = "master"

    hfdreader = (
        HuggingFaceDatasetsReader(config.dataset_name, **kwargs)
        if isinstance(config.dataset_name, str)
        else HuggingFaceDatasetsReader(
            *list(config.dataset_name), **kwargs
        )
    )

    if config.get("multiply_data", None) is not None:
        hfdreader = _multiply_data(hfdreader, config.multiply_data)

    train_dataset = hfdreader("train", text_name, label_name)
    if config.get("no_test"):
        if isinstance(config.get("test_size_split", None), int):
            train_size_split = (len(train_dataset) - config.get("test_size_split")) / len(train_dataset)
        else:
            train_size_split = config.get("train_size_split", 0.8)
        splitted_dataset = train_dataset.train_test_split(
            train_size=train_size_split,
            shuffle=True,
            seed=config.get("seed", 42),
        )
        train_dataset, dev_dataset = splitted_dataset["train"], splitted_dataset["test"]
        test_dataset = deepcopy(dev_dataset)
    elif "validation" in hfdreader.dataset.keys():
        dev_dataset = hfdreader("validation", text_name, label_name)
        # Since on GLUE we do not have gold labels for test data
        if "test" in hfdreader.dataset.keys() and ("glue" not in config.dataset_name):
            test_dataset = hfdreader("test", text_name, label_name)
        else:
            test_dataset = deepcopy(dev_dataset)
    else:
        dev_dataset = hfdreader("test", text_name, label_name)
        test_dataset = deepcopy(dev_dataset)

    log.info(f"Loaded train size: {len(train_dataset)}")
    log.info(f"Loaded dev size: {len(dev_dataset)}")
    log.info(f"Loaded test size: {len(test_dataset)}")
    if dev_dataset is test_dataset:
        log.info("Dev dataset coincides with test dataset")

    if config.labels_to_remove is not None:
        train_dataset = train_dataset.filter(
            lambda x: x[label_name] not in config.labels_to_remove
        )
        dev_dataset = dev_dataset.filter(
            lambda x: x[label_name] not in config.labels_to_remove
        )
        test_dataset = test_dataset.filter(
            lambda x: x[label_name] not in config.labels_to_remove
        )

    if task == "cls" or task == "cnn_cls":
        id2label = {
            i: val for i, val in enumerate(train_dataset.features[label_name].names)
        }
    elif task == "ner":
        id2label = {
            i: val
            for i, val in enumerate(train_dataset.features[label_name].feature.names)
        }
    elif task == "cv_cls":
        id2label = {
            i: val for i, val in enumerate(train_dataset.features[label_name].names)
        }
        train_dataset = train_dataset.map(preprocess_img)
        test_dataset = test_dataset.map(preprocess_img)
        dev_dataset = dev_dataset.map(preprocess_img)
    elif task in ["abs-sum", "nmt"]:
        id2label = None
    else:
        raise NotImplementedError

    if getattr(config, "filter_quantiles", None) is not None:
        train_dataset = _filter_quantiles(
            train_dataset,
            config.filter_quantiles,
            cache_dir,
            text_name,
            config.tokenizer_name,
        )

    if getattr(config, "use_subset", None) is not None:
        train_dataset = _use_train_subset(
            train_dataset,
            config.use_subset,
            getattr(config, "seed", 42),
            task,
            label_name,
        )

    if ("id" not in train_dataset.column_names) and config.get("add_id_column", True):
        train_dataset, dev_dataset, test_dataset = _add_id_column_to_datasets(
            [train_dataset, dev_dataset, test_dataset]
        )

    if getattr(config, "use_test_subset", False):
        test_dataset = _use_test_subset(
            test_dataset,
            config.use_test_subset,
            getattr(config, "seed", 42),
            getattr(config, "subset_fixed_seed", False),
        )

    return [train_dataset, dev_dataset, test_dataset, id2label]
