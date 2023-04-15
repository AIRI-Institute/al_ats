import hydra
import os
from pathlib import Path
from datasets import load_dataset, Dataset
from tqdm import tqdm
import logging


log = logging.getLogger()


@hydra.main(
    config_path=os.environ["HYDRA_CONFIG_PATH"],
    config_name=os.environ["HYDRA_CONFIG_NAME"],
)
def main(config):
    data_config = config.data
    text_column_name = data_config.text_column_name
    dataset_name = data_config.dataset_name
    if "," in dataset_name:
        dataset_name = dataset_name.split(",")

    if data_config.data_path == "datasets":
        if isinstance(dataset_name, str):
            dataset = load_dataset(dataset_name)
        else:
            dataset = load_dataset(*dataset_name)
    else:
        json_path = Path(data_config.data_path) / dataset_name / "train.json"
        if json_path.exists():
            dataset = Dataset.from_json(str(json_path))
        else:
            dataset = Dataset.from_csv(
                str(Path(data_config.data_path) / dataset_name / "train.csv")
            )

    test_data = None
    if data_config.has_validation:
        train_data, val_data = dataset["train"], dataset["validation"]
    elif data_config.get("no_test", False):
        splitted = dataset.train_test_split(
            test_size=data_config.get("test_size", 0.2), seed=config.seed, shuffle=True
        )
        train_data, test_data = splitted["train"], splitted["test"]
        splitted = train_data.train_test_split(
            test_size=data_config.val_size, seed=config.seed, shuffle=True
        )
        train_data, val_data = splitted["train"], splitted["test"]
    else:
        splitted = dataset["train"].train_test_split(
            test_size=data_config.val_size, seed=config.seed, shuffle=True
        )
        train_data, val_data = splitted["train"], splitted["test"]

    log.info(f"Train size: {len(train_data)}")
    log.info(f"Val size: {len(val_data)}")
    if test_data is not None:
        log.info(f"Test size: {len(test_data)}")
    else:
        log.info(f"Test size: {len(dataset['test'])}")

    string = ""
    for inst in tqdm(train_data):
        string += inst[text_column_name] + "\n"

    with open(config.train_data_file, "w") as f:
        f.write(string)

    string = ""
    for inst in tqdm(val_data):
        string += inst[text_column_name] + "\n"

    with open(config.eval_data_file, "w") as f:
        f.write(string)


if __name__ == "__main__":
    main()
