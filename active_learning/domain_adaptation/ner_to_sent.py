import os
from pathlib import Path
import re
import hydra


def read_wnut(file_path):
    file_path = Path(file_path)
    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r"\n\t?\n", raw_text)
    token_docs = []
    for doc in raw_docs:
        tokens = []
        for line in doc.split("\n"):
            line_data = line.split()
            token = line_data[0]
            tokens.append(token)
        token_docs.append(tokens)
    return token_docs


def transform_and_save_data(input_path, output_path):
    texts = read_wnut(input_path)
    sentences = [" ".join(text).strip() for text in texts if text[0] != "-DOCSTART-"]
    with open(output_path, "w") as f:
        f.write("\n".join(sentences))


@hydra.main(
    config_path=os.environ["HYDRA_CONFIG_PATH"],
    config_name=os.environ["HYDRA_CONFIG_NAME"],
)
def main(config):
    args = config
    transform_and_save_data(args.input_train_file, args.train_data_file)
    # load val data
    if args.input_val_file is not None:
        transform_and_save_data(args.input_val_file, args.eval_data_file)


if __name__ == "__main__":
    main()
