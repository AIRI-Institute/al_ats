HYDRA_CONFIG_PATH=../../configs \
HYDRA_CONFIG_NAME=al_cls \
python ../../active_learning/utils/cache_all_necessary_files.py \
data.dataset_name=ag_news acquisition_model.name=distilbert-base-uncased cache_model_and_dataset=True