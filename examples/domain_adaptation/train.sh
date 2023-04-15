export HYDRA_CONFIG_PATH=./configs
export HYDRA_CONFIG_NAME=domain_adaptation.yaml
export output_dir="bert-cola"
export dataset_name="'glue,cola'"

python /home/jovyan/active_learning/active_learning/domain_adaptation/HFDatasets2sent.py \
 output_dir=$output_dir \
 data.dataset_name=$dataset_name \
 data.text_column_name=sentence \
 data.has_validation=True

python /home/jovyan/active_learning/active_learning/domain_adaptation/run_lm.py \
 output_dir=$output_dir \
 data.dataset_name=$dataset_name \
 weight_decay=1e-3 \
 learning_rate=1e-6 \
 gradient_accumulation_steps=4