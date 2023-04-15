## Active Learning for Abstractive Text Summarization
A code to reproduce experiments in out paper Active Learning for Abstractive Text Summarization.

## Installation
Install the requirements:
```
pip install -r requirements.txt
```

## Usage
The `configs` folder contains config files with experiment settings. To run an experiment with chosen configuration specify config file name in `HYDRA_CONFIG_NAME` variable and run 
`train.sh` script. 

For example to launch experiment with IDDS strategy, AESLC data and BART model:
```
HYDRA_CONFIG_PATH=./configs HYDRA_CONFIG_NAME=aescl_idds_bart python 'active_learning/active_learning/run_tasks_on_multiple_gpus.py'
```