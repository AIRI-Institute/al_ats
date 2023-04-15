## Domain adaptation
### Main info
Main code adapted from https://github.com/allenai/dont-stop-pretraining. This code includes main script run_lm.py that runs LM training and ner_to_sent.py script that transforms NER datasets into appropriate format for training model.
### How to run
Just set training parameters in ./configs/domain_adaptation.yaml file and run ./run.sh file. For NER datasets like CoNLL recomended to train on small amount of steps, like 100 or 1000 with 256 batch size.