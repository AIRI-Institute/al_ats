# Example of client_lib usage
List of files:
1. test_config - directory with configs for active learning
2. preprocessing_script.sh - preprocessing script, loads data before training
3. test_run.sh - common script for active learning
4. test.py|test.sh - wrapper for test_run.sh: run some preprocessing in container, after run test_run.sh
5. TestRunClientLib.ipynb - notebook with example of running job on cluster
How to use - look into TestRunClientLib.ipynb.