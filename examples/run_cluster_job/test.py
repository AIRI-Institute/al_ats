import os

# preprocessing part - copy cache of jovyan user (instance) to user user (container)
os.system(
    f"cp -r /home/jovyan/.cache/huggingface {os.path.join(os.path.expanduser('~'), '.cache/huggingface')}"
)
# after build active learning library
os.chdir("./active_learning")
os.system("python setup_offline.py install --user")
os.chdir("./examples/run_cluster_job")
# and after run script
os.system("./test_run.sh")
