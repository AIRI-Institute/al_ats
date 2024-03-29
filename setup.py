from pathlib import Path
from setuptools import setup, find_packages
import os

REQUIREMENTS_PATH = Path(__file__).resolve().parent / "requirements.txt"


with open(str(REQUIREMENTS_PATH), "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

os.system("chmod a+x init.sh examples/*")

setup(
    name="active_learning",
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "*.trainable_sampling*"]
    ),
    version="0.0.1",
    description="A Library for Active Learning. Supports classification, NER, and abstractive summarization tasks for NLP and classification for CV.",
    author="Tsvigun A.O., Shelmanov A.O., Kuzmin G.Yu., Sanochkin L.O.",
    license="Apache",
    long_description=open("README.md").read(),
    install_requires=requirements,
    include_package_data=True,
    setup_requires=["pytest-runner"],
    tests_require=["pytest==4.4.1"],
    test_suite="tests",
    package_data={"active_learning_nlp": ["configs/*", "configs/framework/*"]},
)

os.system("./init.sh")
