import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Medical Diagnosis Training with GRPO"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install torch transformers accelerate trl\n",
                "!git clone https://github.com/birgermoell/trl.git\n",
                "%cd trl/examples/medical_diagnosis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import logging\n",
                "from healthcare_examples import create_dataset, setup_trainer\n",
                "\n",
                "logging.basicConfig(level=logging.INFO)\n",
                "logger = logging.getLogger(__name__)\n",
                "\n",
                "output_dir = \"medical_model_output\"\n",
                "os.makedirs(output_dir, exist_ok=True)\n",
                "\n",
                "train_dataset = create_dataset()\n",
                "eval_dataset = create_dataset()[:2]\n",
                "\n",
                "trainer = setup_trainer(\n",
                "    train_dataset=train_dataset,\n",
                "    eval_dataset=eval_dataset,\n",
                "    output_dir=output_dir\n",
                ")\n",
                "\n",
                "trainer.train()\n",
                "trainer.save_pretrained(output_dir)"
            ]
        }
    ],
    "metadata": {
        "colab": {
            "name": "Medical Diagnosis GRPO"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open('medical_diagnosis_grpo.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2) 