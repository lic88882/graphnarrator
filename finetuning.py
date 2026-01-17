import json
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

from log import logger


def self_training(exp_id, model_id, ft_prefix="graph") -> str:
    # Upload training data
    client = OpenAI()

    # path to the .jsonl dataset for self-training
    finetuning_file = Path(f"outputs/finetuning_jsonl/finetune-{exp_id}.jsonl")

    # Upload training data
    ft_file = client.files.create(file=open(finetuning_file, "rb"), purpose="fine-tune")

    # initialize fine-tuning job
    job = client.fine_tuning.jobs.create(
        training_file=ft_file.id,
        model=model_id,
        seed=42,
        suffix=f"{ft_prefix}-v{exp_id+1}",
        hyperparameters={"n_epochs": 3},
    )

    while True:
        # Retrieve the state of the fine-tune job
        ft_state = client.fine_tuning.jobs.retrieve(job.id)

        if ft_state.status == "succeeded":
            ft_model_id = ft_state.fine_tuned_model
            logger.info("Fine-tuned model: %s", ft_model_id)
            return ft_model_id

        if ft_state.status in ["failed", "cancelled"]:
            raise Exception(f"Fine-tuning job {job.id} {ft_state.status}")  # pylint: disable=broad-exception-raised

        else:
            logger.info("finetuing status:%s (update every 15 secs)", ft_state.status)
            job_events = client.fine_tuning.jobs.list_events(job.id, limit=1).data
            logger.info("message:%s", job_events[0].message)

        time.sleep(15)
