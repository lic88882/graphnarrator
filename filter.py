import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import regex as re
import torch
from openai import OpenAI

from clm import CLM
from log import logger
from tag import TAG


class XAIEvaluator:

    def __init__(
        self,
        device="cuda",
    ):
        self.MLM = None  # lazy initialization
        self.masker = "<mask>"
        self.device = device

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

    def initialize_mlm(self):
        if self.MLM is None:
            self.MLM = CLM(model_id="google/gemma-2-2b", device=self.device)

    def extract_parameters(
        self,
        graph,
        explanation,
        masking_range=(0, 90),
        corpus_range=(5, 100),
        json_style="document",
        with_score=False,  # with importance score in the graph corpus
        with_score_of_masked_tokens=False,  # with importance scores of the masked tokens
    ):
        # Extract the full corpus of the input graph
        full_corpus = graph.text(style=json_style, p_range=corpus_range, masker="", with_score=with_score)

        # Get the model's prediction and normalize it
        raw_prediction = graph.prediction
        normalized_prediction = raw_prediction.replace("_", " ")

        # Update explanation with normalized prediction
        updated_explanation = explanation.replace(raw_prediction, normalized_prediction)

        # Extract the corpus with important tokens masked
        masked_corpus = graph.text(
            p_range=masking_range,
            style=json_style,
            masker=self.masker,
            with_score=with_score,
        )

        # Get the list of masked tokens
        masked_tokens, importance_scores = graph.get_masked_tokens(p_range=masking_range, return_scores=True)

        # Generate a partially masked explanation
        # keywords = self.extract_keywords(updated_explanation, method="random")
        masked_explanation = []
        explanation_masked_tokens = []
        # for word in updated_explanation.split(" "):
        #     if word.lower() in keywords:
        #         explanation_masked_tokens.append(word)
        #         masked_explanation.append(self.masker)
        #     else:
        #         masked_explanation.append(word)
        # masked_explanation = " ".join(masked_explanation)

        params = (
            full_corpus,
            normalized_prediction,
            masked_corpus,
            masked_tokens,
            updated_explanation,
            masked_explanation,
            explanation_masked_tokens,
        )
        if with_score_of_masked_tokens:
            params += (importance_scores,)

        return params

    def metrics(self, graph, E, **kwargs):
        """
            computing saliency, faithfulness, and brevity

        return:
            (saliency, faithfulness, brevity)
        """

        self.initialize_mlm()  # Initialize MLM only when needed

        # prepare the parameters
        S, y_hat, S_M, M_S, E, E_M, M_E = self.extract_parameters(graph, E, **kwargs)

        saliency, _ = self.mlm_saliency(y_hat, S_M, E, M_S)
        faithfulness, _ = self.mlm_faithfulness(y_hat, S_M, E)
        brevity = self.brevity(S, E)

        return saliency, faithfulness, brevity

    def extract_keywords(self, text, method: Literal["LLM", "random"]):
        if method == "LLM":
            # extract important keywords using LLM
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Extract 25 key words from the following text. Choose non-stop words that carry significant meaning and best represent the overall content. Prioritize nouns, verbs, adjectives, and adverbs that capture the main ideas and themes. Provide only the list of 20 words, seperated by comma without explanations. <startoftext>{text}</startoftext>",
                    },
                ],
                max_tokens=4095,
                temperature=0.0,
                top_p=0.95,
            )

            keywords = completion.choices[0].message.content
            keywords = [k.lower() for k in keywords.split(", ")]
        elif method == "random":
            keywords = random.sample(text.split(), 25)

        return keywords

    def mlm_saliency(self, y_hat, S_M, E, M_S):
        saliency = 0
        length = 0

        context1 = (
            f"A human-readable free-text explanation which explains the classification result is given below. <explanation>{E}</explanation>"
            + f"The classification result of the ROOT Node is: {y_hat}"
            + "Try to recover the verbalized graph.\n<Verbalized-Graph>"
            + S_M
            + "</Verbalized-Graph>"
        )
        context2 = (
            f"The classification result of the ROOT Node is: {y_hat}"
            + "Try to recover the verbalized graph.\n<Verbalized-Graph>"
            + S_M
            + "</Verbalized-Graph>"
        )
        prob1 = self.MLM.fill_mask(context1, M_S)
        prob2 = self.MLM.fill_mask(context2, M_S)

        for p1, p2 in zip(prob1, prob2):
            # if p1[-1] < 1e-4 or p2[-1] < 1e-4:
            #     continue
            # print(f"token: {p1[0]} ({p1[1]}), prob: {p1[-1]}, {p2[-1]}")
            saliency += np.log(p1[-1] / p2[-1])
            length += 1

        return saliency / length, (prob1, prob2)

    def mlm_faithfulness(self, y_hat, S_M, E):
        # manually replace explicit y_hat in E
        E = E.replace(y_hat, "<label>")

        faithfulness = 0
        length = 0
        # == category section ==
        context1 = (
            "A human-readable free-text explanation which explains a node classification result is given below."
            + f"\n{E}\n"
            + "The following verbalized graph contains important words in the text of each nodes. These words contributes to the classification of ROOT Node into one of the seven possible categories."
            + f"\n{S_M}\n"
        )
        context2 = (
            "The following verbalized graph contains important words in the text of each nodes. These words contributes to the classification of ROOT Node into one of the seven possible categories."
            + f"\n{S_M}\n"
        )

        prompt = "Given above information, among seven categories (Case Based, Genetic Algorithms, Neural Networks, Probabilistic Methods, Reinforcement Learning, Rule Learning, Theory), the label of the ROOT node is"
        completion = y_hat

        prob1 = self.MLM.sample_next_tokens(context1 + prompt, completion)
        prob2 = self.MLM.sample_next_tokens(context2 + prompt, completion)

        for p1, p2 in zip(prob1, prob2):
            faithfulness += np.log(p1 / p2)
            length += 1

        return faithfulness / length, (prob1, prob2)

        # == category section ==

    def brevity(self, S, E):
        brevity = len(E.split()) / len(S.split())
        return brevity


def cal_metrics(exp_id, device, expl_list):
    quality_dir = Path("outputs/quality")
    quality_dir.mkdir(exist_ok=True)
    quality_log = quality_dir / f"quality-{exp_id}.csv"

    # initialize log file if not exists
    if not quality_log.exists():
        with open(quality_log, "w", encoding="utf-8") as f:
            f.write("INDEX,FILE,SALIENCY,FAITHFULNESS,BREVITY,VALID_TIME\n")

    # load processed nodes
    df = pd.read_csv(quality_log, dtype={'INDEX': str})
    processed_index = df['INDEX'].tolist()

    # prepare xai evaluator
    xai_eval = XAIEvaluator(device=device)

    pkl_dir = Path("outputs/pkls")
    for expl_file in expl_list:
        if expl_file.stem in processed_index:
            continue  # skip the file if it has been processed

        pkl_file = pkl_dir / (expl_file.stem + ".pkl")
        if not pkl_file.exists():
            continue

        graph = TAG.load(pkl_file)

        try:
            with open(expl_file, "r", encoding="utf-8") as f:
                chatting = json.load(f)
        except json.JSONDecodeError:
            logger.error("%s Bad Format!", expl_file)
            continue

        response = chatting["messages"][-1]["content"]
        pattern = r'### Free-Text Explanation\s*(?:```markdown\s*(.*?)\s*```|(.+?))(?=\n##{1,3}\s|$)'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            logger.warning("No match found for the pattern.")
            continue  # skip the file
        explanation = match.group(1) or match.group(2)

        logger.info("Working on %s", pkl_file)

        start = time.time()

        try:
            # _, (saliency, faithfulness, brevity) = fil.valid(graph, explanation)
            saliency, faithfulness, brevity = xai_eval.metrics(
                graph, explanation
            )  # actually [s = precison, f = recall, b = f1]
        except torch.cuda.OutOfMemoryError as e:
            logger.warning("%s\nOut of memory! Skip file-%s", e, pkl_file.stem)
            continue  # skip processing the file due to OOM

        end = time.time()
        duration = end - start

        with open(quality_log, "a", encoding="utf-8") as f:
            f.write(f"{pkl_file.stem},{expl_file},{saliency},{faithfulness},{brevity},{duration:0.3f}\n")

        logger.info("validation-%s completed. Time: %.3fs", pkl_file.stem, duration)

    return quality_log


def dataframe_filter(dataframe, mode: str):
    assert mode is not None, "Proper filtering mode is required!"
    if mode == "auto":
        # auto-mod: select top p% for all metrics
        filtered_df = dataframe[
            (dataframe['SALIENCY'] >= dataframe['SALIENCY'].quantile(0.7))
            & (dataframe['FAITHFULNESS'] >= dataframe['FAITHFULNESS'].quantile(0.5))
            & (dataframe['BREVITY'] <= dataframe['BREVITY'].quantile(0.5))
        ]
    elif mode == "abl_sali":
        # abl_sali-mod: select top p% for all metrics except saliency
        filtered_df = dataframe[
            (dataframe['FAITHFULNESS'] >= dataframe['FAITHFULNESS'].quantile(0.5))
            & (dataframe['BREVITY'] <= dataframe['BREVITY'].quantile(0.5))
        ]
        if len(filtered_df) > 100:
            filtered_df = filtered_df.sample(100, random_state=42)
    elif mode == "abl_fait":
        # abl_fait-mod: select top p% for all metrics except faithfulness
        filtered_df = dataframe[
            (dataframe['SALIENCY'] >= dataframe['SALIENCY'].quantile(0.7))
            & (dataframe['BREVITY'] <= dataframe['BREVITY'].quantile(0.5))
        ]
        if len(filtered_df) > 100:
            filtered_df = filtered_df.sample(100, random_state=42)
    elif mode == "abl_brev":
        # abl_brev-mod: select top p% for all metrics except brevity
        filtered_df = dataframe[
            (dataframe['SALIENCY'] >= dataframe['SALIENCY'].quantile(0.7))
            & (dataframe['FAITHFULNESS'] >= dataframe['FAITHFULNESS'].quantile(0.5))
        ]
        if len(filtered_df) > 100:
            filtered_df = filtered_df.sample(100, random_state=42)
    elif mode == "sali":
        # sali-mode: only consider saliency score
        filtered_df = dataframe.sort_values(by='SALIENCY', ascending=False).head(50)
    elif mode == "brev":
        # brev-mode: only consider brevity score
        filtered_df = dataframe.sort_values(by='BREVITY', ascending=True).head(50)
    elif mode == "fait":
        # fait-mode: only consider faithfulness score
        filtered_df = dataframe.sort_values(by='FAITHFULNESS', ascending=False).head(50)
    elif mode == "identical":  # a special case that DO NOT filter the dataframe
        filtered_df = dataframe
    else:
        raise ValueError(f"Invalid mode: {mode}. Supported modes: auto, sali, brev, identical")

    return filtered_df


def write_ft_dataset(exp_id, filtered_df):
    # load selected explanations
    records = []
    for exp_file in filtered_df['FILE']:
        exp_file = Path(exp_file)
        with open(exp_file, 'r', encoding="utf-8") as f:
            record = json.load(f)
            records.append(record)

    # save the filtered records to .jsonl
    output_dir = Path("outputs/finetuning_jsonl")
    output_dir.mkdir(exist_ok=True)
    ft_dataset = output_dir / f"finetune-{exp_id}.jsonl"
    with open(ft_dataset, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    logger.info("wrote dataset to %s with %d samples", ft_dataset, len(records))

    return ft_dataset


def rejection_sampling(exp_id, device, indexs=None, mode=None):
    # search for available explanations
    expl_dir = Path(f"outputs/generated_expls-{exp_id}")
    expl_list = [expl_dir / f"{index}.json" for index in indexs]
    expl_list = [f for f in expl_list if f.exists()]

    # evaluate explanations
    quality_log = cal_metrics(exp_id, device, expl_list)

    # Load metrics
    df = pd.read_csv(quality_log)

    # filter the dataframe
    filtered_df = dataframe_filter(df, mode)

    # prepare fine-tuning dataset with filtered explanations
    write_ft_dataset(exp_id, filtered_df)


def combine_multiple_explanations(exp_id, node_ids, sample_steps, split=(0.5, 0.3, 0.2)):
    # merge sample_step times quality-dataframes
    quality_files = [Path(f"outputs/quality/quality-{exp_id}s{i}.csv") for i in range(1, sample_steps + 1)]
    quality_dataframes = [pd.read_csv(file) for file in quality_files]
    concat_dataframes = pd.concat(quality_dataframes, ignore_index=True)

    quartile_of_saliency = concat_dataframes["SALIENCY"].quantile(0.5)
    quartile_of_faithfulness = concat_dataframes["FAITHFULNESS"].quantile(0.25)
    quartile_of_brevity = concat_dataframes["BREVITY"].quantile(0.75)

    merged_df = pd.DataFrame(columns=quality_dataframes[0].columns)
    parts = [
        node_ids[: int(len(node_ids) * split[0])],
        node_ids[int(len(node_ids) * split[0]) : int(len(node_ids) * (split[0] + split[1]))],
        node_ids[int(len(node_ids) * (split[0] + split[1])) :],
    ]  # saliency, faithfulness, brevity
    for index in parts[0]:
        records = pd.concat([df.loc[df["INDEX"] == index] for df in quality_dataframes], axis=0, ignore_index=True)
        if records.empty:
            continue  # Skip if no records found for the index

        # select the best record by SCORE
        best_record = records.loc[records["SALIENCY"].idxmax()]
        if best_record["FAITHFULNESS"] > quartile_of_faithfulness and best_record["BREVITY"] < quartile_of_brevity:
            merged_df = pd.concat([merged_df, best_record.to_frame().T], ignore_index=True)

    for index in parts[1]:
        records = pd.concat([df.loc[df["INDEX"] == index] for df in quality_dataframes], axis=0, ignore_index=True)
        if records.empty:
            continue  # Skip if no records found for the index

        # select the best record by SCORE
        best_record = records.loc[records["FAITHFULNESS"].idxmax()]
        if best_record["SALIENCY"] > quartile_of_saliency and best_record["BREVITY"] < quartile_of_brevity:
            merged_df = pd.concat([merged_df, best_record.to_frame().T], ignore_index=True)

    for index in parts[2]:
        records = pd.concat([df.loc[df["INDEX"] == index] for df in quality_dataframes], axis=0, ignore_index=True)
        if records.empty:
            continue  # Skip if no records found for the index

        # select the best record by SCORE
        best_record = records.loc[records["BREVITY"].idxmax()]
        if best_record["FAITHFULNESS"] > quartile_of_faithfulness and best_record["SALIENCY"] > quartile_of_saliency:
            merged_df = pd.concat([merged_df, best_record.to_frame().T], ignore_index=True)

    # save merged dataframe
    merged_quality_file = Path("outputs/quality/quality-" + str(exp_id) + ".csv")
    merged_df.to_csv(merged_quality_file, index=False)

    # copy selected expls
    expls_dir = Path("outputs/generated_expls-" + str(exp_id))
    expls_dir.mkdir(exist_ok=True)
    for expl_file in merged_df["FILE"].values:
        expl_file = Path(expl_file)
        if expl_file.exists():
            shutil.copy(expl_file, expls_dir / expl_file.name)

    # write fine-tuned dataset
    write_ft_dataset(exp_id, merged_df)
