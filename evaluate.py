import argparse
import csv
import json
import random

import regex as re

from filter import XAIEvaluator
from free_text_explanation import FreeTextExpGenerator
from tag import TAG


def retrieve_explanation_from_history(sample_id, archive_path):
    with open(f"{archive_path}/{sample_id}.json", "r", encoding="utf-8") as f:
        chatting = json.load(f)
    return chatting["messages"][-1]["content"]


def generate_explanation_from_openai(generator, graph, model):
    prompt, response = generator.gen(
        label=graph.prediction,
        document=graph.text(style="document", with_score=False, masker=""),
        model=model,
        example_version="v1",
    )
    return prompt, response


def generate_explanation_from_local(generator, graph, finetuned_model):
    prompt, response = generator.gen(
        label=graph.prediction,
        document=graph.text(style="document", with_score=False, masker=""),
        use_finetuned=True,
        model=finetuned_model,
    )
    return prompt, response


def parse_explanation(response):
    pattern = r'### Free-Text Explanation\s*(?:```markdown\s*(.*?)\s*```|(.+?))(?=\n##{1,3}\s|$)'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1) or match.group(2)
    else:
        raise ValueError("No match found for the pattern.")


def evaluate_explanation(graph, explanation, masking_range=(0, 80)):
    fil = XAIEvaluator()
    fil.initialize_mlm()
    return fil.metrics(graph, explanation, masking_range=masking_range)


def main(args):
    # prepare testing node ids
    random.seed(42)
    last_700_indices = list(range(2708 - 700, 2708))
    node_ids = random.sample(last_700_indices, 50)
    metrics = []

    generator = FreeTextExpGenerator()

    for sample_id in node_ids:
        graph = TAG.load(f"archive/pkls/{sample_id}.pkl")

        if args.mode == "history":
            response = retrieve_explanation_from_history(sample_id, args.history_archive)
        elif args.mode == "openai":
            prompt, response = generate_explanation_from_openai(generator, graph, args.model_id)
        elif args.mode == "local":
            prompt, response = generate_explanation_from_local(generator, graph, args.model_id)
        else:
            raise ValueError("Invalid mode specified")

        explanation = parse_explanation(response)

        result = evaluate_explanation(graph, explanation, masking_range=(0, args.masking_range))
        metrics.append(result)

        print(f"Sample {sample_id} evaluated")

    with open(args.output, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "saliency", "faithfulness", "brevity"])
        for index, metric in zip(node_ids, metrics):
            writer.writerow([index, metric[0], metric[1], metric[2]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate explanations")
    parser.add_argument("--mode", choices=["history", "openai", "local"], required=True, help="Evaluation mode")
    parser.add_argument("--model_id", help="OpenAI model name (required for openai mode)")
    parser.add_argument("--history_archive", help="Iteration number (required for history mode)")
    parser.add_argument("--masking_range", type=int, default=80, help="Masking range (required for history mode)")
    parser.add_argument("--output", default="metrics.csv", help="Output CSV file name")
    args = parser.parse_args()

    if (args.mode == "openai" or args.mode == "local") and not args.model_id:
        parser.error("--model_id is required when mode is 'openai'")
    if args.mode == "history" and not args.history_archive:
        parser.error("--history_archive is required when mode is 'history'")
    main(args)
