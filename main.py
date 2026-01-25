import argparse
import math
import multiprocessing
import random
import shutil
from pathlib import Path

import pandas as pd

from filter import combine_multiple_explanations, rejection_sampling
from finetuning import self_training
from free_text_explanation import generate_explanations
from log import logger
from preprocess import generate_dataset


# helper function to initiate multiple processes for explanation generation
def multiprocess_expl_gen(exp_id, model_id, indexs, num_processes=32):
    # assign processes
    procs = []
    for t in range(num_processes):
        params = (
            [indexs[i] for i in range(t, len(indexs), num_processes)],
            model_id,
            exp_id,
        )
        proc = multiprocessing.Process(target=generate_explanations, args=params)
        procs.append(proc)
        proc.start()

    # wait for all processes to finish
    for proc in procs:
        proc.join()

    logger.info("All explanation generation processes finished.")


# helper function to initiate multiple processes for rejection sampling
def multiprocess_rejection_sampling(exp_id, devices, indexs, mode):
    # Set the start method to 'spawn' to use multiprocessing with CUDA
    multiprocessing.set_start_method('spawn', force=True)

    # assign processes
    procs = []
    num_proc = len(devices)
    for t in range(num_proc):
        params = (exp_id, devices[t], [indexs[i] for i in range(t, len(indexs), num_proc)], mode)
        proc = multiprocessing.Process(target=rejection_sampling, args=params)
        procs.append(proc)
        proc.start()

    # wait for all processes to finish
    for proc in procs:
        proc.join()

    logger.info("All rejection sampling processes finished.")


def manual(args):
    # -- step by step -- #
    # Step 0: Preprocess
    if 0 in args.step:
        generate_dataset(args.seed, args.lm_model, args.gnn_model, args.dataset, args)

    # Step 1: Dataset Generation
    if 1 in args.step:
        multiprocess_expl_gen(args.exp_id, args.model_id, indexs=range(2708))

    # Step 2: Rejection Sampling
    if 2 in args.step:
        multiprocess_rejection_sampling(exp_id=args.exp_id, devices=args.devices, indexs=range(2708), mode="auto")

    # # step 3: Self-Training
    if 3 in args.step:
        self_training(exp_id=args.exp_id, model_id=args.model_id)


def auto(args):
    # -- auto mode -- #
    model_id = args.resume_model_id
    assert (args.resume > 0 and args.resume_model_id != "Qwen/Qwen2.5-7B-Instruct") or (
        args.resume == 0 and args.resume_model_id == "Qwen/Qwen2.5-7B-Instruct"
    ), "Invalid model id!"

    for exp_id in range(args.resume, args.num_iterations):
        # -- step 1: expl generation -- #
        multiprocess_expl_gen(exp_id=exp_id, model_id=model_id, indexs=range(0, 2000), num_processes=args.num_processes)
        # -- step 2: rejection sampling -- #
        multiprocess_rejection_sampling(exp_id=exp_id, devices=args.devices, indexs=range(0, 2000), mode="auto")
        # -- step 3: self-training -- #
        model_id = self_training(exp_id=exp_id, model_id=model_id)


def abl_sali(args):
    # -- ablation study for saliency -- #
    model_id = args.resume_model_id
    assert (args.resume > 0 and args.resume_model_id != "Qwen/Qwen2.5-7B-Instruct") or (
        args.resume == 0 and args.resume_model_id == "Qwen/Qwen2.5-7B-Instruct"
    ), "Invalid model id!"

    for exp_id in range(args.resume, args.num_iterations):
        # -- step 1: expl generation -- #
        multiprocess_expl_gen(exp_id=exp_id, model_id=model_id, indexs=range(0, 500), num_processes=args.num_processes)
        # -- step 2: rejection sampling -- #
        multiprocess_rejection_sampling(exp_id=exp_id, devices=args.devices, indexs=range(0, 500), mode="abl_sali")
        # -- step 3: self-training -- #
        model_id = self_training(exp_id=exp_id, model_id=model_id, ft_prefix="graph-abl-sali")


def abl_fait(args):
    # -- ablation study for fait -- #
    model_id = args.resume_model_id
    assert (args.resume > 0 and args.resume_model_id != "Qwen/Qwen2.5-7B-Instruct") or (
        args.resume == 0 and args.resume_model_id == "Qwen/Qwen2.5-7B-Instruct"
    ), "Invalid model id!"

    for exp_id in range(args.resume, args.num_iterations):
        # -- step 1: expl generation -- #
        multiprocess_expl_gen(exp_id=exp_id, model_id=model_id, indexs=range(0, 500), num_processes=args.num_processes)
        # -- step 2: rejection sampling -- #
        multiprocess_rejection_sampling(exp_id=exp_id, devices=args.devices, indexs=range(0, 500), mode="abl_fait")
        # -- step 3: self-training -- #
        model_id = self_training(exp_id=exp_id, model_id=model_id, ft_prefix="graph-abl-fait")


def abl_brev(args):
    # -- ablation study for brev -- #
    model_id = args.resume_model_id
    assert (args.resume > 0 and args.resume_model_id != "Qwen/Qwen2.5-7B-Instruct") or (
        args.resume == 0 and args.resume_model_id == "Qwen/Qwen2.5-7B-Instruct"
    ), "Invalid model id!"

    for exp_id in range(args.resume, args.num_iterations):
        # -- step 1: expl generation -- #
        multiprocess_expl_gen(exp_id=exp_id, model_id=model_id, indexs=range(0, 500), num_processes=args.num_processes)
        # -- step 2: rejection sampling -- #
        multiprocess_rejection_sampling(exp_id=exp_id, devices=args.devices, indexs=range(0, 500), mode="abl_brev")
        # -- step 3: self-training -- #
        model_id = self_training(exp_id=exp_id, model_id=model_id, ft_prefix="graph-abl-brev")


def sali(args):
    # -- saliency-concentrated improvement experiment -- #
    model_id = args.resume_model_id
    assert (args.resume > 0 and args.resume_model_id != "Qwen/Qwen2.5-7B-Instruct") or (
        args.resume == 0 and args.resume_model_id == "Qwen/Qwen2.5-7B-Instruct"
    ), "Invalid model id!"

    for exp_id in range(args.resume, 5):
        multiprocess_expl_gen(exp_id=exp_id, model_id=model_id, indexs=range(0, 500))
        multiprocess_rejection_sampling(exp_id=exp_id, devices=args.devices, indexs=range(0, 500), mode="sali")
        model_id = self_training(exp_id=exp_id, model_id=model_id, ft_prefix="graph-sali")
        logger.info("iteration:%s completed!", exp_id)


def brev(args):
    # -- brev-concentrated improvement experiment -- #
    model_id = args.resume_model_id
    assert (args.resume > 0 and args.resume_model_id != "Qwen/Qwen2.5-7B-Instruct") or (
        args.resume == 0 and args.resume_model_id == "Qwen/Qwen2.5-7B-Instruct"
    ), "Invalid model id!"

    for exp_id in range(args.resume, 5):
        multiprocess_expl_gen(exp_id=exp_id, model_id=model_id, indexs=range(0, 500))
        multiprocess_rejection_sampling(exp_id=exp_id, devices=args.devices, indexs=range(0, 500), mode="brev")
        model_id = self_training(exp_id=exp_id, model_id=model_id, ft_prefix="graph-brev")
        logger.info("iteration:%s completed!", exp_id)


def fait(args):
    # -- fait-concentrated improvement experiment -- #
    model_id = args.resume_model_id
    assert (args.resume > 0 and args.resume_model_id != "Qwen/Qwen2.5-7B-Instruct") or (
        args.resume == 0 and args.resume_model_id == "Qwen/Qwen2.5-7B-Instruct"
    ), "Invalid model id!"

    for exp_id in range(args.resume, 5):
        multiprocess_expl_gen(exp_id=exp_id, model_id=model_id, indexs=range(0, 500))
        multiprocess_rejection_sampling(exp_id=exp_id, devices=args.devices, indexs=range(0, 500), mode="fait")
        model_id = self_training(exp_id=exp_id, model_id=model_id, ft_prefix="graph-fait")
        logger.info("iteration:%s completed!", exp_id)


def rand(args):
    # --random nodes experiments -- #
    model_id = args.resume_model_id
    assert (args.resume > 0 and args.resume_model_id != "Qwen/Qwen2.5-7B-Instruct") or (
        args.resume == 0 and args.resume_model_id == "Qwen/Qwen2.5-7B-Instruct"
    ), "Invalid model id!"

    for exp_id in range(args.resume, args.num_iterations):
        # sample rand_size nodes
        random.seed(4)
        rand_nodes = random.sample(range(2708), args.rand_size)

        # generate sample_step times expls for each sampled node
        for sample_step in range(1, args.sample_steps + 1):
            subexp_id = str(exp_id) + "s" + str(sample_step)
            multiprocess_expl_gen(exp_id=subexp_id, model_id=model_id, indexs=rand_nodes, num_processes=args.num_processes)  # step 1
            multiprocess_rejection_sampling(
                exp_id=subexp_id, devices=args.devices, indexs=rand_nodes, mode="identical"
            )  # step 2

        # combine expl with configurable splits
        combine_multiple_explanations(
            exp_id=exp_id,
            node_ids=rand_nodes,
            sample_steps=args.sample_steps,
            split=(args.train_split, args.val_split, args.test_split),
        )
        model_id = self_training(exp_id=exp_id, model_id=model_id, ft_prefix="graph-rand-weighted-quartile")  # step 3

        # save model_id to local csv file
        with open("outputs/model_ids.csv", "a", encoding="utf-8") as f:
            f.write(f"{exp_id},{model_id}\n")

        logger.info("iteration:%s completed!", exp_id)


def parse():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lm_model", type=str, default="bert-base-uncased")
    parser.add_argument("--gnn_model", type=str, default="SAGE")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--exp_id", type=int, default=None, help="indicate the number of experiment iteration")
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--step", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--devices", nargs='+', type=str, default=["cuda:0", "cuda:1"], help="List of devices to use (e.g., cuda:0 cuda:1)")
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--resume_model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="the model id to resume from")
    parser.add_argument("--mode", type=str, default="auto", required=True, help="the mode of the pipeline [auto;sali;manual]")
    
    parser.add_argument("--num_processes", type=int, default=32, help="Number of processes for explanation generation")
    parser.add_argument("--num_iterations", type=int, default=5, help="Number of iterations for auto/ablation modes")
    parser.add_argument("--rand_size", type=int, default=60, help="Number of random nodes to sample")
    parser.add_argument("--sample_steps", type=int, default=3, help="Number of sample steps for random nodes")
    parser.add_argument("--train_split", type=float, default=0.7, help="Training split ratio for random nodes")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio for random nodes")
    parser.add_argument("--test_split", type=float, default=0.2, help="Test split ratio for random nodes")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoint", help="Directory to store model checkpoints")
    # fmt: on
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse()

    if args.mode == "auto":
        auto(args)  # the automatical pipeline

    elif args.mode == "sali":
        sali(args)  # saliency-impr-exp pipeline

    elif args.mode == "brev":
        brev(args)  # brev-impr-exp pipeline

    elif args.mode == "fait":
        fait(args)  # fait-impr-exp pipeline

    elif args.mode == "rand":
        rand(args)  # random-nodes-exp pipeline

    elif args.mode == "abl_sali":
        abl_sali(args)  # ablation study for saliency

    elif args.mode == "abl_brev":
        abl_brev(args)  # ablation study for brev

    elif args.mode == "abl_fait":
        abl_fait(args)  # ablation study for fait

    elif args.mode == "manual":
        manual(args)  # deprecated step-by-step pipeline
