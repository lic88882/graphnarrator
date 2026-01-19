import os
import time
from pathlib import Path
from xml.parsers.expat import model

import torch
from torch.optim import AdamW
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import CORA, DBLP, BookHistory, Encoder
from log import logger
from models.joint_model import LM_GNN_Joint_Model, init_hooks_lrp
from tag import TAG, Node
from utils import safe_zip


def train(DATASET, ckpt_dir, SEED=42, LM_MODEL="bert-base-uncased", GNN_MODEL="SAGE"):
    # == load dataset ==
    if DATASET == "dblp":
        data, text, num_classes = DBLP.load()
    elif DATASET == "cora":
        data, text, num_classes = CORA.load(seed=SEED)
    elif DATASET == "book_history":
        data, text, num_classes = BookHistory.load()
    else:
        raise NotImplementedError

    data = data.to("cuda")

    encoder = Encoder()  # natural language encoder

    model = LM_GNN_Joint_Model(lm_model_name=LM_MODEL, gnn_model_name=GNN_MODEL, out_dim=num_classes).to("cuda")
    best_val_accuracy = 0
    best_model_state = model.state_dict()

    # Create dataset-specific checkpoint directory
    dataset_ckpt_dir = os.path.join(ckpt_dir, DATASET)
    os.makedirs(dataset_ckpt_dir, exist_ok=True)

    # Try loading canonical best checkpoint if it exists
    best_ckpt_path = f"{dataset_ckpt_dir}/GNN_{DATASET}_seed_{SEED}.pt"

    if os.path.exists(best_ckpt_path):
        print(f"[INFO] Loading existing checkpoint: {best_ckpt_path}")
        # load_from_ckpt
        model.gnn.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
        # Validation Loop
        model.eval()
        with torch.no_grad():
            subgraph_loader = NeighborLoader(data, num_neighbors=[10] * 2, batch_size=512, input_nodes=data.val_mask)
            predictions = []
            ground_truths = []
            for batch in subgraph_loader:
                embs = torch.concat([encoder.encode(text[i]).detach().cpu() for i in batch.n_id.tolist()]).to("cuda")
                edge_index = batch.edge_index.to("cuda")
                logits = model.forward_gnn(embs, edge_index)
                pred = torch.argmax(logits, dim=1)
                predictions.append(pred)
                ground_truths.append(batch.y)
            predictions = torch.concat(predictions, dim=0)
            ground_truths = torch.concat(ground_truths, dim=0)
            correct = (predictions == ground_truths).float()
            accuracy = correct.sum() / len(correct)
        best_val_accuracy = accuracy  # best validation accuracy
        print(f"[INFO] Loaded checkpoint validation accuracy = {accuracy:.4f}")
    else:
        print("[INFO] No checkpoint found. Training from scratch.")

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.00)

    model.train()

    logger.info("Start training")
    # log GPU VRAM usage
    logger.info("GPU VRAM usage: %.2f GB", torch.cuda.memory_allocated() / 1024 / 1024 / 1024)

    EPOCHS = 100
    BATCH_SIZE = 16
    for epoch in range(EPOCHS):
        total_loss = 0

        # sample a subgraph from data
        subgraph_loader = NeighborLoader(
            data, num_neighbors=[10] * 2, batch_size=BATCH_SIZE, input_nodes=data.train_mask
        )

        step = 0
        for batch in subgraph_loader:
            step += 1
            if step >= 100:
                break
            model.train()
            optimizer.zero_grad()

            start = time.time()

            embs = torch.concat([encoder.encode(text[i]).detach().cpu() for i in batch.n_id.tolist()], dim=0).to("cuda")
            edge_index = batch.edge_index.to("cuda")

            # forward pass
            logits = model.forward_gnn(embs, edge_index)

            # labels
            loss = torch.nn.CrossEntropyLoss()(logits, batch.y)

            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            end = time.time()
            # logger.info("GPU VRAM usage: %.2f GB", torch.cuda.memory_allocated() / 1024 / 1024 / 1024)
            logger.info(
                "Epoch: %d, Step: %d, Sampled_Nodes: %d, Training loss: %.4f, Time: %.2fs",
                epoch,
                step,
                len(batch.x),
                loss.item(),
                end - start,
            )

        # Validation Loop
        model.eval()
        with torch.no_grad():
            subgraph_loader = NeighborLoader(
                data, num_neighbors=[10] * 2, batch_size=BATCH_SIZE, input_nodes=data.val_mask
            )
            predictions = []
            ground_truths = []
            for batch in subgraph_loader:
                embs = torch.concat([encoder.encode(text[i]).detach().cpu() for i in batch.n_id.tolist()]).to("cuda")
                edge_index = batch.edge_index.to("cuda")
                logits = model.forward_gnn(embs, edge_index)
                pred = torch.argmax(logits, dim=1)
                predictions.append(pred)
                ground_truths.append(batch.y)
            predictions = torch.concat(predictions, dim=0)
            ground_truths = torch.concat(ground_truths, dim=0)
            correct = (predictions == ground_truths).float()
            accuracy = correct.sum() / len(correct)

        logger.info(f"Validation Accuracy: {accuracy}")
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            best_model_state = model.state_dict()

            # Save canonical best checkpoint
            torch.save(model.gnn.state_dict(), f'{dataset_ckpt_dir}/GNN_{DATASET}_seed_{SEED}.pt')
            logger.info(f"New best model saved with accuracy: {best_val_accuracy}")

        # Save epoch-specific checkpoint
        torch.save(model.gnn.state_dict(), f'{dataset_ckpt_dir}/GNN_{DATASET}_seed_{SEED}_epoch_{epoch}.pt')
        
        avg_train_loss = total_loss / step / BATCH_SIZE
        logger.info(f"Training loss: {avg_train_loss}")

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        subgraph_loader = NeighborLoader(
            data, num_neighbors=[10] * 2, batch_size=BATCH_SIZE, input_nodes=data.test_mask
        )
        predictions = []
        ground_truths = []

        for batch in subgraph_loader:
            embs = torch.concat([encoder.encode(text[i]).detach().cpu() for i in batch.n_id.tolist()]).to("cuda")
            edge_index = batch.edge_index.to("cuda")
            logits = model.forward_gnn(embs, edge_index)
            pred = torch.argmax(logits, dim=1)
            predictions.append(pred)
            ground_truths.append(batch.y)
        predictions = torch.concat(predictions, dim=0)
        ground_truths = torch.concat(ground_truths, dim=0)

        correct = (predictions == ground_truths).float()
        accuracy = correct.sum() / len(correct)
        print(f"Test Accuracy: {accuracy}")

    return accuracy.cpu()


def generate_dataset(SEED, LM_MODEL, GNN_MODEL, DATASET, args):
    # == Load Datasets ==
    if DATASET == "cora":
        data, text, num_classes = CORA.load(seed=SEED)
    elif DATASET == "book_history":
        data, text, num_classes = BookHistory.load()
    elif DATASET == "dblp":
        data, text, num_classes = DBLP.load()
    else:
        raise NotImplementedError

    data = data.to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(LM_MODEL)

    subgraph_loader = NeighborLoader(data, num_neighbors=[10] * 2, batch_size=1, input_nodes=data.train_mask)

    # == Load Model ==
    model = LM_GNN_Joint_Model(lm_model_name=LM_MODEL, gnn_model_name=GNN_MODEL, out_dim=num_classes)
    model.to(args.device)

    # Load checkpoint from dataset-specific directory
    dataset_ckpt_dir = os.path.join(args.ckpt_dir, DATASET)
    checkpoint_path = f"{dataset_ckpt_dir}/GNN_{DATASET}_seed_{SEED}.pt"
    model.gnn.load_state_dict(torch.load(checkpoint_path),)
    model.eval()

    # == Load Optimizer ==
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.00)
    init_hooks_lrp(model)

    # == Explain ==

    def id2label(idx):
        # CORA dataset label mapping
        cora_labels = [
            "Case Based",               # 0
            "Genetic Algorithms",       # 1
            "Neural Networks",          # 2
            "Probabilistic Methods",    # 3
            "Reinforcement Learning",   # 4
            "Rule Learning",            # 5
            "Theory",                   # 6
        ]
        
        if DATASET == "cora":
            return cora_labels[idx]
        elif DATASET == "dblp":
            return data.label_dict[idx]
        else:
            return str(idx)

    for index, batch in enumerate(subgraph_loader):
        if index >= 2500:
            break

        output_pkl = Path(f"outputs/pkls/{index}.pkl")
        if output_pkl.exists():
            continue

        start = time.time()
        edge_index = batch.edge_index
        sampled_node_ids = batch.n_id.tolist()
        X = tokenizer([text[i] for i in sampled_node_ids], padding="max_length", truncation=True, max_length=512)
        input_ids = torch.tensor(X["input_ids"]).to(args.device)
        attention_mask = torch.tensor(X["attention_mask"]).to(args.device)

        # == Saliency-based explanation ==
        try:
            embedding_scores, node_scores, prediction = model.explain(input_ids, attention_mask, edge_index)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"out of memory! Skip sample-{index}")
            continue

        optimizer.zero_grad()
        # token_scores = torch.abs(embedding_scores).sum(dim=-1)
        token_scores = embedding_scores.sum(dim=-1)
        token_scores = token_scores.detach().numpy()

        # initialize the text on output document nodes
        graph = TAG()
        for i in range(input_ids.shape[0]):
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
            # sentence = tokenizer.convert_tokens_to_string(tokens)
            sep_index = tokens.index("[SEP]")
            tokens = tokens[1:sep_index]
            attribution = token_scores[i][1:sep_index]  # skip [CLS] and [SEP]
            node = Node()

            for token, score in safe_zip(tokens, attribution):
                # concat sub-tokens into one English word
                if token.startswith("##"):
                    node.concat(token[2:], score)
                elif token in [",", ".", "!", "?", "-"]:
                    node.concat(token, score)
                else:
                    node.append(token, score)
            graph.add_node(node)

        for dst, src in zip(edge_index[0], edge_index[1]):
            dst = dst.item()
            src = src.item()

            graph.add_edge(src, dst)
            # we also add a revere edge to make the graph undirected
            # as which is the actual data structure that GNN
            # see during training and inference
            graph.add_edge(dst, src)  # pylint: disable=arguments-out-of-order

            # == strong citation direction ==

            # convert local node id to global node id
            # global_dst = sampled_node_ids[dst]
            # global_src = sampled_node_ids[src]

            # if [global_src, global_dst] in directed_edge_index.tolist():
            # graph.add_edge(dst, src)

        graph.set_labels(prediction=id2label(prediction.item()), ground_truth=id2label(batch.y[0].item()))
        # print(saliency_map := graph.text(style="document"))
        graph.save(output_pkl)
        end = time.time()
        logger.info("batch-%d prepared. Time: %.2fs", index, end - start)


def compute_embeddings(batch_size, dataset):
    dataset_lower = dataset.lower()
    
    if dataset_lower == "dblp":
        data, text, num_classes = DBLP.load()
    elif dataset_lower == "cora":
        data, text, num_classes = CORA.load()
    elif dataset_lower == "book_history":
        data, text, num_classes = BookHistory.load()
    else:
        print(f"{dataset} embedding is not yet implemented.")
        return
    
    encoder = Encoder()
    embs = []
    for i in range(0, len(text), batch_size):
        t = text[i : i + batch_size]
        embs.append(encoder.encode(t).detach().cpu())
        print(f"Processed {i} texts, GPU VRAM: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB")

    x = torch.concat(embs, dim=0)
    data.x = x  # install embeddings into the PyG Dataset
    data.text = text  # plug nodes_text back

    # Save to dataset-specific path
    output_path = f"dataset/{dataset_lower}_updated_w_emb.pt"
    torch.save(data, output_path)
    print(f"[INFO] Embeddings saved to {output_path}")


if __name__ == "__main__":
    # == train GNN models ==
    train(DATASET="cora", ckpt_dir="./checkpoint", SEED=42, LM_MODEL="bert-base-uncased", GNN_MODEL="SAGE")

    # == save pkls ==
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--lm_model", type=str, default="bert-base-uncased")
    # parser.add_argument("--gnn_model", type=str, default="SAGE")
    # parser.add_argument("--dataset", type=str, default="dblp")
    # parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # args = parser.parse_args()

    # generate_dataset(args.seed, args.lm_model, args.gnn_model, args.dataset, args)
