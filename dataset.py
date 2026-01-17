import json
import random

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from transformers import AutoTokenizer
from transformers import BertModel as BertModelOfficial

from models.LMs.BERT_LRP.code.model.BERT import BertConfig, MyBertModel


class Encoder:
    def __init__(self, lm_model="bert-base-uncased"):
        config = BertConfig.from_json_file("models/LMs/BERT_LRP/models/BERT-Google/bert_config.json")
        self.bert = MyBertModel(config)
        Hug_bert_model = BertModelOfficial.from_pretrained(lm_model)
        target_state_dict = self.bert.state_dict()  # get the target model's state dict
        for name, param in Hug_bert_model.state_dict().items():
            if name in target_state_dict:
                if target_state_dict[name].shape == param.shape:
                    target_state_dict[name].copy_(param)
                else:
                    print(f"Skipping {name} due to shape mismatch.")
        self.bert.load_state_dict(target_state_dict, strict=False)

        self.tokenizer = AutoTokenizer.from_pretrained(lm_model)
        self.bert = self.bert.to("cuda")

    def encode(self, string):
        inputs = self.tokenizer(string, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(
            "cuda"
        )
        outputs = self.bert(**inputs)
        return outputs[1]


class PubMed:
    @classmethod
    def load(cls):
        data, text = cls.get_raw_text_pubmed(use_text=True)
        return data, text

    @classmethod
    def get_pubmed_casestudy(cls, corrected=False, SEED=0):
        from sklearn.preprocessing import normalize

        _, data_X, data_Y, data_pubid, data_edges = cls.parse_pubmed()
        data_X = normalize(data_X, norm="l1")

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)  # Numpy module.
        random.seed(SEED)  # Python random module.

        # load data
        data_name = "PubMed"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
        dataset = Planetoid("dataset", data_name, transform=T.NormalizeFeatures())
        data = dataset[0]

        # replace dataset matrices with the PubMed-Diabetes data, for which we have the original pubmed IDs
        data.x = torch.tensor(data_X)
        data.edge_index = torch.tensor(data_edges)
        data.y = torch.tensor(data_Y)

        # split data
        node_id = np.arange(data.num_nodes)
        np.random.shuffle(node_id)

        data.train_id = np.sort(node_id[: int(data.num_nodes * 0.6)])
        data.val_id = np.sort(node_id[int(data.num_nodes * 0.6) : int(data.num_nodes * 0.8)])
        data.test_id = np.sort(node_id[int(data.num_nodes * 0.8) :])

        if corrected:
            is_mistake = np.loadtxt("pubmed_casestudy/pubmed_mistake.txt", dtype="bool")
            data.train_id = [i for i in data.train_id if not is_mistake[i]]
            data.val_id = [i for i in data.val_id if not is_mistake[i]]
            data.test_id = [i for i in data.test_id if not is_mistake[i]]

        data.train_mask = torch.tensor([x in data.train_id for x in range(data.num_nodes)])
        data.val_mask = torch.tensor([x in data.val_id for x in range(data.num_nodes)])
        data.test_mask = torch.tensor([x in data.test_id for x in range(data.num_nodes)])

        return data, data_pubid

    @classmethod
    def parse_pubmed(cls):
        path = "dataset/PubMed_orig/data/"

        n_nodes = 19717
        n_features = 500

        data_X = np.zeros((n_nodes, n_features), dtype="float32")
        data_Y = [None] * n_nodes
        data_pubid = [None] * n_nodes
        data_edges = []

        paper_to_index = {}
        feature_to_index = {}

        # parse nodes
        with open(path + "Pubmed-Diabetes.NODE.paper.tab", "r") as node_file:
            # first two lines are headers
            node_file.readline()
            node_file.readline()

            k = 0

            for i, line in enumerate(node_file.readlines()):
                items = line.strip().split("\t")

                paper_id = items[0]
                data_pubid[i] = paper_id
                paper_to_index[paper_id] = i

                # label=[1,2,3]
                label = int(items[1].split("=")[-1]) - 1  # subtract 1 to zero-count
                data_Y[i] = label

                # f1=val1 \t f2=val2 \t ... \t fn=valn summary=...
                features = items[2:-1]
                for feature in features:
                    parts = feature.split("=")
                    fname = parts[0]
                    fvalue = float(parts[1])

                    if fname not in feature_to_index:
                        feature_to_index[fname] = k
                        k += 1

                    data_X[i, feature_to_index[fname]] = fvalue

        # parse graph
        data_A = np.zeros((n_nodes, n_nodes), dtype="float32")

        with open(path + "Pubmed-Diabetes.DIRECTED.cites.tab", "r") as edge_file:
            # first two lines are headers
            edge_file.readline()
            edge_file.readline()

            for i, line in enumerate(edge_file.readlines()):

                # edge_id \t paper:tail \t | \t paper:head
                items = line.strip().split("\t")

                edge_id = items[0]

                tail = items[1].split(":")[-1]
                head = items[3].split(":")[-1]

                data_A[paper_to_index[tail], paper_to_index[head]] = 1.0
                data_A[paper_to_index[head], paper_to_index[tail]] = 1.0
                if head != tail:
                    data_edges.append((paper_to_index[head], paper_to_index[tail]))
                    data_edges.append((paper_to_index[tail], paper_to_index[head]))

        return (
            data_A,
            data_X,
            data_Y,
            data_pubid,
            np.unique(data_edges, axis=0).transpose(),
        )

    @classmethod
    def get_raw_text_pubmed(cls, use_text=False, seed=0):
        data, data_pubid = cls.get_pubmed_casestudy(SEED=seed)
        if not use_text:
            return data, None

        f = open("dataset/PubMed_orig/pubmed.json")
        pubmed = json.load(f)
        df_pubmed = pd.DataFrame.from_dict(pubmed)

        AB = df_pubmed["AB"].fillna("")
        TI = df_pubmed["TI"].fillna("")
        text = []
        for ti, ab in zip(TI, AB):
            t = "Title: " + ti + "\n" + "Abstract: " + ab
            text.append(t)
        return data, text


class CORA:
    """
    CORA dataset
    """

    @classmethod
    def load(cls, seed):
        data, text = cls.get_raw_text_cora(use_text=True, seed=seed)
        return data, text, 7

    @classmethod
    def get_cora_casestudy(cls, SEED=0, training_size=0.6, validation_size=0.2):
        data_X, data_Y, data_citeid, data_edges, data_directed_edges = cls.parse_cora()
        # data_X = sklearn.preprocessing.normalize(data_X, norm="l1")

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)  # Numpy module.
        random.seed(SEED)  # Python random module.

        # load data
        data_name = "cora"
        # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
        dataset = Planetoid("dataset", data_name, transform=T.NormalizeFeatures())
        data = dataset[0]

        data.x = torch.tensor(data_X).float()
        data.edge_index = torch.tensor(data_edges).long()
        data.directed_edge_index = torch.tensor(data_directed_edges).long()
        data.y = torch.tensor(data_Y).long()
        data.num_nodes = len(data_Y)

        # split data
        node_id = np.arange(data.num_nodes)
        np.random.shuffle(node_id)

        data.train_id = np.sort(node_id[: int(data.num_nodes * training_size)])
        data.val_id = np.sort(
            node_id[int(data.num_nodes * training_size) : int(data.num_nodes * (validation_size + training_size))]
        )
        data.test_id = np.sort(node_id[int(data.num_nodes * (validation_size + training_size)) :])

        data.train_mask = torch.tensor([x in data.train_id for x in range(data.num_nodes)])
        data.val_mask = torch.tensor([x in data.val_id for x in range(data.num_nodes)])
        data.test_mask = torch.tensor([x in data.test_id for x in range(data.num_nodes)])

        return data, data_citeid

    # credit: https://github.com/tkipf/pygcn/issues/27, xuhaiyun

    @classmethod
    def parse_cora(cls):
        path = "dataset/cora_orig/cora"
        idx_features_labels = np.genfromtxt("{}.content".format(path), dtype=np.dtype(str))
        data_X = idx_features_labels[:, 1:-1].astype(np.float32)
        labels = idx_features_labels[:, -1]
        class_map = {
            x: i
            for i, x in enumerate(
                [
                    "Case_Based",
                    "Genetic_Algorithms",
                    "Neural_Networks",
                    "Probabilistic_Methods",
                    "Reinforcement_Learning",
                    "Rule_Learning",
                    "Theory",
                ]
            )
        }
        data_Y = np.array([class_map[l] for l in labels])
        data_citeid = idx_features_labels[:, 0]
        idx = np.array(data_citeid, dtype=np.dtype(str))
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}.cites".format(path), dtype=np.dtype(str))
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
        # remove lines with missing value
        data_edges = np.array(edges[~(edges == None).max(1)], dtype="int")

        # keep directed edges
        data_directed_edges = data_edges

        # converting directed graph to undirected graph
        data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
        data_edges = np.unique(data_edges, axis=0)

        # transpose into the shape of (2, num_edges)
        data_edges = data_edges.T

        return data_X, data_Y, data_citeid, data_edges, data_directed_edges

    @classmethod
    def get_raw_text_cora(cls, use_text=False, seed=0, training_size=0.6, validation_size=0.2):
        data, data_citeid = cls.get_cora_casestudy(seed, training_size=training_size, validation_size=validation_size)
        if not use_text:
            return data, None

        with open("dataset/cora_orig/mccallum/cora/papers") as f:
            lines = f.readlines()
        pid_filename = {}
        for line in lines:
            pid = line.split("\t")[0]
            fn = line.split("\t")[1]
            pid_filename[pid] = fn

        path = "dataset/cora_orig/mccallum/cora/extractions/"
        text = []
        for pid in data_citeid:
            fn = pid_filename[pid]
            with open(path + fn) as f:
                lines = f.read().splitlines()

            for line in lines:
                if "Title:" in line:
                    ti = line
                if "Abstract:" in line:
                    ab = line
            text.append(ti + "\n" + ab)
        return data, text


class BookHistory:
    @classmethod
    def load(cls):
        data = torch.load("dataset/Book_History/book-history.pt")
        text = data.texts
        num_classes = len(set(data.categories))
        return data, text, num_classes


class DBLP:
    @classmethod
    def load(cls, with_emb=False):
        if not with_emb:
            data = torch.load("dataset/DBLP/dblp_updated.pt")
        else:
            data = torch.load("dataset/DBLP/dblp_updated_w_emb.pt")

        text = data.text
        num_classes = len(data.label)
        del data.text  # remove text from data
        data.num_nodes = len(data.y)

        return cls.split(data), text, num_classes

    @staticmethod
    def split(data):
        # split PyG dataset into train:val:test = 8:1:1
        num_nodes = data.num_nodes
        indices = torch.arange(num_nodes)

        train_indices = indices[:-2000]
        val_test_indices = indices[-2000:]
        val_indices = val_test_indices[:1000]
        test_indices = val_test_indices[1000:]

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[train_indices] = True
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask[val_indices] = True
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[test_indices] = True

        return data
