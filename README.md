# GraphNarrator

Official implementation of the ACL 2025 Main Conference paper "GraphNarrator: Textual Explanations for Graph Neural Networks". For more details, refer to the paper: [GraphNarrator: Textual Explanations for Graph Neural Networks](https://arxiv.org/pdf/2410.15268).

GraphNarrator is a powerful framework for generating and improving natural language explanations for graph neural networks. It combines the strengths of language models and graph neural networks to provide interpretable explanations for graph-based predictions.

## Project Structure

- `main.py`: Main entry point and pipeline orchestration
- `dataset.py`: Dataset handling and preprocessing
- `filter.py`: Implementation of rejection sampling
- `finetuning.py`: Model fine-tuning functionality
- `free_text_explanation.py`: Natural language explanation generation
- `preprocess.py`: Data preprocessing utilities
- `evaluate.py`: Evaluation metrics and scoring
- `utils.py`: Utility functions
- `models/`: Model implementations


## Installation

1. Clone the repository:
```bash
git clone https://github.com/pb0316/GraphNarrator.git
cd GraphNarrator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the datasets at https://drive.google.com/file/d/1VqAmMueEfbkIOHsYBJj_kq2iWeHiMDhY/view?usp=sharing

5. (Optional): The finetuned model weights are at: (book history) https://drive.google.com/drive/folders/1ujtUSkXzoXcUgyFYv_JUyklpitN0e9Ot?usp=sharing, (dblp) https://drive.google.com/drive/folders/1Af65pOgVypCw2MuwJXG4JvTw_X0kkvxu?usp=sharing

## Usage


### Automatic Pipeline with Custom Configuration
```bash
python main.py --mode auto --devices cuda:0 cuda:1 --num_processes 16 --num_iterations 3
```

### Manual Step-by-Step Pipeline with Custom Splits
```bash
python main.py --mode manual --step 0 1 2 3 --devices cuda:0 cuda:1 --train_split 0.8 --val_split 0.1 --test_split 0.1
```

The manual pipeline consists of the following steps:
- `step 0`: Preprocess the dataset and prepare it for explanation generation
- `step 1`: Generate natural language explanations for the graph nodes
- `step 2`: Perform rejection sampling to filter and improve the generated explanations
- `step 3`: Fine-tune the model using the filtered explanations

### Main Pipeline Arguments
- `--seed`: Random seed for reproducibility
- `--lm_model`: Language model to use (default: "bert-base-uncased")
- `--gnn_model`: Graph neural network model (default: "SAGE"), options: SAGE, GAT, GCN
- `--dataset`: Dataset to use (default: "cora"), options: cora, citeseer, pubmed, dblp
- `--devices`: GPU devices to use
- `--mode`: Pipeline mode (auto, manual, sali, brev, fait, etc.)
- `--num_processes`: Number of processes for explanation generation (default: 32)
- `--num_iterations`: Number of iterations for auto/ablation modes (default: 5)
- `--rand_size`: Number of random nodes to sample (default: 60)
- `--sample_steps`: Number of sample steps for random nodes (default: 3)
- `--train_split`: Training split ratio for random nodes (default: 0.7)
- `--val_split`: Validation split ratio for random nodes (default: 0.1)
- `--test_split`: Test split ratio for random nodes (default: 0.2)


### Distillation Pipeline
```bash
# Preprocess data
python distill.py --preprocess

# Fine-tune model
python distill.py --finetune --model_name "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" --max_seq_length 40000

# Run inference
python distill.py --inference --output_dir "path/to/finetuned/model"
```

### Distillation Arguments
- `--preprocess`: Run data preprocessing
- `--finetune`: Run model fine-tuning
- `--inference`: Run model inference
- `--model_name`: Base model name (default: "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
- `--max_seq_length`: Maximum sequence length (default: 40000)
- `--load_in_4bit`: Load model in 4-bit precision (default: True)
- `--input_file`: Input data file (default: "processed_finetuned_data_dblp.jsonl")
- `--output_dir`: Output directory for fine-tuned model (default: "llama-3.1-8b-graph-expl-finetuned-dblp")

### Ablation Studies
```bash
# Saliency ablation
python main.py --mode abl_sali

# Brevity ablation
python main.py --mode abl_brev

# Faithfulness ablation
python main.py --mode abl_fait
```

## Citation

If you find this work helpful, please consider cite our paper:

```bibtex
@article{pan2024tagexplainer,
  title={TAGExplainer: Narrating Graph Explanations for Text-Attributed Graph Learning Models},
  author={Pan, Bo and Xiong, Zhen and Wu, Guanchen and Zhang, Zheng and Zhang, Yifei and Zhao, Liang},
  journal={arXiv preprint arXiv:2410.15268},
  year={2024}
}
```# graphnarrator
