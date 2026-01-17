import torch
from torch.nn import Module
from torch_geometric.explain import CaptumExplainer, Explainer
from transformers import AutoTokenizer
from transformers import BertModel as BertModelOfficial

from models.GNNs.sage import SAGE
from models.LMs.BERT_LRP.code.model.BERT import (
    BertConfig,
    MyBertModel,
    get_activation,
    get_activation_multi,
    get_inputivation,
)


class LM_GNN_Joint_Model(Module):
    def __init__(self, out_dim, lm_model_name="bert-base-uncased", gnn_model_name="SAGE"):
        super().__init__()

        config = BertConfig.from_json_file("models/LMs/BERT_LRP/models/BERT-Google/bert_config.json")
        self.bert = MyBertModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model_name)

        # load pre-trained weights for Bert
        Hug_bert_model = BertModelOfficial.from_pretrained(lm_model_name)
        target_state_dict = self.bert.state_dict()  # get the target model's state dict
        for name, param in Hug_bert_model.state_dict().items():
            if name in target_state_dict:
                if target_state_dict[name].shape == param.shape:
                    target_state_dict[name].copy_(param)
                else:
                    print(f"Skipping {name} due to shape mismatch.")
        self.bert.load_state_dict(target_state_dict, strict=False)

        self.gnn = SAGE(
            in_channels=768,
            hidden_channels=768,
            out_channels=out_dim,
            num_layers=2,
            dropout=0.0,
        )
        self.gradients = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.register_hooks()

    def forward(self, input_ids, attention_mask, edge_index):
        text_emb = self.bert(input_ids, attention_mask=attention_mask)[1]
        x = self.gnn(text_emb, edge_index)
        return x

    def forward_gnn(self, x, edge_index):
        x = self.gnn(x, edge_index)
        return x

    def register_hooks(self):
        def forward_hook(module, input, output):
            # self.activations.append(output)
            output.register_hook(self.save_gradient)

        self.gnn.convs[0].register_forward_hook(forward_hook)

    def save_gradient(self, grad):
        self.gradients = grad

    def explain(self, input_ids, attention_mask, edge_index):
        # Get Node Importance
        explainer = Explainer(
            self.gnn,  # It is assumed that model outputs a single tensor.
            # algorithm=CaptumExplainer("IntegratedGradients"),
            algorithm=CaptumExplainer("Saliency"),
            # algorithm=CaptumExplainer("InputXGradient"),
            # algorithm=GNNExplainer(),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="probs",  # Model returns probabilities.
            ),
        )

        # Get target class label
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, edge_index)
            predicted_label = torch.argmax(logits[0].view(-1))

        # print(f"---y_target: {y_target}: {['Case_Based','Genetic_Algorithms', 'Neural_Networks', 'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'][y_target]}---")

        with torch.no_grad():
            text_emb = self.bert(input_ids, attention_mask=attention_mask)[1]
        gnn_explanation = explainer(text_emb, edge_index, index=0)

        node_attribution_scores = gnn_explanation.node_mask.float()

        # Do LRP for Bert
        R = torch.zeros(input_ids.shape[0], 512, 768)
        batch_size = 1
        for i in range(0, input_ids.shape[0], batch_size):
            text_emb = self.bert(
                input_ids[i : i + batch_size],
                attention_mask=attention_mask[i : i + batch_size],
            )
            R_ = self.bert.backward_lrp(node_attribution_scores[i : i + batch_size])
            R[i : i + batch_size] = R_
            self.bert.zero_grad()

        return R, node_attribution_scores.sum(-1), predicted_label

    def predict(self, texts, edge_index):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            logits = self.forward(
                torch.tensor(inputs["input_ids"]).to(self.device),
                torch.tensor(inputs["attention_mask"]).to(self.device),
                torch.tensor(edge_index).to(self.device),
            )
            prediction = torch.argmax(logits[0].view(-1))
        return prediction.item()


def init_hooks_lrp(model):
    """
    Initialize all the hooks required for full lrp for BERT model.
    """
    # in order to backout all the lrp through layers
    # you need to register hooks here.

    model.bert.pooler.dense.register_forward_hook(get_inputivation("model.bert.pooler.dense"))
    model.bert.pooler.dense.register_forward_hook(get_activation("model.bert.pooler.dense"))
    model.bert.pooler.register_forward_hook(get_inputivation("model.bert.pooler"))
    model.bert.pooler.register_forward_hook(get_activation("model.bert.pooler"))

    model.bert.embeddings.word_embeddings.register_forward_hook(get_activation("model.bert.embeddings.word_embeddings"))
    model.bert.embeddings.register_forward_hook(get_activation("model.bert.embeddings"))

    layer_module_index = 0
    for module_layer in model.bert.encoder.layer:

        ## Encoder Output Layer
        layer_name_output_layernorm = "model.bert.encoder." + str(layer_module_index) + ".output.LayerNorm"
        module_layer.output.LayerNorm.register_forward_hook(get_inputivation(layer_name_output_layernorm))

        layer_name_dense = "model.bert.encoder." + str(layer_module_index) + ".output.dense"
        module_layer.output.dense.register_forward_hook(get_inputivation(layer_name_dense))
        module_layer.output.dense.register_forward_hook(get_activation(layer_name_dense))

        layer_name_output = "model.bert.encoder." + str(layer_module_index) + ".output"
        module_layer.output.register_forward_hook(get_inputivation(layer_name_output))
        module_layer.output.register_forward_hook(get_activation(layer_name_output))

        ## Encoder Intermediate Layer
        layer_name_inter = "model.bert.encoder." + str(layer_module_index) + ".intermediate.dense"
        module_layer.intermediate.dense.register_forward_hook(get_inputivation(layer_name_inter))
        module_layer.intermediate.dense.register_forward_hook(get_activation(layer_name_inter))

        layer_name_attn_layernorm = "model.bert.encoder." + str(layer_module_index) + ".attention.output.LayerNorm"
        module_layer.attention.output.LayerNorm.register_forward_hook(get_inputivation(layer_name_attn_layernorm))

        layer_name_attn = "model.bert.encoder." + str(layer_module_index) + ".attention.output.dense"
        module_layer.attention.output.dense.register_forward_hook(get_inputivation(layer_name_attn))
        module_layer.attention.output.dense.register_forward_hook(get_activation(layer_name_attn))

        layer_name_attn_output = "model.bert.encoder." + str(layer_module_index) + ".attention.output"
        module_layer.attention.output.register_forward_hook(get_inputivation(layer_name_attn_output))
        module_layer.attention.output.register_forward_hook(get_activation(layer_name_attn_output))

        layer_name_self = "model.bert.encoder." + str(layer_module_index) + ".attention.self"
        module_layer.attention.self.register_forward_hook(get_inputivation(layer_name_self))
        module_layer.attention.self.register_forward_hook(get_activation_multi(layer_name_self))

        layer_name_value = "model.bert.encoder." + str(layer_module_index) + ".attention.self.value"
        module_layer.attention.self.value.register_forward_hook(get_inputivation(layer_name_value))
        module_layer.attention.self.value.register_forward_hook(get_activation(layer_name_value))

        layer_name_query = "model.bert.encoder." + str(layer_module_index) + ".attention.self.query"
        module_layer.attention.self.query.register_forward_hook(get_inputivation(layer_name_query))
        module_layer.attention.self.query.register_forward_hook(get_activation(layer_name_query))

        layer_name_key = "model.bert.encoder." + str(layer_module_index) + ".attention.self.key"
        module_layer.attention.self.key.register_forward_hook(get_inputivation(layer_name_key))
        module_layer.attention.self.key.register_forward_hook(get_activation(layer_name_key))

        layer_module_index += 1
