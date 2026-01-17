import re
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HybridCache


class CLM:
    def __init__(self, model_id: str = "google/gemma-2-2b", device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float16,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float16,
        )

        self.device = self.model.device

    def generate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=20)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def sample_next_tokens(self, text: str, completion: str, verbose: bool = False) -> List[float]:
        """
        Param:
        - text: input text. e.g., "Paris is the"
        - completion: completion text. e.g., "capital of France"

        Return:
        - probs: list of probabilities for each token in the completion text. e.g., [0.56, 0.98, 0.97]
        """

        inputs = self.tokenizer(text.rstrip(), return_tensors="pt").to(self.device)

        completion = " " + completion.strip()
        completion_ids = self.tokenizer(completion, add_special_tokens=False, return_tensors="pt")["input_ids"]
        completion_ids = completion_ids.view(-1)

        probs = []
        kv_cache = HybridCache(
            config=self.model.config,
            max_batch_size=1,
            max_cache_len=inputs.input_ids.shape[1] + completion_ids.shape[0],
            device=self.device,
            dtype=torch.float16,
        )
        next_token_id = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        cache_position = torch.arange(attention_mask.shape[1]).to(self.device)

        for step, completion_id in enumerate(completion_ids):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_token_id,
                    attention_mask=attention_mask,
                    past_key_values=kv_cache,
                    cache_position=cache_position,
                    use_cache=True,
                )
                kv_cache = outputs.past_key_values
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=-1)
                cache_position = cache_position[-1:] + 1
                next_token_probs = outputs.logits[:, -1, :].softmax(dim=-1)

            if verbose:
                # == show top-5 candidates ==
                topK, topK_ids = next_token_probs.topk(k=5, dim=-1)
                print(f"topK={topK}")
                print(f"topK_ids={topK_ids}")

                for score, token_id in zip(topK[0], topK_ids[0]):
                    next_token = self.tokenizer.decode(token_id)
                    print(f"\t{next_token}-({score*100:.2f}%)")
                # == show top-5 candidates ==

            probs.append(next_token_probs[0][completion_id].item())
            next_token_id = torch.tensor([[completion_id]]).to(self.device)

        return probs

    def fill_mask(
        self,
        text: str,
        labels: List[str],
        masker: str = "<mask>",
        verbose: bool = False,
    ) -> List[float]:
        """
        Param:
        - text: partially masked text. e.g., "Paris is the <mask> of <mask>."
        - labels: list of labels to fill in the blanks. e.g., ["capital", "France"]
        - masker: the masker token. e.g., "<mask>"

        Return:
        - label_probs: list of probabilities for each label. e.g., [0.99, 0.98]
        """
        # Check if the number of masks matches the number of labels
        mask_count = text.count(masker)
        if mask_count != len(labels):
            raise ValueError(f"Number of masks ({mask_count}) does not match number of labels ({len(labels)})")

        label_ids = []
        label_subwords = []
        num_subwords = []
        # reformat the masked text
        for label in labels:
            label_id = self.tokenizer.encode(" " + label, add_special_tokens=False)
            label_ids.extend(label_id)
            label_subwords.extend([self.tokenizer.decode(id) for id in label_id])
            num_subwords.append(len(label_id))

        # Tokenize the input text
        input_ids = self.tokenizer.encode(text)

        # Find the positions of the mask tokens
        masker_id = self.tokenizer.encode(masker, add_special_tokens=False)[0]
        mask_positions = [i for i, id in enumerate(input_ids) if id == masker_id]

        # insert multi-masker into the input_ids
        new_input_ids = []
        for input_id in input_ids:
            if input_id == masker_id:
                if len(new_input_ids) > 0:
                    new_input_ids.pop()  # remove the whitespace before the masker token
                new_input_ids.extend(
                    [masker_id] * num_subwords.pop(0)
                )  # replace single mask token with subword number of masks
            else:
                new_input_ids.append(input_id)
        input_ids = new_input_ids

        # update mask_positions based on new input_ids
        mask_positions = [i for i, id in enumerate(input_ids) if self.tokenizer.decode([id]) == masker]
        # convert input_ids to tensor
        input_ids = torch.tensor(input_ids).to(self.device).unsqueeze(0)
        # create attention mask
        attention_mask = torch.ones_like(input_ids)

        # Initialize HybridCache
        kv_cache = HybridCache(
            config=self.model.config,
            max_batch_size=1,
            max_cache_len=input_ids.shape[1],
            device=self.device,
            dtype=torch.float16,
        )

        label_probs = []
        cache_position = torch.arange(input_ids.shape[1]).to(self.device)

        last_mask_pos = 0

        for mask_pos in mask_positions:
            # Process up to the current mask
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids[:, last_mask_pos:mask_pos],
                    attention_mask=attention_mask[:, :mask_pos],
                    past_key_values=kv_cache,
                    cache_position=cache_position[last_mask_pos:mask_pos],
                    use_cache=True,
                )
                kv_cache = outputs.past_key_values
                logits = outputs.logits[0, -1]

            # Calculate probability for the current label
            label_id = label_ids.pop(0)
            label_prob = torch.softmax(logits, dim=-1)[label_id].item()
            label_probs.append((label_subwords.pop(0), label_id, label_prob))

            if verbose:
                # == show top-5 candidates ==
                topK, topK_ids = torch.softmax(logits, dim=-1).topk(k=5, dim=-1)
                print(f"Expected: {label_probs[-1][0]}, Top-5 Candidates:")
                for order, (score, token_id) in enumerate(zip(topK, topK_ids)):
                    next_token = self.tokenizer.decode(token_id)
                    print(f"\t{order}. {next_token}-({score*100:.2f}%) {'*' if token_id == label_id else ''}")
                # == show top-5 candidates ==

            # Update input_ids with the predicted label
            input_ids[0, mask_pos] = label_id
            # update last_mask_pos
            last_mask_pos = mask_pos

        return label_probs
