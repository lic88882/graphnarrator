import regex as re
from datasets import load_dataset
from transformers import TextStreamer, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import argparse


class LLMFinetuner:
    def __init__(self, model_name, max_seq_length=8 * 1024, dtype=None, load_in_4bit=True):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit

        # lazy initialization
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if self.model is None:
            print(f"Loading model from {self.model_name}")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=self.dtype,
                load_in_4bit=self.load_in_4bit,
            )

    def setup_peft(self):
        if self.model is None:
            raise ValueError("Model must be loaded before setting up PEFT")

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

    def prepare_data(self, data_file):
        training_dataset = load_dataset("json", data_files=data_file, split="train")

        if self.tokenizer is None:
            self.load_model()

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="llama-3",
            mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
        )

        def formatting_prompts_func(examples):
            messages = examples["messages"]
            texts = [
                self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
                for message in messages
            ]
            return {"text": texts}

        training_dataset = training_dataset.map(formatting_prompts_func, batched=True)
        return training_dataset

    def train(self, training_dataset):
        self.load_model()
        self.setup_peft()

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=training_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=4,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                num_train_epochs=3,
                # max_steps=60,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="distillation/training",
            ),
        )
        return trainer.train()

    def save_model(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def inference(self, messages, max_new_tokens=4 * 1024, stream=False):
        self.load_model()
        FastLanguageModel.for_inference(self.model)

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="llama-3",
            mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        )

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        attention_mask = inputs.ne(self.tokenizer.pad_token_id).long().to("cuda")

        if stream:
            text_streamer = TextStreamer(self.tokenizer)
            _ = self.model.generate(
                input_ids=inputs,
                streamer=text_streamer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        else:
            outputs = self.model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
            generation = self.tokenizer.decode(outputs[0])

            pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>'
            match = re.search(pattern, generation, re.DOTALL)

            if match and match.group(1):
                response = match.group(1)
                return response

            print("Assistant response not found. Return generation instead.")
            return generation


def remove_importance_score(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        data = file.read()

    pattern = r'\(\d+\.\d\d\)'
    processed_data = re.sub(pattern, '', data)

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(processed_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true", help="Run data preprocessing")
    parser.add_argument("--finetune", action="store_true", help="Run model fine-tuning")
    parser.add_argument("--inference", action="store_true", help="Run model inference")
    parser.add_argument("--model_name", type=str, default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", help="Base model name")
    parser.add_argument("--max_seq_length", type=int, default=40_000, help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load model in 4-bit precision")
    parser.add_argument("--input_file", type=str, default="processed_finetuned_data_dblp.jsonl", help="Input data file")
    parser.add_argument("--output_dir", type=str, default="llama-3.1-8b-graph-expl-finetuned-dblp", help="Output directory for fine-tuned model")
    args = parser.parse_args()

    # Data preprocessing
    if args.preprocess:
        remove_importance_score("distillation_dataset.jsonl", "processed_finetuned_data.jsonl")

    # Fine-tuning
    if args.finetune:
        print("start load model")
        finetuner = LLMFinetuner(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
        )
        print("load model finished")
        training_dataset = finetuner.prepare_data(args.input_file)
        print("prepare dataset finished")
        finetuner.train(training_dataset)
        finetuner.save_model(args.output_dir)

    # Inference
    if args.inference:
        finetuner = LLMFinetuner(
            model_name=args.output_dir,  # saved finetuned model
            max_seq_length=8 * 1024,
            load_in_4bit=args.load_in_4bit,
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        print(finetuner.inference(messages, stream=False))


if __name__ == "__main__":
    main()
