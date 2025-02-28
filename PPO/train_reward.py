import warnings
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,

)  
if __name__ == "__main__":
    
    model_path="/share/project/chenglongkai/datasets/llama3.2-3b"
    model_path2="/share/project/chenglongkai/datasets/qwen25_7B"
    model_path3="/share/project/chenglongkai/datasets/qwen0.5B"
    tokenizer=AutoTokenizer.from_pretrained(model_path3)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model=AutoModelForSequenceClassification.from_pretrained(model_path3,num_labels=1,device_map="auto")
    model.config.pad_token_id = tokenizer.pad_token_id
    dataset=load_dataset("json",data_files="/share/project/daliwang/daliwang/GCRRL/new/align_train.json")
    # print(dataset)
    tokenizer.pad_token = tokenizer.eos_token
    def preprocess_function(examples):
        chosen_inputs = tokenizer(
            [f"{p} {c}" for p, c in zip(examples['question'], examples['chosen'])],
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        rejected_inputs = tokenizer(
            [f"{p} {r}" for p, r in zip(examples['question'], examples['rejected'])],
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
        
        # 返回两个字典，分别对应 chosen 和 rejected 输入
        return {
            "input_ids_chosen": chosen_inputs["input_ids"],
            "attention_mask_chosen": chosen_inputs["attention_mask"],
            "input_ids_rejected": rejected_inputs["input_ids"],
            "attention_mask_rejected": rejected_inputs["attention_mask"]
        }

# 对数据集进行预处理
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    train_config=RewardConfig(
        num_train_epochs=10,
        gradient_checkpointing=True,
        learning_rate=1.0e-4,
        per_device_train_batch_size=64,
        logging_steps=25,
        output_dir="/share/project/daliwang/daliwang/RL/loraqwen/reward",
        max_length=2048,
        gradient_accumulation_steps=2,
        fp16=True
    )
    trainer=RewardTrainer(model=model,
        train_dataset=tokenized_datasets["train"],
        args=train_config,
        processing_class=tokenizer
        )
    trainer.train()
    trainer.save_model(train_config.output_dir)
    
    