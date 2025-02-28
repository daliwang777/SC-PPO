import warnings
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, AutoModelForCausalLM
from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
    PPOTrainer,
    PPOConfig
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator

if __name__ == "__main__":
    accelerator=Accelerator()
    model_path = "/share/project/chenglongkai/datasets/qwen25_7B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    dataset = load_dataset('json', data_files='/share/project/daliwang/daliwang/GCRRL/new/align_train.json')
    print(dataset)
    dataset_text_field = "question"
    def modify_question(item):
        return '''你是一位智能运维诊断专家，能够根据历史经验使用以下检测工具：
1. 'GPUMemoryBw'（用于检测计算卡主存带宽）
2. 'NetworkP2pBw'（用于检测跨服务器点对点通信异常）
3. 'GPUComputeFP'（用于检测计算卡算力异常）
4. 'NetworkAllreduceBw'（用于检测跨服务器通信性能异常）
5. 'HostD2hBw'（用于检测数据传输速率下降（设备到主机））
6. 'HostH2dBw'（用于检测数据传输速率下降（主机到设备））
7. 'BusAllreduceBw'（用于检测单服务器内的通信效率异常）
8. 'NetworkTrainllm'（用于检测单机多卡训练性能异常）
9. 'NetworkTrainllama'（用于检测跨服务器单并行方案训练异常）
你会经历多轮情况，请结合提供的问题与已使用过的函数结果信息向用户提供下一步需要执行的函数。注意，在回答时需遵循以下规则：
- 使用<>括起来的函数名形式，例如<function_name>。
- 仅输出一个函数名，不包含额外文本。
- 每次只输出一个函数名。
- 你只需要返回符合格式的函数即可
用户的日志如下，你需要返回的函数是''' + item

# 更新 dataset 中的 question 字段
    dataset["train"] = dataset["train"].map(
        lambda example: {dataset_text_field: modify_question(example[dataset_text_field])}
    )
    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}
        
        print(dataset.column_names)
        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=1
        )
    
    tokenized_datasets = prepare_dataset(dataset["train"], tokenizer)

    # 加载基础Reward Model并准备用于K位训练
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "/share/project/daliwang/daliwang/RL/loraqwen/reward",
        num_labels=1,
    )
    # value_model = AutoModelForSequenceClassification.from_pretrained(
    #     model_path,
    #     num_labels=1,
    # )

    # 定义LoRA配置
    # lora_config = LoraConfig(
    #     r=8,
    #     target_modules=[
    #         "q_proj", "v_proj", "k_proj", "o_proj",
    #         "gate_proj", "down_proj", "up_proj"
    #     ],
    #     task_type=TaskType.SEQ_CLS,
    #     lora_alpha=16,
    #     lora_dropout=0.05
    # )

    # # 准备基础模型用于K位训练，并应用LoRA配置
    # reward_model = prepare_model_for_kbit_training(reward_model)
    # reward_model = get_peft_model(reward_model, lora_config)

    # value_model = prepare_model_for_kbit_training(value_model)
    # value_model = get_peft_model(value_model, lora_config)

    # 加载LoRA适配器
    # adapter_path = '/share/project/daliwang/daliwang/RL/qwen7-reward'
    # reward_model.load_adapter(adapter_path, adapter_name="default")
    # value_model.load_adapter(adapter_path, adapter_name="default")

    # 加载基础Policy Model
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,  
    )

    # 定义LoRA配置
    policy_lora_config =  LoraConfig(
            r=64,  # 增大秩,qwq是64,qwen是256
            lora_alpha=128,  # 增大缩放因子
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],  # 增加目标模块
            lora_dropout=0.0,  # 减少 Dropout
            bias="all",  # 添加偏置项
            task_type="CAUSAL_LM",
        )
    # 准备基础模型用于K位训练，并应用LoRA配置
    policy = prepare_model_for_kbit_training(policy)
    policy = get_peft_model(policy, policy_lora_config)
    #policy,dataset=accelerator.prepare(policy,dataset)
    #policy,tokenized_datasets = accelerator.prepare(policy, tokenized_datasets)
    training_Args = PPOConfig(
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        warmup_ratio=0.05,
        group_by_length=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        total_episodes=10000,
        output_dir="/share/project/daliwang/daliwang/RL/loraqwen/policy",
    )

    trainer = PPOTrainer(
        args=training_Args,
        value_model=reward_model,
        model=policy,
        reward_model=reward_model,
        ref_policy=None,
        train_dataset=tokenized_datasets,  # 确保使用'train' split
        eval_dataset=tokenized_datasets ,
        processing_class=tokenizer,
        peft_config=policy_lora_config
    )

    trainer.train()
    trainer.save_model(training_Args.output_dir)



