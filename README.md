# 1. Overview
This repository implements the SC-PPO (Specific Context-Proximal Policy Optimization) algorithm for reinforcement learning experiments. It contains training modules, testing scenarios, dataset management and comparative implementations of PPO/RAG methods.

# 2. Repository Structure
```
SC-PPO/ 
        ├── train/ # SC-PPO training scripts \
        ├── test/ # Evaluation scripts for different test scenarios\
        ├── dataset/ # Preprocessed training/validation datasets \
        ├── RAG/ # Retrieval-Augmented Generation implementation \
        └── PPO/ # Baseline PPO algorithm implementation\
```
# 3. Experimental Results 
The comparative performance of different methods across models is shown below:
 
| Method       | Model          | Mean  | Variance |
|--------------|----------------|-------|----------|
| **SC-PPO**   | **Qwen2.5-7B** | 0.62  | 0.03     |
| **SC-PPO**   | **QwQ-32B**    | 0.56  | 0.03     |
| Baseline     | Qwen2.5-7B     | 0.23  | 0.14     |
| Baseline     | QwQ-32B        | 0.26  | 0.13     |
| RAG(train)   | Qwen2.5-7B     | 0.46  | 0.70     |
| RAG(train)   | QwQ-32B        | 0.46  | 0.09     |
| RAG(all)     | Qwen2.5-7B     | 0.43  | 0.09     |
| RAG(all)     | QwQ-32B        | 0.47  | 0.06     |
| PPO          | Qwen2.5-7B     | 0.01  | 0.10     |
| PPO          | QwQ-32B        | -     | -        |
# 4. Quick Start
```
### Train
python train/sc_ppo.py  \
 
### Evaluation 
bash 
python test/[xxx]test.py  \
```
 
# 5 . License 
This project is open-sourced under [Apache License 2.0](LICENSE). Commercial use requires written permission.
 
