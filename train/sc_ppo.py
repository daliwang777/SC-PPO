import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from matplotlib import pyplot as plt
from peft import LoraConfig, get_peft_model,PeftModel
import random
from new_pool import ExperiencePool
import json
import wandb
import wandb
import ast
wandb.init(
    # set the wandb project where this run will be logged
    project="RLGNN",
    name='qwen2.5-7b-tr-flag1-2d',
    # track hyperparameters and run metadata
    config={
    "architecture": "GNN",
    "dataset": "CIFAR-100",
    }
)
#torch.manual_seed(114514)
#torch.cuda.manual_seed(3407)
import re
class ResidualValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        x = F.relu(self.linear1(hidden_states))
        residual = x
        x = F.relu(self.linear2(x))
        x = x + residual  # 残差连接
        x = self.dropout(x)
        value = self.linear3(x)
        return value
judge_prompt="""
如果上一步使用的函数返回的答案是错误的，请从我提供的终止函数选择合适的输出（即<StopNetwork>(停止网络探测), <StopVendor>(报告厂商), <StopJob>(暂停业务),<StopServer>(终止服务器检测)）。如果是正确的，请你选择下一步将要执行的诊断函数即("<GPUMemoryBw>", "<NetworkP2pBw>"
, "<GPUComputeFP>", "<NetworkAllreduceBw>", "<HostD2hBw>", "<HostH2dBw>","<BusAllreduceBw>","<NetworkTrainllm>","<NetworkTrainllama>")进行下一步的探测或者你认为当前服务器没问题则输出"<StopSuccess>"函数。请注意：
- 输出格式必须为<>包裹起来的函数名称。
- 你只能从已知的函数中选择一个输出，如果你确信可以通过压测的方式探测则可以重复使用函数
无需提供分析，但每次回答必须包含一个符合上述格式的函数名称。你接下来的使用的函数是:
"""
class PPOTrainer:
    def __init__(self, model_path, data, gamma=0.8, clip=0.2, entropy=0.1):
        ## 预训练模型
        self.pool=[]
        self.pool_build()
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 加载预训练模型（使用 device_map 自动分配设备）
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",  # 自动分配设备
        )
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 添加 LoRA 模块
        lora_config = LoraConfig(
            r=64,  # 增大秩,qwq是64,qwen是256
            lora_alpha=128,  # 增大缩放因子
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],  # 增加目标模块
            lora_dropout=0.0,  # 减少 Dropout
            bias="all",  # 添加偏置项
            task_type="CAUSAL_LM",
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        # lora_path = "/share/project/daliwang/daliwang/GCRRL/new/lora_qwq_tr3"
        # self.base_model = PeftModel.from_pretrained(self.base_model, lora_path)
        
        # 添加 value 头（Critic 网络）
        self.value_head = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 1),
        ).to(self.base_model.device)
        #self.value_head = ResidualValueHead(self.base_model.config.vocab_size).to(self.base_model.device)
        # 定义两个优化器
        #self.value_optimizer = optim.Adam(self.value_head.parameters(), lr=1e-8)  # 优化 value_head
        #self.actor_optimizer = optim.Adam(self.base_model.parameters(), lr=1e-8)  # 优化 actor_head
        all_params = list(self.base_model.parameters()) + list(self.value_head.parameters())
        
        # 定义一个优化器
        self.optimizer = optim.Adam(all_params, lr=1e-6)
        # PPO 超参数
        self.gamma = gamma  # 折扣因子
        self.clip_epsilon = clip  # PPO 剪裁范围
        self.entropy_coef = entropy  # 熵正则化系数
        self.lam=0.8
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 使用 eos_token 作为 pad_token
        self.data_process(data)
    def data_process(self, data):
        # 处理数据，数据列分别是question、answer、reward
        if isinstance(data, list):
            self.data = Dataset.from_list(data)
        else:
            raise ValueError("data must be a list")
        return self.data
    def pool_build(self):
        with open('/share/project/daliwang/daliwang/GCRRL/new/train_2.json', 'r',encoding='utf-8') as f:
            data = json.load(f)
        for elem in data:
            elem["best_traj"]=ast.literal_eval(elem["best_traj"])
            self.pool.append(elem)
        return
        
    def find_action(self,action):
        #print(f"action:{action}")
        json_pattern = re.compile(r'<.*?>', re.DOTALL)  # 匹配 [...] 之间的内容
        match = re.findall(json_pattern, action)
        action_step=''
        if match==None or len(match)==0:
            action_step=''
        else:
            action_step=match[0].replace('<','').replace('>','')
            #print(f"action_step:{action_step}")
        return action_step
    def save_lora_weights(self, save_dir):
        """
        保存 LoRA 权重到指定目录
        """
        self.base_model.save_pretrained(save_dir)
        print(f"LoRA 权重已保存到 {save_dir}")
    def get_reward(self,generated_state,generate_action,max_step_length=7,index=0,flag=0):
        """
        改进后的奖励函数,注意这一步是在先生成一次generate之后传进来的，因此我们需要先judge生成答案。
        """
        experience_pool=ExperiencePool()
        step=1
        reward=0
        temp_generated_state=torch.clone(generated_state)
        generate_ans=self.tokenizer.decode(temp_generated_state[0],skip_special_tokens=True)
        generate_action=self.tokenizer.decode(generate_action[0],skip_special_tokens=True)
        #print(f"generate_ans:{generate_ans}")
        real_step=[]
        infer_step=1
        action_step=self.find_action(generate_action)
        #print(generate_ans)
        while step<= max_step_length:
            if action_step =='':
                reward-=0
                if type(generate_ans)!=str:
                    generate_ans=self.tokenizer.decode(generate_ans[0],skip_special_tokens=True)
                if type(generate_ans)==str:
                    encode_ans=self.tokenizer(generate_ans,return_tensors='pt').to(self.base_model.device)
                #with self.base_model.disable_adapter():
                self.base_model.eval()
                generate_ans = self.base_model.generate(input_ids=encode_ans["input_ids"],attention_mask=encode_ans["attention_mask"])
                new_tokens = generate_ans[:, encode_ans["input_ids"].shape[1]:]
                infer_step+=new_tokens.shape[1]
                new_tokens = self.tokenizer.decode(new_tokens[0],skip_special_tokens=True)
                action_step=self.find_action(new_tokens)
                if "<StopNetwork>" in new_tokens:
                    real_step.append('StopNetwork')
                    break
                elif "<StopVendor>" in new_tokens:
                    real_step.append('StopVendor')
                    break
                elif "<StopServer>" in new_tokens:
                    real_step.append('StopServer')
                    break
                elif "<StopSuccess>" in new_tokens:
                    real_step.append('StopSuccess')
                    break
                elif '<StopJob>' in new_tokens:
                    real_step.append('StopJob')
                    break
                step+=1
            else:
                if action_step=='StopNetwork' or action_step=='StopVendor' or action_step=='StopServer' or action_step=='StopSuccess' or action_step=='StopJob':
                    real_step.append(action_step)
                    break
                real_step.append(action_step)
                result_judge,num=self.judge()
                feedback=''
                if num<0.7:
                    feedback='True'
                else:
                    feedback='False'
                real_step.append(feedback)
                if type(generate_ans)!=str:
                    generate_ans=self.tokenizer.decode(generate_ans[0],skip_special_tokens=True)
                generate_ans=generate_ans+result_judge+judge_prompt
                if type(generate_ans)==str:
                    encode_ans=self.tokenizer(generate_ans,return_tensors='pt').to(self.base_model.device)
                #with self.base_model.disable_adapter():
                self.base_model.eval()
                generate_ans = self.base_model.generate(input_ids=encode_ans["input_ids"],attention_mask=encode_ans["attention_mask"])
                new_tokens = generate_ans[:, encode_ans["input_ids"].shape[1]:]
                infer_step+=new_tokens.shape[1]
                new_tokens = self.tokenizer.decode(new_tokens[0],skip_special_tokens=True)
                action_step=self.find_action(new_tokens)
                    #print(f"new_tokens:{new_tokens}")
                del result_judge,encode_ans
                # if "<StopNetwork>" in new_tokens:
                #     real_step.append('StopNetwork')
                #     break
                # elif "<StopVendor>" in new_tokens:
                #     real_step.append('StopVendor')
                #     break
                # elif "<StopServer>" in new_tokens:
                #     real_step.append('StopServer')
                #     break
                # elif "<StopSuccess>" in new_tokens:
                #     real_step.append('StopSuccess')
                #     break
                # elif '<StopJob>' in new_tokens:
                #     real_step.append('StopJob')
                #     break
                # else:
                step+=1
                
        print(f"real_step:{real_step}")
        if flag==0:
            reward=experience_pool.get_node_reward(real_step)
        else:
            reward1=experience_pool.get_node_reward(real_step)
            if reward1<=0:
                return reward1
            reward=experience_pool.reward_caculate_strength(real_step)
        return reward
            
            
        #print(f"reward:{reward}")
                
        #     result_judge,num=self.judge()
        #     generate_ans=generate_ans+result_judge+judge_prompt
        #     print(f"ans:{generate_ans}")
        #     print("---------------------------------------")
        #     encode_ans=self.tokenizer(generate_ans,return_tensors='pt').to(self.base_model.device)
        #     with self.base_model.disable_adapter():
        #         self.base_model.eval()
        #         output = self.base_model.generate(input_ids=encode_ans["input_ids"],attention_mask=encode_ans["attention_mask"])
        #         new_tokens = output[:, encode_ans["input_ids"].shape[1]:]
        #         new_tokens = self.tokenizer.decode(new_tokens[0],skip_special_tokens=True)
        #         print(f"new_tokens:{new_tokens}")
        #     del result_judge,encode_ans,output
        #     if "<ALARMED>" in new_tokens:
        #         if num<0.5:
        #             reward-=0.5
        #         else:
        #             reward=1/(step)
        #         break
        #     else:
        #         if num>0.5:
        #             reward+=0.5
        #         else:
        #             reward-=0.5
        #         step+=1
        # #print(f"reward:{reward}")
        # if step>max_step_length and reward ==0:
        #     reward-=0.2
        # print(f"reward:{reward}")
        

    def generate(self,input_token):
        #传入的tokenize分词后的input id
        #input_id=input_token["input_ids"]
        #attention_mask=input_token["attention_mask"]
        output = self.base_model.generate(
            input_ids=input_token,
            attention_mask=None,
            num_return_sequences=1,
            eos_token_id=None,
            output_hidden_states=True,
        )
        # print(len(input_token[0]))
        # print(type(output))
        #print(output)
        new_tokens = output[:, input_token.shape[1]:]
        #print(new_tokens)
        return output,new_tokens
    def forward(self, input_token):
        # 前向传播，获取 actor_logits 和 value,这里的input token其实是当前的state
        # input_ids = input_token["input_ids"]
        # attention_mask = input_token["attention_mask"]
        actor_logits = self.base_model(input_ids=input_token).logits
        state_=self.base_model(input_ids=input_token,output_hidden_states=True).hidden_states
        state_=state_[-1][:,-1,:]
        # print("state_:")
        # print(state_)
        # print("state_.shape:")
        # print(state_.shape)
        values = self.value_head(state_).squeeze(-1)
        next_logits=torch.clone(actor_logits[:,-1,:])
        del next_logits
        return actor_logits, values
    def compute_advantage(self,rewards, values): #**
    # 计算 GAE
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
        
        #advantages = torch.tensor(advantages, dtype=torch.float32).to(self.base_model.device)
        return advantages
        
        
        
    def compute_loss(self, logits,values,advantage, old_log_probs): #**
    # 计算策略损失
        next_token_logits = logits[:, -1, :]
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.argmax(next_token_probs, dim=-1)
        log_prob = torch.log(next_token_probs.gather(1, next_token.unsqueeze(-1)).squeeze(-1))
        
        ratio = torch.exp(log_prob - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 计算价值损失
        value_target=advantage+values
        value_loss = F.mse_loss(values, value_target)
        
        # 计算熵正则化损失
        entropy = -torch.sum(next_token_probs * torch.log(next_token_probs + 1e-10), dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # 总损失
        total_loss = policy_loss + 0.1*value_loss + 0.1*entropy_loss
        
        return total_loss, policy_loss, value_loss, entropy_loss
    def train(self, epochs=5,step=1,update_steps=3):
        self.question= self.data["question"]
        total_loss_list=[]
        policy_loss_list=[]
        value_loss_list=[]
        entropy_loss_list=[]
        all=0
        correct=0
        correct1=0
        correct2=0
        correct3=0
        sums=0
        for epoch in tqdm(range(epochs)):
            for i in range(len(self.question)):
                for _ in range(step):
                    question=self.question[i]
                    question_token=self.tokenizer(question,return_tensors='pt').to(self.base_model.device)
                    state=[]
                    action=[]
                    reward=[]
                    log_probs=[]
                    values=[]
                    old_log_probs=[]
                    state.append(question_token["input_ids"])
                    gen_state,gen_action=self.generate(question_token["input_ids"])
                    for j in range(1,gen_state.shape[1]-question_token["input_ids"].shape[1]):
                        state.append(gen_state[:,0:question_token["input_ids"].shape[1]+j])
                    for st in state:
                        logits,vl=self.forward(st)
                        next_token_logits = logits[:, -1, :]
                        next_token_probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.argmax(next_token_probs, dim=-1)
                        log_prob=torch.log(next_token_probs.gather(1, next_token.unsqueeze(-1)).squeeze(-1))
                        old_log_probs.append(log_prob.detach())
                        values.append(vl.detach())
                    tmp_reward=self.get_reward(gen_state,gen_action,index=i,flag=1)
                    all+=1
                    if tmp_reward>0.01:
                        correct+=1
                    if tmp_reward>0.2:
                        correct1+=1
                    if tmp_reward>0.4:
                        correct2+=1
                    if tmp_reward>0.7:
                        correct3+=1
                    ratio=0.2*float(correct/all)-0.1
                    for _ in range(len(values)-1):
                        reward.append(tmp_reward)
                    advantages=self.compute_advantage(reward,values)
                    self.base_model.train()
                    for m in range(update_steps):
                        self.optimizer.zero_grad()
                        all_total_loss=[]
                        all_policy_loss=[]
                        all_value_loss=[]
                        all_entropy_loss=[]
                        for k in range(1,len(state)):
                            logits,vl=self.forward(state[k-1])
                            old_log_prob=old_log_probs[k-1]  #**
                            total_loss,policy_loss,value_loss,entropy_loss=self.compute_loss(logits,values[k-1],advantages[k-1],old_log_prob)
                            all_total_loss.append(total_loss)
                            all_policy_loss.append(policy_loss)
                            all_value_loss.append(value_loss)
                            all_entropy_loss.append(entropy_loss)
                        total_loss=torch.stack(all_total_loss).mean()
                        policy_loss=torch.stack(all_policy_loss).mean()
                        value_loss=torch.stack(all_value_loss).mean()
                        entropy_loss=torch.stack(all_entropy_loss).mean()
                        log_interval = 5  # 每10步上传一次日志
                        if True:
                            try:
                                wandb.log({
                                "epoch": epoch,
                                "step": m,
                                "total_loss": total_loss.item(),
                                "policy_loss": policy_loss.item(),
                                "value_loss": value_loss.item(),
                                "entropy_loss": entropy_loss.item(),
                                "reward": tmp_reward,
                                "no exception": float(correct/all),
                                "rag level":float(correct1/all),
                                "ppo level":float(correct2/all),
                                "human level":float(correct3/all)
                                
                        })
                            except BrokenPipeError as e:
                                print(f" Wandb logging failed: {e}")
                        sums+=1
                        print(f"Epoch {epoch}, question {i},Step {m}+{sums}:Total Loss:{total_loss.item()},Policy Loss:{policy_loss.item()},Value Loss:{value_loss.item()},Entropy Loss:{entropy_loss.item()}")
                        total_loss_list.append(total_loss.item())
                        policy_loss_list.append(policy_loss.item())
                        value_loss_list.append(value_loss.item())
                        entropy_loss_list.append(entropy_loss.item())
                        total_loss.backward()
                        self.optimizer.step()
                        if sums%300==0:
                            self.save_lora_weights(f"/share/project/daliwang/daliwang/GCRRL/new/lora_qwen25_tr7B")
        self.plot_and_save_loss(total_loss_list,"Total Loss")
        self.plot_and_save_loss(policy_loss_list,"Policy Loss")
        self.plot_and_save_loss(value_loss_list,"Value Loss")
        self.plot_and_save_loss(entropy_loss_list,"Entropy Loss")  
    def judge(self):
        random_num=random.random()
        right="当前工具结果正确。"
        wrong="当前工具结果错误。"
        if random_num<0.7:
            return right,random_num
        else:
            return wrong,random_num

            
    def plot_and_save_loss(self, losses, loss_name):
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label=loss_name)
        plt.title(loss_name)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{loss_name}.png')
        plt.close()
        wandb.log({f"{loss_name}_plot": wandb.Image(f'{loss_name}.png')})

# 定义问题模板
question_templates = [
    "Why is {object} slow?",
    "How can I speed up {object}?",
    "Is there a faster way to {object}?",
    "Can you optimize {object}?",
    "What is the bottleneck in {object}?",
    "How to improve the performance of {object}?",
    "Why is the performance of {object} poor?",
    "Can we make {object} faster?",
    "Is there a way to reduce latency in {object}?",
    "How to optimize {object}?"
]

# 定义对象列表
objects = [
    "Compute",
    "Memory",
    "Chip",
    "Algorithm",
    "Pipeline",
    "Cache",
    "Data processing",
    "Network",
    "Execution",
    "System"
]

# 定义答案列表
answers = [
    "ComputeFP16",
    "ChipNumCheck",
    "MemoryCheck",
    "ChipTopoCheck",
    "CacheOptimize",
    "Parallelize",
    "ReduceOverhead",
    "AlgorithmOptimize",
    "DataPrefetch",
    "PipelineOptimize"
]

# 生成1万条数据
def generate_data(num_samples):
    data = []
    for _ in range(num_samples):
        # 随机选择一个模板和一个对象
        template = random.choice(question_templates)
        obj = random.choice(objects)
        # 组合生成完整的问题
        question = template.format(object=obj)
        # 随机选择一个答案和奖励
        answer = random.choice(answers)
        reward = random.choice([0, 1])
        data.append({"question": question, "answer": answer, "reward": reward})
    return data

# 生成1万条数据

# 打印前10条生成的数据
if __name__ == "__main__":
    #3. 'NetworkTrainllama'（用于检测跨服务器单并行方案训练异常）
    #8. 'NetworkTrainllm'（用于检测单机多卡训练性能异常）
    #5. 'BusAllreduceBw'（用于检测单服务器内的通信效率异常）
    prompt_template = """
    你是一位智能运维诊断专家，能够根据历史经验使用以下检测工具：
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
用户的问题是{}, 你的回答是:
    """
    # data = [{'question': 'How can we verify the utilization of half-precision floating-point computations in the system?', 'answer': 'ComputeFP16'}, 
    #         {'question': 'What tool should be used to monitor the performance of single-precision floating-point computations?', 'answer': 'ComputeFP32'}, 
    #         {'question': 'Which tool is appropriate for checking the memory integrity and detecting any memory leaks?', 'answer': 'MemoryCheck'}, 
    #         {'question': 'How can we determine the number of chips installed in the system?', 'answer': 'ChipNumCheck'}, 
    #         {'question': 'What tool is used to inspect the topology and arrangement of chips within the system?', 'answer': 'ChipTopoCheck'}]

# 将每个问题插入到模板中
    with open("/share/project/daliwang/daliwang/GCRRL/new/alignment_final_train1_can_reward.json",'r',encoding='utf-8') as file:
        data=json.load(file)
    generated_data = []
    for i in data:
        generated_data.append({"question":i["question"]})
    #generated_data.sort(key=lambda x: len(x["question"]), reverse=True)
    for item in generated_data:
        item["question"] = prompt_template.format(item["question"])
    model_path="/share/project/chenglongkai/datasets/QwQ-32B-Preview"
    model_path1="/share/project/chenglongkai/datasets/llama3.2-3b"
    model_path2="/share/project/chenglongkai/datasets/qwen25_7B"
    trainer=PPOTrainer(model_path2,generated_data)
    test1=trainer.tokenizer("当前业务速度较慢",return_tensors="pt").to(trainer.base_model.device)
    ans1=trainer.tokenizer("ComputeFP16",return_tensors="pt").to(trainer.base_model.device)
    gen1=trainer.tokenizer("<ComputeFP16>.",return_tensors="pt").to(trainer.base_model.device)
    #trainer.generate(test1["input_ids"])
    trainer.train()
    wandb.finish()
    #trainer.forward(test1["input_ids"])
    #trainer.get_reward(test1["input_ids"],gen1["input_ids"])
    # #print(test1)
    # # trainer.generate(test1)
    # # trainer.batch_generate([test1,test1])
    # # trainer.forward(test1)
    # #nodes=["ComputeFP16","ComputeFP32","MemoryCheck","ChipNumCheck","ChipTopoCheck"]
    # nodes=answers.copy()
    # trainer.get_reward(gen1["input_ids"],ans1["input_ids"],nodes)
    # trainer.train(batch_size=1,step=32,epochs=100)
    
    
