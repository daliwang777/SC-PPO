import json
import torch
import torch.nn.functional as F
class ExperiencePool:
    def __init__(self, file_path="/share/project/daliwang/daliwang/GCRRL/new/pool.json"):
        self.file_path = file_path
        self.experiences = self.load_experiences()
        self.trajectory = []
        self.reward = []
        self.split_trajectory()
        self.node=self.read_in_node()
        self.data=self.load_data_pool()
        

    def load_experiences(self):
        """
        从pool.json文件中读取经验数据
        """
        experiences = []
        with open(self.file_path, 'r',encoding='utf-8') as file:
            experiences = json.load(file)
        return experiences
    def load_data_pool(self):#换数据集的
        with open('train4.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
        
    def get_experiences(self):
        """
        返回所有经验数据
        """
        return self.experiences
    def split_trajectory(self):
        """
        将经验数据拆分为trajectory和reward
        """
        
        for experience in self.experiences:
            del experience["故障"]
            self.reward.append(experience["reward"])
            del experience["reward"]
            step=[]
            feedback=[]
            trajectory=[]
            for key, value in experience.items():
                if 'step' in key:
                    step.append(value)
                elif 'result' in key:
                    feedback.append(value)
            step.append('<ALARMED>')
            for i in range(len(feedback)):
                trajectory.append((step[i],feedback[i],step[i+1]))
                #print(type(feedback[i]))
            self.trajectory.append(trajectory)
        #print(self.trajectory)
        # print(len(self.trajectory))
        # print(len(self.reward))
        return 

    def calculate_distance(self,real_step: list):
        """
        计算真实轨迹与预先轨迹的距离
        real_step:['func1','feedback1','func2','feedback2',...]
        """
        # 这里预留距离计算的逻辑
        # 你可以根据具体的需求来实现距离计算
        # 例如，可以计算trajectory之间的相似度，或者reward之间的差异
        
        real_trajectory=[]
        for i in range(0,len(real_step),2):
            if i+2>=len(real_step):
                break
            real_trajectory.append((real_step[i],real_step[i+1],real_step[i+2]))
        distance=[]
        real_trajectory=list(set(real_trajectory))
        #print(real_trajectory)
        for pool_traj in self.trajectory:
            dist=0
            for real_step in real_trajectory:
                for pool_step in pool_traj:
                    if real_step[0]==pool_step[0] and real_step[1]==pool_step[1] and real_step[2]!=pool_step[2]:
                        #print(real_step[0],real_step[1],real_step[2])
                        dist+=1
                    # elif real_step[0]==pool_step[0] and real_step[2]==pool_step[2]:
                    #     dist-=1.5
            distance.append(-dist)
        return distance
    def get_reward(self,real_step: list):
        """
        计算真实轨迹与预先轨迹的距离
        real_step:['func1','feedback1','func2','feedback2',...]
        """
        # 这里预留距离计算的逻辑
        # 你可以根据具体的需求来实现距离计算
        # 例如，可以计算trajectory之间的相似度，或者reward之间的差异
        distance=self.calculate_distance(real_step)
        reward=torch.tensor(self.reward,dtype=torch.float)
        distance=torch.tensor(distance,dtype=torch.float)
        distance=F.softmax(distance,dim=0)
        new_reward=torch.dot(reward,distance)
        return new_reward
    def get_state_reward(self,real_step: list):
        real_trajectory=[]
        if len(real_step)<3:
            return -0.1
        for i in range(0,len(real_step),2):
            if i+2>=len(real_step):
                break
            real_trajectory.append((real_step[i],real_step[i+1],real_step[i+2]))
        real_trajectory=list(set(real_trajectory))
        last_state=real_trajectory[-1]
        if len(last_state)!=3:
            return -0.1
        else:
            if last_state[1]=='True':
                return -0.1
            elif last_state[1]=='False':
                if last_state[2]=='<ALARMED>':
                    return 0.5
                else:
                    return 0.3
    def read_in_node(self,node_path='/share/project/daliwang/daliwang/GCRRL/new/newnode.json'):
        with open(node_path, 'r',encoding='utf-8') as file:
            node = json.load(file)
        return node
    def levenshtein_distance(self,seq1, seq2):
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = [[0] * size_y for _ in range(size_x)]

        for x in range(size_x):
            matrix[x][0] = x
        for y in range(size_y):
            matrix[0][y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x - 1] == seq2[y - 1]:
                    matrix[x][y] = min(
                        matrix[x - 1][y] + 1,
                        matrix[x - 1][y - 1],
                        matrix[x][y - 1] + 1
                    )
                else:
                    matrix[x][y] = min(
                        matrix[x - 1][y] + 1,
                        matrix[x - 1][y - 1] + 1,
                        matrix[x][y - 1] + 1
                    )

        return matrix[size_x - 1][size_y - 1]

# 示例函数调用序列 **

    def get_node_reward(self,real_step: list):
        real_trajectory=[]
        if len(real_step)<3:
            return -0.2
        for i in range(0,len(real_step),2):
            if i+2>=len(real_step):
                break
            real_trajectory.append((real_step[i],real_step[i+1],real_step[i+2]))
        print("real_trajectory:",real_trajectory)
        last_state=real_trajectory[-1]
        reward=0
        flag=0
        flag1=0
        real_trajectory=list(set(real_trajectory))
        for i in range(len(real_trajectory)):
            if real_trajectory[i][0] in self.node['node'] and real_trajectory[i][2] in self.node['node']:
                reward+=0.1
                if 'Stop' in real_trajectory[i][2]:
                    flag1=1
            else:
                reward=-0.2
                return reward
            if real_trajectory[i][1]=='False':
                flag=1
        if flag1==1:
            reward+=0.1
        else:
            reward+=0
        if flag==0:
            reward+=0
        return reward
    def get_strengthen_reward(self,real_step: list,answer):
        distance=[]
        for ans in answer:
            distance.append(self.levenshtein_distance(ans,real_step))
        print(distance)
        mean_dis=torch.exp(-torch.tensor(9))
        all_distance=torch.tensor(sum(distance)/len(distance))
        # reward=torch.exp(-all_distance)-mean_dis
        # if reward<0:
        #     reward=-64*torch.exp(-all_distance)-0.01
        # return reward
        reward=torch.exp(-all_distance)
        lamma1=0.8
        lamma2=0.9
        real_trajectory=[]
        if len(real_step)<3:
            return -0.2
        for i in range(0,len(real_step),2):
            if i+2>=len(real_step):
                break
            real_trajectory.append((real_step[i],real_step[i+1],real_step[i+2]))
        print("real_trajectory:",real_trajectory)
    def gen_traj(self,real_step):
        real_trajectory=[]
        if len(real_step)<3:
            return []
        for i in range(0,len(real_step),2):
            if i+2>=len(real_step):
                break
            real_trajectory.append((real_step[i],real_step[i+1],real_step[i+2]))
        return real_trajectory
    # 读取output.json文件 **
    def reward_caculate_strength(self,real_step):
        real_trajectory=self.gen_traj(real_step)
        print("real_trajectory:",real_trajectory)
        if len(real_trajectory)==0:
            return -0.1
        
            #print(len(data))
        trajectory={}
        point={}
        for i in range(len(self.data)):
            traj=self.data[i][0]
            tmp=[]
            for step in traj:
                flag=""
                if step[1]==1:
                    flag="True"
                else:
                    flag="False"
                tmp.append((step[0],flag,step[2]))
            trajectory[i]=tmp
            point[i]=torch.tensor(self.data[i][1])
        #print(trajectory)
        #print(trajectory)
        distance=[]
        for key in trajectory.keys():
            dist=0.0
            flag1=0
            for j in range(len(real_trajectory)):
                for i in range(len(trajectory[key])):
                    if real_trajectory[j][0]==trajectory[key][i][0] and real_trajectory[j][1]==trajectory[key][i][1]:
                        flag1=1
                        if real_trajectory[j][2]!=trajectory[key][i][2]:
                            dist+=1.0
                        else:
                            continue
            if flag1==0:
                dist=20
            distance.append(dist)
        distance=torch.tensor(distance)
        distance=torch.exp(-distance*2)
        #print(distance)
        px=0
        py=0
        for key in point.keys():
            px+=point[key][0]*distance[key]/torch.sum(distance)
            py+=point[key][1]*distance[key]/torch.sum(distance)
        reward=1-(px**2+py**2)**0.5
        print(reward)
        #print(reward)
        return reward

# 示例用法
if __name__ == "__main__":
    pool = ExperiencePool('/share/project/daliwang/daliwang/GCRRL/new/pool.json')
    experiences = pool.get_experiences()
    pool.get_state_reward(['MemoryCheck','False','ChipNumCheck','True'])
    print(pool.read_in_node())
    print(pool.get_node_reward(['MemoryCheck','False','ChipNumCheck']))
    
    t1 = ['A1', 'B1', 'C1']
    t2 = ['A2', 'B2', 'C2', 'D2']

    distance = pool.levenshtein_distance(t1, t2)
    print(f"The Levenshtein distance between the two sequences is: {distance}")
    print(pool.get_strengthen_reward(['MemoryCheck','False','ChipNumCheck'],[['MemoryCheck','False','ChipNumCheck','True'],['MemoryCheck','False','ChipNumCheck','True']]))
    print(pool.reward_caculate_strength(['NetworkP2pBw', 'True', 'NetworkAllreduceBw', 'False', 'StopNetwork']))
    print(pool.reward_caculate_strength(['NetworkP2pBw', 'False', 'NetworkAllreduceBw', 'True', 'StopNetwork']))
    # 示例：计算两个经验之间的距离
    