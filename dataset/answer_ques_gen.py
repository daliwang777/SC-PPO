import json
import re
from tqdm import tqdm
import time
# 设置智谱AI API密钥
import requests

url = "https://api.siliconflow.cn/v1/chat/completions"
df=[]
with open("/share/project/daliwang/daliwang/GCRRL/new/alignment_final_test1_can_reward.json",'r',encoding='utf-8') as f:
    raw_Data = json.load(f)
for i in tqdm(range(len(raw_Data))):
    payload = {
    "model": "deepseek-ai/DeepSeek-V2.5",
    "messages": [
        {"role": "system", "content": '''现在你是一个诊断数据的收集大师。我们能够使用的诊断函数为:
            1. 'GPUMemoryBw'（用于检测计算卡主存带宽）
            2. 'NetworkP2pBw'（用于检测跨服务器点对点通信异常）
            3. 'GPUComputeFP'（用于检测计算卡算力异常）
            4. 'NetworkAllreduceBw'（用于检测跨服务器通信性能异常）
            5. 'HostD2hBw'（用于检测数据传输速率下降（设备到主机））
            6. 'HostH2dBw'（用于检测数据传输速率下降（主机到设备））
            7. 'BusAllreduceBw'（用于检测单服务器内的通信效率异常）
            8. 'NetworkTrainllm'（用于检测单机多卡训练性能异常）
            9. 'NetworkTrainllama'（用于检测跨服务器单并行方案训练异常）
            10.'StopNetwork'(停止网络探测)
            11.'StopVendor'(报告厂商)
            12.'StopJob'(暂停业务)
            13.'StopServer'(终止服务器检测)
            14.'StopSuccess'(如果当前)
            现在我需要你能够针对我所提供给你的真实日志信息来确定几组最佳的调用过程，返回的要求如下:
            我需要你按照如下样例为我生成一批数据。
            {
                "question":"<填入我提供给你的问题>",
                'chosen':"<符合要求的调用过程与分析过程>"(如step1:我应该先调用<GPUMemoryBw>函数，因为该函数可以检测计算卡主存带宽是否正常。step2:<GPUMemoryBw>函数的返回结果是正确的。接下来，我应该调用<GPUComputeFP>函数，去检测芯片算力问题。step3:<GPUComputeFP>函数的返回结果是错误的。接下来，我应该调用<StopVendor>函数，去告诉芯片厂商这个问题),
                'rejected':"<不符合要求的调用过程>"(不符合chosen类型的都是rejected)
            }
            请注意有如下几个要求:
            1. 请按照上述样例生成数据，其中question字段为当前日志信息即我提供给你的数据，chosen字段为你认为正确的诊断步骤，rejected字段为你认为错误的诊断步骤。你可以根据自己的判断是生成这样的一批数据。
            2. 你只需要向我提供一条这样的样例即可，你应该基于我提供给你的工具大胆构造不同的故障和诊断步骤,并可以适当拓展多步操作,即有二三个返回正确的，然后再探测到错的，最后终止。
            3. 请保证你的数据是被'{}'包围的json格式，否则我将无法解析你的数据。
            4. 生成chosen的基本要求如下:
                (1)当发现是属于检测GPU问题的函数报错后，应该输出<StopVendor>来告诉芯片厂商这个问题。
                (2)当发现是属于检测网络问题的函数报错后，应该输出<StopNetwork>来停止网络探测。
                (3)当发现是属于检测服务器问题的函数报错后，应该输出<StopServer>来终止服务器检测。
                (4)当发现是属于检测业务问题的函数报错后，应该输出<StopJob>来暂停业务。
                (5)当发现现在所有函数都是正确的时候，如果你确定没有问题应该输出<StopSuccess>来停止探测。
            5. 生成rejected的基本要求如下:
                (1)没有终止类型的操作
                (2)重复使用某一个正确函数
                (3)使用了不存在于我当前调用函数的工具
                (4)没有及时输出终止操作
                (5)检查逻辑不符合正常的人类逻辑
                (6)请你大胆的构造不同的错误类型！不要局限于样例！
            现在你需要根据上面的要求向使用者提供数据。
            '''},
            {"role": "user", "content":"当前问题是:"+raw_Data[i]['question']+'.'+'请你给出一组数据'}
    ],
    "stream": False,
    "max_tokens": 1024,
    "stop": ["null"],
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "frequency_penalty": 0.5,
    "n": 1,
    "response_format": {"type": "text"},
}
    headers = {
        "Authorization": "Bearer sk-gabvhehdyrigkgxohuwktnefxxgbvbyisamcazjukjtmqtep",
        "Content-Type": "application/json"
    }
    try:
        response = requests.request("POST", url, json=payload, headers=headers)
        response_data = response.json()
        message_content = response_data['choices'][0]['message']['content']
        print(message_content)
        print(type(message_content))
        json_pattern = re.compile(r'\{.*\}', re.DOTALL)  # 匹配 [...] 之间的内容
        match = json_pattern.search(message_content)
        json_str=match.group(0)
        print(json_str)
        data = json.loads(json_str)
        df.append(data)
        json_data = json.dumps(df, indent=4,ensure_ascii=False)  # indent=4 用于美化输出，可选
        with open("align_test.json", "w",encoding='utf-8') as json_file:
            json_file.write(json_data)
    except Exception as e:
        print(f"Error: {e}")
        continue
    time.sleep(0.5)
    # # 将 JSON 数据保存到文件
    #     with open("train_2.json", "w",encoding='utf-8') as json_file:
    #         json_file.write(json_data)
    # except (KeyError, json.JSONDecodeError) as e:
    #     print(f"Error parsing response: {e}")
    # time.sleep(0.5)
    

    
