import json 
from transformers import AutoTokenizer, AutoModel 
from langchain.document_loaders  import JSONLoader 
from langchain.text_splitter  import RecursiveCharacterTextSplitter 
from langchain.embeddings  import HuggingFaceEmbeddings 
from langchain.vectorstores  import FAISS 
from langchain.schema import Document

# === 1. 加载 BGE-m3 模型 === 
model_name = "/share/project/chenglongkai/datasets/bge-m3"
 
# === 2. 自定义 JSON 加载器 === 
class CustomJSONLoader:
    def __init__(self, file_path):
        self.file_path  = file_path 
 
    def load(self):
        documents = []
        # with open(self.file_path,  'r',encoding='utf-8') as f:
        #     data=json.load(f)
        # for i in data:
        #     content = i['question'] + '\n' + i['chosen']
        #     documents.append(Document(page_content=content))
        with open("/share/project/daliwang/daliwang/GCRRL/new/align_train.json",  'r',encoding='utf-8') as f:
            data=json.load(f)
        with open("/share/project/daliwang/daliwang/GCRRL/new/align_test.json","r",encoding='utf-8') as f:
            data+=json.load(f)
        for i in data:
            content = i['question'] + '\n' + i['chosen']
            documents.append(Document(page_content=content))
        return documents 
 
# === 3. 加载和处理数据 === 
file_path = "/share/project/daliwang/daliwang/GCRRL/new/align_train.json" 
loader = CustomJSONLoader(file_path)
texts = loader.load() 
 
# 打印前几条数据以验证 
print("Loaded texts:")
for i, text in enumerate(texts[:3]):
    print(f"Text {i+1}:")
    print(text.page_content)
    print("\n")
 
# === 4. 分割文本 === 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 每个 chunk 的最大长度 
    chunk_overlap=200  # 相邻 chunk 之间的重叠长度 
)
chunks = text_splitter.split_documents(texts)
 
# === 5. 创建嵌入模型 === 
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
)
 
# === 6. 构建向量数据库 === 
# 初始化向量数据库 
db = FAISS.from_documents(chunks,  embeddings)
 
# 保存向量数据库到指定路径 
db.save_local("/share/project/daliwang/daliwang/GCRRL/new/vec_test_train") 
 
# === 7. 加载并验证向量数据库 === 
loaded_db = FAISS.load_local("/share/project/daliwang/daliwang/GCRRL/new/vec_test_train",  embeddings,allow_dangerous_deserialization=True)
 
# 示例：进行相似度搜索 
query = "What is the best way to improve my English?"
results = loaded_db.similarity_search(query,  k=3)
 
# 打印结果 
print("Search Results:")
for result in results:
    print(result.page_content) 
    print("\n")



