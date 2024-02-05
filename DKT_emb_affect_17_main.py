from torch.utils.data import DataLoader
from data_load_affect_17  import read_file

from DKT_emb_affect_17 import myKT_DKT
import torch
import numpy as np
import random
import  json
import math

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(1025)  # 你可以选择你想要的任何种子值

# assist 2017
NUM_PROBLEMS = 3162
NUM_QUESTIONS = 102

#assist 2009
# NUM_PROBLEMS = 17751
# NUM_QUESTIONS = 123

NUM_CLUSTERS=7
EMBED_SIZE = 256
BATCH_SIZE = 32
auc_max=0
loaded_total_dict={}
# with open('../data/bloom_categories_total_assist2009.json', 'r') as f:
#     loaded_total_dict = json.load(f)
#
# loaded_correct_dict = {}
# with open('../data/bloom_categories_correct_assist2009.json', 'r') as f:
#     loaded_correct_dict = json.load(f)

#读取数据
train_students = read_file("data_add_affect/train_affect_assist2017.txt")  # (有效长度，学生分割的组，分割个数，问题，知识点，答案，做题时间，做题开始时间，尝试次数）
test_students = read_file("data_add_affect/test_affect_assist2017.txt")     # (有效长度，学生分割的组，分割个数，问题，知识点，答案，做题时间，做题开始时间，尝试次数）

#EdNet
# train_students = read_file("../data/EdNet/train_ednet_kt1_4LPKT.txt")  # (有效长度，问题，知识点，答案）
# test_students = read_file("../data/EdNet/test_ednet_kt1_4LPKT.txt")    # (有效长度，问题，知识点，答案）



train_data_loader = DataLoader(train_students, batch_size=BATCH_SIZE, shuffle=True)  # 创建数据加载器
test_data_loader = DataLoader(test_students, batch_size=BATCH_SIZE, shuffle=True)  # 创建数据加载器



dkt = myKT_DKT(NUM_PROBLEMS,NUM_QUESTIONS,NUM_CLUSTERS,EMBED_SIZE)

dkt.train(train_data_loader,test_data_loader, epoch=40)
dkt.save("dkt.params")
dkt.load("dkt.params")
auc = dkt.eval(test_data_loader)
print("auc: %.6f" % auc)