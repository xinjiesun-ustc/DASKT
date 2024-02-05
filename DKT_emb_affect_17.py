# DKT使用了embedding进行了复现

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,1,3"
from EduKTM import KTM
import logging
import torch
import torch.nn as nn
from torch.nn import Module, LSTM, Linear, Dropout
import tqdm
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import Embedding, Linear
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch.nn import LSTM
from torch_geometric.data import Data
from torch_geometric.data import Batch

import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

seqlen = 500  #2017:500 2009:50  EdNet;120

def genr_edge_index(length):
    in_vec = []
    out_vec = []
    for i in range(length):
        for j in range(i - 1, i + 2):  #情绪与前后一个有关系，包括自己
            if j >= 0 and j < length:
                in_vec.append(i)
                out_vec.append(j)
    final_list = [in_vec, out_vec]
    return final_list


def batch_genr_edge_index(true_length,batch_data):
    batch_size, _ = batch_data.shape
    batch_edge_index = []

    for i in range(batch_size):
        edge_index = genr_edge_index(int(true_length[i])-1)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        batch_edge_index.append(edge_index)

    return batch_edge_index

class DKT(Module):
    def __init__(self,num_p, num_q, num_c,emb_size, dropout=0.2):
        super().__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.lstm_layer = LSTM(self.emb_size * 4, self.hidden_size, batch_first=True)
        # self.lstm_layer_qc = LSTM(self.emb_size * 2 + 1, self.hidden_size, batch_first=True)
        self.dropout_layer = Dropout(dropout)
        # self.dropout_layer_qc = Dropout(dropout)
        self.out_layer = Linear(self.hidden_size*3, 1 )
        self.embedding_problem = nn.Embedding(num_p+1,emb_size)
        self.embedding_question = nn.Embedding(num_q + 1, emb_size)
        self.answer_embedding = nn.Embedding(2, emb_size)
        self.embedding_affect = nn.Embedding(num_c + 1, emb_size)
        # self.embedding_affect = nn.Embedding
        #新增GAT
        self.affconcat_layer = Linear(self.hidden_size * 2, self.hidden_size)  # New fully connected layer
        self.conv1 = GATConv(self.hidden_size, 128, heads=8, dropout=0.2)
        self.conv2 = GATConv(128 * 8, self.hidden_size, heads=1, concat=False, dropout=0.2)



    def forward(self, true_length,p,q, r, aff,q_next,p_next):
        p_emb=self.embedding_problem(p)
        q_emb = self.embedding_question(q)
        r_emb = self.answer_embedding(r)
        q_next_emb=self.embedding_question(q_next)
        p_next_emb = self.embedding_problem(p_next)
        #新增
        aff_emb = self.embedding_affect(aff)

        batch_edge_index_p=batch_genr_edge_index(true_length,p)
        aff_con_data = torch.cat((p_emb, aff_emb), dim=-1) #和前后的问题，知识点，情感综合在一起的有关
        aff_con_data_fc = self.affconcat_layer(aff_con_data)
        #批量装载数据和对应的边
        batch_data_list = []
        for i in range(len(aff_con_data_fc)):
            data_obj = Data(x=aff_con_data_fc[i], edge_index=batch_edge_index_p[i])
            batch_data_list.append(data_obj)
        batch_aff = Batch.from_data_list(batch_data_list)
        batch_x,batch_edge_index = batch_aff.x.to(device),batch_aff.edge_index.to(device)
        #使用GAT进行情感的聚合和更新
        x = F.dropout(batch_x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, batch_edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, batch_edge_index)  # Output x is the new node features
        aff = x.view(-1, 499, 256)

        concatenated_qr = torch.cat((p_emb, q_emb,r_emb,aff), dim=-1)

        h, _ = self.lstm_layer(concatenated_qr)

        h = self.dropout_layer(h)

        concat_atn = torch.cat((h, q_next_emb, p_next_emb), dim=-1)  #(2017 0.3较好 达到0.7945   )  (2009 0.8较好 达到0.8357)
        y = self.out_layer(concat_atn)
        y = torch.sigmoid(y)
        return y


#新增 直接预测下一个知识点
def process_raw_pred_one(question, true_answer, answer):  #question, true_answer是一个学生所有的知识点和对应的答案，answer从第二个知识点开始的预测值
    mask = torch.zeros_like(question, dtype=torch.bool)
    mask[question != 0] = True  #找出一个学生所有知识点中非填充的知识点，为了下面的找真正知识点的真实值和预测值做准备
    count = torch.sum(mask)     #统计一个学生所有知识点中非填充的知识点的个数
    final_true_answer = torch.masked_select(true_answer[1:count], mask[1:count]).to(device) #[1:count]从第二个知识点对应的答案开始找
    final_answer = torch.masked_select(torch.flatten(answer)[0:count-1], mask[0:count-1]).to(device)#[0:count-1] 从第一个预测值开始使用，count-1是因为本来answer就是少一位 你可以用count=seqlen来举例，马上理解
    return final_answer, final_true_answer


class myKT_DKT(KTM):
    def __init__(self, num_problem,num_questions,num_cluster, emb_size):
        super(myKT_DKT, self).__init__()
        self.num_questions = num_questions
        self.dkt_model = DKT(num_problem,num_questions,num_cluster, emb_size).to(device)

    def train(self, train_data, test_data=None, *, epoch: int, lr=0.003) -> ...:
        auc_max=0
        count_e=0
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.dkt_model.parameters(), lr)

        for e in range(epoch):
            all_pred, all_target = torch.Tensor([]).to(device), torch.Tensor([]).to(device)
            for batch in tqdm.tqdm(train_data, "Epoch %s" % e):
                all_pred = torch.Tensor([]).to(device)  # 清空all_pred张量
                all_target = torch.Tensor([]).to(device)  # 清空all_target张量
                true_length,batch_stu,batch_num,batch_p, batch_q, batch_a , batch_at,batch_st,batch_ac,batch_affect= batch
                # 将每个字符串转换为整数列表

                batch_p = [list(map(int, p.split(','))) for p in batch_p]
                batch_q = [list(map(int, kp.split(','))) for kp in batch_q]
                # batch_a = [list(map(int, answer.split(','))) for answer in batch_a]
                batch_a = [list(map(lambda x: int(float(x)), answer.split(','))) for answer in batch_a]

                batch_affect = [list(map(int, c.split(','))) for c in batch_affect]
                # batch_total = [list(map(int, t.split(','))) for t in batch_total]

                # 将列表转换为张量（tensor）
                batch_p = torch.tensor(batch_p).to(device)
                batch_q = torch.tensor(batch_q).to(device)
                batch_a = torch.tensor(batch_a).to(device)
                #新增
                batch_q_next = batch_q[:,1:batch_q.shape[1]]
                batch_p_next = batch_p[:, 1:batch_p.shape[1]]

                batch_affect = torch.tensor(batch_affect).to(device)
                # batch_total = torch.tensor(batch_total).to(device)


                pred_y = self.dkt_model(true_length,batch_p[:,0:batch_p.shape[1]-1].to(device),batch_q[:,0:batch_q.shape[1]-1].to(device), batch_a[:,0:batch_q.shape[1]-1].to(device), batch_affect[:,0:batch_q.shape[1]-1].to(device),batch_q_next.to(device),batch_p_next.to(device))

                batch_size = batch_q.shape[0]
                for student in range(batch_size):
                    pred, truth = process_raw_pred_one(batch_q[student].to(device), batch_a[student].to(device),
                                                   pred_y[student].to(device))
                    all_pred = torch.cat([all_pred, pred])
                    all_target = torch.cat([all_target, truth.float().to(device)])
                # print(f"预测长度{all_pred.size()}")
                loss = loss_function(all_pred, all_target)
                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # 再次检查特定层的参数是否发生了更新
            # for name, param in self.dkt_model.named_parameters():
            #     if name == 'out_cluser.weight':
            #         if param.grad is not None:
            #             print("fc1.weight 已更新")
            #         else:
            #             print("fc1.weight 未更新")
            #     elif name == 'out_layer.weight':
            #         if param.grad is not None:
            #             print("fc2.weight 已更新")
            #         else:
            #             print("fc2.weight 未更新")

            print("[Epoch %d] LogisticLoss: %.6f" % (e, loss))

            if test_data is not None:
                auc = self.eval(test_data)
                print("[Epoch %d] auc: %.6f" % (e, auc))
                if auc > auc_max:
                    auc_max = auc
                    count_e = e+1
        print(f"最大的auc是在第{count_e}轮出现的：{auc_max}")

    def eval(self, test_data) -> float:
        # self.dkt_model.eval()

        y_pred = torch.Tensor([]).to(device)
        y_truth = torch.Tensor([]).to(device)
        for batch in tqdm.tqdm(test_data):
            true_length, batch_stu, batch_num, batch_p, batch_q, batch_a, batch_at, batch_st, batch_ac, batch_affect = batch
            # 将每个字符串转换为整数列表

            batch_p = [list(map(int, p.split(','))) for p in batch_p]
            batch_q = [list(map(int, kp.split(','))) for kp in batch_q]
            # batch_a = [list(map(int, answer.split(','))) for answer in batch_a]
            batch_a = [list(map(lambda x: int(float(x)), answer.split(','))) for answer in batch_a]

            batch_affect = [list(map(int, c.split(','))) for c in batch_affect]
            # batch_total = [list(map(int, t.split(','))) for t in batch_total]

            # 将列表转换为张量（tensor）
            batch_p = torch.tensor(batch_p).to(device)
            batch_q = torch.tensor(batch_q).to(device)
            batch_a = torch.tensor(batch_a).to(device)
            # 新增
            batch_q_next = batch_q[:, 1:batch_q.shape[1]]
            batch_p_next = batch_p[:, 1:batch_p.shape[1]]

            batch_affect = torch.tensor(batch_affect).to(device)
            # batch_total = torch.tensor(batch_total).to(device)

            pred_y = self.dkt_model(true_length,batch_p[:, 0:batch_p.shape[1] - 1].to(device),
                                    batch_q[:, 0:batch_q.shape[1] - 1].to(device),
                                    batch_a[:, 0:batch_q.shape[1] - 1].to(device),
                                    batch_affect[:, 0:batch_q.shape[1] - 1].to(device), batch_q_next.to(device),
                                    batch_p_next.to(device))

            batch_size = batch_q.shape[0]
            for student in range(batch_size):
                pred, truth = process_raw_pred_one(batch_q[student].to(device), batch_a[student].to(device),
                                                   pred_y[student].to(device))
                y_pred = torch.cat([y_pred, pred])
                y_truth = torch.cat([y_truth, truth])



        return roc_auc_score(y_truth.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

    def save(self, filepath):
        torch.save(self.dkt_model.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dkt_model.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)





