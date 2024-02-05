#!/usr/bin/env python
# coding: utf-8



from EduData import get_data
import  random
import  os
from sklearn.model_selection import train_test_split, KFold

if not os.path.exists('./data/anonymized_full_release_competition_dataset.csv'):
    get_data("assistment-2017", "../data")

import pandas as pd
import tqdm

data = pd.read_csv(
    './data/anonymized_full_release_competition_dataset.csv',
    usecols=['startTime', 'timeTaken', 'attemptCount','studentId', 'skill', 'problemId', 'correct']
).dropna(subset=['skill', 'problemId']).sort_values('startTime')


skills = data.skill.unique().tolist()
problems = data.problemId.unique().tolist()


# # question id from 1 to #num_skill
# skill2id = { p: i+1 for i, p in enumerate(skills) }
# problem2id = { p: i+1 for i, p in enumerate(problems) }
# at2id = { a: i for i, a in enumerate(at) }

# question id from 1 to (num_skill )
questions = { q: i+1  for i, q in enumerate(skills) }  #字典

# problem id from 1 to (num_problem )
problem = { p: i+1 for i, p in enumerate(problems) }

print("number of skills: %d" % len(skills))
print("number of problems: %d" % len(problems))

import numpy as np


def parse_all_seq(students):
    all_sequences = []
    for student_id in tqdm.tqdm(students, 'parse student sequence:\t'):
        student_sequence = parse_student_seq(data[data.studentId == student_id])
        all_sequences.extend([student_sequence])
    return all_sequences


def parse_student_seq(student):
    seq = student
    s = [questions[q] for q in seq.skill.tolist()]
    a = seq.correct.tolist()
    p = [problem[p] for p in seq.problemId.tolist()]
    at = seq.timeTaken.tolist()  #做题时间
    st = seq.startTime.tolist()   #做题开始时间
    ac=seq.attemptCount.tolist()

    return p, s, a, at, st ,ac  #题目id, 知识点，答案, 做题时间，做题开始时间（为了后面计算间隔使用），做题尝试次数

sequences = parse_all_seq(data.studentId.unique())

def train_test_split(data, train_size=.8, shuffle=True,seed=1025):
    random.seed(seed)
    if shuffle:
        random.shuffle(data)
    boundary = round(len(data) * train_size)
    return data[: boundary], data[boundary:]


# split train data and test data
train_data, test_data = train_test_split(sequences)

# try:
#     train_data = np.array(train_data)
#     test_data = np.array(test_data)
# except ValueError:
#     pass  # 忽略错误


def sequences2tl(sequences, trg_path):
    with open(trg_path, 'w', encoding='utf8') as f:
        for seq in tqdm.tqdm(sequences, 'write data into file: %s' % trg_path):
            p_seq,s_seq, a_seq,at_seq,st_seq ,ac_seq= seq
            seq_len =  len([q for q in s_seq if q != 0])
            f.write(str(seq_len) + '\n')
            f.write(','.join([str(p) for p in p_seq]) + '\n')
            f.write(','.join([str(s) for s in s_seq]) + '\n')
            f.write(','.join([str(a) for a in a_seq]) + '\n')
            f.write(','.join([str(at) for at in at_seq]) + '\n')
            f.write(','.join([str(st) for st in st_seq]) + '\n')
            f.write(','.join([str(ac) for ac in ac_seq]) + '\n')

def split_and_pad_records(sequences, max_step):
    result = []
    for triplet in sequences:
        problem, knowledge_points, answers, at, st ,ac = triplet
        for i in range(0, len(knowledge_points), max_step):
            sub_p = list(problem[i:i + max_step])
            sub_k = list(knowledge_points[i:i + max_step])
            sub_a = list(answers[i:i + max_step])
            sub_at = list(at[i:i + max_step])
            sub_st = list(st[i:i + max_step])
            sub_ac = list(ac[i:i + max_step])
            # pad the sub-record with 0 if its length is less than step
            if len(sub_p) < max_step:
                sub_p = list(np.pad(sub_p, (0, max_step - len(sub_p)), 'constant', constant_values=0))
                sub_k = list(np.pad(sub_k, (0, max_step - len(sub_k)), 'constant', constant_values=0))
                sub_a = list(np.pad(sub_a, (0, max_step - len(sub_a)), 'constant', constant_values=0))
                sub_at = list(np.pad(sub_at, (0, max_step - len(sub_at)), 'constant', constant_values=0))
                sub_st = list(np.pad(sub_st, (0, max_step - len(sub_st)), 'constant', constant_values=0))
                sub_ac = list(np.pad(sub_ac, (0, max_step - len(sub_ac)), 'constant', constant_values=0))
            result.append((sub_p, sub_k, sub_a, sub_at, sub_st,sub_ac))
    return result


# sequences2tl(split_and_pad_records(train_data,500), './data//train_assist2017.txt')
# sequences2tl(split_and_pad_records(test_data,500), './data/test_assist2017.txt')

sequences2tl(train_data, './data/train_assist2017.csv')
sequences2tl(test_data, './data/test_assist2017.csv')

