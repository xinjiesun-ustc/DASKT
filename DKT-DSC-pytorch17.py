import torch
import numpy as np
import  math
from sklearn.preprocessing import normalize
from itertools import groupby
# 设置随机种子
torch.manual_seed(1025)
torch.cuda.manual_seed_all(1025)

#读取所有学生列表，并返回所有学生的知识点和答案
def  huizong(data):  #data=students 看返回的students的组成 比原始csv数据多了一个n_split
    # 创建一个新的列表来存储转换后的元组
    students_problem= []
    students_quesion = []
    students_answer = []
    students_at = []
    students_st = []
    students_ac = []
    # 遍历每个学生
    for student in data:

        # length=int(student[0][])
        # 获取学生元组的题目，答案（一个字符串），并去掉末尾的换行符
        # problem = student[1].strip()
        question = student[3]
        answer=student[4]
        at = student[5]
        st = student[6]
        ac = student[7]
        students_quesion.extend(question)
        students_answer.extend(answer)
        students_at.extend(at)
        students_st.extend(st)
        students_ac.extend(ac)

    return  students_quesion,students_answer,students_at,students_st,students_ac


def writefile(students,file,problem_len):
    rows=students
    index = 0
    with open(f'data_add_affect/{file}_affect_assist2017.txt', 'w') as f:
        while (index < len(rows)):

            student_id = rows[index][0]
            # stu_iden.append(student_id)
            n_split  = rows[index][1]
            problem_ids = rows[index][2]
            exercise_length = len(problem_ids)  # 有效练习序列长度
            kc = rows[index][3]
            answer = rows[index][4]
            at = rows[index][5]
            st = rows[index][6]
            ac = rows[index][7]
            affect = rows[index][8]
            affect_len =len(affect)   #单独计算的原因在于 在处理affect的时候 最后一个动态分组的  不够dymaic_length（10）个 也给他补了10个 这样在不能整除的时候就会比真正的练习序列长   在  new_values = [item for item in values for _ in range(dymaic_length)]这句话中可以看到


            # 方法 2: `*` 运算符
            # affect_list = affect * exercise_length



            f.write(str(exercise_length)+ '\n')
            f.write(','.join(map(str, student_id)) + '\n')
            f.write(','.join(map(str, [n_split])) + '\n')
            if exercise_length < problem_len:  #不够练习序列的补0
                problem_ids+=[0] * (problem_len - exercise_length)
                kc += [0] * (problem_len - exercise_length)
                answer += [0] * (problem_len - exercise_length)
                at += [0] * (problem_len - exercise_length)
                st += [0] * (problem_len - exercise_length)
                ac += [0] * (problem_len - exercise_length)
                affect += [0] * (problem_len - affect_len)

            f.write(','.join(map(str, problem_ids)) + '\n')
            f.write(','.join(map(str, kc)) + '\n')
            f.write(','.join(map(str, answer)) + '\n')
            f.write(','.join(map(str, at)) + '\n')
            f.write(','.join(map(str, st)) + '\n')
            f.write(','.join(map(str, ac)) + '\n')
            f.write(','.join(map(str, affect)) + '\n')
            index += 1
    print("Finish writing data")

#拼接函数，目的是想把正确率的差值，做题时间的差值，做题时间的间隔差值拼接成成一个完整的特征数组
def merge_datasets(dataset1, dataset2, dataset3):
         dataset1 = np.expand_dims(dataset1, axis=1)
         dataset2 = np.expand_dims(dataset2, axis=1)
         dataset3 = np.expand_dims(dataset3, axis=1)
         # dataset4 = np.expand_dims(dataset4, axis=1)
         return np.concatenate((dataset1, dataset2, dataset3), axis=0)



def read_data_from_csv_file(path,skill_num):
    # config = HyperParamsConfig()
    problem_len = 500    #max_step 每个学生的连续序列是保留多长

    f_data = open(path, 'r')
    max_seg = 0
    max_skills = 0
    max_attempts= 0
    students = []
    students_all = []
    studentids = []
    #读取csv数据并进行按学生编号，题目编号，知识点、答案，做题时间，开始时间，尝试次数进行拆分
    for lineid, line in enumerate(f_data):
        if lineid % 7 == 0:
           stu = line.split(',')
           stu_id=int(stu[1])
           studentids.append(stu_id) #学生编号
        elif lineid % 7 == 1:
            problem_list = line.split(',')  # 题目编号
        elif lineid % 7 == 2:
             q_tag_list = line.split(',')     #知识点
        elif lineid % 7 == 3:
             answer_list = line.split(',')   #答案
        elif lineid % 7 == 4:
             at_list = line.split(',')  # 做题时间
        elif lineid % 7 == 5:
             st_list = line.split(',')  # 做题开始时间
        elif lineid % 7 == 6:
             ac_list = line.split(',')  # 尝试次数

             s1=[stu_id]
             t_all=(s1,problem_list,q_tag_list,answer_list,at_list,st_list,ac_list)
             students_all.append(t_all)

             tmp_max_attempts = int(len(q_tag_list))   #
             if(tmp_max_attempts> max_attempts):
                max_attempts = tmp_max_attempts

             if len(q_tag_list) > problem_len:
                n_split = len(q_tag_list) // problem_len

                if len(q_tag_list) % problem_len:
                   n_split += 1   # 有剩余的知识点，需要一个额外的分割
                   tmp_max_seg = int(n_split)
                   if(tmp_max_seg > max_seg):
                      max_seg = tmp_max_seg
             else:
                  n_split = 1

                #把学生进行切割，并按【学生编号，学生被分成的第几个片段】如：[2,0] 2号学生的第0个片段   并把相应数据进行存储
                #tuple_data存储的是每个切分后的相应数据
                #students存储的是所有学生tuple_data每个被切割后的汇总
             for k in range(n_split):
                q_container = []
                a_container = []
                #新增
                pro_container = []
                at_container = []
                st_container = []
                ac_container = []

                if k == n_split - 1:
                   end_index = len(answer_list)
                else:
                   end_index = (k+1)*problem_len
                #把每个学生被拆分完后的所有问题，知识点，答案等存入对应的list中
                for i in range(k*problem_len, end_index):
                    q_container.append(int(q_tag_list[i])) #去掉了原来的加1  因为的我们的数据处理方式已经在预处理的时候对所有知识点和题目编号从1开始进行编号了
                    a_container.append(int(answer_list[i]))
                    #新增
                    pro_container.append(int(problem_list[i]))
                    at_container.append(float(at_list[i]))
                    st_container.append(float(st_list[i]))
                    ac_container.append(int(ac_list[i]))



                if len(q_container)>0:
                   s1=[stu_id,k]
                   tuple_data=(s1,n_split,pro_container,q_container,a_container,at_container,st_container,ac_container)
                   # print("n_split=",n_split)
                   students.append(tuple_data)
                   tmp_max_skills = max(q_container)
                   if(tmp_max_skills > max_skills):
                      max_skills = tmp_max_skills

    f_data.close()

    ##计算所有学生的每个知识点的平均正确率，平均做题时间，平均做题时间间隔

    q, a,at,st,ac = huizong(students)
    students_split_data=students
    from collections import defaultdict
    # 初始化字典
    question_dict = defaultdict(lambda: {'total': 0, 'correct': 0}) #计算知识点
    at_dict = defaultdict(lambda: {'total_time': 0, 'count': 0}) #计算答题时间
    ac_dict = defaultdict(lambda: {'total_count': 0, 'count': 0})  # 计算平均尝试次数
    # st_dict = defaultdict(lambda: {'total_interval': 0, 'count': 0})  #计算开始时间

    # 遍历所有的答题结果
    old_st = 0
    count = 0
    for question, result,at,st,ac  in zip( q, a,at,st,ac):
        question_dict[question]['total'] += 1  # 更新题目的出现次数
        at_dict[question]['total_time'] += at
        at_dict[question]['count'] += 1
        ac_dict[question]['total_count'] += ac
        ac_dict[question]['count'] += 1
        st_interal=st-old_st
        old_st=st
        count+=1

        if int(result) == 1:
            question_dict[question]['correct'] += 1  # 如果答题结果是正确的，更新题目的正确次数

    new_dict = {}

    # 假设 question_dict, at_dict, 和 ac_dict 的键是相同的
    for question in question_dict:
        # 从 question_dict 中取得统计数据
        stats = question_dict[question]
        correct_ratio = stats['correct'] / stats['total'] if stats['total'] != 0 else 0

        # 从 at_dict 中取得统计数据
        stats = at_dict.get(question, {'count': 0, 'total_time': 0})  # 使用 get 防止 KeyError
        time_ratio = stats['count'] / stats['total_time'] if stats['total_time'] != 0 else 0

        # 从 ac_dict 中取得统计数据
        stats = ac_dict.get(question, {'count': 0, 'total_count': 0})  # 使用 get 防止 KeyError
        ac_ratio = stats['count'] / stats['total_count'] if stats['total_count'] != 0 else 0

        # 将结果存入新的字典
        new_dict[question] = {
            'correct_ratio': correct_ratio,
            'time_ratio': time_ratio,
            'ac_ratio': ac_ratio,
        }



    #输出新的字典
    # for question, stats in sorted(new_dict.items(), key=lambda item: int(item[0])):
    #     print(f"Question ID: {question}, Correct Ratio: {stats['correct_ratio']}, time_ratio : {stats['time_ratio']}, ac_ratio : {stats['ac_ratio']}")

    max_steps= np.round(max_attempts)

    max_stu=max(studentids)+1  #加1的目的是给0腾出一个空间 因为没有0号学生
    index=0
    cluster_data = []
   #这些行创建了三个二维numpy数组，用于存储每个学生对每个技能的答题总数，正确次数和错误次数
    xtotal = np.zeros((max_stu,skill_num+1)) #这里加1的目的和上面一样 因为我们是从1开始编号的 但是系统会从0开始计算
    x1 = np.zeros((max_stu,skill_num+1))  #该生所作题目的自己内部计算的正确率
    x0 = np.zeros((max_stu,skill_num+1))  #该生所作题目的所有学生的先验统计的平均正确率

    #新增做题时间，做题时间间隔,尝试次数
    x4 = np.zeros((max_stu,skill_num+1))  #该生所作题目的自己内部计算的做题时间
    x3 = np.zeros((max_stu, skill_num + 1))  #该生所作题目的所有学生的先验统计的平均做题时间

    # x6 = np.zeros((max_stu, max_skills + 1))  # 该生所作题目的自己内部计算的做题时间间隔
    # x5 = np.zeros((max_stu, max_skills + 1))  # 该生所作题目的所有学生的先验统计的平均做题时间间隔

    x8 = np.zeros((max_stu, skill_num + 1))  # 该生所作题目的自己内部计算的做题尝试次数
    x7 = np.zeros((max_stu, skill_num + 1))  # 该生所作题目的所有学生的先验统计的平均尝试次数

    while(index < len(students)):
         student = students[index]
         stu_id = int(student[0][0])  #每个学生的学生编号
         seg_id = int(student[0][1])  #每个学生被分成的第几个片段
         relative_participation_rate = float(student[1]/max_seg)  # 相对参与率   每个学生的被切割的份数n_split/所有数据集中的被切割的最大分数max_seg
         # print("relative_participation_rate=",relative_participation_rate)
         problem_ids = student[3]    #知识点
         correctness = student[4]     #答案
         at = student[5]  #做题时间
         st = student[6]  # 做题间隔
         ac = student[7]  # 尝试次数
         old_st_interal_every_stu = 0
         count_every_stu = 0
         for j in range(len(problem_ids)):
             key =problem_ids[j]
             xtotal[stu_id,key] +=1

             #新增
             x4[stu_id, key] += at[j]
             x3[stu_id, key] = new_dict[key]['time_ratio']

             st_interal_every_stu = st[j]-old_st_interal_every_stu
             old_st_interal_every_stu = st[j]
             count_every_stu+=1
             # x6[stu_id, key] = 13
             # x5[stu_id, key] = 9

             x8[stu_id, key] += ac[j]
             x7[stu_id, key] = new_dict[key]['ac_ratio']

             if(int(correctness[j]) == 1):  #统计每个学生的所涉及到的知识点的正确的个数
                x1[stu_id,key] +=1
                x0[stu_id, key] = new_dict[key]['correct_ratio']   #如果该知识点正确把对应平均正确率进行写入
             else:
                 x0[stu_id,key]=new_dict[key]['correct_ratio']   #如果该知识点错误也把对应平均正确率进行写入  ，无论正确错误都把平均正确率写入 目的就是不遗漏每个学生自己的所涉及到的知识点与整体平均之间的下面做差
                  # x0[stu_id,key] +=1


         #这两行代码计算当前学生对每个问题的答题成功率和所有学生对该知识点的平均成功率。
         xsr=[x/y for x, y in zip(x1[stu_id], xtotal[stu_id])]
         xfr = [x  for x  in x0[stu_id]]

         #新增
         x4r = [x/y for x, y in zip(x4[stu_id], xtotal[stu_id])] #做题时间
         x3r = [x for x in x3[stu_id]]   #平均做题时间

         st_interal_every_stu_ratio = st_interal_every_stu/count_every_stu
         # x6r = [x for x in x6[stu_id]]   #做题间隔
         # x5r = [x for x in x5[stu_id]]   #平均做题间隔

         x8r = [x/y for x, y in zip(x8[stu_id], xtotal[stu_id])] # 尝试次数
         x7r = [x for x in x7[stu_id]]  # 平均尝试次数



         x=np.nan_to_num(xsr)-np.nan_to_num(xfr)   #这行代码计算知识点成功率和该知识点的平均正确率之间的差值。

         xx = np.nan_to_num(x4r) - np.nan_to_num(x3r)  # 这行代码计算知识点做题时间和该知识点的平均做题时间之间的差值。

         # xxx= np.nan_to_num(x6r) - np.nan_to_num(x5r)  # 这行代码计算知识点做题时间间隔和该知识点的平均做题时间间隔之间的差值。
         xxxx = np.nan_to_num(x8r) - np.nan_to_num(x7r)  # 这行代码计算知识点尝试次数与该知识点的平均尝试次数之间的差值。


            #拼接正确率差值，做题时间差值，尝试次数差值
         merged_dataset = merge_datasets(x, xx,xxxx)

         # 把相对参与率添加到每个学生参与聚类运算的数据中  虚拟学生和所对应的真实学生具有同样的参与率
         merged_dataset = np.append(merged_dataset, relative_participation_rate)
         #把该生的做题间隔离与所有学生的做题间隔离作差插进来  st_interal/count代表所有学生汇总后的做题间隔离
         merged_dataset = np.append(merged_dataset, st_interal_every_stu_ratio-st_interal/count)

         #下面这两行代码将学生ID和段ID添加到x数组的末尾。
         merged_dataset=np.append(merged_dataset, stu_id)
         merged_dataset=np.append(merged_dataset, seg_id)
         cluster_data.append(merged_dataset)
         index+=1

    del students

    return students_all,students_split_data, cluster_data, studentids, max_skills, max_seg, max_steps
#students_all  原始的csv中的数据装入list
#cluster_data  包含成功率和失败率的差值等其他属性，以及将学生ID和段ID添加到结果数组中
#studentids  学生ID的集合
#max_skills  知识点的最大值
#max_seg  每个学生的练习序列被切割成了多少份的最大值 本方法是20个一份
#max_steps  没有切割之前的最大练习序列是多大

# def euclideanDistance(a, b, w):
#     return np.sqrt(w*np.sum((a-b)**2))


# 定义一个函数，用于检查一个集合是否等于 {0}
def is_zero_set(s):
    return s == {0}

def euclideanDistance(instance1, instance2,select_kc):  #length代表的那多长的属性值参与长度的计算
    length = len(instance1)  #测试集和训练集已经强制把所有知识点都应用上了，所以两种数据集的大小是一样的，即使不包含的知识点 用0值代替该知识点的值
    distance=0
    new_select_kc = [num + i * (skill_num + 1) for num in select_kc for i in range(3)]
    for x in new_select_kc:
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def dymaic_cluster_std(sudents): ##处理students分割成分割成多个虚拟学生，每个学生保留被分割后的知识点，为按知识点筛选进行聚类做准备
    kc_set_list=[]
    # 创建一个空列表，用于存放结果
    kc_set_list_divid = []
    count_num = 1
    count_std = 1
    for std in sudents:
        KC_set_num = set()  # 存放一个被动态分割后的学生的学习序列  如10个练习过程  进行一次aff类别判断
        count_num = 1
        if count_std != 1:
            kc_set_list.append({0})
        for element in std[3]:  #找到知识点所在的索引位置
            if count_num <= dymaic_length:
                KC_set_num.add(element)
                count_num += 1
            else:
                if KC_set_num:  # 在这里检查KC_set_num是否非空
                    kc_set_list.append(KC_set_num.copy())
                KC_set_num.clear()
                KC_set_num.add(element)
                count_num = 2

        # 如果最后一个序列的元素数量不足dymaic_length，我们也将其添加到kc_set_list中
        if KC_set_num:  # 在这里检查KC_set_num是否非空
            kc_set_list.append(KC_set_num.copy())

        count_std += 1
    # 使用 groupby 函数将列表分割成多个子列表
    groups = groupby(kc_set_list, is_zero_set)

    # 遍历 groups，将每个子列表添加到结果中
    for is_zero, group in groups:
        if not is_zero:
            kc_set_list_divid.append(list(group))
    # print(kc_set_list[:200])
    return kc_set_list_divid

def k_means_clust(train_students, test_students,train_cluster_data, test_cluster_data, max_stu, max_seg, num_clust, num_iter):
    identifiers=2  #把train_students中的最后两个维度去掉，因为最后两个维度存放的是学生的编号和相应被分割的片段
    max_stu=int(max_stu)
    max_seg=int(max_seg)
    cluster= {}
    data=[]
    kc_set_list_divid_train = dymaic_cluster_std(train_students)
    kc_set_list_divid_test = dymaic_cluster_std(test_students)

    for ind,i in enumerate(train_cluster_data):
        data.append(i[:-identifiers])

    data = torch.Tensor(data)   # 转化成tensor
    data = normalize(data, norm='l2')  #进行L2范式归一化（单位归一化）对于需要计算距离的聚类方法很有效
    data = torch.Tensor(data)


    centroids = data[torch.randperm(len(data))[:num_clust]]  # 初始化 k-means 聚类算法的聚类中心  打乱之后 随机选择前num_clust个

    for _ in range(num_iter):
        distances = ((data[:, None] - centroids[None, :])**2).sum(-1)  # 计算每个数据点到每个聚类中心的欧氏距离的平方，最后一个维度上进行求和
        indices = distances.argmin(1)  # torch.argmin函数返回的是指定维度上最小值的索引。
        clusters = [data[indices == i] for i in range(num_clust)]  # 把points中的点分配到相应的簇中
        centroids = torch.stack([c.mean(0) for c in clusters])  # 这段代码是在计算每个簇的新的质心，这是K-means聚类算法的关键步骤。

    # cluster for training data
    for ind,i in enumerate(train_cluster_data):
        inst=torch.Tensor(i[:-identifiers])
        min_dist=float('inf')
        closest_clust=None

        for virtual_std in range(len(kc_set_list_divid_train[ind])):
            select_kc = kc_set_list_divid_train[ind][virtual_std]

            for j in range(num_clust):
                if euclideanDistance(inst,centroids[j],select_kc)< min_dist:
                   cur_dist=euclideanDistance(inst,centroids[j],select_kc)
                   if cur_dist<min_dist:
                      min_dist=cur_dist
                      closest_clust=j
            # 检查键是否存在，如果不存在则创建一个空列表
            if (int(i[-2]), int(i[-1])) not in cluster:
                cluster[(int(i[-2]), int(i[-1]))] = []

            # 现在你可以安全地添加 closest_clust
            cluster[(int(i[-2]), int(i[-1]))].append(closest_clust)

        # 将元组转化为列表

        list_i = list(train_students[ind])
        # 添加 closest_clust
        # 假设我们有一个键
        key = (int(i[-2]), int(i[-1]))
        # 获取这个键对应的列表
        values = cluster[key]
        # 使用列表解析和 * 操作符来复制每个元素
        new_values = [item for item in values for _ in range(dymaic_length)]
        # 更新 cluster 中的值
        cluster[key] = new_values
        list_i.append(cluster[key])
        # 将列表转化回元组并替换原来的元组
        train_students[ind] = tuple(list_i)




    # cluster for testing data
    for ind,i in enumerate(test_cluster_data):
        inst=torch.Tensor(i[:-identifiers])
        min_dist=float('inf')
        closest_clust=None
        for virtual_std in range(len(kc_set_list_divid_test[ind])):
            select_kc = kc_set_list_divid_test[ind][virtual_std]

            for j in range(num_clust):
                if euclideanDistance(inst, centroids[j], select_kc) < min_dist:
                    cur_dist = euclideanDistance(inst, centroids[j], select_kc)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = j
            # 检查键是否存在，如果不存在则创建一个空列表
            if (int(i[-2]), int(i[-1])) not in cluster:
                cluster[(int(i[-2]), int(i[-1]))] = []

            # 现在你可以安全地添加 closest_clust
            cluster[(int(i[-2]), int(i[-1]))].append(closest_clust)

        # 将元组转化为列表
        test_list_i = list(test_students[ind])
        # 添加 closest_clust
        # 假设我们有一个键
        key = (int(i[-2]), int(i[-1]))
        # 获取这个键对应的列表
        values = cluster[key]
        # 使用列表解析和 * 操作符来复制每个元素
        new_values = [item for item in values for _ in range(dymaic_length)]
        # 更新 cluster 中的值
        cluster[key] = new_values
        test_list_i.append(cluster[key])
        # 将列表转化回元组并替换原来的元组
        test_students[ind] = tuple(test_list_i)

    writefile(test_students, 'test',problem_len=500)
    writefile(train_students, 'train',problem_len=500)

    return cluster


if __name__ == "__main__":

    model_name = 'DKT-DSC'
    # data_name = 'Assist_09'
    skill_num=102  #2017数据集

    dymaic_length =5  #截取多少个学生计算一次情感  （2017数据集 测试下来10个算一次 然后再和GAT结合最好 在第15轮 可以达到0.8729的AUC）

    train_students_csv_data, train_students,train_cluster_data, train_ids, train_max_skills, train_max_seg, train_max_steps = read_data_from_csv_file(
        'new/train_assist2017_train.csv',skill_num)
    test_students_csv_data, test_students, test_cluster_data, test_ids, test_max_skills, test_max_seg, test_max_steps = read_data_from_csv_file(
        'new/test_assist2017_test.csv',skill_num)
    max_skills = max([int(train_max_skills), int(test_max_skills)]) + 1
    num_skills = max_skills
    max_stu = max(train_ids + test_ids) + 1
    max_seg = max([int(train_max_seg), int(test_max_seg)]) + 1
    # config.num_steps= max([int(train_max_steps),int(test_max_steps)])
    num_steps = 500
    print('Shape of train data : %s,  test data : %s, max_step : %s ' % (
    len(train_students), len(test_students), num_steps))
   # length=(max_skills)*3+2  #max_skills 是最大知识点个数（实际上应该测试集和训练集中所有知识点的去重后之和）*3（对于每个知识点分别计算  代表的是正确率差值，做题时间差值，尝试率差值）+2（以人为单位计算   代表的是参与率和做题间隔）
    cluster = k_means_clust(train_students, test_students,train_cluster_data, test_cluster_data, max_stu, max_seg,7, max_skills)  #7代表想聚成几个类别