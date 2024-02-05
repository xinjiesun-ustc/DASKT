# The code is rewritten based on source code from tensorflow tutorial for Recurrent Neural Network.
# https://www.tensorflow.org/versions/0.6.0/tutorials/recurrent/index.html
# You can get source code for the tutorial from
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/ptb_word_lm.py
#
# There is dropout on each hidden layer to prevent the model from overfitting
#
# Here is an useful practical guide for training dropout networks
# https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
# You can find the practical guide on Appendix A
#该模块是给学生进行编号
import csv
import os

def read_data_from_csv_file(file_path,file_name,file_type,student_id=1):
    
    rows = []
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)

    new_file_path = "./new/{}_{}.csv".format(file_name,file_type)
    output_file = open(new_file_path, 'w',newline='')
    csv_writer = csv.writer(output_file, delimiter=',')       
    
    index = 0       
    while(index < len(rows)-1):
          stu_iden=[]
          problems_length = int(rows[index][0])   #有效练习序列长度
          stu_iden.append(problems_length)
          stu_iden.append(student_id)

          problem_ids = rows[index+1]          
          kc = rows[index+2]
          answer = rows[index+3]
          at = rows[index + 4]
          st = rows[index + 5]
          ac = rows[index + 6]
          
          csv_writer.writerow(stu_iden)
          csv_writer.writerow(problem_ids)
          csv_writer.writerow(kc)
          csv_writer.writerow(answer)
          csv_writer.writerow(at)
          csv_writer.writerow(st)
          csv_writer.writerow(ac)

          student_id+=1
          index += 7
          
    max_num_student= student_id
    print ("Finish reading and writing data")   
    return max_num_student
    


def main():    

    train_file_path='data/train_assist2017.csv'
    student_id=1
    max_num_stud =read_data_from_csv_file(train_file_path,'train_assist2017','train',student_id)
    test_file_path='data/test_assist2017.csv'
    max_num_stud =read_data_from_csv_file(test_file_path,"test_assist2017",'test',max_num_stud)

if __name__ == "__main__":
    main()
