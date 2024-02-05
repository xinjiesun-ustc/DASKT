def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = []
    for i in range(0, len(lines), 10):  #10代表多少行是一个完整的学生数据
        student_data = tuple(lines[i:i + 10])
        data.append(student_data)

    return data


