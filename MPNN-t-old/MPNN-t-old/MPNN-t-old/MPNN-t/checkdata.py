# 检查数据集分布
data_path = '/home/WH/modeldata/data1.4.txt'
def getinfo(data_path):

    data_cnt = 0
    data_cls = {}
    with open(data_path) as inf:
        for idx, line in enumerate(inf):
            graph = eval(line)
            data_cnt += 1
            if data_cls.__contains__(graph['label']):
                data_cls[graph['label']].append(graph)
            else:
                data_cls[graph['label']] = [graph]
    print(f"{data_path}一共包含{data_cnt}个智能合约")
    print(f"{data_path}一共包含{len(data_cls)}个类\n")
    return data_cls
data = getinfo(data_path)
for key,val in data.items():
    if len(val) < 2:
        print(key)
        print("有问题啊")
data_path = '/home/WH/modeldata/data1.3.txt'
#getinfo(data_path)
data_path = '/home/WH/modeldata/data1.2.txt'
#getinfo(data_path)
data_path = '/home/WH/modeldata/data1.1.txt'
#getinfo(data_path)

data_path = '/home/WH/modeldata/data1.0.txt'
#getinfo(data_path)
