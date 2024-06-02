import torch
import torch.nn as nn
import numpy as np
import os
import math
import warnings
import collections
import datasets_new as dn
import models_new as mn
from run import reshape_and_split_tensor, get_graph_new
from log import logger
from parser_args import parse_args
warnings.filterwarnings("ignore")
args = parse_args()
ll = args.raw_node_feat_dim
conweidu = ll

def build_model(args):
    graph_encoder = mn.GraphEncoder(args)
    block_node_prop_module = mn.NodePropLayer
    model = mn.MLM(args, graph_encoder, block_node_prop_module)
    return model



def euclidean_distance(x, y):
    """This is the squared Euclidean distance."""
    return torch.sum((x - y) ** 2, dim=-1)

def exact_hamming_similarity(x, y):
    """Compute the binary Hamming similarity."""
    match = ((x > 0) * (y > 0)).float()
    return torch.mean(match, dim=1)

def compute_similarity(args, x, y):   #计算距离
    """Compute the distance between x and y vectors.

    The distance will be computed based on the training loss type.

    Args:
      config: a config dict.
      x: [n_examples, feature_dim] float tensor.
      y: [n_examples, feature_dim] float tensor.

    Returns:
      dist: [n_examples] float tensor.

    Raises:
      ValueError: if loss type is not supported.
    """
    if args.sim_fun == 'margin':
        # similarity is negative distance
        return euclidean_distance(x, y)
    elif args.sim_fun == 'hamming':
        return exact_hamming_similarity(x, y)
    else:
        raise ValueError('Unknown loss type')



#def read_test(self, test_file_path):
def read_test(test_file_path):
    """load positive and negative samples"""
    x = []

    with open(test_file_path) as inf:
        for line in inf:
            line = list(map(int, line.strip().split("\t")))

            x.append(line)
    return x

def sample_test(cfg_data, test_file_path):
    x = []
    with open(test_file_path, 'w') as f:
        for idx, cfg in enumerate(cfg_data):
            # 写入索引到文件，每行一个
            x.append(idx)
            f.write(str(idx) + '\n')
    return x




#def triplets_detector(self, batch_size, cfg_data, gx_list):
#def triplets_detector(self, batch_size, cfg_data):
def triplets_detector(batch_size, cfg_data):  # 生成测试集的Dataloader
        """Yields batches of triplet test data.
        Args:
            batch_size: size of batch
            cfg_data:
        """
        while True:
            cfg_data = np.array(cfg_data)

            batch_num = math.ceil(len(cfg_data) / batch_size)

            # one epoch
            for i in range(batch_num):
                st = i * batch_size
                ed = st + batch_size if st + batch_size < len(cfg_data) else len(cfg_data)

                batch_g = cfg_data[st:ed]
                #batch_gp_idx = gp_list[st:ed]
                #batch_gx_idx = gx_list[st:ed]
                #batch_gn_idx = gn_list[st:ed]

                one_batch_graphs = []
                num_graph = ed - st
                for j in range(num_graph):
                    g1 = batch_g[j]
                    #g2 = cfg_data[batch_gp_idx[j]]
                    #g2 = cfg_data[batch_gx_idx[j]]
                    #g3_idx = batch_gn_idx[j]
                    one_batch_graphs.append(g1)
                    #one_batch_graphs.append(g2)
                    #for jj in range(self.args.neg_num):
                        #one_batch_graphs.append(cfg_data[g3_idx[jj]])
                yield _pack_batch(one_batch_graphs)

def single_batch_triplets_detector(batch_size, cfg_data):
    cfg_data = np.array(cfg_data)
    print('cfg_data size:', len(cfg_data))
    if len(cfg_data) > 0:
        st = 0
        ed = min(batch_size, len(cfg_data))
        batch_g = cfg_data[st:ed]
        return _pack_batch(batch_g)
    return None

def _pack_batch(one_batch):
        """Pack a batch of graphs into a single `GraphData` instance.
        Args:
          graphs: a list of generated networkx graphs.
        Returns:
          graph_data: a `GraphData` instance, with node and edge indices properly
            shifted.
        """
        node_from_idx = []
        node_to_idx = []
        node_time_list = []
        node_edge_type_list = []
        node_feature_list = []
        node_graph_idx = []
        edge_num_per_g = [] # total edge number of cur g and g's before
        node_num_per_g = []
        node_num_per_f = []

        fun_from_idx = []
        fun_to_idx = []
        fun_time_list = []
        fun_edge_type_list = []
        fun_feature_list = []
        fun_graph_idx = []
        fun_edge_num_per_g = []
        fun_node_num_per_g = []
        fun_node_num_per_f = []
        fun_indexs = []

        con_feature_list = []


        n_total_nodes = 0
        n_total_edges = 0

        fun_total_nodes = 0
        fun_total_edges = 0
        #_edge_num = 0
        for i, g in enumerate(one_batch):
            n_nodes = g.block_node_num
            n_edges = g.block_edge_num
            fn_nodes = g.func_node_num
            fn_edges = g.func_edge_num

            n_sort_idx = np.array(g.block_sort_idx)  # 节点排序
            fun_sort_idx = np.array(g.func_sort_idx)

            edges = np.array(g.get_block_edges())[n_sort_idx]
            times = np.array(g.get_block_times())[n_sort_idx]
            edge_types = np.array(g.get_block_edge_types())[n_sort_idx]
            node_features = g.block_features
            #print(fun_sort_idx)
            fun_edges = np.array(g.get_func_edges())[fun_sort_idx]
            fun_times = np.array(g.get_func_times())[fun_sort_idx]
            fun_edge_types = np.array(g.get_func_edge_types())[fun_sort_idx]
            fun_features = g.func_features

            #con_features特征修改----
            #con_features = g.con_features #合约特征---
            #------------------------------------------此处构造不包含SSTORE的合约特征
            demo_con=g.func_features
            result = [sum(values) for values in zip(*demo_con)]
            ll=[]
            ll.append(result)
            con_features=ll
            #-----------------------------------------
            tensordata = torch.tensor(con_features[0])
            Change_vector=nn.Linear(conweidu,128)
            new_vector = Change_vector(tensordata)
            con_features=[new_vector.tolist()]

            #print(f'con_features的特征2：{con_features},长度--{len(con_features)},维度：{len(con_features[0])}')
            fun_index = g.func_indexs

            # shift the node indices for the edges
            # fun_indexs.append(fun_index)
            #print(f"fun_index:{fun_index}")
            for fi in fun_index:
                # tmp = []
                node_num_per_f.append(len(fi)) # 每个函数中的节点个数
                #fun_indexs.append(np.ones(len(fi),dtype=np.int32) * i)
                fun_indexs.append(np.array(fi,dtype=np.int32)+n_total_nodes)  # 节点序号，每个函数包含的节点序号
            #print(f"fun_indexs:{fun_indexs}")
            node_from_idx.append(edges[:, 0] + n_total_nodes)
            node_to_idx.append(edges[:, 1] + n_total_nodes)
            node_time_list.append(times)
            node_edge_type_list.append(edge_types)
            node_feature_list.append(node_features)
            node_graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)  # 这个的作用？
            # graph_idx.append(np.ones(n_edges, dtype=np.int32) * i)  # 结点或边信息
            con_feature_list.append(con_features)

            fun_from_idx.append(fun_edges[:, 0] + fun_total_nodes)
            fun_to_idx.append(fun_edges[:, 1] + fun_total_nodes)
            fun_time_list.append(fun_times)
            fun_edge_type_list.append(fun_edge_types)
            fun_feature_list.append(fun_features)
            fun_graph_idx.append(np.ones(fn_nodes, dtype=np.int32) * i)
            #_edge_num = _edge_num + n_edges
            edge_num_per_g.append(n_edges)
            node_num_per_g.append(n_nodes)
            fun_node_num_per_g.append(fn_nodes)
            fun_edge_num_per_g.append(fn_edges)

            fun_total_nodes += fn_nodes
            fun_total_edges += fn_edges
            n_total_nodes += n_nodes
            n_total_edges += n_edges
        # print(f"total node number is {n_total_nodes}, total edge number is {n_total_edges}.")

        GraphData = collections.namedtuple('GraphData',
            ['node_from_idx',       #block起始id
            'node_to_idx',          #block终点id
            'node_time_list',       #block时序信息
            'node_edge_type_list',  #block边类型
            'node_feature_list',    #block向量编码
            'node_graph_idx',       #block所属图id---
            'edge_num_per_g',       #图中block边的数量
            'node_num_per_g',       #图中block的数量---
            'node_num_per_f',       #每个函数中block个数---

            'fun_from_idx',         #function起始id
            'fun_to_idx',           #function终点id
            'fun_time_list',        #function时序信息
            'fun_edge_type_list',   #function边类型
            'fun_feature_list',     #function向量编码
            'fun_graph_idx',        #function所属图id---
            'fun_node_num_per_g',   #图中function数量---
            'fun_edge_num_per_g',   #图中function边的数量
            'fun_indexs',           #当前函数包括的所有block的index
            'con_feature_list',     #合约图编码
            'n_graphs'])  #模型缺失n_graphs
        one_batch_GraphData = GraphData(
            node_from_idx = np.concatenate(node_from_idx, axis = 0),
            node_to_idx = np.concatenate(node_to_idx, axis = 0),
            node_time_list = np.concatenate(node_time_list, axis = 0),
            node_edge_type_list = np.concatenate(node_edge_type_list, axis = 0),
            node_feature_list = np.concatenate(node_feature_list, axis = 0),
            node_graph_idx = np.concatenate(node_graph_idx, axis = 0), #----
            edge_num_per_g = edge_num_per_g,
            node_num_per_g = node_num_per_g,
            node_num_per_f = node_num_per_f,
            fun_from_idx=np.concatenate(fun_from_idx, axis = 0),
            fun_to_idx=np.concatenate(fun_to_idx, axis = 0),
            fun_time_list=np.concatenate(fun_time_list, axis=0),
            fun_edge_type_list=np.concatenate(fun_edge_type_list, axis=0),
            fun_feature_list=np.concatenate(fun_feature_list, axis=0),
            fun_graph_idx=np.concatenate(fun_graph_idx, axis=0),          #----
            fun_node_num_per_g=fun_node_num_per_g,
            fun_edge_num_per_g=fun_edge_num_per_g,
            fun_indexs=np.concatenate(fun_indexs, axis=0),               #----
            con_feature_list= np.concatenate(con_feature_list, axis = 0),
            n_graphs = len(one_batch)
        )

        return one_batch_GraphData

def detector(args,model,x_iter,y_iter):
    # 加载预训练的参数
    #state_dict = torch.load('bert_348-MPNN.pth')
    #model = torch.load('bert_348-MPNN.pth', map_location=torch.device('cpu'))
    #model = build_model(args)
    #model.load_state_dict(state_dict)
    # 模型设为评估模式
    model.eval()
    #graph_vectors_xs = []
    #graph_vectors_ys = []
    #batch = next(x_iter)
    #for batch in x_iter:
        #print('batch:', batch)
    node_features, edge_features, node_from_idx, node_to_idx, t, node_graph_idx, edge_num_per_g, node_num_per_g, node_num_per_f, con_features, \
    fun_from_idx, fun_to_idx, fun_t, fun_edge_features, fun_features, fun_graph_idx, fun_node_num_per_g, \
    fun_edge_num_per_g, fun_indexs = get_graph_new(x_iter) #yuanlaishi batch
    graph_vectors_x = model(node_features.to(device),
                      edge_features.to(device),
                      node_from_idx.to(device),
                      node_to_idx.to(device),
                      t.to(device),
                      node_graph_idx.to(device),
                      edge_num_per_g,
                      node_num_per_g,
                      node_num_per_f,
                      con_features.to(device),
                      fun_from_idx.to(device),
                      fun_to_idx.to(device),
                      fun_t.to(device),
                      fun_edge_features.to(device),
                      fun_features.to(device),
                      fun_graph_idx.to(device),
                      fun_node_num_per_g,
                      fun_edge_num_per_g,
                      fun_indexs.to(device))
        #graph_vectors_xs.append(graph_vectors_x)


    #batch = next(y_iter)
    #for batch in y_iter:
    node_features, edge_features, node_from_idx, node_to_idx, t, node_graph_idx, edge_num_per_g, node_num_per_g, node_num_per_f, con_features, \
    fun_from_idx, fun_to_idx, fun_t, fun_edge_features, fun_features, fun_graph_idx, fun_node_num_per_g, \
    fun_edge_num_per_g, fun_indexs = get_graph_new(y_iter)
    graph_vectors_y = model(node_features.to(device),
                      edge_features.to(device),
                      node_from_idx.to(device),
                      node_to_idx.to(device),
                      t.to(device),
                      node_graph_idx.to(device),
                      edge_num_per_g,
                      node_num_per_g,
                      node_num_per_f,
                      con_features.to(device),
                      fun_from_idx.to(device),
                      fun_to_idx.to(device),
                      fun_t.to(device),
                      fun_edge_features.to(device),
                      fun_features.to(device),
                      fun_graph_idx.to(device),
                      fun_node_num_per_g,
                      fun_edge_num_per_g,
                      fun_indexs.to(device))
        #graph_vectors_ys.append(graph_vectors_y)

    #print('graph_vectors_x:', graph_vectors_x, 'graph_vectors_y:', graph_vectors_y)
    #初始化结果数组
    num_graphs_x = x_iter.n_graphs
    num_graphs_y = y_iter.n_graphs
    result_array = np.zeros((num_graphs_x, num_graphs_y))     #可能有问题
    for i, x in enumerate(graph_vectors_x):
        # 计算 x 与 y_iter 中每个元素的欧氏距离
        for j, y in enumerate(graph_vectors_y):
            # 计算欧氏距离
            #distance = np.linalg.norm(x - y)
            sim_pos = compute_similarity(args, x, y)
            # 将结果保存到结果数组中的对应位置
            result_array[i, j] = sim_pos
    print('result_array_shape:', result_array.shape)
    print('result:', result_array)
    for i in range(result_array.shape[0]):  # 遍历行（128个元素）
        for j in range(result_array.shape[1]):  # 遍历列（10个数值）
            value = result_array[i, j]  # 获取当前元素的值

            # 检查值并打印相应的消息
            if 0 <= value < 0.1:
                # 根据列的索引（即“维度”的索引）打印不同的消息
                if j == 0:
                    print("block dependency detected: x-", i)
                elif j == 1:
                    #print("unsafe delegatecall detected")
                    print("reentrancy detected")
                elif j == 2:
                    #print("leaking ether detected")
                    print("unsafe delegatecall detected")
                elif j == 3:
                    #print("unprotected selfdestruct detected")
                    print("leaking ether detected")
                elif j == 4:
                    #print("locking ether  detected")
                    print("unprotected selfdestruct detected")
                elif j == 5:
                    #print("assertion failure detected")
                    print("locking ether  detected")
                elif j == 6:
                    #print("unhandled exception detected")
                    print("assertion failure detected")
                elif j == 7:
                    #print("integer overflows detected")
                    print("unhandled exception detected")
                elif j == 8:
                    #print("transaction order dependency detected")
                    print("integer overflows detected")
                elif j == 9:
                    #print("reentrancy detected")
                    print("transaction order dependency detected")
                else:
                    # 对于其他维度，可以打印一个通用的消息或者选择不打印任何内容
                    print(f"Detect vulnerability at dimension {j}")
            elif value > 0.1:
                # 如果当前值大于0.1，则继续检查下一个值
                continue
            else:
                # 如果当前值小于0，则不打印任何内容（或者可以选择打印一个消息）
                print("sim value is wrong")

                # 检查完当前行的所有值后，如果都没有小于0.1的值，则打印“No vulnerabilities found”
        if all(value > 0.1 for value in result_array[i, :]):
            print("No vulnerabilities found")

    return True


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    #待测的合约地址
    data_path_1 = './data1.6_x_test.txt'
    SCdataset_1 = dn.SmartContrastDataset(data_path_1,args)
    old_dict_1 = SCdataset_1.func_reindex()
    #已知漏洞类型的合约地址
    data_path_2 = './data1.6_y_test.txt'
    SCdataset_2 = dn.SmartContrastDataset(data_path_2,args)
    old_dict_2 = SCdataset_2.func_reindex()

    perm_file_x = f'x_perm{args.data_type}.npy'
    if os.path.isfile(perm_file_x):
        # 如果文件存在，则加载排列
        perm_x = np.load(perm_file_x)
        list_perm_x = perm_x.tolist()
        print(f'perm_list_x输出：{list_perm_x}')
    else:
        # 如果文件不存在，则创建一个顺序列表
        perm_x = np.arange(len(SCdataset_1.graph_idx_by_fun))
        list_perm_x = perm_x.tolist()
        np.save(perm_file_x, perm_x)

        # 下面的长度检查实际上是不必要的，因为我们是创建了一个顺序列表
    # 但为了保持代码完整性，我仍然保留它（尽管现在它不会执行任何操作）
    if len(perm_x) != len(SCdataset_1.graph_idx_by_fun):
        # 这部分代码现在不会执行，因为我们已经创建了一个正确长度的顺序列表
        # 但为了完整性，我还是保留它
        perm_x = np.arange(len(SCdataset_1.graph_idx_by_fun))
        list_perm_x = perm_x.tolist()
        np.save(perm_file_x, perm_x)

    logger.info("生成顺序完成")

    perm_file_y = f'y_perm{args.data_type}.npy'
    if os.path.isfile(perm_file_y):
        # 如果文件存在，则加载排列
        perm_y = np.load(perm_file_y)
        list_perm_y = perm_y.tolist()
        print(f'perm_list_y输出：{list_perm_y}')
    else:
        # 如果文件不存在，则创建一个顺序列表
        perm_y = np.arange(len(SCdataset_2.graph_idx_by_fun))
        list_perm_y = perm_y.tolist()
        np.save(perm_file_y, perm_y)

        # 下面的长度检查实际上是不必要的，因为我们是创建了一个顺序列表
    # 但为了保持代码完整性，我仍然保留它（尽管现在它不会执行任何操作）
    if len(perm_y) != len(SCdataset_2.graph_idx_by_fun):
        # 这部分代码现在不会执行，因为我们已经创建了一个正确长度的顺序列表
        # 但为了完整性，我还是保留它
        perm_y = np.arange(len(SCdataset_2.graph_idx_by_fun))
        list_perm_y = perm_y.tolist()
        np.save(perm_file_y, perm_y)

    logger.info("生成顺序完成")

    graph_x, fpos_x, x_dict = SCdataset_1.partition_test(perm_x, old_dict_1)
    graph_y, fpos_y, y_dict = SCdataset_2.partition_test(perm_y, old_dict_2)

    '''test_file_path_1 = f'x_test-neg{args.data_type}.txt'
    if os.path.isfile(test_file_path_1):
        x = read_test(test_file_path_1)
    else:
        x = sample_test(x,test_file_path_1)'''

    test_file_path_1 = f'x_test-neg{args.data_type}.txt'
    if os.path.isfile(test_file_path_1):
       x = read_test(test_file_path_1)
    else:
       x = sample_test(graph_x, test_file_path_1)
    x_iter = _pack_batch(graph_x)
    count = 0
    for _ in x_iter:
        count += 1
    print("Number of elements in x_iter:", count)
    print(x_iter)
    logger.info("生成预测模型输入数据格式完成")

    test_file_path_2 = f'y_test-neg{args.data_type}.txt'
    if os.path.isfile(test_file_path_2):
       y = read_test(test_file_path_2)
    else:
       y = sample_test(graph_y, test_file_path_2)
    #y_iter = triplets_detector(batch_size=args.batch_size_test, cfg_data=graph_test, gp_list=gp_test, gn_list = gn_test)
    y_iter = _pack_batch(graph_y)
    logger.info("生成预测模型输入数据格式完成")



    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = build_model(args)  ##############
    model = torch.load('./bert_348-MPNN.pth', map_location = torch.device('cpu'))
    model.to(device)
    #print('x_iter:', x_iter, 'y_iter:', y_iter)
    detector(args, model, x_iter, y_iter)    #修改







