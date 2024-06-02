import abc
import random
import collections
import math
import json
import os
import gc
import datetime
import numpy as np
import torch
import torch.nn as nn
from parser_args import parse_args
args = parse_args()
ll=args.raw_node_feat_dim
conweidu=ll

class graph(object):
    def __init__(self, block_node_num:int = 0, func_node_num:int = 0,label = None):
        self.label = label                    ###   合约标签（--labe--）
        self.block_node_num = block_node_num  ###   节点个数
        self.block_succs = []                 ###   节点的边
        self.block_times = []                 ###   节点时序信息
        self.block_edge_attributes = []       ###   节点属性
        self.block_features = []              ###   节点编码

        self.func_node_num = func_node_num    ###   函数个数
        self.func_succs = []                  ###   函数的边
        self.func_times = []                  ###   函数时序信息
        self.func_edge_attributes = []        ###   函数边属性
        self.func_features = []               ###   函数编码
        self.func_indexs = []                 ###   函数标签 （--labe--）???

        self.con_features = []                ###   合约编码

        self.block_preds = []                 ###   ???     节点头
        self.block_edge_num = 0               ###   读取时填充
        self.block_sort_idx = []              ###   ???
        self.func_preds = []                  ###   ???     函数头
        self.func_edge_num = 0
        self.func_sort_idx = []               ###   ???
        
        if (block_node_num > 0):
            for i in range(block_node_num):
                self.block_features.append([])
                self.block_succs.append([])
                self.block_preds.append([])
                self.block_times.append([])
                self.block_edge_attributes.append([])

        if (func_node_num > 0):
            for i in range(func_node_num):
                self.func_features.append([])
                self.func_succs.append([])
                self.func_preds.append([])
                self.func_times.append([])
                self.func_edge_attributes.append([])

    def add_con(self, con_instrs):
        self.con_features.append(con_instrs)
        
    def add_block_node(self, feature = []):
        self.block_node_num += 1
        self.block_features.append(feature)
        self.block_succs.append([])
        self.block_preds.append([])
        
    def add_block_edge(self, u, v, t, a):
        self.block_succs[u].append(v) 
        self.block_preds[v].append(u)
        self.block_times[u].append(t)
        self.block_edge_attributes[u].append(a)

    def add_func_node(self, feature = []):
        self.func_node_num += 1
        self.func_features.append(feature)
        self.func_succs.append([])
        self.func_preds.append([])
        
    def add_func_edge(self, u, v, t, a):
        self.func_succs[u].append(v) 
        self.func_preds[v].append(u)
        self.func_times[u].append(t)
        self.func_edge_attributes[u].append(a)

    def get_block_edges(self):
        edges = []
        for pred in range(self.block_node_num):
            for suc in self.block_succs[pred]:
                edges.append([pred,suc])
        return edges

    def get_block_times(self):
        times = []
        for node in range(self.block_node_num):
            for t in self.block_times[node]:
                times.append(t)
        return times
    
    def get_block_edge_types(self):
        edge_attributes = []
        for node in range(self.block_node_num):
            for a in self.block_edge_attributes[node]:
                edge_attributes.append(a)
        return edge_attributes

    def get_func_edges(self):
        edges = []
        for pred in range(self.func_node_num):
            for suc in self.func_succs[pred]:
                edges.append([pred,suc])
        return edges

    def get_func_times(self):
        times = []
        for node in range(self.func_node_num):
            for t in self.func_times[node]:
                times.append(t)
        return times
    
    def get_func_edge_types(self):
        edge_attributes = []
        for node in range(self.func_node_num):
            for a in self.func_edge_attributes[node]:
                edge_attributes.append(a)
        return edge_attributes

    def toString(self):
        ret = '{} {}\n'.format(self.node_num, self.fname)
        for u in range(self.node_num):
            for fea in self.features[u]:
                ret += '{} '.format(fea)
            ret += str(len(self.succs[u]))
            for succ in self.succs[u]:
                ret += ' {}'.format(succ)
            ret += '\n'
        return ret


"""A general Interface"""

class Dataset(object):
    """Base class for all the graph similarity learning datasets.
  This class defines some common interfaces a graph similarity dataset can have,
  in particular the functions that creates iterators over pairs and triplets. 
  """

    @abc.abstractmethod
    def triplets(self, batch_size):
        """Create an iterator over triplets.
    Args:
      batch_size: int, number of triplets in a batch.
    Yields:
      graphs: a `GraphData` instance.  The batch of triplets put together.  Each
        triplet has 3 graphs (x, y, z).  Here the first graph is duplicated once
        so the graphs for each triplet are ordered as (x, y, x, z) in the batch.
        The batch contains `batch_size` number of triplets, hence `4*batch_size`
        many graphs.
    """
        pass
    '''
    @abc.abstractmethod
    def pairs(self, batch_size):
        """Create an iterator over pairs.
    Args:
      batch_size: int, number of pairs in a batch.
    Yields:
      graphs: a `GraphData` instance.  The batch of pairs put together.  Each
        pair has 2 graphs (x, y).  The batch contains `batch_size` number of
        pairs, hence `2*batch_size` many graphs.
      labels: [batch_size] int labels for each pair, +1 for similar, -1 for not.
    """
        pass
    '''

class SmartContrastDataset(Dataset):

    def __init__(self, data_path, args):
        """Constructor
        Args:
            data_path: dataset file path
        """
        self.args = args

        # data preprocessing
        self.data_path = data_path
        
        self.cfg_graph_list = []      # request1:  cfg图保存
        self.graph_idx_by_fun = []    # request2:  graph_idx_by_fun:  [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]...........]

        self.biggest_edge_num = 0

        g_idx_dict = self.func_reindex()  # 使用label定义新的数值，标签对应成数字  func_dict:  {'10001_IndTokenPayment': 0, '10001_Ownable': 1, '10013_OCCC': 2, '10021_Ownable': 3}
        self.read(g_idx_dict)
        
    # dataset preprocessing
    def func_reindex(self):     #保存每个合约名字lable,对应唯一序号
        """dict all function name"""
        time1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'开始运行func_reindex，当前时间：{time1}')
        func_num = 0
        func_dict = {}
        count = 0
        with open(self.data_path) as inf:
            for line in inf:
                count = count+1
                line = line.replace("'", "\"")  # 将单引号替换为双引号
                g_info = json.loads(line.strip())
                #g_info = eval(line.strip())
                if (g_info['label'] not in func_dict):
                    func_dict[g_info['label']] = func_num
                    func_num += 1
        time3 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'func_reindex执行结束：{time3}')
        #print(f'func_dict打印输出：{func_dict}')
        print(f'去重合约类别数量：{len(func_dict)}')
        print(f'原始ID：{func_dict}')
        return func_dict                                #{'10001_IndTokenPayment': 0, '10001_Ownable': 1, '10013_OCCC': 2, '10021_Ownable': 3}


    def read(self, reidx_dict):  #输入：func_dict:  {'10001_IndTokenPayment': 0, '10001_Ownable': 1, '10013_OCCC': 2, '10021_Ownable': 3}
        """load dataset"""
        time3 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'开始运行read，当前时间：{time3}')

        if reidx_dict != None:
            for f in range(len(reidx_dict)):
                self.graph_idx_by_fun.append([])  # 保存每类cfg图的idx

        with open(self.data_path) as inf:
            for line in inf:

                line = line.replace("'", "\"")  # 将单引号替换为双引号
                g_info = json.loads(line.strip())
                #g_info = eval(line.strip())
                # i_f_idx : int, function index
                i_f_idx = reidx_dict[g_info['label']]   #合约id
                self.graph_idx_by_fun[i_f_idx].append(len(self.cfg_graph_list))      # 每条数据所对应的图，在所有graph里的位置的检索编号
                                                                                     # graph_idx_by_fun:  [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]...........]
                block_edge_num = 0
                func_edge_num = 0
                cur_graph = graph(g_info['n_num'],g_info['fn_num'], i_f_idx)         #实例化graph   （block个数，function个数、合约id）
                
                cur_graph.func_indexs = g_info['fn_indexs']
                # 加入函数层信息------------------------------------------------------------------------------------------
                f_pred, f_succ, f_t, f_a = [], [], [], []

                for u in range(g_info['fn_num']):
                    #cur_graph.features[u] = np.array(g_info['features'][u])
                    cur_graph.func_features[u] = g_info['fn_feat'][u]

                    func_edge_num += len(g_info['fn_succ'][u])
                    for v in range(len(g_info['fn_succ'][u])):
                        # get pred, succ, time, attribute of a graph
                        pred = u
                        succ = g_info['fn_succ'][u][v]
                        t = g_info['fn_time'][u][v]
                        a = g_info['fn_attribute'][u][v]

                        cur_graph.add_func_edge(pred, succ, t, a) 

                        f_pred.append(pred)
                        f_succ.append(succ)
                        f_t.append(t)
                        f_a.append(a)

                idx_sort = np.array(f_t).argsort().tolist()
                cur_graph.func_sort_idx = idx_sort
                if len(idx_sort) == 0:
                    continue
                # 加入合约层信息----------------------------------------------------------------------------------------
                cur_graph.add_con(g_info['con_feat'])

                l_pred, l_succ, l_t, l_a = [], [], [], []

                for u in range(g_info['n_num']):
                    #cur_graph.features[u] = np.array(g_info['features'][u])
                    cur_graph.block_features[u] = g_info['blo_feat'][u]   #读取节点向量

                    block_edge_num += len(g_info['n_succ'][u])            #计算block的边数量
                    for v in range(len(g_info['n_succ'][u])):
                        # get pred, succ, time, attribute of a graph
                        pred = u
                        succ = g_info['n_succ'][u][v]
                        t = g_info['n_time'][u][v]
                        a = g_info['n_attribute'][u][v]

                        cur_graph.add_block_edge(pred, succ, t, a) 

                        l_pred.append(pred)
                        l_succ.append(succ)
                        l_t.append(t)
                        l_a.append(a)

                idx_sort = np.array(l_t).argsort().tolist()
                cur_graph.block_sort_idx = idx_sort

                if block_edge_num > self.biggest_edge_num:
                    self.biggest_edge_num = block_edge_num
                cur_graph.block_edge_num = block_edge_num

                if func_edge_num > self.biggest_edge_num:
                    self.biggest_edge_num = func_edge_num
                cur_graph.func_edge_num = func_edge_num
                self.cfg_graph_list.append(cur_graph)
            print(f"biggest edge number of a graph is {self.biggest_edge_num}.")
        time4 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'原始ID--列表：{self.graph_idx_by_fun}')
        print(f'结束运行read，当前时间：{time4}')

    def partition(self, plist, permutation_list,old_dict):    #输入( [0.8,0.2] , perm1.npy)    main函数里调用->用来划分数据集   训练集-测试集
        """partition data by 'plist' list.
            Args:
            plist: list, describe how to partition. e.g. [train, test]: [0.8,0.2]
            permutation_list: list, permute all data according to this permutation list
        """
        C = len(self.graph_idx_by_fun)   ## graph_idx_by_fun:  [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]...........]
        st = 0.0
        partitioned_data = []
        for part in plist:
            cur_g = []
            new1_dict={}

            cur_g_idx_by_f = []
            ed = st + part * C
            iindex=0
            for cls in range(int(st), int(ed)):
                prev_class = self.graph_idx_by_fun[permutation_list[cls]]  #prev_class [4,5]  (原始graph_idx_by——fun中的)

                flag_index=permutation_list[cls]  #原始字典中的值
                for key, value in old_dict.items():
                    if value == flag_index:
                        flag_name=key             #原始字典中的值
                        break
                new1_dict[flag_name]=iindex
                iindex+=1
                cur_g_idx_by_f.append([])
                for i in range(len(prev_class)):
                    cur_g.append(self.cfg_graph_list[prev_class[i]])  #cur_g.append(cfg_graph_list[4])
                    cur_g[-1].fname = len(cur_g_idx_by_f)-1
                    cur_g_idx_by_f[-1].append(len(cur_g)-1)

            partitioned_data.append(cur_g)
            partitioned_data.append(cur_g_idx_by_f)
            partitioned_data.append(new1_dict)
            st = ed
        
        # release memory 
        del self.graph_idx_by_fun, self.cfg_graph_list
        gc.collect()

        return partitioned_data

    def partition_test(self, permutation_list, old_dict):
        """Partition data into a single test set based on the permutation list.

        Args:
            permutation_list: list, permute all data according to this permutation list
            old_dict: dict, original mapping from some name to indices in graph_idx_by_fun

        Returns:
            tuple: containing the test set graphs (`cur_g`), indices by function (`cur_g_idx_by_f`),
                   and a new dictionary mapping names to indices in the test set (`new_dict`).
        """
        C = len(self.graph_idx_by_fun)
        cur_g = []
        cur_g_idx_by_f = [[] for _ in range(C)]  # Initialize with empty lists for each class
        new_dict = {}

        for idx in permutation_list:
            prev_class = self.graph_idx_by_fun[idx]

            # Find the original name for this class index
            for key, value in old_dict.items():
                if value == idx:
                    flag_name = key
                    break

                    # Add the new index to the new dictionary
            new_index = len(cur_g)
            new_dict[flag_name] = new_index

            # Add the graphs to the test set
            for i in range(len(prev_class)):
                graph = self.cfg_graph_list[prev_class[i]]
                graph.fname = len(cur_g_idx_by_f) - 1  # Assuming fname is the index in cur_g_idx_by_f
                cur_g.append(graph)
                cur_g_idx_by_f[graph.fname].append(len(cur_g) - 1)

                # You don't need to delete self.graph_idx_by_fun and self.cfg_graph_list unless you're sure
        # But if you do, make sure to remove any references to them in the rest of the class

        return cur_g, cur_g_idx_by_f, new_dict


    def search_key(self,flag_index,old_dict):  #通过字典中的值定位合约名字
        for key, value in old_dict.items():
            if value == flag_index:
                return str(key)

    # data sampling
    def sample_train_np(self, cfg_data, cfg_fun_pos,train_dict):   #输入：CFG列表   合约正样本标签列表      从给定的CFG列表中进行采样，生成正样本和负样本。
        g_n, g_p = [], []
        #return g_p, g_n

        for cfg in cfg_data:
            idx_g_n = random.randrange(0, len(cfg_data))  #生成随机整数          将注释部分打开，并且while(cfg_data[idx_g_n].fname == cfg.fname):注释掉即为负样本重构
            # contract_name1=(self.search_key(cfg_data[idx_g_n].fname,train_dict)).split("_")[1]
            # contract_name2=(self.search_key(cfg.fname,train_dict)).split("_")[1]
            while(cfg_data[idx_g_n].fname == cfg.fname):
                idx_g_n = random.randrange(0, len(cfg_data))
                #contract_name1 = (self.search_key(cfg_data[idx_g_n].fname, train_dict)).split("_")[1]
            g_n.append(idx_g_n)

            idx_g_p = random.choice(cfg_fun_pos[cfg.fname])
            cnt = 0
            while(cfg_data[idx_g_p] == cfg):
                cnt += 1
                if cnt > 500:
                    print("没有找到相似样本")
                    print(cfg_fun_pos[cfg.fname])
                    print(idx_g_p)
                    break
                idx_g_p = random.choice(cfg_fun_pos[cfg.fname])
            g_p.append(idx_g_p)

        return g_p, g_n

    def triplets(self, batch_size, cfg_data, gp_list, gn_list):      #生成寻训练集的Dataloader  输入（batch_size, graph_train, 训练集——正样本索引, 训练集-负样本索引）
        """Yields batches of triplet data.
        Args:
            batch_size: size of batch
            cfg_data:
        """
        while True:
            cfg_data = np.array(cfg_data)

            perm = np.random.permutation(len(cfg_data))
            perm_g = cfg_data[perm]
            perm_gp = cfg_data[np.array(gp_list)[perm]]
            perm_gn = cfg_data[np.array(gn_list)[perm]]

            batch_num = math.ceil(len(cfg_data)/batch_size)

            # one epoch
            for i in range(batch_num):
                st = i*batch_size
                ed = st+batch_size if st+batch_size<len(perm_g) else len(perm_g)-1
                
                batch_g = perm_g[st:ed]
                batch_gp = perm_gp[st:ed]
                batch_gn = perm_gn[st:ed]
                
                one_batch_graphs = []
                num_graph = ed-st
                for j in range(num_graph):
                    g1 = batch_g[j]
                    g2 = batch_gp[j]
                    g3 = batch_gn[j]
                    one_batch_graphs.append(g1)
                    one_batch_graphs.append(g2)
                    one_batch_graphs.append(g3)

                yield self._pack_batch(one_batch_graphs)

    def sample_test_np(self, cfg_data, cfg_fun_pos,test_file_path,test_dict):      #如果没有test.npy将生成正负样本序列，并生成该文件
        g_p, g_n = [], []

        for cfg in cfg_data:
            # positive g sample
            idx_g_p = random.choice(cfg_fun_pos[cfg.fname])
            cnt = 0
            while(cfg_data[idx_g_p] == cfg):
                cnt += 1
                if cnt > 500:
                    print("没有找到相似对")
                    break
                idx_g_p = random.choice(cfg_fun_pos[cfg.fname])
            g_p.append(idx_g_p)

            # negative g sample
            _g_n = []
            for _ in range(self.args.neg_num):
                idx_g_n = random.randrange(0, len(cfg_data))                  #将注释部分打开，并将while(cfg_data[idx_g_n].fname == cfg.fname):注释掉即为负样本重构
                #contract_name3 = (self.search_key(cfg_data[idx_g_n].fname, test_dict)).split("_")[1]
                #contract_name4 = (self.search_key(cfg.fname, test_dict)).split("_")[1]
                while(cfg_data[idx_g_n].fname == cfg.fname):
                #while(contract_name3==contract_name4):
                    idx_g_n = random.randrange(0, len(cfg_data))
                    #contract_name3 = (self.search_key(cfg_data[idx_g_n].fname, test_dict)).split("_")[1]
                _g_n.append(idx_g_n)
            g_n.append(_g_n)

        with open(test_file_path, 'w') as f:
            for idx in range(len(g_p)):
                # write positive sample idx
                f.write(str(g_p[idx]) + '\t')

                # write negative sample idx
                for i_neg in range(self.args.neg_num):
                    f.write(str(g_n[idx][i_neg]) + '\t')

                f.write('\n')
        f.close()

        return g_p, g_n

    def read_test(self, test_file_path):  #输入test.npy进行读取，生成正负样本序列
        """load positive and negative samples"""
        g_p, g_n = [], []

        with open(test_file_path) as inf:
            for line in inf:
                line = list(map(int, line.strip().split("\t")))

                g_p.append(line[0])
                g_n.append(line[1:])

        return g_p, g_n

    def triplets_test(self, batch_size, cfg_data, gp_list, gn_list):  #生成测试集的Dataloader
        """Yields batches of triplet test data.
        Args:
            batch_size: size of batch
            cfg_data:
        """
        while True:
            cfg_data = np.array(cfg_data)

            batch_num = math.ceil(len(cfg_data)/batch_size)

            # one epoch
            for i in range(batch_num):
                st = i*batch_size
                ed = st+batch_size if st+batch_size<len(cfg_data) else len(cfg_data)
                
                batch_g = cfg_data[st:ed]
                batch_gp_idx = gp_list[st:ed]
                batch_gn_idx = gn_list[st:ed]
                
                one_batch_graphs = []
                num_graph = ed-st
                for j in range(num_graph):
                    g1 = batch_g[j]
                    g2 = cfg_data[batch_gp_idx[j]]
                    g3_idx = batch_gn_idx[j]
                    one_batch_graphs.append(g1)
                    one_batch_graphs.append(g2)
                    for jj in range(self.args.neg_num):
                        one_batch_graphs.append(cfg_data[g3_idx[jj]])
                yield self._pack_batch(one_batch_graphs)

    def _pack_batch(self, one_batch):
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


if __name__ == "__main__":
    from log import logger
    from parser_args import parse_args
    import datasets_old as do
    import datasets_new as dn
    # print(time())
    logger.info("开始执行")
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # 选择数据处理方法，0代表task2，1代表新数据
    if args.data_type == 1:
        data_path = './data1.4-0.1.txt'
        SCdataset = dn.SmartContrastDataset(data_path,args)
        graph_idx_by_fun1=SCdataset.graph_idx_by_fun
        print(f'graph_idx_by_fun输出{graph_idx_by_fun1}')
        # print(f'graph_idx_by_fun输出长度{len(graph_idx_by_fun1)}')
        # print(f'cfg图长度：{len(SCdataset.cfg_graph_list)}')
        # print(f'cfg图示例{SCdataset.cfg_graph_list[0]}')
        #print(SCdataset.cfg_graph_list[0].__dict__)
        logger.info("加载数据完成")
    else:
        data_path = './data2.0.txt'
        SCdataset = do.SmartContrastDataset(data_path, args)

    # 打乱数据集
    perm_file = f'perm{args.data_type}.npy'
    if os.path.isfile(perm_file):
         perm = np.load(perm_file)
    else:
         perm = np.random.permutation(len(SCdataset.graph_idx_by_fun))
         np.save(perm_file, perm)
    if len(perm) != len(SCdataset.graph_idx_by_fun):
         perm = np.random.permutation(len(SCdataset.graph_idx_by_fun))
         np.save(perm_file, perm)
    logger.info("生成顺序完成")
    graph_train, fpos_train, graph_test, fpos_test = SCdataset.partition([0.8, 0.2], perm)
    print(f'fpos_train输出 :{fpos_train}')
    print(f'fos_test输出：{fpos_test}')
    logger.info("生成训练数据与测试数据完成")
    gp_train, gn_train = SCdataset.sample_train_np(graph_train, fpos_train)
    print(f'正样本索引{gp_train},  正样本长度{len(gp_train)}')
    print(f'负样本索引{gn_train},  负样本长度{len(gn_train)}')
    train_iter = SCdataset.triplets(args.batch_size, graph_train, gp_train, gn_train)  # 生成训练模型输入数据格式
    batch = next(train_iter)
    # print(f'node_graph_idx{batch.node_graph_idx},{len(batch.node_graph_idx)}')
    # print(f'node_num_per_f{batch.node_num_per_f}')
    # print(f'fun_graph_idx{batch.fun_graph_idx}, {len(batch.fun_graph_idx)}')
    test_file_path = f'test-neg{args.data_type}.txt'
    if os.path.isfile(test_file_path):
        gp_test, gn_test = SCdataset.read_test(test_file_path)
    else:
        gp_test, gn_test = SCdataset.sample_test_np(graph_test, fpos_test, test_file_path)
    test_iter = SCdataset.triplets_test(batch_size=args.batch_size_test, cfg_data=graph_test, gp_list=gp_test,
                                        gn_list=gn_test)
    batch_t = next(test_iter)

