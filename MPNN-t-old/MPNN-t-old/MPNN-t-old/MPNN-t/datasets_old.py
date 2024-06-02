import abc
import random
import collections
import math
import json
import os
import gc

import numpy as np

class graph(object):
    def __init__(self, node_num:int = 0, fname = None):
        self.fname = fname              ###
        self.node_num = node_num        ###
        self.succs = []                 ###
        self.times = []                 ###
        self.edge_attributes = []       ###
        self.features = []              ###

        self.preds = []
        self.edge_num = 0   # 读取时填充
        self.sort_idx = []
        
        if (node_num > 0):
            for i in range(node_num):
                self.features.append([])
                self.succs.append([])
                self.preds.append([])
                self.times.append([])
                self.edge_attributes.append([])
                
    def add_node(self, feature = []):
        self.node_num += 1
        self.features.append(feature)
        self.succs.append([])
        self.preds.append([])
        
    def add_edge(self, u, v, t, a):
        self.succs[u].append(v) 
        self.preds[v].append(u)
        self.times[u].append(t)
        self.edge_attributes[u].append(a)

    def get_edges(self):
        edges = []
        for pred in range(self.node_num):
            for suc in self.succs[pred]:
                edges.append([pred,suc])
        return edges

    def get_times(self):
        times = []
        for node in range(self.node_num):
            for t in self.times[node]:
                times.append(t)
        return times
    
    def get_edge_types(self):
        edge_attributes = []
        for node in range(self.node_num):
            for a in self.edge_attributes[node]:
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
        
        self.cfg_graph_list = []
        self.graph_idx_by_fun = []    # position of each function in self.cfg_graph_list

        self.biggest_edge_num = 0

        g_idx_dict = self.func_reindex()
        self.read(g_idx_dict)
        
    # dataset preprocessing
    def func_reindex(self):
        """dict all function name"""
        func_num = 0
        func_dict = {}
        count = 0
        with open(self.data_path) as inf:
            for line in inf:
                count = count+1
                g_info = json.loads(line.strip())
                if (g_info['fname'] not in func_dict):
                    func_dict[g_info['fname']] = func_num
                    func_num += 1
        return func_dict

    def read(self, reidx_dict):
        """load dataset"""

        if reidx_dict != None:
            for f in range(len(reidx_dict)):
                self.graph_idx_by_fun.append([])

        with open(self.data_path) as inf:

            for line in inf:
                g_info = json.loads(line.strip())

                # i_f_idx : int, function index
                i_f_idx = reidx_dict[g_info['fname']] 
                self.graph_idx_by_fun[i_f_idx].append(len(self.cfg_graph_list))      # 每条数据所对应的图，在所有graph里的位置的检索编号

                edge_num = 0
                cur_graph = graph(g_info['n_num'], i_f_idx)
                l_pred, l_succ, l_t, l_a = [], [], [], []

                for u in range(g_info['n_num']):
                    #cur_graph.features[u] = np.array(g_info['features'][u])
                    cur_graph.features[u] = g_info['features'][u]

                    edge_num += len(g_info['succ'][u])
                    for v in range(len(g_info['succ'][u])):
                        # get pred, succ, time, attribute of a graph
                        pred = u
                        succ = g_info['succ'][u][v]
                        t = g_info['time'][u][v]
                        a = g_info['attribute'][u][v]

                        cur_graph.add_edge(pred, succ, t, a) 

                        l_pred.append(pred)
                        l_succ.append(succ)
                        l_t.append(t)
                        l_a.append(a)

                idx_sort = np.array(l_t).argsort().tolist()
                cur_graph.sort_idx = idx_sort

                if edge_num > self.biggest_edge_num:
                    self.biggest_edge_num = edge_num
                cur_graph.edge_num = edge_num
                self.cfg_graph_list.append(cur_graph)
            print(f"biggest edge number of a graph is {self.biggest_edge_num}.")

    def partition(self, plist, permutation_list):
        """partition data by 'plist' list.
            Args:
            plist: list, describe how to partition. e.g. [train, test]: [0.8,0.2]
            permutation_list: list, permute all data according to this permutation list
        """
        C = len(self.graph_idx_by_fun)
        st = 0.0
        partitioned_data = []
        for part in plist:
            cur_g = []
            cur_g_idx_by_f = []
            ed = st + part * C
            for cls in range(int(st), int(ed)):
                prev_class = self.graph_idx_by_fun[permutation_list[cls]]
                cur_g_idx_by_f.append([])
                for i in range(len(prev_class)):
                    cur_g.append(self.cfg_graph_list[prev_class[i]])
                    cur_g[-1].fname = len(cur_g_idx_by_f)-1
                    cur_g_idx_by_f[-1].append(len(cur_g)-1)

            partitioned_data.append(cur_g)
            partitioned_data.append(cur_g_idx_by_f)
            st = ed
        
        # release memory 
        del self.graph_idx_by_fun, self.cfg_graph_list
        gc.collect()

        return partitioned_data


    # data sampling
    def sample_train_np(self, cfg_data, cfg_fun_pos):
        g_n, g_p = [], []
        #return g_p, g_n

        for cfg in cfg_data:
            idx_g_n = random.randrange(0, len(cfg_data))
            while(cfg_data[idx_g_n].fname == cfg.fname):
                idx_g_n = random.randrange(0, len(cfg_data))
            g_n.append(idx_g_n)

            idx_g_p = random.choice(cfg_fun_pos[cfg.fname])
            while(cfg_data[idx_g_p] == cfg):
                idx_g_p = random.choice(cfg_fun_pos[cfg.fname])
            g_p.append(idx_g_p)

        return g_p, g_n

    def triplets(self, batch_size, cfg_data, gp_list, gn_list):
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

    def sample_test_np(self, cfg_data, cfg_fun_pos, test_file_path):
        g_p, g_n = [], []

        for cfg in cfg_data:
            # positive g sample
            idx_g_p = random.choice(cfg_fun_pos[cfg.fname])
            while(cfg_data[idx_g_p] == cfg):
                idx_g_p = random.choice(cfg_fun_pos[cfg.fname])
            g_p.append(idx_g_p)

            # negative g sample
            _g_n = []
            for _ in range(self.args.neg_num):
                idx_g_n = random.randrange(0, len(cfg_data))
                while(cfg_data[idx_g_n].fname == cfg.fname):
                    idx_g_n = random.randrange(0, len(cfg_data))
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

    def read_test(self, test_file_path):
        """load positive and negative samples"""
        g_p, g_n = [], []

        with open(test_file_path) as inf:
            for line in inf:
                line = list(map(int, line.strip().split("\t")))

                g_p.append(line[0])
                g_n.append(line[1:])

        return g_p, g_n

    def triplets_test(self, batch_size, cfg_data, gp_list, gn_list):
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
        from_idx = []
        to_idx = []
        time_list = []
        edge_type_list = []
        node_feature_list = []
        graph_idx = []
        edge_num_per_g = [] # total edge number of cur g and g's before

        n_total_nodes = 0
        n_total_edges = 0
        #_edge_num = 0
        for i, g in enumerate(one_batch):
            n_nodes = g.node_num
            n_edges = g.edge_num

            sort_idx = np.array(g.sort_idx)

            edges = np.array(g.get_edges())[sort_idx]
            times = np.array(g.get_times())[sort_idx]
            edge_types = np.array(g.get_edge_types())[sort_idx]
            node_features = g.features

            # shift the node indices for the edges
            from_idx.append(edges[:, 0] + n_total_nodes)
            to_idx.append(edges[:, 1] + n_total_nodes)
            time_list.append(times)
            edge_type_list.append(edge_types)
            node_feature_list.append(node_features)
            graph_idx.append(np.ones(n_edges, dtype=np.int32) * i)

            #_edge_num = _edge_num + n_edges
            edge_num_per_g.append(n_edges)

            n_total_nodes += n_nodes
            n_total_edges += n_edges
        # print(f"total node number is {n_total_nodes}, total edge number is {n_total_edges}.")

        GraphData = collections.namedtuple('GraphData', 
            ['from_idx',
            'to_idx',
            'times',
            'edge_types',
            'node_features',
            'graph_idx',
            'edge_num_per_g',
            'n_graphs'])
        one_batch_GraphData = GraphData(
            from_idx = np.concatenate(from_idx, axis = 0),
            to_idx = np.concatenate(to_idx, axis = 0),
            times = np.concatenate(time_list, axis = 0),
            edge_types = np.concatenate(edge_type_list, axis = 0),
            node_features = np.concatenate(node_feature_list, axis = 0),
            graph_idx = np.concatenate(graph_idx, axis = 0),
            edge_num_per_g = edge_num_per_g,
            n_graphs = len(one_batch)
        )

        return one_batch_GraphData


#if __name__ == '__main__':

