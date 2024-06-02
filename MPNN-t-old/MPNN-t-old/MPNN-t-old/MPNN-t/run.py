import torch
import torch.nn as nn
import numpy as np
import os
import sys
from time import time
import warnings
warnings.filterwarnings("ignore")
from parser_args import parse_args
import datasets_old as do
import datasets_new as dn
import models_new as mn
import models_old as mo
from utils import triplet_loss, compute_similarity,compute_metrics,judge_pre_stop
import datetime
import torch.optim as optim
import collections
import copy
from log import logger

def build_model(args):

    if args.data_type == 1:  #MPNN主要看这个
        #一： 图编码器（节点-边）   输出：block的节点编码，block边的编码,function的节点编码，function边的编码，contract的节点编码（5个）
        graph_encoder = mn.GraphEncoder(args)

        # 二： 节点传播 （block层）
        block_node_prop_module = mn.NodePropLayer

        # 三：模型构造
        model = mn.MLM(args,graph_encoder,block_node_prop_module)


    else:
        graph_encoder = mo.GraphEncoder(args)
        node_prop_module = mo.NodePropLayer

        time_encoder = mo.TimeEncoder(args.time_emb_dim)
        event_emb_module = mo.EventEmbLayer(args, time_encoder)
        graph_emb_module = mo.GraphEmbLayer(args)

        model = mo.MLM(args,
                         graph_encoder,
                         node_prop_module,
                         event_emb_module,
                         graph_emb_module)

    optimizer = optim.Adam((model.parameters()),
                            lr = args.lr,
                            weight_decay=1e-5)
    return model,optimizer

def reshape_and_split_tensor(tensor, n_splits):
    """Reshape and split a 2D tensor along the last dimension.

    Args:
      tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
        multiple of `n_splits`.
      n_splits: int, number of splits to split the tensor into.

    Returns:
      splits: a list of `n_splits` tensors.  The first split is [tensor[0],
        tensor[n_splits], tensor[n_splits * 2], ...], the second split is
        [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
    """
    feature_dim = tensor.shape[-1]
    tensor = torch.reshape(tensor, [-1, feature_dim * n_splits])
    tensor_split = []
    for i in range(n_splits):
        tensor_split.append(tensor[:, feature_dim * i: feature_dim * (i + 1)])
    return tensor_split


def get_graph_new(batch):

    graph = batch
    node_from_idx = torch.from_numpy(graph.node_from_idx).long()        #节点起始id
    node_to_idx = torch.from_numpy(graph.node_to_idx).long()            #节点结束id
    t = torch.from_numpy(graph.node_time_list[:,None]).float()          #节点时序信息
    edge_features = torch.from_numpy(graph.node_edge_type_list).long()  #节点边特征
    # node_features = torch.from_numpy(graph.node_features).float()     # 不一样用得到
    node_features = torch.from_numpy(graph.node_feature_list).float()   # 不一样用得
    con_features = torch.from_numpy(graph.con_feature_list).float()    #合约特征
    node_graph_idx = torch.from_numpy(graph.node_graph_idx).long()     #block所属图id  ***
    edge_num_per_g = graph.edge_num_per_g                              #图中block边数量
    node_num_per_g = graph.node_num_per_g                              #图中block的数量
    node_num_per_f = graph.node_num_per_f                              #函数中block的数量
    fun_from_idx = torch.from_numpy(graph.fun_from_idx).long()         #函数起始节点
    fun_to_idx = torch.from_numpy(graph.fun_to_idx).long()             #函数终点节点
    fun_t = torch.from_numpy(graph.fun_time_list).long()               #函数时序信息
    fun_edge_features = torch.from_numpy(graph.fun_edge_type_list).long() #函数边信息
    fun_features = torch.from_numpy(graph.fun_feature_list).float()       #函数特征
    fun_graph_idx = torch.from_numpy(graph.fun_graph_idx).long()          #function所属图id  ***
    fun_node_num_per_g = graph.fun_node_num_per_g                         #函数节点个数
    fun_edge_num_per_g = graph.fun_edge_num_per_g                         #函数边的个数
    fun_indexs = torch.from_numpy(graph.fun_indexs).long()                #当前函数包括的所有block的index  ***


    # return node_features, edge_features, from_idx, to_idx, t, graph_idx, event_num_per_graph
    return node_features,edge_features, node_from_idx, node_to_idx, t, node_graph_idx, edge_num_per_g, node_num_per_g,node_num_per_f, con_features, \
    fun_from_idx, fun_to_idx, fun_t,fun_edge_features,fun_features, fun_graph_idx, fun_node_num_per_g, fun_edge_num_per_g,fun_indexs

def get_graph_old(batch):
    graph = batch
    
    from_idx = torch.from_numpy(graph.from_idx).long()
    to_idx = torch.from_numpy(graph.to_idx).long()
    t = torch.from_numpy(graph.times[:,None]).float()
    edge_features = torch.from_numpy(graph.edge_types).long()
    node_features = torch.from_numpy(graph.node_features).float()
    graph_idx = torch.from_numpy(graph.graph_idx).long()
    event_num_per_graph = graph.edge_num_per_g

    return node_features, edge_features, from_idx, to_idx, t, graph_idx, event_num_per_graph    

def train(args,model,optimizer,train_iter,test_iter,dataset_info,model_save_path):
    iter_num_per_epoch_train = dataset_info['train_size'] // args.batch_size + 1
    iter_num_per_epoch_text = dataset_info['test_size'] // args.batch_size_test + 1

    pre_stop_list = []
    max_acc, max_epoch = 0.0, 0
    accumulated_metrics = collections.defaultdict(list)

    for iter in range(args.n_iter):
        model.train(mode=True)
        
        batch = next(train_iter)

        if args.data_type == 0:
            raw_node_features, raw_edge_features, from_idx, to_idx, t, graph_idx = get_graph_old(batch)
            graph_vectors = model(raw_node_features.to(device),
                            raw_edge_features.to(device), 
                            from_idx.to(device), 
                            to_idx.to(device),
                            t.to(device),
                            graph_idx.to(device),
                            event_num_per_graph)
        else:
            node_features,edge_features, node_from_idx, node_to_idx, t, node_graph_idx, edge_num_per_g, node_num_per_g,node_num_per_f, con_features, \
            fun_from_idx, fun_to_idx, fun_t,fun_edge_features,fun_features, fun_graph_idx, fun_node_num_per_g, \
            fun_edge_num_per_g,fun_indexs = get_graph_new(batch)
            graph_vectors= model(node_features.to(device),
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
        
        # 三个层面的汇总向量都有了，在不同层面分别做相似性比较和loss计算吗

        graph_vectors=graph_vectors
        x, pos, neg = reshape_and_split_tensor(graph_vectors, 3)  # 划分向量

        # loss计算方式？？？----------------------------
        loss = triplet_loss(x, pos, neg, args.loss)

        #计算距离
        sim_pos = torch.mean(compute_similarity(args,x,pos))
        sim_neg = torch.mean(compute_similarity(args,x,neg))
        sim_diff = sim_pos - sim_neg


        graph_vec_scale = torch.mean(graph_vectors ** 2)
        if args.graph_vec_regularizer_weight > 0:
            loss = loss + args.graph_vec_regularizer_weight * 0.5 * graph_vec_scale
        optimizer.zero_grad()
        loss.backward(torch.ones_like(loss))
        nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)   #对模型的参数梯度进行裁剪，  [-10,10]
        optimizer.step()

        print('-----------------------------------------------------------------------')
        accumulated_metrics['loss'].append(loss)
        accumulated_metrics['sim_pos'].append(sim_pos)
        accumulated_metrics['sim_neg'].append(sim_neg)
        accumulated_metrics['sim_diff'].append(sim_diff)
        if (iter + 1) % iter_num_per_epoch_train == 0:
        
            metrics_to_print = {k: torch.mean(v[0]) for k, v in accumulated_metrics.items()}
            info_str = f'Traininf: epoch {(iter + 1) // iter_num_per_epoch_train}, ' + \
                ', '.join([f'{k} {v:.4f}' for k,v in metrics_to_print.items()])
            logger.info(info_str)
            print(info_str)
            accumulated_metrics = collections.defaultdict(list)
        
        # Do evaluation, per 5 epoch
        eval_per_epo = 2
        if ((iter + 1) % (iter_num_per_epoch_train * eval_per_epo) == 0):
            t_start = time()
            model.eval()
            with torch.no_grad():
                accumulated_auc = []
                accumulated_acc = []
                accumulated_hr = []
                accumulated_ndcg = []
                accumulated_mrr = []

                for i, batcch in enumerate(test_iter):
                    if args.data_type == 0:
                        node_features, edge_features, from_idx, to_idx, t, graph_idx, event_num_per_graph = get_graph_old(batcch)
                        eval_triplets = model(node_features.to(device),
                                          edge_features.to(device), 
                                          from_idx.to(device),
                                          to_idx.to(device),
                                          t.to(device),
                                          graph_idx.to(device),
                                          event_num_per_graph)
                    else:
                        node_features,edge_features, node_from_idx, node_to_idx, t, node_graph_idx, edge_num_per_g, node_num_per_g,node_num_per_f, con_features, \
                        fun_from_idx, fun_to_idx, fun_t,fun_edge_features,fun_features, fun_graph_idx, fun_node_num_per_g, \
                        fun_edge_num_per_g,fun_indexs = get_graph_new(batcch)

                        eval_triplets= model(node_features.to(device),
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

                    eval_triplets=eval_triplets

                    x_pos_neg = reshape_and_split_tensor(eval_triplets, args.neg_num+2)  # 1个模型输出， 1个正样本， args.neg_num个负样本

                    sim_pos = compute_similarity(args, x_pos_neg[0], x_pos_neg[1])   #x_pos_neg[0]是样本x、 x_pos_neg[1]是pos、剩下的99个是neg
                    sim_pos = sim_pos.cpu().numpy()
                    #循环来计算一组负样本与一个给定正样本之间的相似性
                    sim_neg = [compute_similarity(args,x_pos_neg[0], x_pos_neg[i_neg+2]) \
                            for i_neg in range(args.neg_num)]
                    sim_neg = np.array([sim_neg[i_neg].cpu().numpy() for i_neg in range(args.neg_num)]).T

                    acc, auc, hr_k, ndcg_k = compute_metrics(args, sim_pos, sim_neg)
                    #MRR计算--------------------
                    queries = []
                    queries.append(sim_pos[0])
                    for j in range(len(sim_neg)):
                        queries.append(sim_neg[j][0])
                    mrr = 0.0
                    for query in queries:
                        distances = [np.linalg.norm(query - pair) for pair in queries]
                        sorted_indices = np.argsort(distances)
                        rank = np.where(sorted_indices == 0)[0][0] + 1
                        mrr += 1 / rank
                    mrr /= len(queries)


                    accumulated_acc.append(acc)   #准确率
                    accumulated_auc.append(auc)
                    accumulated_hr.append(hr_k)   #命中率
                    accumulated_ndcg.append(ndcg_k)
                    accumulated_mrr.append(mrr)

                    if i == iter_num_per_epoch_text - 1:
                        break
                
                eval_metrics = {
                    'auc': np.mean(accumulated_auc),
                    'acc': np.mean(accumulated_acc),
                    'mrr': np.mean(accumulated_mrr),
                    f'hr@{args.hrk}': np.mean(accumulated_hr),
                    f'ndcg@{args.hrk}': np.mean(accumulated_ndcg),
                    'time': time() - t_start
                }

                info_str = 'Testing: ' + ', '.join([f'{k}: {v:.4f}' for k, v in eval_metrics.items()])
                # print(info_str)
                logger.info(info_str)

                # best metric update
                if max_acc < eval_metrics['acc']:
                    max_acc = eval_metrics['acc']
                    max_epoch = (iter + 1) // iter_num_per_epoch_train
                    best_model = copy.deepcopy(model)
                    torch.save(best_model, model_save_path)
                # pre-stop judgement
                if (len(pre_stop_list) < args.patience):
                    pre_stop_list.append(round(eval_metrics['acc'], 4))
                else:
                    pre_stop_list.pop(0)
                    pre_stop_list.append(round(eval_metrics['acc'],4))

                    if_stop = judge_pre_stop(pre_stop_list)
                    if if_stop:
                        if args.save_best_model:
                            torch.save(best_model, model_save_path)
                        break
            model.train()
    
    sinfo = f"training finished! Best ACC is {max_acc:.4f} in epoch {max_epoch}"
    # print(sinfo)
    logger.info(sinfo)
    return True

if __name__ == "__main__":

    logger.info("开始执行")
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    Data_type='bert_348'       #_________________________________________
    # 选择数据处理方法，0代表task2，1代表新数据
    if args.data_type == 1:
        data_path = './data1.8-demo-embe.txt'
        SCdataset = dn.SmartContrastDataset(data_path,args)
        old_dict=SCdataset.func_reindex() #加载初始字典
        #graph_idx=SCdataset.graph_idx_by_fun
        logger.info("加载数据完成")
    else:
        data_path = './data2.0.txt'
        SCdataset = do.SmartContrastDataset(data_path, args)


    # 打乱数据集
    perm_file = f'perm{args.data_type}.npy'
    if os.path.isfile(perm_file):
        perm = np.load(perm_file)
        list_perm=perm.tolist()
        print(f'perm_list输出：{list_perm}')

    else:
        perm = np.random.permutation(len(SCdataset.graph_idx_by_fun))
        list_perm = perm.tolist()
        np.save(perm_file,perm)
    if len(perm) != len(SCdataset.graph_idx_by_fun):
        perm = np.random.permutation(len(SCdataset.graph_idx_by_fun))
        list_perm = perm.tolist()
        np.save(perm_file,perm)
    logger.info("生成顺序完成")


    #生成训练集并记录合约标签
    graph_train, fpos_train,train_dict, graph_test, fpos_test,test_dict = SCdataset.partition([0.8,0.2],perm,old_dict)
    logger.info("生成训练数据与测试数据完成")
    dataset_info = {"train_size":len(graph_train), 'test_size':len(graph_test)}
    # 生成triplet 形式的train/test batch
    gp_train, gn_train = SCdataset.sample_train_np(graph_train, fpos_train,train_dict)  # 从给定的CFG列表中进行采样，生成正样本和负样本。
    train_iter = SCdataset.triplets(args.batch_size, graph_train, gp_train, gn_train) #生成训练模型输入数据格式
    logger.info("生成训练模型输入数据格式完成")

    #生成测试集并记录合约标签
    test_file_path = f'test-neg{args.data_type}.txt'
    if os.path.isfile(test_file_path):
        gp_test, gn_test = SCdataset.read_test(test_file_path)
    else:
        gp_test, gn_test = SCdataset.sample_test_np(graph_test,fpos_test,test_file_path,test_dict)
    test_iter = SCdataset.triplets_test(batch_size=args.batch_size_test, cfg_data=graph_test,gp_list=gp_test,gn_list=gn_test)
    logger.info("生成预测模型输入数据格式完成")

    #配置模型
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model,optimizer = build_model(args) ##############
    #model = torch.load('./tmp_pth')
    model.to(device)
    date = datetime.datetime.now()
    date = datetime.datetime.strftime(date,'%Y%m%d-%H%M%S')

    #开始训练
    model_save_path = f'./{Data_type}-MPNN.pth'
    train(args, model, optimizer, train_iter, test_iter, dataset_info, model_save_path)
    print(f'全局运行结束----')















