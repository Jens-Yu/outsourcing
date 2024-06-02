import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from parser_args import parse_args
from utils import unsorted_segment_sum,new_contract
import torch.nn.functional as F



Global_flag=0                    #0代表CPU  1代表GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphEncoder(nn.Module):   #一：  图编码器模块，将节点和边进行编码  （节点是block）
    
    """Encoder module that projects node and edge features to embeddings."""
    def __init__(self, args, name='GEncoder'):
        """Constructor.

        Args:
          args: parse all parameters this class need from args
          name: name of this module.
        """
        super(GraphEncoder, self).__init__()

        self._paraParse(args)

        # initialize edge feature embedding table
        self._edge_emb_tab = nn.Embedding(self._n_edge_type, self._edge_emb_dim)  #将边类型转换为固定维度的嵌入向量，边类型的数量就是边类型嵌入的维度

        self._build_model()

    def _paraParse(self, args):
        # parse parameters from args

        self._raw_node_feat_dim = args.raw_node_feat_dim  #原始节点特征的维度
        self._raw_edge_feat_dim = args.raw_edge_feat_dim  #原始边特征的维度
        self._node_emb_dim = args.node_emb_dim            #节点嵌入的维度
        self._edge_emb_dim = args.edge_type_emb_dim       #边类型嵌入的维度
        self._n_edge_type = args.num_edge_type            #边类型的数量

    def _build_model(self):

        # Node Embedding MLP
        layer = []
        layer.append(nn.Linear(self._raw_node_feat_dim, self._node_emb_dim))
        #layer.append(nn.ReLU())
        self.MLP1 = nn.Sequential(*layer)

        layer = []
        layer.append(nn.Linear(self._edge_emb_dim, self._edge_emb_dim))
        #layer.append(nn.ReLU())
        self.MLP2 = nn.Sequential(*layer)



    def forward(self, raw_node_features, raw_edge_features):

        # Node Embedding
        node_outputs = self.MLP1(raw_node_features)

        # Edge Embedding 
        _edge_emb = self._edge_emb_tab(raw_edge_features)
        edge_outputs = self.MLP2(_edge_emb)

        return node_outputs, edge_outputs


class NodePropLayer(nn.Module):   #该类为MPNN---------重点看！！！
    """Implementation of a graph propagation (message passing) layer."""

    def __init__(self, args, name='propLayer'):
        """Constructor.
        Args:
          args: parse all parameters this class need from args
        """
        super(NodePropLayer, self).__init__()

        self._paraParse(args)

        self.build_model()

        if self._layer_norm:
            self.layer_norm1 = nn.LayerNorm()  #归一化
            self.layer_norm2 = nn.LayerNorm()


    def _paraParse(self, args):
        # parse parameters from args
        self.node_emb_dim = args.node_emb_dim
        self.edge_type_emb_dim = args.edge_type_emb_dim
        self.msg_emb_dim = args.msg_emb_dim   #时序传播      消息嵌入维度
        self._node_update_type = args.node_update_type  #节点是否更新    默认 mlp
        self._use_reverse_direction = args.use_reverse_direction  #是否使用反向传播  默认0
        self._reverse_dir_param_different = args.reverse_dir_param_different  #反向传播是否使用不用参数  默认 0
        self._layer_norm = False


    def build_model(self):

        """message network"""
        layer = []
        layer.append(nn.Linear(self.node_emb_dim * 2 + self.edge_type_emb_dim, self.msg_emb_dim))  #原始消息嵌入由两个节点嵌入和边类型嵌入拼接而成
        layer.append(nn.ReLU())     # 看看是否去掉
        self._message_net = nn.Sequential(*layer)

        # optionally compute message vectors in the reverse direction 是否选择计算反向消息向量
        if self._use_reverse_direction:
            if self._reverse_dir_param_different:
                layer = []
                layer.append(nn.Linear(self.node_emb_dim * 3, self.msg_emb_dim))
                layer.append(nn.ReLU())
                self._reverse_message_net = nn.Sequential(*layer)
            else:
                self._reverse_message_net = self._message_net

        """propagation network."""
        if self._node_update_type == 'gru':
            self.GRU = nn.GRU(self.msg_emb_dim, self.node_emb_dim)
        else:
            layer = []
            layer.append(nn.Linear(self.msg_emb_dim, self.node_emb_dim))    #mk那步
            layer.append(nn.ReLU())
            self.MLP = nn.Sequential(*layer)


    def _compute_aggregated_messages(self,                 #    1.消息聚合
                                     node_features,
                                     edge_features, 
                                     from_idx, 
                                     to_idx):
        """Compute aggregated messages for each node.
        Args:
          node_features: [n_nodes, input_node_state_dim] float tensor, node states.节点特征集
          edge_features: if not None, should be [n_edges, edge_embedding_dim]
            tensor, edge features.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.

        Returns:
          aggregated_messages: [n_nodes, aggregated_message_dim] float tensor, the
            aggregated messages for each node.
        """
        aggregated_messages = self.graph_prop_once(node_features,            #消息传递
                                                   edge_features,
                                                   from_idx,
                                                   to_idx,
                                                   self._message_net)

        # optionally compute message vectors in the reverse direction
        if self._use_reverse_direction:
            reverse_aggregated_messages = self.graph_prop_once(node_features,
                                                               edge_features,
                                                               from_idx,
                                                               to_idx,
                                                               self._reverse_message_net)

            aggregated_messages += reverse_aggregated_messages

        if self._layer_norm:
            aggregated_messages = self.layer_norm1(aggregated_messages)

        return aggregated_messages


    def _compute_node_update(self,                      #2. 节点更新
                             node_features,
                             update_msg):
        """Compute node updates.
        Args:
          node_features: [n_nodes, node_emb_dim] float tensor, the input node
            states.
          update_msg: a list of tensors used to compute node updates.  Each
            element tensor should have shape [n_nodes, feat_dim], where feat_dim can
            be different.  These tensors will be concatenated along the feature
            dimension.

        Returns:
          new_node_states: [n_nodes, node_state_dim] float tensor, the new node
            state tensor.
        """
        if self._node_update_type in ('mlp', 'residual'):
            update_msg.append(node_features)

        update_msg = update_msg[0] #if len(update_msg) == 1 else torch.cat(update_msg, dim=-1)

        if self._node_update_type == 'gru':
            update_msg = torch.unsqueeze(update_msg, 0)
            node_features = torch.unsqueeze(node_features, 0)

            _, new_node_states = self.GRU(update_msg, node_features)
            new_node_states = torch.squeeze(new_node_states)
            return new_node_states
        else:
            mlp_output = self.MLP(update_msg)
            if self._layer_norm:
                mlp_output = self.layer_norm2(mlp_output)
            if self._node_update_type == 'mlp':
                return mlp_output
            elif self._node_update_type == 'residual':
                return node_features + mlp_output
            else:
                raise ValueError('Unknown node update type %s' % self._node_update_type)

    @staticmethod
    def graph_prop_once(node_features,    ##消息传递
                        edge_features,
                        from_idx,
                        to_idx,
                        message_net):
        """One round of propagation (message passing) in a graph.

        Args:
            node_features: [n_nodes, node_state_dim] float tensor, node state vectors, one
                row for each node.
            edge_features: if provided, should be a [n_edges, edge_feature_dim] float
                tensor, extra features for each edge.
            from_idx: [n_edges] int tensor, index of the from nodes.
            to_idx: [n_edges] int tensor, index of the to nodes.
            message_net: a network that maps concatenated edge inputs to message
                vectors.

        Returns:
          aggregated_messages: an [n_nodes, edge_message_dim] float tensor, the
            aggregated messages, one row for each node.
        """
        #print("node_features shape:", node_features.shape)
        #print("edge_features shape:", edge_features.shape)
        #print("from_idx shape:", from_idx.shape)
        #print("to_idx shape:", to_idx.shape)

        from_emb = node_features[from_idx]
        to_emb = node_features[to_idx]
        edge_inputs = [from_emb, to_emb, edge_features]

        edge_inputs = torch.cat(edge_inputs, dim=-1)
        messages = message_net(edge_inputs)

        tensor = unsorted_segment_sum(messages, to_idx, node_features.shape[0])
        return tensor


    def forward(self,
                node_features,
                edge_features,
                from_idx,
                to_idx):
        """Run one propagation step.

        Args:
          node_features: extra node features if provided, should be of size
            [n_nodes, extra_node_feat_dim] float tensor, can be used to implement
            different types of skip connections.
          edge_features: if not None, should be [n_edges, edge_embedding_dim]
            tensor, edge features.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.

        Returns:
          node_features: [n_nodes, node_state_dim] float tensor, new node states.
        """
        aggregated_messages = self._compute_aggregated_messages(node_features,         #1.消息聚合
                                                                edge_features, 
                                                                from_idx, 
                                                                to_idx)

        updated_node = self._compute_node_update(node_features,[aggregated_messages])   #2.节点更新
        return updated_node







class MLM(nn.Module):        #模型整体构造,主要包括上述两类  图编码和MPNN
    def __init__(self,
                args,
                graph_encoder,
                block_node_prop_module,
                ):
        super(MLM,self).__init__()

        # parse parameters
        self._paraParse(args)

        # get modules needed
        self.M_graph_encoder = graph_encoder
        self.M_block_prop_layer = block_node_prop_module


        # n run prop layers list （block 与  function 传播列表）
        self._block_prop_layers = []
        self._block_prop_layers = nn.ModuleList()
        self._function_prop_layers = []
        self._function_prop_layers = nn.ModuleList()

        # build model
        self.build_model()

    def _paraParse(self, args):
        """parse parameters from args.
        
        Args: 
            args: an parameter class, get all hyper parameters from it
        """
        self.args = args
        self._n_run_block_prop = args.block_n_prop_runs   #5层
        self._n_run_function_prop = args.function_n_prop_runs  #3层
        self._share_prop_params = args.share_prop_params  #1层

        #定义注意力机制参数
        #self._attention_hidden_dim = args.attention_hidden_dim
        #self._attention_output_dim = args.attention_output_dim

    def build_model(self):
        if len(self._block_prop_layers) < self._n_run_block_prop:
            # build the layers  #构造block层传播
            for i in range(self._n_run_block_prop):
                if i == 0 or not self._share_prop_params:
                    one_prop_layer = self.M_block_prop_layer(self.args)
                else:
                    one_prop_layer = self._block_prop_layers[0]  #重用已经创建的传播层
                self._block_prop_layers.append(one_prop_layer)

        #注意力机制
        #self.attention_layer = nn.Linear(self._attention_hidden_dim, self._attention_output_dim)
        #self.attention_layer = nn.Linear(128, 64)




        #Block层额外配件
        layer_b = []
        layer_b.append(nn.Linear(128, 128))
        self.MLP_g_node = nn.Sequential(*layer_b)



    def forward(self,
                block_node_features, 
                block_edge_features,
                block_node_from_idx,
                block_node_to_idx,
                t,
                node_graph_idx,
                edge_num_per_g,
                node_num_per_g,
                node_num_per_f,
                con_features,
                fun_from_idx,
                fun_to_idx,
                fun_t,
                fun_edge_features,
                fun_node_features,
                fun_graph_idx,
                fun_node_num_per_g,
                fun_edge_num_per_g,
                fun_indexs):
        """Compute graph representations.

        Args:

        Returns:
          graph_representations: [batch_num, graph_representation_dim] float tensor,
            graph representations.
        """

        """第零步：针对节点和边进行编码"""
        #第一类：图编码器输出（5输入，5输出）
        block_node_features1, block_edge_features1 = self.M_graph_encoder(block_node_features, block_edge_features)
        self.blocks_node_non=block_node_features1




        """第一步：横向Block层处理，利用MPNN模型"""
        block_prop_feature_list = [self.blocks_node_non]
        # block层传播--传递5层  (MPNN)
        for i, layer in enumerate(self._block_prop_layers):
            # node_features could be wired in here as well, leaving it out for now as
            # it is already in the inputs
            block_node_features1 = layer(block_node_features1,block_edge_features1,block_node_from_idx,block_node_to_idx)


            #print("block_node_features1 shape:", block_node_features1.shape)  # 检查 block_node_features1 的形状
            #block_node_features1 = block_node_features1.unsqueeze(0)  # 在最前面添加一个维度


            #注意力机制
            #att_weights = F.softmax(self.attention_layer(block_node_features1), dim=1)
            #att_weights = F.softmax(self.attention_layer(block_node_features1), dim=2)  # 将注意力机制的输出维度调整为与 block_node_features1 的最后一个维度匹配

            #print("att_weights shape:", att_weights.shape)  # 检查注意力权重的形状

            # 将 block_node_features1 的维度与 att_weights 匹配
            #block_node_features1 = block_node_features1.transpose(1, 2)  # 调换维度顺序
            #print("block_node_features1 shape after transpose:", block_node_features1.shape)  # 检查调换维度后的形状

            # 获取原始张量的形状
            #batch_size, num_nodes, hidden_dim = block_node_features1.size()
            #batch_size = block_node_features1.size(0)
            #num_nodes = block_node_features1.size(1)

            #att_weights = att_weights.squeeze(0)  # 将第一个维度（批次维度）展平
            #att_weights = att_weights.unsqueeze(-1)
            #att_weights = att_weights.unsqueeze(-1).expand_as(block_node_features1)
            #att_weights = att_weights.unsqueeze(-1).expand(-1, -1, block_node_features1.size(-1))
            # 手动指定每个维度的扩展倍数
            #att_weights = att_weights.unsqueeze(-1).expand(batch_size, num_nodes, self._attention_output_dim,hidden_dim)

            #block_node_features1 = torch.matmul(att_weights.transpose(1, 2),block_node_features1.unsqueeze(-1)).squeeze(-1)
            #block_node_features1 = torch.matmul(att_weights.transpose(1, 2), block_node_features1)
            #block_node_features1 = torch.matmul(att_weights, block_node_features1.transpose(1, 2)).transpose(1, 2)
            #block_node_features1 = torch.matmul(att_weights, block_node_features1.transpose(1, 2))
            #block_node_features1 = torch.matmul(block_node_features1.transpose(1, 2),att_weights)
            #block_node_features1 = torch.matmul(att_weights, block_node_features1.transpose(-2, -1)).transpose(-2, -1)
            #block_node_features1 = torch.matmul(att_weights, block_node_features1.unsqueeze(-1)).squeeze(-1)
            #block_node_features1 = torch.matmul(att_weights.unsqueeze(-1), block_node_features1.unsqueeze(1)).squeeze(1)

            #block_node_features1 = torch.matmul(att_weights.unsqueeze(1), block_node_features1).squeeze(1)

            #print("block_node_features1 shape:", block_node_features1.shape)  # 检查 block_node_features1 的形状
            #block_node_features1 = block_node_features1.squeeze(0)
            #print("block_node_features1 shape:", block_node_features1.shape)  # 检查 block_node_features1 的形状

            block_prop_feature_list.append(block_node_features1)

        self.block_layer_outputs = block_prop_feature_list # may be used for visualization
        self.block_node_prop_features = block_node_features1   # node feature after propagation----  ##主要用这个


        #if Global_flag==0:
            #node_graph_emb = torch.div(
                #unsorted_segment_sum(self.block_node_prop_features, node_graph_idx, node_graph_idx[node_graph_idx.shape[0] - 1].item() + 1),
                #torch.tensor(node_num_per_g)[:, None])

        #else:
        node_graph_emb = torch.div(
                unsorted_segment_sum(self.block_node_prop_features, node_graph_idx,node_graph_idx[node_graph_idx.shape[0] - 1].item() + 1),
                torch.tensor(node_num_per_g)[:, None].cuda())  #得到每个图的平均节点嵌入

        self.MPNN_BLOCK_OUT = self.MLP_g_node(node_graph_emb)    #最终BLck表征


        return self.MPNN_BLOCK_OUT



    def reset_n_prop_layers(self, n_prop_layers): #没调用
        """Set n_prop_layers to the provided new value.

        This allows us to train with certain number of propagation layers and
        evaluate with a different number of propagation layers.

        This only works if n_prop_layers is smaller than the number used for
        training, or when share_prop_params is set to True, in which case this can
        be arbitrarily large.

        Args:
          n_prop_layers: the new number of propagation layers to set.
        """
        self._n_prop_layers = n_prop_layers

    @property
    def n_prop_layers(self):     #没调用
        return self._n_prop_layers

    def get_layer_outputs(self): #没调用
        """Get the outputs at each prop layer."""
        if hasattr(self, '_layer_outputs'):
            return self._layer_outputs
        else:
            raise ValueError('No layer outputs available.')

   

