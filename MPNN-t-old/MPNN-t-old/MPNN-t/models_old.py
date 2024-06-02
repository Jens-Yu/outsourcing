import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from parser_args import parse_args
from utils import unsorted_segment_sum
args = parse_args()

class GraphEncoder(nn.Module):   # 一：图编码器   对节点和边进行编码       输入：args   输出：节点编码、边的编码
    
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
        self._edge_emb_tab = nn.Embedding(self._n_edge_type, self._edge_emb_dim)

        self._build_model()

    def _paraParse(self, args):
        # parse parameters from args

        self._raw_node_feat_dim = args.raw_node_feat_dim1
        self._raw_edge_feat_dim = args.raw_edge_feat_dim
        self._node_emb_dim = args.node_emb_dim
        self._edge_emb_dim = args.edge_type_emb_dim
        self._n_edge_type = args.num_edge_type

    def _build_model(self):

        # Node Embedding MLP
        layer = []
        layer.append(nn.Linear(self._raw_node_feat_dim, self._node_emb_dim))
        #layer.append(nn.ReLU())
        self.MLP1 = nn.Sequential(*layer)

        # Edge Embedding MLP
        layer = []
        layer.append(nn.Linear(self._edge_emb_dim, self._edge_emb_dim))
        #layer.append(nn.ReLU())
        self.MLP2 = nn.Sequential(*layer)



    def forward(self, raw_node_features, raw_edge_features):
        """Encode node and edge features.

        Args:
          raw_node_features: [n_nodes, node_feat_dim] float tensor.
          raw_edge_features: if provided, should be [n_edges, edge_feat_dim] float
            tensor.

        Returns:r
          node_outputs: [n_nodes, node_embedding_dim] float tensor, node embeddings.
          edge_outputs: if raw_edge_features is not None and edge_hidden_sizes is not
            None, this is [n_edges, edge_embedding_dim] float tensor, edge
            embeddings; otherwise just the input raw_edge_features.
        """

        # Node Embedding
        node_outputs = self.MLP1(raw_node_features)

        # Edge Embedding 
        _edge_emb = self._edge_emb_tab(raw_edge_features)
        edge_outputs = self.MLP2(_edge_emb)

        return node_outputs, edge_outputs

class NodePropLayer(nn.Module):     #二：节点传播
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
            self.layer_norm1 = nn.LayerNorm()
            self.layer_norm2 = nn.LayerNorm()


    def _paraParse(self, args):
        # parse parameters from args
        self.node_emb_dim = args.node_emb_dim
        self.edge_type_emb_dim = args.edge_type_emb_dim
        self.msg_emb_dim = args.msg_emb_dim
        self._node_update_type = args.node_update_type
        self._use_reverse_direction = args.use_reverse_direction
        self._reverse_dir_param_different = args.reverse_dir_param_different
        self._layer_norm = False


    def build_model(self):

        """message network"""
        layer = []
        layer.append(nn.Linear(self.node_emb_dim * 2 + self.edge_type_emb_dim, self.msg_emb_dim))
        layer.append(nn.ReLU())     # 看看是否去掉
        self._message_net = nn.Sequential(*layer)

        # optionally compute message vectors in the reverse direction
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
            layer.append(nn.Linear(self.msg_emb_dim, self.node_emb_dim))
            layer.append(nn.ReLU())
            self.MLP = nn.Sequential(*layer)


    def _compute_aggregated_messages(self, 
                                     node_features,
                                     edge_features, 
                                     from_idx, 
                                     to_idx):
        """Compute aggregated messages for each node.
        Args:
          node_features: [n_nodes, input_node_state_dim] float tensor, node states.
          edge_features: if not None, should be [n_edges, edge_embedding_dim]
            tensor, edge features.
          from_idx: [n_edges] int tensor, from node indices for each edge.
          to_idx: [n_edges] int tensor, to node indices for each edge.

        Returns:
          aggregated_messages: [n_nodes, aggregated_message_dim] float tensor, the
            aggregated messages for each node.
        """
        aggregated_messages = self.graph_prop_once(node_features,
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


    def _compute_node_update(self,
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
    def graph_prop_once(node_features,
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
        aggregated_messages = self._compute_aggregated_messages(node_features, 
                                                                edge_features, 
                                                                from_idx, 
                                                                to_idx)

        updated_node = self._compute_node_update(node_features,[aggregated_messages])
        return updated_node

class GraphEmbLayer(nn.Module):            #五：图表征
    """Encoder module that projects node and edge features to some embeddings."""

    def __init__(self,
                 args,
                 name='graph-encoder'):
        """Constructor.

        Args:
          node_hidden_sizes: if provided should be a list of ints, hidden sizes of
            node encoder network, the last element is the size of the node outputs.
            If not provided, node features will pass through as is.
          edge_hidden_sizes: if provided should be a list of ints, hidden sizes of
            edge encoder network, the last element is the size of the edge outptus.
            If not provided, edge features will pass through as is.
          name: name of this module.
        """
        super(GraphEmbLayer, self).__init__()

        # this also handles the case of an empty list
        self.args = args
        self._paraParse(args)

        self._build_model()

    def _paraParse(self, args):
        # parse parameters from args
        self._event_dim = args.node_emb_dim if self.args.wot_EventEmbedding else args.node_emb_dim * 2 + args.edge_type_emb_dim + args.time_emb_dim
        

    def _build_model(self):
        self.linear_K = nn.Linear(self._event_dim, self._event_dim, bias=False)
        self.linear_Q = nn.Linear(self._event_dim, self._event_dim, bias=False)
        
        self.softmax_att = nn.Softmax(dim=1)

        # aggregation event emb to graph emb
        layer = []
        layer.append(nn.Linear(self._event_dim, self._event_dim*2))
        self.MLP_gate= nn.Sequential(*layer)

        layer = []
        layer.append(nn.Linear(self._event_dim, self._event_dim))
        self.MLP_g = nn.Sequential(*layer)

    def event_attention_layer(self, event_emb, event_num_per_graph):

        event_K = self.linear_K(event_emb)
        event_Q = self.linear_Q(event_emb)

        event_emb_per_g = torch.split(event_emb, event_num_per_graph, dim=0)
        event_K_per_g = torch.split(event_K, event_num_per_graph, dim=0)
        event_Q_per_g = torch.split(event_Q, event_num_per_graph, dim=0)

        att_emb = []

        for i, e_emb in enumerate(event_emb_per_g):
            # self attention calculation
            att_multi = event_Q_per_g[i] @ event_K_per_g[i].t()
            att_score = self.softmax_att(
                            torch.div(
                                att_multi, torch.sqrt(torch.tensor(self._event_dim).to(torch.double)))
                        )
            tmp_att_emb = att_score @ e_emb
            att_emb.append(tmp_att_emb)
        
        att_event_emb = torch.cat(att_emb, 0)

        gate_e_emb = self.MLP_gate(att_event_emb)

        gate = torch.sigmoid(gate_e_emb[:, :self._event_dim])
        event_emb = gate_e_emb[:, self._event_dim:] * gate

        return event_emb


    def forward(self, event_emb, graph_idx, event_num_per_graph):
        """Encode graph.

        Args:
          event_emb: [n_events, node_feat_dim*2 + edge_feat_dim + time_feat_dim] float tensor.
          graph_idx: [n_events] index tensor, each dim means which graph did the event belongs to

        Returns:
          graph_emb: [n_graphs, graph_emb_dim] float tensor, graph embeddings.

        """
        if not self.args.wot_EventAttention:
            event_emb = self.event_attention_layer(event_emb, event_num_per_graph)

        if self.args.wot_AveragePooling:
            graph_emb = unsorted_segment_sum(event_emb, graph_idx, graph_idx[graph_idx.shape[0]-1].item()+1)
        else:
            graph_emb = torch.div(unsorted_segment_sum(event_emb, graph_idx, graph_idx[graph_idx.shape[0]-1].item()+1), 
                                  torch.tensor(event_num_per_graph)[:,None])

        graph_emb = self.MLP_g(graph_emb)

        # average pooling
        # gpu version
        '''
        graph_emb = torch.div(unsorted_segment_sum(att_event_emb, graph_idx, graph_idx[graph_idx.shape[0]-1].item()+1), 
                              torch.tensor(event_num_per_graph)[:,None].cuda())
        
        # cpu version
        graph_emb = torch.div(unsorted_segment_sum(att_event_emb, graph_idx, graph_idx[graph_idx.shape[0]-1].item()+1), 
                              torch.tensor(event_num_per_graph)[:,None])    
        '''

        return graph_emb

class EventEmbLayer(nn.Module):             #四：事件建模

    def __init__(self, 
                args, 
                t_encoder, 
                name='event-emb'):

        super(EventEmbLayer, self).__init__()

        self.args = args
        self._paraParse(args)
        self.time_encoder = t_encoder

    def _paraParse(self, args):
        # parse parameters from args
        self._time_dim = args.time_emb_dim

    def forward(self, 
                update_node_emb, 
                edge_emb, 
                from_idx, 
                to_idx, 
                t):

        from_emb = update_node_emb[from_idx]
        to_emb = update_node_emb[to_idx]
        time_emb = self.time_encoder(t)

        if self.args.wot_EventEmbedding:
            event = [from_emb]
            event_emb = torch.cat(event, dim = -1)
        else:
            event = [from_emb, edge_emb, to_emb, time_emb]
            event_emb = torch.cat(event, dim = -1)
        
        
        return event_emb

class TimeEncoder(nn.Module):          #三：时序编码器
    # Time Encoding proposed by TGAT
    def __init__(self, time_dim):
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        self.build_model()
    
    def build_model(self):
        self.linear = nn.Linear(1, 1)   # output dimention: 1
        self.periodic = nn.Linear(1, self.time_dim-1)

    def forward(self, t):
        # output has shape [edge_num, dimension]
        li_dim = self.linear(t)
        peri_dim = torch.sin(self.periodic(t))
        t2vec = torch.cat((li_dim, peri_dim), dim=1)

        return t2vec

class MLM(nn.Module):                    #final: 构造模型
    def __init__(self,
                args,
                graph_encoder,
                block_node_prop_module,
                event_Module,
                graph_Module):
        super(MLM,self).__init__()

        # parse parameters
        self._paraParse(args)


        # get modules needed
        self.M_graph_encoder = graph_encoder
        self.M_block_prop_layer = block_node_prop_module
        # n run prop layers list
        self._block_prop_layers = []
        self._block_prop_layers = nn.ModuleList()

        self.M_event_emb_layer = event_Module
        self.M_graph_emb_layer = graph_Module

        # build model
        self.build_model()

    def _paraParse(self, args):
        """parse parameters from args.
        
        Args: 
            args: an parameter class, get all hyper parameters from it
        """
        self.args = args
        self._n_run_block_prop = args.block_n_prop_runs
        self._share_prop_params = args.share_prop_params


    def build_model(self):
        if len(self._block_prop_layers) < self._n_run_block_prop:
            # build the layers
            for i in range(self._n_run_block_prop):
                if i == 0 or not self._share_prop_params:
                    one_prop_layer = self.M_block_prop_layer(self.args)
                else:
                    one_prop_layer = self._block_prop_layers[0]
                self._block_prop_layers.append(one_prop_layer)

    def forward(self,
                node_features, 
                edge_features,
                node_from_idx,
                node_to_idx,
                t,
                graph_idx,
                event_num_per_graph):
        """Compute graph representations.

        Args:

        Returns:
          graph_representations: [batch_num, graph_representation_dim] float tensor,
            graph representations.
        """

        """node && edge encode module."""
        block_node_features, block_edge_features= self.M_graph_encoder(node_features, edge_features)

        """node propagation module."""
        block_prop_feature_list = [block_node_features]

        for i, layer in enumerate(self._block_prop_layers):
            # node_features could be wired in here as well, leaving it out for now as
            # it is already in the inputs
            block_node_features = layer(block_node_features,block_edge_features,node_from_idx,node_to_idx)

            block_prop_feature_list.append(block_node_features)

        self.block_layer_outputs = block_prop_feature_list # may be used for visualization
        self.block_node_prop_features = block_node_features   # node feature after propagation
        
        # self.event_features = self.M_event_emb_layer(self.block_node_prop_features,
        #                                              block_edge_features,
        #                                              node_from_idx,
        #                                              node_to_idx,
        #                                              t)

        self.graph_features = self.M_graph_emb_layer(block_node_features, graph_idx, event_num_per_graph)
        return self.graph_features

    def reset_n_prop_layers(self, n_prop_layers):
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
    def n_prop_layers(self):
        return self._n_prop_layers

    def get_layer_outputs(self):
        """Get the outputs at each prop layer."""
        if hasattr(self, '_layer_outputs'):
            return self._layer_outputs
        else:
            raise ValueError('No layer outputs available.')

   

