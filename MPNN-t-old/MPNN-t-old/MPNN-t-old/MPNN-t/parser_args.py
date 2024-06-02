import argparse

def parse_args():
    parser = argparse.ArgumentParser('parameter receiver')

    """model hyper parameters."""
    parser.add_argument('--lr', type=float, default=6e-5, help='Learning rate')
    parser.add_argument('--graph_vec_regularizer_weight', type=float, default=1e-6, help='l2 regulizer')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch_size')
    parser.add_argument('--batch_size_test', type=int, default=10, help='Test Batch_size')
    parser.add_argument('--n_iter', type=int, default=5000000, help='Number of iteration, better be multiple of batch number')
    parser.add_argument('--block_n_prop_runs', type=int, default=5, help='Number of propagation runs')
    parser.add_argument('--function_n_prop_runs', type=int, default=3, help='Number of propagation runs')
    parser.add_argument('--contract_n_prop_runs', type=int, default=5, help='Number of propagation runs')
    parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--patience', type=int, default=4, help='Patience for early stopping')
    parser.add_argument('--hrk', type=int, default=5, help='hit ratio @ K')
    parser.add_argument('--data_type', type=int, default=1, help='choose different dataset')

    ''' Ablation Study parameters'''
    parser.add_argument('--wot_EventAttention', type=int, default=0, help='1 without event attention layer, 0 with it')
    parser.add_argument('--wot_AveragePooling', type=int, default=0, help='1 without average pooling layer, 0 with it')
    parser.add_argument('--wot_EventEmbedding', type=int, default=0, help='1 without Event Embedding, 0 with it')

    """model parameters."""
    # dimentions
    parser.add_argument('--raw_node_feat_dim', type=int, default=348, help='Dimensions of the raw node features')  # 节点特征维度
    parser.add_argument('--raw_edge_feat_dim', type=int, default=1, help='Dimensions of the raw node features')
    parser.add_argument('--node_emb_dim', type=int, default=128, help='Dimensions of the node embedding')
    #parser.add_argument('--node_emb_dim', type=int, default=64, help='Dimensions of the node embedding')
    parser.add_argument('--edge_type_emb_dim', type=int, default=64, help='Dimensions of the edge embedding')
    #parser.add_argument('--edge_type_emb_dim', type=int, default=32, help='Dimensions of the edge embedding')
    parser.add_argument('--time_emb_dim', type=int, default=64, help='Dimensions of the time embedding')
    parser.add_argument('--msg_emb_dim', type=int, default=64, help='Dimensions of the message embedding')

    #注意力机制
    #parser.add_argument('--attention_hidden_dim', type=int, default=64, help='Description of attention_hidden_dim')
    #parser.add_argument('--attention_hidden_dim', type=int, default=128, help='Description of attention_hidden_dim')
    #parser.add_argument('--attention_output_dim', type=int, default=128, help='Description of attention_output_dim')
    #parser.add_argument('--attention_output_dim', type=int, default=64, help='Description of attention_output_dim')
    #parser.add_argument('--attention_output_dim', type=int, default=32, help='Description of attention_output_dim')



    # others
    parser.add_argument('--num_edge_type', type=int, default=7, help='Number of edge types')
    parser.add_argument('--neg_num', type=int, default=49, help='Number of edge types')               #负样本数量

    """choices."""
    parser.add_argument('--gpu', type=int, default=0, help='Idx of the gpu to use')
    parser.add_argument('--loss_type', 
                        type=str, 
                        default="triplet", 
                        choices=["triplet", "pair"], 
                        help='Type of loss')
    parser.add_argument('--sim_fun', 
                        type=str, 
                        default="margin", 
                        choices=["hamming", "margin"], 
                        help='similarity calculation function choices')            
    parser.add_argument('--loss', 
                        type=str, 
                        default="margin", 
                        choices=["hamming", "margin", "bpr"], 
                        help='loss type')                  
    parser.add_argument('--node_update_type', 
                        type=str, 
                        default="mlp", 
                        choices=["gru", "mlp", "residual"], 
                        help='node update function choice')

    """judge flag."""
    parser.add_argument('--use_reverse_direction', 
                        type=int, 
                        default=0, 
                        help='whether use reverse direction node propagation')
    parser.add_argument('--reverse_dir_param_different', 
                        type=int, 
                        default=0, 
                        help='whether use different params when using reverse direction node propagation')
    parser.add_argument('--share_prop_params', 
                        type=int, 
                        default=1, 
                        help='whether share parameters in n run propagation layer')
    parser.add_argument('--save_best_model', 
                        type=int, 
                        default=0, 
                        help='whether to save the best model or not')

    """others, not used temporally."""
    parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')


    return parser.parse_args()
