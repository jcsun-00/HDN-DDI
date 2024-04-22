import torch

from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import (
                                GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool,
                                )

from layers import (
                    CoAttentionLayer, 
                    RESCAL, 
                    IntraGraphAttention,
                    InterGraphAttention,
                    )

def get_node(node_rep, batch, type, needed_type):
    emb_dim = node_rep.shape[-1]
    node_rep = node_rep.masked_select((type==needed_type).unsqueeze(-1))
    batch = batch.masked_select(type==needed_type)
    node_rep = node_rep.reshape(-1, emb_dim)
    return node_rep, batch


class HDN_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.n_blocks = len(blocks_params)
        
        self.initial_norm = LayerNorm(self.in_features)
        self.blocks = []
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = HDN_DDI_Block(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            in_features = head_out_feats * n_heads
        
        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)
        self.initial_conv = GATConv(self.in_features, heads_out_feat_params[0], blocks_params[0])

    def forward(self, triples):
        h_data, t_data, rels, b_graph = triples

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)
        h_data.x = self.initial_conv(h_data.x, h_data.edge_index)
        t_data.x = self.initial_conv(t_data.x, t_data.edge_index)
        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out = block(h_data,t_data,b_graph)

            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            repr_h.append(r_h)
            repr_t.append(r_t)
        
        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)
        kge_heads = repr_h
        kge_tails = repr_t
        attentions = self.co_attention(kge_heads, kge_tails)
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)
        return scores     

    def forward_with_weight(self, triples):
        h_data, t_data, rels, b_graph = triples

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)
        h_data.x = self.initial_conv(h_data.x, h_data.edge_index)
        t_data.x = self.initial_conv(t_data.x, t_data.edge_index)
        repr_h = []
        repr_t = []
        weight_h = []
        weight_t = []
        ei_h = []
        ei_t = []

        for i, block in enumerate(self.blocks):
            out = block(h_data,t_data,b_graph)

            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            repr_h.append(r_h)
            repr_t.append(r_t)
            weight_h.append(out[4][1])
            weight_t.append(out[5][1])
            ei_h = out[4][0]
            ei_t = out[5][0]
        
        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)
        kge_heads = repr_h
        kge_tails = repr_t
        attentions = self.co_attention(kge_heads, kge_tails)
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)

        weight_h = torch.stack(weight_h).mean(dim=0).mean(dim=-1, keepdim=True)
        weight_t = torch.stack(weight_t).mean(dim=0).mean(dim=-1, keepdim=True)
        return scores, ((ei_h, weight_h),(ei_t, weight_t))
    
#intra+inter
class HDN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        self.intraAtt = IntraGraphAttention(head_out_feats*n_heads)
        self.interAtt = InterGraphAttention(head_out_feats*n_heads)
        self.pool = GATConv(n_heads*head_out_feats, head_out_feats, n_heads)
        self.norm = LayerNorm(n_heads*head_out_feats)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)
    
    def forward(self, h_data,t_data,b_graph):
   
        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)
        
        h_interRep,t_interRep = self.interAtt(h_data,t_data,b_graph)
        
        h_rep = torch.cat([h_intraRep,h_interRep],1)
        t_rep = torch.cat([t_intraRep,t_interRep],1)
        h_data.x = F.elu(self.norm(h_rep, h_data.batch))
        t_data.x = F.elu(self.norm(t_rep, t_data.batch))

        h_data.x, h_weight = self.pool(h_data.x, h_data.edge_index, return_attention_weights=True)
        t_data.x, t_weight = self.pool(t_data.x, t_data.edge_index, return_attention_weights=True)
        # # Molecular-level Node
        # h_global_graph_emb = get_node(h_data.x, h_data.batch, h_data.y, 2)[0]
        # t_global_graph_emb = get_node(t_data.x, t_data.batch, t_data.y, 2)[0]

        # # SAGPool
        # h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores= self.readout(h_data.x, h_data.edge_index, batch=h_data.batch)
        # t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores= self.readout(t_data.x, t_data.edge_index, batch=t_data.batch)
        # h_global_graph_emb = global_add_pool(h_att_x, h_att_batch)
        # t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)
        # # Sum Pooling
        h_global_graph_emb = global_add_pool(h_data.x, h_data.batch)
        t_global_graph_emb = global_add_pool(t_data.x, t_data.batch)        

        return h_data, t_data, h_global_graph_emb, t_global_graph_emb, h_weight, t_weight
