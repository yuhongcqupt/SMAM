import torch
import torch.nn as nn 

import numpy as np
import copy
import math
from torch.nn import Parameter
import random
import torch.nn.functional as F




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Multi-head attention implementation.
        Args:
            d_model: Dimensionality of input features.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimensionality of each head

        # Learnable projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

    def forward(self, queries, keys, values, mask=None):
        """
        Forward pass for multi-head attention.
        Args:
            queries: Query tensor [B, input_length, d_model].
            keys: Key tensor [B, label_number, d_model].
            values: Value tensor [B, label_number, d_model].
            mask: Attention mask [B, input_length, label_number] (optional).
        Returns:
            Output tensor [B, input_length, d_model].
        """
        B, input_length, _ = queries.size()
        _, label_number, _ = keys.size()

        # Linear projections
        Q = self.W_q(queries).view(B, input_length, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, input_length, d_k]
        K = self.W_k(keys).view(B, label_number, self.num_heads, self.d_k).transpose(1, 2)    # [B, num_heads, label_number, d_k]
        V = self.W_v(values).view(B, label_number, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, label_number, d_k]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, num_heads, input_length, label_number]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)  # [B, num_heads, input_length, label_number]
        attn = self.dropout(attn)

        # Weighted sum of values
        context = torch.matmul(attn, V)  # [B, num_heads, input_length, d_k]

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(B, input_length, self.d_model)  # [B, input_length, d_model]
        output = self.W_o(context)  # [B, input_length, d_model]

        return output, attn





class Element_Wise_Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Element_Wise_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        for i in range(self.in_features):
            self.weight[i].data.uniform_(-stdv, stdv)
        if self.bias is not None:
            for i in range(self.in_features):
                self.bias[i].data.uniform_(-stdv, stdv)

    def forward(self, input):
        x = input * self.weight
        x = torch.sum(x, 2)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5) #sum(1),矩阵在维度1上相加
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


class GraphAttentionLayer11(nn.Module):
    def __init__(self, in_features, out_features, num_heads=1, dropout=0.1, alpha=0.2):
        """
        Multi-head Graph Attention Layer.
        Args:
            in_features: Input feature size.
            out_features: Output feature size per head.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            alpha: Negative slope for LeakyReLU.
        """
        super(GraphAttentionLayer11, self).__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters for each head
        self.attn_weights = nn.Parameter(torch.Tensor(num_heads, in_features, out_features))
        self.attn_bias = nn.Parameter(torch.Tensor(num_heads, out_features))

        # Learnable attention scoring parameters
        self.attn_coeffs = nn.Parameter(torch.Tensor(num_heads, 2 * out_features, 1))

        # Activation and dropout
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.attn_weights)
        nn.init.xavier_uniform_(self.attn_coeffs)
        nn.init.zeros_(self.attn_bias)

    def forward(self, h, adj):
        """
        Forward pass.
        Args:
            h: Input node features of shape [B, N, in_features].
            adj: Adjacency matrix of shape [B, N, N].
        Returns:
            Updated node features of shape [B, N, num_heads * out_features].
        """
        B, N, _ = h.size()
        H = self.num_heads
        O = self.out_features

        # Initialize output for all heads
        outputs = []

        for head in range(H):
            # Linear projection of input features
            Wh = torch.matmul(h, self.attn_weights[head])  # [B, N, O]

            # Pairwise concatenation for attention calculation
            Wh_i = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, N, O]
            Wh_j = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, O]

            # Compute attention scores
            e = torch.cat([Wh_i, Wh_j], dim=-1)  # Concatenate node pairs [B, N, N, 2*O]
            e = torch.matmul(e, self.attn_coeffs[head]).squeeze(-1)  # Compute scores [B, N, N]
            e = self.leaky_relu(e)

            # Mask non-neighbors
            e = e.masked_fill(adj == 0, float('-inf'))

            # Compute normalized attention coefficients
            alpha = F.softmax(e, dim=-1)  # [B, N, N]
            alpha = self.dropout(alpha)

            # Aggregate neighbor features
            head_output = torch.matmul(alpha, Wh)  # [B, N, O]

            # Add bias
            head_output += self.attn_bias[head]

            # Collect head output
            outputs.append(head_output)

        # Concatenate all head outputs
        return torch.cat(outputs, dim=-1)  # [B, N, H * O]

class GraphAttentionNetwork11(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads=1, dropout=0.1, alpha=0.2):
        """
        Two-layer Graph Attention Network.
        Args:
            in_features: Size of input features (dimensionality of input node features).
            hidden_features: Size of hidden layer features (per head).
            out_features: Size of output features (dimensionality of output node features).
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            alpha: Negative slope for LeakyReLU in the attention mechanism.
        """
        super(GraphAttentionNetwork11, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # First Graph Attention Layer
        self.layer1 = GraphAttentionLayer11(
            in_features=in_features,
            out_features=hidden_features,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha
        )

        # Second Graph Attention Layer
        self.layer2 = GraphAttentionLayer11(
            in_features=hidden_features * num_heads,  # Output of layer 1 has `num_heads * hidden_features`
            out_features=out_features,
            num_heads=1,  # Single head for final output
            dropout=dropout,
            alpha=alpha
        )

    def forward(self, h, adj):
        """
        Forward pass.
        Args:
            h: Input node features of shape [B, N, in_features].
            adj: Adjacency matrix of shape [B, N, N].
        Returns:
            Output node features of shape [B, N, out_features].
        """
        # Layer 1: Multi-head graph attention
        h1 = self.layer1(h, adj)  # [B, N, num_heads * hidden_features]
        h1 = F.elu(h1)  # Non-linear activation
        h1 = self.dropout(h1)  # Dropout for regularization

        # Layer 2: Final graph attention
        h2 = self.layer2(h1, adj)  # [B, N, out_features]
        return h2


# Implementing a simple Graph Attention Layer (GAT)
class GraphAttentionLayer1(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(GraphAttentionLayer1, self).__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.out_features = out_features
        # self.adj_matrix = adj_matrix  # Shape: [B, num_tags, num_tags]

        self.attn_weights = nn.Parameter(torch.Tensor(self.num_heads, in_features, out_features))
        self.attn_bias = nn.Parameter(torch.Tensor(self.num_heads, out_features))

        self.dropout = nn.Dropout(p=0.1)  # Dropout for regularization

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.attn_weights)
        nn.init.zeros_(self.attn_bias)

    def forward(self, tag_embeddings, adj_matrix):
        """
        tag_embeddings: shape [B, num_tags, in_features]
        """
        B, num_tags, _ = tag_embeddings.size()

        # Initialize output tensor
        output = torch.zeros(B, num_tags, self.out_features).to(tag_embeddings.device)

        for head in range(self.num_heads):
            # Attention mechanism
            Q = torch.matmul(tag_embeddings, self.attn_weights[head])  # [B, num_tags, out_features]
            K = Q  # Graph attention uses Q as both Q and K (self-attention style)
            V = Q  # Graph attention uses Q as both V (self-attention style)

            # Calculate attention scores
            attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, num_tags, num_tags]
            attention_scores = attention_scores / (self.out_features ** 0.5)  # Scaling

            # Apply adjacency matrix to attention scores (label relationships)
            attention_scores = attention_scores * adj_matrix  # [B, num_tags, num_tags]
            attention_weights = F.softmax(attention_scores, dim=-1)  # Apply softmax to get attention probabilities

            # Apply attention to values V
            head_output = torch.matmul(attention_weights, V)  # [B, num_tags, out_features]

            # Apply dropout for regularization
            head_output = self.dropout(head_output)

            # Aggregate head outputs
            output += head_output

        return output  # [B, num_tags, out_features]


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # [B, N, out_features]
        Wh_i = Wh.unsqueeze(1)  # [B, 1, N, out_features]
        Wh_j = Wh.unsqueeze(2)  # [B, N, 1, out_features]

        # Compute attention coefficients (pairwise scores)
        e = self.leaky_relu((Wh_i + Wh_j).sum(dim=-1))  # Broadcasting for efficient computation [B, N, N]

        # Mask non-existent edges
        e = e.masked_fill(adj == 0, float('-inf'))
        alpha = F.softmax(e, dim=-1)  # [B, N, N]
        alpha = self.dropout(alpha)

        # Aggregate features using sparse adjacency matrix
        h_prime = torch.matmul(alpha, Wh)  # [B, N, out_features]
        return h_prime



class GraphAttentionLayer2(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        """
        Single layer of Graph Attention Network (GAT).
        Args:
            in_features: Input feature size (d_model).
            out_features: Output feature size.
            dropout: Dropout probability.
            alpha: LeakyReLU negative slope.
        """
        super(GraphAttentionLayer2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout)

        # Learnable weight matrices
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))

        # LeakyReLU for attention coefficient activation
        self.leaky_relu = nn.LeakyReLU(alpha)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, adj):
        """
        Forward pass.
        Args:
            h: Node features [B, N, d_model].
            adj: Adjacency matrix [B, N, N] (0s and 1s).
        Returns:
            Updated node features [B, N, out_features].
        """
        B, N, _ = h.size()
        Wh = torch.matmul(h, self.W)  # [B, N, out_features]

        # Compute attention scores
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, out_features]
        Wh2 = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, N, out_features]
        e = self.leaky_relu(torch.matmul(torch.cat([Wh1, Wh2], dim=-1), self.a).squeeze(-1))  # [B, N, N]

        # Mask out attention scores for non-connected nodes
        e = e.masked_fill(adj == 0, float('-inf'))

        # Softmax normalization
        alpha = F.softmax(e, dim=-1)  # [B, N, N]
        alpha = self.dropout(alpha)

        # Compute new features as a weighted sum of neighbors
        h_prime = torch.matmul(alpha, Wh)  # [B, N, out_features]
        return h_prime


class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=2, dropout=0.1, alpha=0.2):
        """
        Multi-layer Graph Attention Network.
        Args:
            in_features: Input feature size.
            hidden_features: Hidden feature size.
            out_features: Output feature size.
            num_layers: Number of attention layers.
            dropout: Dropout probability.
            alpha: LeakyReLU negative slope.
        """
        super(GraphAttentionNetwork, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(GraphAttentionLayer2(in_features, hidden_features, dropout, alpha))

        # Hidden layers
        # for _ in range(num_layers - 2):
        #     self.layers.append(GraphAttentionLayer(hidden_features, hidden_features, dropout, alpha))

        # Output layer
        self.layers.append(GraphAttentionLayer2(hidden_features, out_features, dropout, alpha))

    def forward(self, x, adj):
        """
        Forward pass.
        Args:
            x: Input features [B, N, in_features].
            adj: Adjacency matrix [B, N, N].
        Returns:
            Output features [B, N, out_features].
        """
        for layer in self.layers:
            x = layer(x, adj)
            x = F.elu(x)  # Apply activation between layers
        return x


class MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout=0.1, alpha=0.2, concat=True):
        """
        Multi-head Graph Attention Layer.
        Args:
            in_features: Input feature size (d_model).
            out_features: Output feature size per head.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            alpha: LeakyReLU negative slope.
            concat: If True, concatenate head outputs. If False, average them.
        """
        super(MultiHeadGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        # Learnable parameters for each head
        self.W = nn.ParameterList([nn.Parameter(torch.empty(in_features, out_features)) for _ in range(num_heads)])
        self.a = nn.ParameterList([nn.Parameter(torch.empty(2 * out_features, 1)) for _ in range(num_heads)])
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self._init_weights()

    def _init_weights(self):
        for w, a in zip(self.W, self.a):
            nn.init.xavier_uniform_(w)
            nn.init.xavier_uniform_(a)

    def forward(self, h, adj):
        """
        Forward pass.
        Args:
            h: Node features [B, N, d_model].
            adj: Adjacency matrix [B, N, N] (0s and 1s).
        Returns:
            Updated node features [B, N, num_heads * out_features] if concat=True,
                                  [B, N, out_features] if concat=False.
        """
        B, N, _ = h.size()
        head_outputs = []

        for i in range(self.num_heads):
            Wh = torch.matmul(h, self.W[i])  # [B, N, out_features]

            # Compute attention scores
            Wh1 = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # [B, N, N, out_features]
            Wh2 = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, N, out_features]
            e = self.leaky_relu(torch.matmul(torch.cat([Wh1, Wh2], dim=-1), self.a[i]).squeeze(-1))  # [B, N, N]

            # Mask attention scores for non-edges
            e = e.masked_fill(adj == 0, float('-inf'))
            alpha = F.softmax(e, dim=-1)  # [B, N, N]
            alpha = self.dropout(alpha)

            # Compute head output
            head_output = torch.matmul(alpha, Wh)  # [B, N, out_features]
            head_outputs.append(head_output)

        if self.concat:
            # Concatenate outputs of all heads
            return torch.cat(head_outputs, dim=-1)  # [B, N, num_heads * out_features]
        else:
            # Average outputs of all heads
            return torch.mean(torch.stack(head_outputs, dim=-1), dim=-1)  # [B, N, out_features]


class MultiHeadGraphAttentionNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_heads, num_layers, dropout=0.1, alpha=0.2):
        """
        Multi-layer Graph Attention Network with multi-head attention.
        Args:
            in_features: Input feature size.
            hidden_features: Hidden feature size (per head).
            out_features: Output feature size.
            num_heads: Number of attention heads.
            num_layers: Number of attention layers.
            dropout: Dropout probability.
            alpha: LeakyReLU negative slope.
        """
        super(MultiHeadGraphAttentionNetwork, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            MultiHeadGraphAttentionLayer(in_features, hidden_features, num_heads, dropout, alpha, concat=True))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                MultiHeadGraphAttentionLayer(hidden_features * num_heads, hidden_features, num_heads, dropout, alpha,
                                             concat=True))

        # Output layer
        self.layers.append(
            MultiHeadGraphAttentionLayer(hidden_features * num_heads, out_features, num_heads, dropout, alpha,
                                         concat=False))

    def forward(self, x, adj):
        """
        Forward pass.
        Args:
            x: Input features [B, N, in_features].
            adj: Adjacency matrix [B, N, N].
        Returns:
            Output features [B, N, out_features].
        """
        for layer in self.layers:
            x = layer(x, adj)
            x = F.elu(x)  # Apply activation function between layers
        return x












def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def setEmbedingModel(d_list, d_out):
    return nn.ModuleList([Mlp(d, d, d_out) for d in d_list])


class Mlp(nn.Module):
    """ Transformer Feed-Forward Block """

    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.2):
        super(Mlp, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        # print(x.shape)
        out = self.fc1(x)
        out = self.act(out)

        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        if self.dropout1:
            out = self.dropout2(out)
        return out


class CFTransforer(nn.Module):
    def __init__(self, num_layers=2, d_model=256, nhead=4, dim_feedforward=1024,
                 dropout=0.1):
        super(CFTransforer, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, activation='relu', layer_norm_eps=1e-12)
        self.layers = get_clones(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, src):

        output = src.permute(1, 0, 2)

        for mod in self.layers:
            output = mod(output)

        if self.norm is not None:
            output = self.norm(output)

        return output.permute(1, 0, 2)


class CrossmodalAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.scale = dim // heads

        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        Q = Q.reshape(batch_size, -1, self.heads, self.scale)
        K = K.reshape(batch_size, -1, self.heads, self.scale)
        V = V.reshape(batch_size, -1, self.heads, self.scale)

        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = F.softmax(energy, dim=-1)

        V = V.permute(0, 2, 1, 3)
        attention = self.dropout(attention)
        output = torch.matmul(attention, V)

        output = output.permute(0, 2, 1, 3).reshape(batch_size, -1, self.heads * self.scale)

        return output






class Model(nn.Module):
    def __init__(self, input_len, d_model, n_layers, heads, d_list, classes_num, dropout, exponent=2):
        super().__init__()

        self.Trans = CFTransforer(num_layers=n_layers, d_model=d_model, nhead=heads, dropout=dropout)

        # Define GAT model
        self.gat11 = GraphAttentionNetwork11(
            in_features=d_model,
            hidden_features=2*d_model,
            out_features=d_model,
            num_heads=4,  # Multi-head attention for layer 1
            dropout=0.1,
            alpha=0.2
        )


        # cross-modal attention
        self.cross_modal_att = CrossmodalAttention(dim=d_model)

        sizes = [3072] + list(map(int, '512-512-512'.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        self.norm = nn.LayerNorm(d_model)

        self.flatten = nn.Flatten(start_dim=1)

        self.embeddinglayers = setEmbedingModel(d_list, d_model)
        self.view_num = input_len
        # self.CFTrans = TransformerWoDecoder(classes_num+1, d_model, n_layers, heads, dropout)

        self.classifiers = nn.ModuleList([nn.Linear(d_model,1)for _ in range(classes_num)])
        self.classifier2 = nn.Linear(d_model, classes_num)
        self.adaptive_weighting = nn.Linear(d_model,1,bias=False)
        self.weights = Parameter(torch.softmax(torch.randn([1,self.view_num,1]),dim=1))
        self.exponent = exponent
        self.cls_tokens = nn.Parameter(torch.randn(1, classes_num, d_model))
        torch.nn.init.xavier_uniform_(self.cls_tokens, gain=1)

        self.embeddings = nn.Parameter(torch.randn((classes_num, 512)))
        nn.init.normal_(self.embeddings)


        # co-occurrence matrix
        self.register_buffer('comats', torch.from_numpy(np.load('/home8T/meiqiuyu/data/multi-view multi-label/corel5k/corel5k_adj_bi.npy')).float())

        self.relu = nn.LeakyReLU(0.2)


        self.embeddings = nn.Parameter(torch.randn(classes_num, d_model))
        nn.init.normal_(self.embeddings)


        _feat_dim = d_model * 2
        self.fc = nn.Sequential(
            nn.Linear(_feat_dim, d_model),
            nn.Tanh()
        )
        self.classifier = Element_Wise_Layer(classes_num, d_model)



    def mask_view(self, x, mask_ratio):
        # random.seed()

        data_shape = x.shape  # batch x Mod x embedding_size
        repeats = data_shape[-1]

        rand_a_md = torch.rand((data_shape[0], data_shape[1])).to(x.device)
        mask_a_ = (rand_a_md < mask_ratio).float().unsqueeze(-1)
        mask_a_md = mask_a_.repeat(1, 1, repeats)

        rand_b_md = torch.rand((data_shape[0], data_shape[1])).to(x.device)
        mask_b_ = (rand_b_md < mask_ratio).float().unsqueeze(-1)
        mask_b_md = mask_b_.repeat(1, 1, repeats)

        mask_emb_a = mask_a_md * x
        mask_emb_b = mask_b_md * x

        return mask_emb_a, mask_emb_b


    def forward(self, x):


        # encoder
        for i in range(self.view_num): # 线性映射
            x[i] = self.embeddinglayers[i](x[i])
        x = torch.stack(x, dim=1) # B,view,d
        B = x.shape[0]



        # view-mask

        mask_threshold = 0.5
        mask_a, mask_b = self.mask_view(x, mask_threshold)

        x_trans_a = self.Trans(mask_a)
        x_trans_b = self.Trans(mask_b)
        x_trans = self.Trans(x)

        x_trans_view = x_trans


        # ap_a = self.projector(x_trans_a.mean(dim=1))

        ap_a = self.projector(self.flatten(x_trans_a))
        ap_b = self.projector(self.flatten(x_trans_b))

        # for ssl
        # empirical cross-correlation matrix
        c = self.bn(ap_a).T @ self.bn(ap_b)
        # # sum the cross-correlation matrix between all gpus
        c.div_(B)

        # label embedding
        cls_tokens = self.cls_tokens.expand(B, -1, -1).to(x.device)


        # x_trans_pj = self.projector(self.flatten(x_trans)).unsqueeze(1)
        # de_cls = self.CFDecoder(cls_tokens.permute(1, 0, 2), x_trans)
        # x = de_cls

        # cma, _ = self.multi_head_attention(cls_tokens, x_trans, x_trans)

        cma = self.cross_modal_att(cls_tokens, x_trans, x_trans)  # [B, classes, d_model]
        #
        cma = self.norm(cma + cls_tokens)

        # cma, alphas = self.attention(x_trans, cls_tokens)

        adj = gen_adj(self.comats).unsqueeze(0).repeat(B, 1, 1).to(x_trans.device)



        # x = self.gan_decoder(cls_tokens, x_trans, adj)

        x = self.gat11(cma, adj)

        x = self.norm(cma + x)

        pred2 = [torch.sigmoid(classifier(x[:, i])) for i, classifier in enumerate(self.classifiers)]
        pred2 = torch.stack(pred2, dim=1).squeeze(-1)

        return [pred2, pred2], x_trans_view, c


def get_model(args, input_len, d_list, d_model=768, n_layers=2, heads=4,classes_num=10,dropout=0.2,exponent=1):
    
    """
    input_len: 视图数目
    d_list: 视图的维度

    """
    assert d_model % heads == 0
    assert dropout < 1

    model = Model(input_len, d_model, n_layers, heads, d_list, classes_num, dropout, exponent)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model).to(args.device)
    else:
        model.to(args.device)

    return model
