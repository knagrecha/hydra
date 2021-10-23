import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm, TransformerEncoder

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        #print(x.shape)
        S, N = x.size()
        pos = torch.arange(S,
                           dtype=torch.long,
                           device=x.device).unsqueeze(0).expand((N, S)).t()
        return self.pos_embedding(pos)

class TokenTypeEncoding(nn.Module):
    def __init__(self, type_token_num, d_model):
        super(TokenTypeEncoding, self).__init__()
        self.token_type_embeddings = nn.Embedding(type_token_num, d_model)

    def forward(self, seq_input, token_type_input):
        S, N = seq_input.size()
        if token_type_input is None:
            token_type_input = torch.zeros((S, N),
                                           dtype=torch.long,
                                           device=seq_input.device)
        return self.token_type_embeddings(token_type_input)


class BertEmbedding(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5, transpose=True):
        super().__init__()
        self.ninp = ninp
        self.ntoken = ntoken
        self.pos_embed = PositionalEncoding(ninp)
        self.embed = nn.Embedding(ntoken, ninp)
        self.tok_type_embed = TokenTypeEncoding(2, ninp)  # Two sentence type
        self.norm = nn.LayerNorm(ninp)
        self.dropout = nn.Dropout(dropout)
        self.transpose = transpose

    def forward(self, src, token_type_input=None):

        embed = self.embed(src)
        pos_embed = self.pos_embed(src)
        tok_type_embed = self.tok_type_embed(src, token_type_input)
        if self.transpose:
            src = embed + pos_embed + tok_type_embed.transpose(0, 1)
        else:
            src = embed + pos_embed + tok_type_embed
        src = self.dropout(self.norm(src))
        return src
    
    
class BertEmbeddingNoTOK(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5, transpose=True):
        super().__init__()
        self.ninp = ninp
        self.ntoken = ntoken
        self.pos_embed = PositionalEncoding(ninp)
        self.embed = nn.Embedding(ntoken, ninp)
        self.tok_type_embed = TokenTypeEncoding(2, ninp)  # Two sentence type
        self.norm = nn.LayerNorm(ninp)
        self.dropout = nn.Dropout(dropout)
        self.transpose = transpose

    def forward(self, src):

        embed = self.embed(src)
        pos_embed = self.pos_embed(src)
        token_type_input = None
        tok_type_embed = self.tok_type_embed(src, token_type_input)
        if self.transpose:
            src = embed + pos_embed + tok_type_embed.transpose(0, 1)
        else:
            src = embed + pos_embed + tok_type_embed
        src = self.dropout(self.norm(src))
        return src

