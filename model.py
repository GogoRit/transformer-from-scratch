import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model) # scale the embeddings by the square root of the model dimension
    
class PositionalEmbeddings(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
    
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # even indices
        pe[:, 1::2] = torch.cos(position * div_term) # odd indices
        pe = pe.unsqueeze(0) # (1, seq_len, d_model) -> add a batch dimension
        self.register_buffer('pe', pe) # register as a buffer to be saved and loaded with the model

    def forward(self, x): 
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # add positional embeddings to the input
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #multiply by this
        self.bias = nn.Parameter(torch.zeros(1)) #add this

    def forward(self, x):
        return self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias # normalize the input
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, d_ff) # linear layer to map the input to the hidden layer
        self.linear_2 = nn.Linear(d_ff, d_model) # linear layer to map the input to the output

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) # apply the feed forward block

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h # dimension of the key, query, and value
        self.q_linear = nn.Linear(d_model, d_model) # linear layer to map the input to the query
        self.k_linear = nn.Linear(d_model, d_model) # linear layer to map the input to the key
        self.v_linear = nn.Linear(d_model, d_model) # linear layer to map the input to the value
        self.out_linear = nn.Linear(d_model, d_model) # linear layer to map the input to the output
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9) # mask out the padding tokens
        attention_scores = F.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask=None):
        query = self.q_linear(q) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        key = self.k_linear(k) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        value = self.v_linear(v) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model) # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, d_model)
        
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        return self.out_linear(x)


class ResidualConnection(nn.Module): 
 
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # apply the residual connection
    
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x # return the output of the encoder block
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList, norm: LayerNormalization) -> None:
        super().__init__()
        self.layers = layers
        self.norm = norm

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) # return the output of the encoder

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x # return the output of the decoder block

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList, norm: LayerNormalization) -> None:
        super().__init__()
        self.layers = layers
        self.norm = norm

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x) # return the output of the decoder
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size) # linear layer to map the input to the output

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1) # return the output of the projection layer

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEmbeddings, tgt_pos: PositionalEmbeddings, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src_embeddings = self.src_embed(src)
        src_embeddings = self.src_pos(src_embeddings)
        encoder_output = self.encoder(src_embeddings, src_mask)
        return encoder_output # return the output of the encoder

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt_embeddings = self.tgt_embed(tgt)
        tgt_embeddings = self.tgt_pos(tgt_embeddings)
        decoder_output = self.decoder(tgt_embeddings, encoder_output, src_mask, tgt_mask)
        return decoder_output # return the output of the decoder

    def project(self, x):
        return self.projection_layer(x) # return the output of the projection layer


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, h: int = 8, N: int = 6, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    # create the positional encoding layers
    src_pos = PositionalEmbeddings(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEmbeddings(d_model, tgt_seq_len, dropout)
    # create the encoder blocks
    encoder_blocks = nn.ModuleList([EncoderBlock(MultiHeadAttention(d_model, h, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(N)])
    # create the decoder blocks
    decoder_blocks = nn.ModuleList([DecoderBlock(MultiHeadAttention(d_model, h, dropout), MultiHeadAttention(d_model, h, dropout), FeedForwardBlock(d_model, d_ff, dropout), dropout) for _ in range(N)])
    # create the encoder
    encoder = Encoder(nn.ModuleList(encoder_blocks), LayerNormalization())
    # create the decoder
    decoder = Decoder(nn.ModuleList(decoder_blocks), LayerNormalization())
    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    # initialize the transformer
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # return the transformer
    return transformer