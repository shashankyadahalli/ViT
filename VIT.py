import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x) # (batch_size, emb_size, num_patches, num_patches)
        x = x.permute(0, 2, 3, 1) # (batch_size, num_patches, num_patches, emb_size)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        
        self.q_linear = nn.Linear(emb_size, emb_size)
        self.k_linear = nn.Linear(emb_size, emb_size)
        self.v_linear = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(emb_size, emb_size)

    def forward(self, queries, keys, values, mask=None):
        queries = self.q_linear(queries) # (batch_size, num_queries, emb_size)
        keys = self.k_linear(keys) # (batch_size, num_keys, emb_size)
        values = self.v_linear(values) # (batch_size, num_keys, emb_size)

        # Split the embedding into num_heads and transpose to perform attention on each head
        queries = queries.view(queries.shape[0], queries.shape[1], self.num_heads, self.head_dim)
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.view(keys.shape[0], keys.shape[1], self.num_heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)
        values = values.view(values.shape[0], values.shape[1], self.num_heads, self.head_dim)
        values = values.permute(0, 2, 1, 3)

        # Calculate attention scores and apply mask
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Apply attention dropout and compute weighted sum of values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, values)

        # Reshape and apply output linear layer
        attn_output = attn_output.permute(0, 2, 1, 3)
        attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1], self.emb_size)
        attn_output = self.out_linear(attn_output)

        return attn_output

class FeedForward(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(emb_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, hidden_size, dropout=0.1):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(emb_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        self.feedforward = FeedForward(emb_size, hidden_size, dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention and residual connection
        attn_output = self.multihead_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feedforward network and residual connection
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class ViT(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, num_classes=10, 
                 num_layers=12, num_heads=12, hidden_size=3072, dropout=0.1):
        super().__init__()

        # Calculate number of patches
        self.num_patches = (224 // patch_size) ** 2

        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_size))

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(emb_size, num_heads, hidden_size, dropout) for _ in range(num_layers)
        ])

        # Classification head
        self.fc = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embedding(x)

        # Flatten patches into sequence
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = torch.cat((self.positional_encoding.repeat(x.shape[0], 1, 1), x), dim=1)

        # Transformer layers
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Classification head
        x = x[:, 0, :]
        x = self.fc(x)

        return x
