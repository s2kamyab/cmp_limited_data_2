
import numpy as np
import torch
import torch.nn as nn

class GPT2TimeSeries(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, d_model=64, nhead=4, num_layers=4):
        super(GPT2TimeSeries, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len  # number of future steps

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        self.decoder_input = nn.Parameter(torch.randn(1, pred_len, d_model))  # learnable decoder start

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, 3)  # predict trend , seasonal, residual

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        batch_size = x.size(0)

        x_embed = self.embedding(x) + self.pos_embedding  # [batch, seq_len, d_model]
        memory = x_embed.permute(1, 0, 2)  # [seq_len, batch, d_model]

        # Prepare decoder input (repeat learnable decoder input across batch)
        tgt = self.decoder_input.expand(batch_size, -1, -1)  # [batch, pred_len, d_model]
        tgt = tgt.permute(1, 0, 2)  # [pred_len, batch, d_model]

        # Causal mask for decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.pred_len).to(x.device)

        # Decode
        decoded = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)  # [pred_len, batch, d_model]
        decoded = decoded.permute(1, 0, 2)  # [batch, pred_len, d_model]

        out = self.output_layer(decoded)  # [batch, pred_len, 3]
        return out
    

class CNNTimeSeriesModel(nn.Module):
    def __init__(self, input_channels, seq_len, pred_len):
        super(CNNTimeSeriesModel, self).__init__()
        layers = []
        self.pred_len = pred_len
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3),  # (batch, 2, 49) -> (batch, 64, 47)
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3),  # -> (batch, 32, 45)
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),  # -> (batch, 32, 43)
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),  # -> (batch, 32, 41)
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=32 * 41, out_features=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert to (batch, channels, seq_len)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x[:, -self.pred_len:, :]
class GRUTimeSeriesModel(nn.Module):
    def __init__(self, pred_len):
        super(GRUTimeSeriesModel, self).__init__()
        self.gru1 = nn.GRU(input_size=2, hidden_size=100, batch_first=True, dropout=0.0, bidirectional=False)
        self.gru2 = nn.GRU(input_size=100, hidden_size=100, batch_first=True, dropout=0.0, bidirectional=False)
        self.gru3 = nn.GRU(input_size=100, hidden_size=100, batch_first=True, dropout=0.0, bidirectional=False)
        self.gru4 = nn.GRU(input_size=100, hidden_size=100, batch_first=True, dropout=0.0, bidirectional=False)
        self.dropout = nn.Dropout(p=0.2)
        self.dense = nn.Linear(100, 1)
        self.pred_len = pred_len

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x, _ = self.gru3(x)
        x, _ = self.gru4(x)
        x = self.dropout(x)
        x = self.dense(x)  # Output shape: (batch, seq_len, 1)
        return x[:, -self.pred_len:, :]  # Keep last 3 time steps, shape: (batch, 3, 1)



