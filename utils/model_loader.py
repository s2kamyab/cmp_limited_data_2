
import numpy as np
import torch
import torch.nn as nn
from utils.tst import Transformer
from statsmodels.tsa.api import VAR
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from transformers import GPT2Model, GPT2Config
from transformers import InformerForPrediction, InformerConfig
from transformers import AutoformerForPrediction, AutoformerConfig
def autocorrelation(query_states, key_states):
    """
    Computes autocorrelation(Q,K) using `torch.fft`. 
    Think about it as a replacement for the QK^T in the self-attention.
    
    Assumption: states are resized to same shape of [batch_size, time_length, embedding_dim].
    """
    query_states_fft = torch.fft.rfft(query_states, dim=1)
    key_states_fft = torch.fft.rfft(key_states, dim=1)
    attn_weights = query_states_fft * torch.conj(key_states_fft)
    attn_weights = torch.fft.irfft(attn_weights, dim=1)  
    
    return attn_weights
class DecompositionLayer(nn.Module):
    """
    Returns the trend and the seasonal parts of the time series.
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0) # moving average 

    def forward(self, x):
        """Input shape: Batch x Time x EMBED_DIM"""
        # padding on the both ends of time series
        num_of_pads = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, num_of_pads, 1)
        end = x[:, -1:, :].repeat(1, num_of_pads, 1)
        x_padded = torch.cat([front, x, end], dim=1)

        # calculate the trend and seasonal part of the series
        x_trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend
        return x_seasonal, x_trend
class InformerTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, pred_len, output_dim, seq_len=96, label_len=48, d_model=512):
        super(InformerTimeSeriesModel, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.label_len = label_len

        config = InformerConfig(
            prediction_length=pred_len,
            context_length=seq_len,
            label_length=label_len,
            input_size=input_dim,
            d_model=d_model,
            target_size=output_dim
        )
        self.model = InformerForPrediction(config)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        Returns:
            (batch_size, pred_len, output_dim)
        """
        inputs = {
            "past_values": x,
            'past_observed_mask': torch.ones(x.size(0), self.seq_len, dtype=torch.bool).to(x.device),
            "past_time_features": torch.zeros_like(x),
            "future_time_features": torch.zeros(x.size(0), self.pred_len, x.size(2)).to(x.device),
        }
        out = self.model(**inputs)
        return out.predictions  # (batch, pred_len, output_dim)
    
class AutoformerTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, pred_len, output_dim, seq_len,  d_model=512):
        super(AutoformerTimeSeriesModel, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        # self.label_len = label_len
        self.label_len = int(seq_len//2)

        config = AutoformerConfig(
            prediction_length=pred_len,
            context_length=seq_len,
            label_length=self.label_len,
            input_size=input_dim,
            d_model=d_model,
            target_size=output_dim
        )
        self.model = AutoformerForPrediction(config)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        Returns:
            (batch_size, pred_len, output_dim)
        """
        inputs = {
            "past_values": x,
            'past_observed_mask': torch.ones(x.size(0), self.seq_len, dtype=torch.bool).to(x.device),
            "past_time_features": torch.zeros_like(x),
            "future_time_features": torch.zeros(x.size(0), self.pred_len, x.size(2)).to(x.device),
        }
        out = self.model(**inputs)
        return out.predictions  # (batch, pred_len, output_dim)    
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from huggingface_hub import login
# login(token="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")
# login(token="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")
# class FinGPTTimeSeriesForecast(nn.Module):
#     def __init__(self, 
#                  model_name: str = "FinGPT/FinGPT-Forecaster", 
#                  input_dim: int = None, 
#                  pred_len: int = 3, 
#                  output_dim: int = 1,
#                  device: str = None):
#         """
#         Adapt FinGPT forecaster for numerical time-series.
        
#         Args:
#             model_name: pretrained FinGPT model checkpoint
#             input_dim: # of features per timestep (optional, for assertion)
#             pred_len: steps ahead to forecast
#             output_dim: # of target dimensions
#         """
#         super().__init__()
#         self.pred_len = pred_len
#         self.output_dim = output_dim
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Load pretrained FinGPT forecasting model
#         self.model = AutoModelForSequenceClassification.from_pretrained(model_name , token="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")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
#         self.model.to(self.device)
        
#         if input_dim and self.model.config.num_labels != (pred_len * output_dim):
#             raise ValueError("Model pred_len/output_dim doesn't match pretrained checkpoint.")
        
#     def forward(self, x: torch.Tensor):
#         """
#         x: shape (batch_size, seq_len, input_dim) numeric tensor
        
#         Steps:
#         1. Quantize input to text tokens
#         2. Tokenize and feed FinGPT
#         3. Decode logits back into numeric forecasts
#         """
#         batch, seq_len, feat = x.shape
        
#         # 1. Quantization (example: simple rounding or binning; needs domain-specific tuning)
#         quant = (x.detach().cpu().numpy() * 100).round().astype(int)
#         text_sequences = [" ".join(map(str, row.flatten())) for row in quant]
        
#         # 2. Tokenize and model inference
#         encoded = self.tokenizer(text_sequences, padding=True, truncation=True, return_tensors="pt")
#         encoded = {k: v.to(self.device) for k, v in encoded.items()}
#         outputs = self.model(**encoded)
        
#         # 3. Reshape logits → [batch, pred_len, output_dim]
#         logits = outputs.logits  # shape: (batch, pred_len*output_dim)
#         fc_output = logits.view(batch, self.pred_len, self.output_dim)
        
#         # Convert to tensor
#         return torch.sigmoid(fc_output)  # or apply appropriate activation
class GPT2TimeSeriesModel(nn.Module):
    def __init__(self, input_dim, pred_len, output_dim, d_model=768):
        super(GPT2TimeSeriesModel, self).__init__()
        self.pred_len = pred_len
        self.input_projection = nn.Linear(input_dim, d_model)

        # Load GPT-2 configuration and model (no embedding layer, we handle it ourselves)
        config = GPT2Config(
            n_embd=d_model,
            n_layer=6,
            n_head=8,
            n_positions=1024,
            n_ctx=1024,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1
        )
        self.transformer = GPT2Model(config)
        # Freeze the first 4 layers of GPT-2
        for i in range(5):
            for param in self.transformer.h[i].parameters():
                param.requires_grad = False

        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        x: shape (batch_size, seq_len, input_dim)
        """
        # Project input to GPT2's expected hidden size
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # GPT-2 expects input shape: (batch, seq_len, d_model) → need to pass attention mask optionally
        outputs = self.transformer(inputs_embeds=x)
        hidden_states = outputs.last_hidden_state  # shape: (batch, seq_len, d_model)

        # Final linear projection to output_dim
        out = self.output_layer(hidden_states)  # (batch, seq_len, output_dim)

        # Return only the last pred_len steps
        return out[:, -self.pred_len:, :]  # shape: (batch, pred_len, output_dim)
class ETSTimeSeries:
    def __init__(self, pred_len, seq_len):
        """
        VAR model wrapper for multivariate time series forecasting.

        Args:
            pred_len (int): Number of future steps to forecast.
            maxlags (int): Maximum lags to consider for VAR model.
        """
        self.pred_len = pred_len
        self.seq_len = seq_len
        # self.test_size = test_size
        # self.maxlags = maxlags

    def forward(self, train1, test1, normalization, target_index):
        self.seq_len = int(train1.shape[0])
        history = train1.iloc[-self.seq_len:,:]
        prediction = []
        gt = []
        for step in range(50):#len(test1)):
            if normalization == 'standard':
                x_mean = history.mean(axis=0)#, keepdim=True)  # (N, F)
                x_std = history.std(axis=0)#, keepdim=True)
                x_std[x_std == 0] = 1e-8
                x_norm = (history - x_mean) / x_std
                # --- Normalize target to same scale (optional but typical) ---
                # y_norm = (test1.iloc[:, target_index] - np.tile(np.expand_dims(x_mean[ target_index], axis=0), [test1.shape[0], 1])) / np.tile(np.expand_dims(x_std[ target_index], axis=0), [test1.shape[0], 1])  # Assuming y relates to 1st feature
                y_norm = (test1 - np.tile(np.expand_dims(x_mean, axis=0), [test1.shape[0], 1])) / np.tile(np.expand_dims(x_std, axis=0), [test1.shape[0], 1])  # Assuming y relates to 1st feature
            elif normalization == 'minmax':
                x_min = history.min(axis=0)#dim=1, keepdim=True)[0]
                x_min = np.tile(np.expand_dims(x_min[:,  target_index], axis=1), [1,history.shape[1],1] )
                x_max = history.max(axis=0)#(dim=1, keepdim=True)[0]
                x_max = np.tile(np.expand_dims(x_max[:,  target_index], axis=1), [1,history.shape[1],1] )
                x_norm = (history - x_min) / (x_max - x_min + 1e-8)
                # y_norm = (test1.iloc[:, target_index]  - x_min.iloc[:, target_index] ) / (x_max - x_min + 1e-8)
                y_norm = (test1 - x_min ) / (x_max - x_min + 1e-8)
            elif normalization == 'relative':
                ref = history.iloc[-1, :]  # last time step
                ref[ref == 0] = 1e-8
                x_norm = history / np.tile(np.expand_dims(ref,axis=0),[history.shape[0],1])  # assuming x relates to 1st feature
                # y_norm = test1.iloc[:,target_index] / np.tile(np.expand_dims(ref.iloc[target_index], axis=0), [test1.shape[0],1] ) # assuming y relates to 1st feature
                y_norm = test1 / np.tile(np.expand_dims(ref, axis=0), [test1.shape[0],1] ) # assuming y relates to 1st feature
            history = x_norm
            model = Holt(np.asarray(history.iloc[:, target_index].values))
            model_fit = model.fit()
            # make prediction on validation
            tt = model_fit.forecast(steps=self.pred_len)
            # tt[abs(tt)>4]=0
            prediction.append(tt)#[:, target_index])
            gt.append(y_norm.iloc[step:step+self.pred_len,target_index].values)
            ########################### Update train  history#################################
            hist = train1.iloc[-self.seq_len:,:]
            # hist_index = history.index#[(-training_window):]
            # print(hist.index)
            for i in range(step+1):

                # obs = y_norm.iloc[i,:].values
                obs = test1.iloc[i,:].values
                # if normalization == 'standard':
                #     obs = (obs - x_mean) / (x_std + 1e-8)
                # elif normalization == 'minmax':
                #     obs = (obs - x_min) / (x_max - x_min + 1e-8)  
                # elif normalization == 'relative':
                #     obs = obs / ref   
                obs = np.expand_dims(obs , axis=0)
                
                hist= np.concatenate((hist, obs), axis=0)#append(obs)
                hist = hist[1:]

                # hist_index = pd.DatetimeIndex(np.append(hist_index, pd.to_datetime(test1.index[i])))

                # hist_index = hist_index[1:]
            history = pd.DataFrame(data=np.array(hist), columns = train1.columns)#, index = hist_index)

        return np.array(prediction), np.array(gt)
    
class VARTimeSeries:
    def __init__(self, pred_len, seq_len):
        """
        VAR model wrapper for multivariate time series forecasting.

        Args:
            pred_len (int): Number of future steps to forecast.
            maxlags (int): Maximum lags to consider for VAR model.
        """
        self.pred_len = pred_len
        self.seq_len = seq_len
        # self.test_size = test_size
        # self.maxlags = maxlags

    def forward(self, train1, test1, normalization, target_index):
        self.seq_len = int(train1.shape[0]/3)
        history = train1.iloc[-self.seq_len:,:]
        prediction = []
        gt = []
        for step in range(50):#len(test1)):
            if normalization == 'standard':
                x_mean = history.mean(axis=0)#, keepdim=True)  # (N, F)
                x_std = history.std(axis=0)#, keepdim=True)
                x_std[x_std == 0] = 1e-8
                x_norm = (history - x_mean) / x_std
                # --- Normalize target to same scale (optional but typical) ---
                # y_norm = (test1.iloc[:, target_index] - np.tile(np.expand_dims(x_mean[ target_index], axis=0), [test1.shape[0], 1])) / np.tile(np.expand_dims(x_std[ target_index], axis=0), [test1.shape[0], 1])  # Assuming y relates to 1st feature
                y_norm = (test1 - np.tile(np.expand_dims(x_mean, axis=0), [test1.shape[0], 1])) / np.tile(np.expand_dims(x_std, axis=0), [test1.shape[0], 1])  # Assuming y relates to 1st feature
            elif normalization == 'minmax':
                x_min = history.min(axis=0)#dim=1, keepdim=True)[0]
                x_min = np.tile(np.expand_dims(x_min[:,  target_index], axis=1), [1,history.shape[1],1] )
                x_max = history.max(axis=0)#(dim=1, keepdim=True)[0]
                x_max = np.tile(np.expand_dims(x_max[:,  target_index], axis=1), [1,history.shape[1],1] )
                x_norm = (history - x_min) / (x_max - x_min + 1e-8)
                # y_norm = (test1.iloc[:, target_index]  - x_min.iloc[:, target_index] ) / (x_max - x_min + 1e-8)
                y_norm = (test1 - x_min ) / (x_max - x_min + 1e-8)
            elif normalization == 'relative':
                ref = history.iloc[-1, :]  # last time step
                ref[ref == 0] = 1e-8
                x_norm = history / np.tile(np.expand_dims(ref,axis=0),[history.shape[0],1])  # assuming x relates to 1st feature
                # y_norm = test1.iloc[:,target_index] / np.tile(np.expand_dims(ref.iloc[target_index], axis=0), [test1.shape[0],1] ) # assuming y relates to 1st feature
                y_norm = test1 / np.tile(np.expand_dims(ref, axis=0), [test1.shape[0],1] ) # assuming y relates to 1st feature
            history = x_norm
            model = VAR(endog=history)#).iloc[:, target_index])#, freq='d')
            model_fit = model.fit()
            # make prediction on validation
            tt = model_fit.forecast(model_fit.endog, steps=self.pred_len)
            # tt[abs(tt)>4]=0
            prediction.append(tt[:, target_index])
            gt.append(y_norm.iloc[step:step+self.pred_len,target_index].values)
            print(tt[:, target_index])
            print(y_norm.iloc[step:step+self.pred_len,target_index].values)
            ########################### Update train  history#################################
            # move the training window
            # print(train.values.shape, train.index.shape)
            # hist = history.values#[x for x in train]
            hist = train1.iloc[-self.seq_len:,:]
            # hist_index = history.index#[(-training_window):]
            # print(hist.index)
            for i in range(step+1):

                # obs = y_norm.iloc[i,:].values
                obs = test1.iloc[i,:].values
                # if normalization == 'standard':
                #     obs = (obs - x_mean) / (x_std + 1e-8)
                # elif normalization == 'minmax':
                #     obs = (obs - x_min) / (x_max - x_min + 1e-8)  
                # elif normalization == 'relative':
                #     obs = obs / ref   
                obs = np.expand_dims(obs , axis=0)
                
                hist= np.concatenate((hist, obs), axis=0)#append(obs)
                hist = hist[1:]

                # hist_index = pd.DatetimeIndex(np.append(hist_index, pd.to_datetime(test1.index[i])))

                # hist_index = hist_index[1:]
            history = pd.DataFrame(data=np.array(hist), columns = train1.columns)#, index = hist_index)

        return np.array(prediction), np.array(gt)
    

class GPT2TimeSeries(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, pred_len, d_model=64, nhead=4, num_layers=4):
        super(GPT2TimeSeries, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len  # number of future steps

        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        self.decoder_input = nn.Parameter(torch.randn(1, pred_len, d_model))  # learnable decoder start

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, output_dim)  # predict trend , seasonal, residual

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        # x = x.float()  # Ensure input is float
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
    def __init__(self, input_channels, output_dim, seq_len, pred_len):
        super(CNNTimeSeriesModel, self).__init__()
        layers = []
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3),  # (batch, 2, 49) -> (batch, 64, 47)
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
        self.dense = None#nn.Linear( out_features=output_dim)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = x.float()  # Ensure input is float
        x = x.permute(0, 2, 1)  # Convert to (batch, channels, seq_len)
        x = self.conv_layers(x)
        # x = x.permute(0, 2, 1)
        x = self.flatten(x)
        in_features = x.shape[-1]
        # if self.dense is None:
        in_features = x.shape[-1]
        self.dense = nn.Linear(in_features=in_features, out_features=self.pred_len*self.output_dim).to(x.device)

        x = self.dense(x)   
        x = x.view(-1, self.pred_len, self.output_dim)  # (batch, pred_len, output_dim)
        return x
    
class GRUTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, pred_len, output_dim=1):
        super(GRUTimeSeriesModel, self).__init__()
        self.gru1 = nn.GRU(input_size=input_dim, hidden_size=100, batch_first=True, dropout=0.0, bidirectional=False)
        self.gru2 = nn.GRU(input_size=100, hidden_size=100, batch_first=True, dropout=0.0, bidirectional=False)
        self.gru3 = nn.GRU(input_size=100, hidden_size=100, batch_first=True, dropout=0.0, bidirectional=False)
        self.gru4 = nn.GRU(input_size=100, hidden_size=100, batch_first=True, dropout=0.0, bidirectional=False)
        self.dropout = nn.Dropout(p=0.2)
        self.dense = nn.Linear(100, output_dim)
        self.pred_len = pred_len

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x, _ = self.gru3(x)
        x, _ = self.gru4(x)
        x = self.dropout(x)
        x = self.dense(x)  # Output shape: (batch, seq_len, 1)
        return x[:, -self.pred_len:, :]  # Keep last 3 time steps, shape: (batch, 3, 1)
    
class LSTMTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, pred_len, output_dim=1):
        super(LSTMTimeSeriesModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=100, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.2)
        self.lstm2 = nn.LSTM(input_size=100, hidden_size=100, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=100, hidden_size=100, batch_first=True)
        self.lstm4 = nn.LSTM(input_size=100, hidden_size=100, batch_first=True)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dense = nn.Linear(100, output_dim)
        self.pred_len = pred_len

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = self.dropout2(x)
        x = self.dense(x)  # Output shape: (batch, seq_len, 1)
        return x[:, -self.pred_len:, :]  # Keep last 3 time steps, shape: (batch, 3, 1)

class SimpleRNNTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, pred_len, output_dim):
        super(SimpleRNNTimeSeriesModel, self).__init__()
        self.rnn1 = nn.RNN(input_size=input_dim, hidden_size=100, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.2)
        self.rnn2 = nn.RNN(input_size=100, hidden_size=100, batch_first=True)
        self.rnn3 = nn.RNN(input_size=100, hidden_size=100, batch_first=True)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dense = nn.Linear(100, output_dim)
        self.pred_len = pred_len

    def forward(self, x):
        x, _ = self.rnn1(x)
        x = self.dropout1(x)
        x, _ = self.rnn2(x)
        x, _ = self.rnn3(x)
        x = self.dropout2(x)
        x = self.dense(x)  # Output shape: (batch, seq_len, 1)
        return x[:, -self.pred_len:, :]  # Keep last 3 time steps, shape: (batch, 3, 1)
    

#Building the TimesNet Model with corrected architecture
class TimesNet(nn.Module):
    def __init__(self, input_features, sequence_length, output_length, num_layers=4):
        super(TimesNet, self).__init__()
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_features if i == 0 else 64
            self.conv_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(64 * sequence_length, output_length)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert to (batch, channels, seq_len)
        for conv in self.conv_layers:
            x = torch.relu(conv(x))
        x = self.flatten(x)
        x = self.dense(x)
        return x

def load_model(model_type, input_dim, output_dim, seq_len, pred_len, lr=0.0001):
    if model_type == 'finspd_transformer':
        chunk_mode = None
        output_length = pred_len#3
        d_output = output_length * input_dim# prediction length be 6, this is confirmed
        d_model = 32 # Lattent dim
        q = 8 # Query size
        v = 8 # Value size
        h = 8 # Number of heads
        N = 4 # Number of encoder and decoder to stack
        attention_size = 50 # Attention window size 这个和形状没有关系
        dropout = 0.1 # Dropout rate
        pe = 'regular' # Positional encoding

        # Creating the model
        model = Transformer(input_dim, d_model, d_output, q, v, h, N, attention_size=attention_size, dropout=dropout, chunk_mode=chunk_mode, pe=pe)
        
    elif model_type == 'GPT2like_transformer':

        model = GPT2TimeSeries(input_dim,output_dim, seq_len, pred_len)
    elif model_type == 'cnn':
        model = CNNTimeSeriesModel(input_dim,output_dim, seq_len, pred_len)
    elif model_type == 'gru':
        model = GRUTimeSeriesModel(input_dim, pred_len,output_dim)
    elif model_type == 'lstm':
        model = LSTMTimeSeriesModel(input_dim,pred_len,output_dim)
    elif model_type == 'rnn':
        model = SimpleRNNTimeSeriesModel(input_dim,pred_len,output_dim)
    elif model_type == 'times_net':
        model = TimesNet(input_features=input_dim, sequence_length=seq_len, output_length=pred_len)
    elif model_type == 'var':
        # Initialize VAR model
        model = VARTimeSeries(seq_len=seq_len, pred_len=pred_len)
    elif model_type == 'ets':
        # Initialize VAR model
        model = ETSTimeSeries(seq_len=seq_len, pred_len=pred_len)
    elif model_type == 'pretrained_gpt2':
        model = GPT2TimeSeriesModel(input_dim, pred_len, output_dim)
    elif model_type == 'pretrained_autoformer':
        input_dim, pred_len, output_dim, seq_len
        model = AutoformerTimeSeriesModel(input_dim, pred_len, output_dim, seq_len)
    elif model_type == 'pretrained_informer':
        model = InformerTimeSeriesModel(input_dim=input_dim, pred_len=pred_len, output_dim=output_dim)
    else:
        raise ValueError(f"Model type '{model_type}' is not recognized.")
    # Define loss function and optimizer
    # criterion = nn.MSELoss()
    if model_type not in ['var', 'ets'] :
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adjust learning rate as needed
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
    else:
         optimizer = []
    
    return model, optimizer