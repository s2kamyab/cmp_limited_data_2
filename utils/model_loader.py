
import numpy as np
import torch
import torch.nn as nn
from utils.tst import Transformer
from statsmodels.tsa.api import VAR
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt

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
        history = train1.iloc[-self.seq_len:,:]
        if normalization == 'standard':
            x_mean = history.mean(axis=0)#, keepdim=True)  # (N, F)
            x_std = history.std(axis=0)#, keepdim=True)
            x_std[x_std == 0] = 1e-8
            x_norm = (history - np.tile(np.expand_dims(x_mean, axis = 0), [history.shape[0],1])) / np.tile(np.expand_dims(x_std, axis = 0), [history.shape[0],1])
            # --- Normalize target to same scale (optional but typical) ---
            # y_norm = (test1.iloc[:, target_index] - np.tile(np.expand_dims(x_mean[ target_index], axis=0), [test1.shape[0], 1])) / np.tile(np.expand_dims(x_std[ target_index], axis=0), [test1.shape[0], 1])  # Assuming y relates to 1st feature
            y_norm = (test1 - np.tile(np.expand_dims(x_mean, axis=0), [test1.shape[0], 1])) / np.tile(np.expand_dims(x_std, axis=0), [test1.shape[0], 1])  # Assuming y relates to 1st feature
        elif normalization == 'minmax':
            x_min = history.min(dim=1, keepdim=True)[0]
            x_min = np.tile(np.expand_dims(x_min[:,  target_index], axis=1), [1,history.shape[1],1] )
            x_max = history.max(dim=1, keepdim=True)[0]
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
        history = x_norm#pd.DataFrame(x_norm, columns = train1.columns)
        prediction = []
        gt = []
        for step in range(50):#len(test1)):
            # model = SimpleExpSmoothing(np.asarray(history.iloc[:, target_index].values))#, freq='d')
            model = Holt(np.asarray(history.iloc[:, target_index].values))
            model_fit = model.fit()
            # make prediction on validation
            tt = model_fit.forecast(steps=self.pred_len)
            # tt[abs(tt)>4]=0
            prediction.append(tt)#[:, target_index])
            gt.append(y_norm.iloc[step:step+self.pred_len,target_index].values)
            ########################### Update train  history#################################
            # move the training window
            # print(train.values.shape, train.index.shape)
            hist = history.values#[x for x in train]
            for i in range(step+1):

                obs = y_norm.iloc[i,:].values
                obs = np.expand_dims(obs , axis=0)
                hist= np.concatenate((hist, obs), axis=0)#append(obs)
                hist = hist[1:]

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
        history = train1.iloc[-self.seq_len:,:]
        if normalization == 'standard':
            x_mean = history.mean(axis=0)#, keepdim=True)  # (N, F)
            x_std = history.std(axis=0)#, keepdim=True)
            x_std[x_std == 0] = 1e-8
            x_norm = (history - x_mean) / x_std
            # --- Normalize target to same scale (optional but typical) ---
            # y_norm = (test1.iloc[:, target_index] - np.tile(np.expand_dims(x_mean[ target_index], axis=0), [test1.shape[0], 1])) / np.tile(np.expand_dims(x_std[ target_index], axis=0), [test1.shape[0], 1])  # Assuming y relates to 1st feature
            y_norm = (test1 - np.tile(np.expand_dims(x_mean, axis=0), [test1.shape[0], 1])) / np.tile(np.expand_dims(x_std, axis=0), [test1.shape[0], 1])  # Assuming y relates to 1st feature
        elif normalization == 'minmax':
            x_min = history.min(dim=1, keepdim=True)[0]
            x_min = np.tile(np.expand_dims(x_min[:,  target_index], axis=1), [1,history.shape[1],1] )
            x_max = history.max(dim=1, keepdim=True)[0]
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
        history = x_norm#pd.DataFrame(x_norm, columns = train1.columns)
        prediction = []
        gt = []
        for step in range(50):#len(test1)):
            model = VAR(endog=history)#, freq='d')
            model_fit = model.fit()
            # make prediction on validation
            tt = model_fit.forecast(model_fit.endog, steps=self.pred_len)
            tt[abs(tt)>4]=0
            prediction.append(tt[:, target_index])
            gt.append(y_norm.iloc[step:step+self.pred_len,target_index].values)
            ########################### Update train  history#################################
            # move the training window
            # print(train.values.shape, train.index.shape)
            hist = history.values#[x for x in train]
            # hist_index = history.index#[(-training_window):]
            # print(hist.index)
            for i in range(step+1):

                obs = y_norm.iloc[i,:].values
                obs = np.expand_dims(obs , axis=0)
                hist= np.concatenate((hist, obs), axis=0)#append(obs)
                hist = hist[1:]

                # hist_index = pd.DatetimeIndex(np.append(hist_index, pd.to_datetime(test1.index[i])))

                # hist_index = hist_index[1:]
            history = pd.DataFrame(data=np.array(hist), columns = train1.columns)#, index = hist_index)


            ###############################################################################
            ###############################################################################

                    # """
        # Forward pass to generate forecasts.

        # Args:
        #     x (np.ndarray): Input of shape (batch_size, seq_len, features)

        # Returns:
        #     np.ndarray: Forecasts of shape (batch_size, pred_len, features)
        # """
        # N, n_feat = x.shape
        # preds = np.zeros((self.pred_len, n_feat))

        # # for i in range(N):
        # try:
        #     series_df = np.array(x.values)#[i]
        #     model = VAR(series_df)
        #     model_fit = model.fit(maxlags=self.maxlags, ic='aic')
        #     forecast = model_fit.forecast(y=series_df, steps=self.pred_len)
        #     preds = forecast
        # except Exception as e:
        #     print(f"Error in sample : {e}")
        #     preds = np.nan
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

    else:
        raise ValueError(f"Model type '{model_type}' is not recognized.")
    # Define loss function and optimizer
    # criterion = nn.MSELoss()
    if model_type not in{'var', 'ets'} :
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adjust learning rate as needed
    else:
         optimizer = []
    
    return model, optimizer