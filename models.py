import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(hidden_dim * 2, input_dim)

    def forward(self, x):
        # encode
        enc_out, (hn, cn) = self.encoder(x)
        # decode
        dec_out, _ = self.decoder(enc_out)
        # reconstruct
        reconstruction = self.output_layer(dec_out)
        return reconstruction, enc_out

class ScorePredictor(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, num_layers=2):
        super(ScorePredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True, # BiLSTM
            batch_first=True,
            dropout=0.5
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        scores = self.fc(lstm_out)
        return scores.squeeze(-1)
