import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, dev):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.device = dev

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=True)  # encoder

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes),
            # nn.ReLU()
            # nn.LeakyReLU()
        ) 
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.Tanh()

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers * 2, x.size(0), self.hidden_size)).to(self.device)

        c_0 = Variable(torch.zeros(
            self.num_layers * 2, x.size(0), self.hidden_size)).to(self.device)
        # c_0: (num_layers, batch size, hidden_size)

        # Propagate input through LSTM
        h_out, _ = self.lstm(x, (h_0, c_0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # h_out = h_out.view(-1, self.hidden_size)
        h_out = self.fc(self.dropout(h_out[:, -1, :]))  # only reserve the last h_n

        h_out = self.activation(h_out)
        return h_out