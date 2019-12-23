from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import string
import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def save_checkpoint(self, folder="Checkpoints", file_name="checkpt.pth.tar"):
        filepath = os.path.join(folder, file_name)
        if not os.path.exists(folder):
            os.mkdir(folder)

        save_content = {
            'rnn_weights' : RNN.state_dict(self),
        }
        torch.save(save_content, filepath)

    def load_checkpoint(self, folder="Checkpoints", file_name="checkpt.pth.tar"):
        file_path = os.path.join(folder, file_name)
        if not os.path.exists(file_path): 
            raise Exception(f"No model in path {folder}")

        saved_content = torch.load(file_path)
        self.load_state_dict(saved_content['rnn_weights'])