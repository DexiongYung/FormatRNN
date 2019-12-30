import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    """
    Class for standard neural net in PyTorch
    input_size: The input nodes of the neural net
    hidden_size: The size of hidden layer for neural net
    output_size: The output size of the neural net
    soft_max_dim: Dimension for soft max of neural net
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, soft_max_dim: int = 1):
        super(NeuralNet, self).__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(input_size, hidden_size)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(hidden_size, output_size)
            
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=soft_max_dim)

        self.hidden_size = hidden_size #new

    def forward(self, x: list):
        """
        Forward data through the neural net
        x: Inputted tensor to forward through neural net
        """
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(nn.Sigmoid()(nn.Linear(self.hidden_size,self.hidden_size)(x)))#x = self.output(x)
        x = self.softmax(x)
            
        return x
