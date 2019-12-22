
import torch
import string
import torch.nn as nn
import pandas as pd
import random
from RNN import RNN
import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

n_iters = 1000000
print_every = 5000
plot_every = 1000

ALL_LETTERS = string.printable
LEARN_RATE = 0.005
LETTERS_COUNT = len(ALL_LETTERS)
# Formats:
# 0 = first last
# 1 = first middle last
# 2 = last, first
# 3 = last, first middle
# 4 = first middle_initial. last
# 5 = last, first middle_initial.
ALL_CATEGORIES = ["first last", "first middle last", "last, first", "last, first middle", "first middle_init. last", "last, first middle_init"]

def letterToIndex(letter):
    return ALL_LETTERS.find(letter)

# Just for demonstration, turn a letter into a <1 x LETTERS_COUNT> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, LETTERS_COUNT)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x LETTERS_COUNT>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, LETTERS_COUNT)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return ALL_CATEGORIES[category_i], category_i

def randomChoice(df):
    return df.loc[random.randint(0, len(df) - 1)]

def randomTrainingExample(df):
    random_choice = randomChoice(df)
    choice_format = random_choice["format"]
    choice_name = random_choice["name"]
    category_tensor = torch.tensor([choice_format], dtype=torch.long)
    line_tensor = lineToTensor(choice_name)
    return choice_format, choice_name, category_tensor, line_tensor

def train(rnn, category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    criterion = nn.NLLLoss()
    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-LEARN_RATE, p.grad.data)

    return output, loss.item()

def plot_losses(all_losses, folder:str="Results", filename:str="checkpoint.png"):
    x = list(range(len(all_losses)))
    plt.plot(x, all_losses, 'b--', label="Loss")
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{filename}")
    plt.close()

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
df = pd.read_csv("Data/balanced.csv")

n_hidden = 128
rnn = RNN(LETTERS_COUNT, n_hidden, 6)
hidden = torch.zeros(1, n_hidden)
# Keep track of losses for plotting
current_loss = 0
all_losses = []

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample(df)
    output, loss = train(rnn, category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess_i == category else '✗ (%s)' % ALL_CATEGORIES[category]
        print('%d %d%% (%s) %.4f %s / %s %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

plot_losses(all_losses,folder="generators/result",filename=f"{RNN_TEST}.png")