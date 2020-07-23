import torch
import torch.nn as nn
import torch.utils.data as tud
from torch.autograd import Variable

import string

import matplotlib.pyplot as plt
import numpy as np


# 将数据读入Dataset
class BabyNameDataset(tud.Dataset):
    def __init__(self):
        self.names = []
        self.load_data()

    def load_data(self):
        files = ["female.txt", "male.txt"]
        for filename in files:
            with open(filename, 'r', encoding='utf-8') as file:
                for line in file:
                    if not line.startswith('#'):
                        name = line.strip()
                        if len(name) != 0:
                            self.names.append(name + '$')

    def __getitem__(self, item):
        return self.names[item]

    def __len__(self):
        return len(self.names)


# 定义RNN结构
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.input2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.input2output = nn.Linear(input_size + hidden_size, output_size)
        self.output_update = nn.Linear(output_size + hidden_size, output_size)

        self.dropout = nn.Dropout()
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.input2hidden(input_combined)
        output = self.input2output(input_combined)
        output_combined = torch.cat((output, hidden), 1)
        output = self.output_update(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


# 生成用于输入的tensor
def generate_input_tensor(name):
    tensor = torch.zeros(len(name) - 1, 1, LETTER_NUM)

    for i in range(len(name) - 1):
        letter = name[i]
        tensor[i][0][all_letters.find(letter)] = 1

    return tensor


# 生成目标tensor
def generate_target_tensor(name):
    return torch.LongTensor([all_letters.find(letter) for letter in name[1:]])


# 训练
def train(input_tensor, target_tensor):
    hidden = rnn.init_hidden()
    rnn.zero_grad()
    loss = 0
    # 训练时采用整个单词在网络中过完后再进行参数更新
    for input_letter, target_letter in zip(input_tensor, target_tensor):
        output, hidden = rnn(input_letter, hidden)
        loss += criterion(output, target_letter.reshape(1))
    loss.backward()
    for para in rnn.parameters():
        para.data.add_(-learning_rate, para.grad.data)
    return loss.item() / input_tensor.size()[0]


# 使用训练完成的网络生成名字
def generate_name(start_letters):
    length_limit = 20

    indices = []
    values = []
    input_ = Variable(generate_input_tensor(start_letters + "$"))
    hidden = rnn.init_hidden()

    output_name = start_letters
    for letter in input_:
        output, hidden = rnn(letter, hidden)
    top_value, top_index = output.data.topk(5)
    indices.append(top_index)
    values.append(top_value)

    for i in range(length_limit - len(start_letters)):
        best_letter = all_letters[top_index[0][0]]
        if best_letter == '$':
            return output_name, indices, values
        output_name = output_name + best_letter
        letter = Variable(generate_input_tensor(best_letter + "$"))[0]
        output, hidden = rnn(letter, hidden)
        top_value, top_index = output.data.topk(5)
        indices.append(top_index)
        values.append(top_value)

    return output_name, indices, values

# 可视化
def visualization(start_letters, output_name, indices, values):
    area = np.pi ** 2
    tmp_y = [v - v[0][4] + 0.5 for v in values]
    y = [v.numpy()[0] for v in tmp_y]
    x = [i for i in range(1, len(y) + 1)]
    for i in range(len(y[0])):
        tmp_y = [j[i] for j in y]
        plt.scatter(x, tmp_y, s=area * (5 - i), alpha=(5 - i) / 5)
        print(x, tmp_y)
        tmp_l = [k[0][i] for k in indices]
        for j in range(len(tmp_l)):
            letter = all_letters[tmp_l[j]]
            print(x[j], tmp_y[j], letter)
            plt.text(x[j], tmp_y[j], letter, ha='center', va='center', fontsize=10)
    plt.title("Input: " + start_letters + "   Output: " + output_name)
    plt.grid(linestyle='-.')
    plt.show()


dataset = BabyNameDataset()

data_loader = tud.DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=True
)

# 参数
all_letters = string.ascii_letters + "-' $"
LETTER_NUM = len(all_letters)
learning_rate = 1e-4
epochs = 50
criterion = nn.CrossEntropyLoss()

rnn = RNN(LETTER_NUM, 128, LETTER_NUM)

all_losses = []
for epoch in range(epochs):
    print("第%s次训练开始" % (epoch + 1))
    loss_total = 0
    for i, name in enumerate(data_loader):
        input_tensor = generate_input_tensor(name[0])
        target_tensor = generate_target_tensor(name[0])
        loss_name = train(input_tensor, target_tensor)
        loss_total += loss_name
    print(loss_total)
    all_losses.append(loss_total)

# 保存训练后的模型
torch.save(rnn.state_dict(), "rnn.mdl")

# 生成名字
start_letters = "Jon"
output_name, indices, values = generate_name(start_letters)

# 可视化
visualization(start_letters, output_name, indices, values)
