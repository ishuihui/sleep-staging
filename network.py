import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
from torch.nn import Parameter
import torchvision.models as models

#Graph Convolution Network
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        adj = adj.to(device)

        support = torch.matmul(input, self.weight)
        output = torch.matmul(support, adj)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, adj=np.zeros((5,5)), in_channel=5):
        super(GCNResnet, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.in_channel = in_channel
        self.pooling = nn.MaxPool2d(14, 14)

        self.lstm = nn.LSTM(input_size = 5, hidden_size = 5, num_layers = 1)
        self.gc1 = GraphConvolution(self.in_channel, 5)
        self.gc2 = GraphConvolution(5, 5)
        self.relu = nn.LeakyReLU(0.2)

        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, input, adj):
        #adj:transition rule matrix
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        state =(torch.zeros(1, input.shape[0], 5).to(device), torch.zeros(1, input.shape[0], 5).to(device))
        input, state_hidden = self.lstm(input.unsqueeze(0), state)
        input = input.squeeze(0)
        x = self.gc1(input, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]

def gcn_resnet101(num_classes, pretrained=False, adj_file=np.zeros((5,5)), in_channel=5):

    model = models.resnet101(pretrained=pretrained)
    return GCNResnet(model, num_classes, adj_file, in_channel=in_channel)

#label generate model
class label_gen(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):

        super(label_gen, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        lstm_out, self.hidden = self.lstm(lstm_out, self.hidden)
        output = self.linear(lstm_out.squeeze(0))
        output = self.linear(output)

        return output, self.hidden

#representation learning network
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(-1,24,32)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    def __init__(self,in_ch, out_ch, std=2):
        super(BasicBlock, self).__init__()
        self.stride = std

        self.conv1 = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=13, stride=1, padding=6, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=out_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size= 13, stride=std, padding=6, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=out_ch)
        self.relu2 = nn.ReLU()

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=std, padding=1)

    def forward(self, x):
        signal = x
        signal = self.conv1(signal)
        signal = self.bn1(signal)
        signal = self.relu1(signal)
        signal = self.conv2(signal)
        signal = self.bn2(signal)
        signal = self.relu2(signal)

        if self.stride!=1:
            x = self.maxpool(x)
        signal = signal+x

        return x

class ResnetBlock(nn.Module):
    def __init__(self, block, block_stride_1, block_stride_2, in_channel=1, out_channel=32, adj=np.zeros((5,5))):
        super(ResnetBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.head = self.make_head()
        self.in_channel = self.out_channel
        self.adj = adj

        self.layers1 = self.make_layers(block, block_stride_1, self.out_channel)
        self.layers2 = self.make_layers(block, block_stride_2, self.out_channel)

        self.tail = self.make_tail()
        self.tail_maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.lstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True)

        self.MaxPool1d1 = nn.MaxPool1d(kernel_size=13, stride=2, padding=6)

        self.linear = nn.Linear(in_features=375, out_features=24)

        self.SELayer = SELayer(24,4)
        self.label_gen = label_gen(5,5,1)
        self.gcn_resnet101 = gcn_resnet101(5)
        
        self.classifier = self.make_classifier()

    def init_state(self):
        state = (torch.zeros(size=(1, self.current_batch, 32)),
                 torch.zeros(size=(1, self.current_batch, 32)))
        return state

    def make_classifier(self):
        classifier = nn.Sequential()
        classifier.add_module('fc1', nn.Linear(in_features=32*24, out_features=32*6))
        classifier.add_module('bn1', nn.BatchNorm1d(32*6))
        classifier.add_module('relu1', nn.ReLU())
        classifier.add_module('fc2', nn.Linear(in_features=32*6, out_features=48))
        classifier.add_module('fc3', nn.Linear(in_features=48, out_features=5))
        return classifier

    def make_layers(self, block, block_stride, out_channel):
        layers = []
        for i in range(len(block_stride)):
            stride = block_stride[i]
            layers.append(('basicblock_{}'.format(i), block(self.in_channel, out_channel, std=stride)))
            self.in_channel = out_channel
        return nn.Sequential(OrderedDict(layers))

    def make_head(self):
        head = nn.Sequential()
        head.add_module('conv1', nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel,\
                                           kernel_size=33, stride=1, padding=16, bias=False))
        head.add_module('bn1', nn.BatchNorm1d(num_features=self.out_channel))
        head.add_module('relu1', nn.ReLU())

        #Optional
        head.add_module('conv2', nn.Conv1d(in_channels=self.out_channel, out_channels=self.out_channel, kernel_size=33, stride=1, padding=16, bias=False))
        head.add_module('bn2', nn.BatchNorm1d(num_features=self.out_channel))
        head.add_module('relu2', nn.ReLU())
        return head

    def make_tail(self):
        tail = nn.Sequential()
        tail.add_module('conv1', nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, \
                                           kernel_size=3, stride=2, padding=1, bias=False))
        tail.add_module('bn1', nn.BatchNorm1d(num_features=self.out_channel))
        tail.add_module('relu1', nn.ReLU())
        return tail

    def forward(self, x, y, train):
        device = torch.device('cuda:0' if x.is_cuda else "cpu")
        self.current_batch = len(x)
        self.state = self.init_state()
        self.state = (self.state[0].to(device), self.state[1].to(device))

        out_head = self.head(x)
        out_res1 = self.layers1(out_head)
        out_res2 = self.layers2(out_res1)

        #Optional
        out = self.tail(out_res2)
        out = self.tail_maxpool(out_res2) + out
        out_res = torch.permute(out, [0,2,1])

        out_lstm, state = self.lstm(out_res, self.state)
        out_lstm, state = self.lstm(out_lstm, state)

        out_lstm = out_lstm.reshape(self.current_batch, -1)

        out_res1= self.MaxPool1d1(out_res1)
        out_res1= self.linear(out_res1)
        out_res1 = out_res1.reshape(out_res.shape[0],-1)

        out = out_res1 + out_lstm
        #out = self.SELayer(out).reshape(self.current_batch,-1) + out #Optional
        out = self.SELayer(out).reshape(self.current_batch,-1)

        tensor_adj = torch.tensor(self.adj).float()

        linear_lear1 = nn.Linear(in_features=5, out_features=5).to(device)

        y = y.to(device)
        y = F.one_hot(y.to(torch.int64), num_classes=5).float()
        y = linear_lear1(y).to(device)
        g_out = self.gcn_resnet101(y, tensor_adj)

        out = self.classifier(out)
        _out = out

        hidden_state =(torch.zeros(1, y.shape[0], 5).to(device), torch.zeros(1, y.shape[0], 5).to(device))
        _y, y_state = self.label_gen(out, hidden_state)

        if train == True:
            #out = out + 0.3*g_out
            out = out + g_out
        else:
            #out = out + 0.3*_y
            out = out + _y

        return out, _y, _out

