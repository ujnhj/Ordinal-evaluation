# coding: utf-8

#############################################
# Consistent Cumulative Logits with ResNet-34
#############################################

# Imports

import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import numpy as np
from scipy.stats import pearsonr
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.build import build_model
from torchvision import transforms
from PIL import Image

torch.backends.cudnn.deterministic = True

TRAIN_CSV_PATH = './IC9600/train_5.csv'
TEST_CSV_PATH = './IC9600/test_5.csv'
#TEST_CSV_PATH = './cacd_test.csv'
IMAGE_PATH = './IC9600/images'


# Argparse helper

parser = argparse.ArgumentParser()
parser.add_argument('--cuda',
                    type=int,
                    default=-1)

parser.add_argument('--seed',
                    type=int,
                    default=-1)

parser.add_argument('--numworkers',
                    type=int,
                    default=3)


parser.add_argument('--outpath',
                    type=str,
                    required=True)

parser.add_argument('--imp_weight',
                    type=int,
                    default=0)

args = parser.parse_args()

NUM_WORKERS = args.numworkers

if args.cuda >= 0:
    DEVICE = torch.device("cuda:%d" % args.cuda)
else:
    DEVICE = torch.device("cpu")

if args.seed == -1:
    RANDOM_SEED = None
else:
    RANDOM_SEED = args.seed

IMP_WEIGHT = args.imp_weight

PATH = args.outpath
if not os.path.exists(PATH):
    os.mkdir(PATH)
LOGFILE = os.path.join(PATH, 'training.log')
TEST_PREDICTIONS = os.path.join(PATH, 'test_predictions.log')
TEST_ALLPROBAS = os.path.join(PATH, 'test_allprobas.tensor')

# Logging

header = []

header.append('PyTorch Version: %s' % torch.__version__)
header.append('CUDA device available: %s' % torch.cuda.is_available())
header.append('Using CUDA device: %s' % DEVICE)
header.append('Random Seed: %s' % RANDOM_SEED)
header.append('Task Importance Weight: %s' % IMP_WEIGHT)
header.append('Output Path: %s' % PATH)
header.append('Script: %s' % sys.argv[0])

with open(LOGFILE, 'w') as f:
    for entry in header:
        print(entry)
        f.write('%s\n' % entry)
        f.flush()


##########################
# SETTINGS
##########################

# Hyperparameters
learning_rate = 0.0002
num_epochs = 150

# Architecture
NUM_CLASSES = 6
BATCH_SIZE = 32
GRAYSCALE = False

df = pd.read_csv(TRAIN_CSV_PATH)
ages = df['class'].values
del df
ages = torch.tensor(ages, dtype=torch.float)


def task_importance_weights(label_array):
    uniq = torch.unique(label_array)
    num_examples = label_array.size(0)

    m = torch.zeros(uniq.shape[0])

    for i, t in enumerate(torch.arange(torch.min(uniq), torch.max(uniq))):
        m_k = torch.max(torch.tensor([label_array[label_array > t].size(0), 
                                      num_examples - label_array[label_array > t].size(0)]))
        m[i] = torch.sqrt(m_k.float())

    imp = m/torch.max(m)
    return imp


# Data-specific scheme
if not IMP_WEIGHT:
    imp = torch.ones(NUM_CLASSES-1, dtype=torch.float)
elif IMP_WEIGHT == 1:
    imp = task_importance_weights(ages)
    imp = imp[0:NUM_CLASSES-1]
else:
    raise ValueError('Incorrect importance weight parameter.')
imp = imp.to(DEVICE)


###################
# Dataset
###################

class CACDDataset(Dataset):
    """Custom Dataset for loading CACD face images"""

    def __init__(self,
                 csv_path, img_dir, transform=None):

        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['name'].values
        self.y = df['class'].values
        self.s = df['score'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        levels = [1]*label + [0]*(NUM_CLASSES - 1 - label)
        levels = torch.tensor(levels, dtype=torch.float32)
        targets = self.s[index]

        return img, label, levels, targets 

    def __len__(self):
        return self.y.shape[0]


custom_transform = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = CACDDataset(csv_path=TRAIN_CSV_PATH,
                            img_dir=IMAGE_PATH,
                            transform=custom_transform)


custom_transform2 = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_dataset = CACDDataset(csv_path=TEST_CSV_PATH,
                           img_dir=IMAGE_PATH,
                           transform=custom_transform2)

#valid_dataset = CACDDataset(csv_path=VALID_CSV_PATH,
#                            img_dir=IMAGE_PATH,
#                            transform=custom_transform2)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)

#valid_loader = DataLoader(dataset=valid_dataset,
#                          batch_size=BATCH_SIZE,
#                          shuffle=False,
#                          num_workers=NUM_WORKERS)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)



##########################
# MODEL
##########################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.num_classes = num_classes
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes-1).float())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        y = logits
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas, y


def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock,
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model


###########################################
# Initialize Cost, Model, and Optimizer
###########################################

def loss_corn(logits, y_train, num_classes):
    sets = []
    for i in range(num_classes-1):
        label_mask = y_train > i-1
        label_tensor = (y_train[label_mask] > i).to(torch.int64)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]

        if len(train_labels) < 1:
            continue

        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]

        loss = -torch.sum(F.logsigmoid(pred)*train_labels
                          + (F.logsigmoid(pred) - pred)*(1-train_labels)
                          )
        losses += loss
    return losses/num_examples


loss = torch.nn.MSELoss()
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
#model = resnet34(NUM_CLASSES, GRAYSCALE)
model = build_model()
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 


def compute_mae_and_mse(model, data_loader, device):
    mae, mse, num_examples, pc = 0, 0, 0, 0
    for i, (features, targets, levels, x) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)
        x = x.to(device)

        logits, score, probas = model(features)
        #predict_levels = probas > 0.5
        #predicted_labels = torch.sum(predict_levels, dim=1)
        num_examples += targets.size(0)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        preb = predicted_labels + score
        #mae += torch.sum(torch.abs(predicted_labels - targets))
        mae += torch.sum(torch.abs(score - x))
        #mse += torch.sum((predicted_labels - targets)**2)
        mse += torch.sum((score - x)**2)
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    score = score.cpu().numpy()
    x = x.cpu().numpy()
    pc = torch.sum(torch.tensor(pearsonr(score, x)[0]))
    return mae, mse, pc


start_time = time.time()

best_mae, best_rmse, best_epoch = 999, 999, -1
best_mse = 999
best_pc = -1
for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, targets, levels, x) in enumerate(train_loader): 
        features = features.to(DEVICE)
        targets = targets
        targets = targets.to(DEVICE)
        levels = levels.to(DEVICE) 
        x = x.to(DEVICE)
        # FORWARD AND BACK PROP
        logits, score, probas = model(features)
        
        cost1 = loss_corn(logits, targets, NUM_CLASSES)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        
        
        cost2 = loss(score.float(), x.float())
        cost = 0.2 * cost1 + 0.8 * cost2

        optimizer.zero_grad()

        cost.backward()

        # UPDATE MODEL PARAMETERS
        optimizer.step()

        # LOGGING
        if not batch_idx % 50:
            s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                 % (epoch+1, num_epochs, batch_idx,
                     len(train_dataset)//BATCH_SIZE, cost))
            print(s)
            with open(LOGFILE, 'a') as f:
                f.write('%s\n' % s)

    model.eval()
    with torch.set_grad_enabled(False):
        test_mae, test_mse, test_pc = compute_mae_and_mse(model, test_loader,
                                                   device=DEVICE)

    if test_pc > best_pc:
        best_mae, best_rmse, best_pc, best_epoch = test_mae, torch.sqrt(test_mse), test_pc, epoch
        ########## SAVE MODEL #############
        torch.save(model.state_dict(), os.path.join(PATH, 'best_model.pt'))


    s = 'MAE/RMSE/PCC: | Current Valid: %.4f/%.4f/%.4f Ep. %d | Best Valid : %.4f/%.4f/%.4f Ep. %d' % (
        test_mae, torch.sqrt(test_mse), test_pc, epoch, best_mae, best_rmse, best_pc, best_epoch)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

    s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

model.eval()
with torch.set_grad_enabled(False):  # save memory during inference

    train_mae, train_mse, train_pc = compute_mae_and_mse(model, train_loader,
                                               device=DEVICE)
    #valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader,
    #                                           device=DEVICE)
    test_mae, test_mse, test_pc = compute_mae_and_mse(model, test_loader,
                                             device=DEVICE)

    s = 'MAE/RMSE/PCC: | Train: %.4f/%.4f/%.4f | Test: %.4f/%.4f/%.4f' % (
        train_mae, torch.sqrt(train_mse), train_pc,
        #valid_mae, torch.sqrt(valid_mse),
        test_mae, torch.sqrt(test_mse), test_pc)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

s = 'Total Training Time: %.2f min' % ((time.time() - start_time)/60)
print(s)
with open(LOGFILE, 'a') as f:
    f.write('%s\n' % s)


########## EVALUATE BEST MODEL ######
model.load_state_dict(torch.load(os.path.join(PATH, 'best_model.pt')))
model.eval()

with torch.set_grad_enabled(False):
    train_mae, train_mse, train_pc = compute_mae_and_mse(model, train_loader,
                                               device=DEVICE)
    #valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader,
    #                                           device=DEVICE)
    test_mae, test_mse, test_pc = compute_mae_and_mse(model, test_loader,
                                             device=DEVICE)

    s = 'MAE/RMSE/PCC: | Best Train: %.4f/%.4f/%.4f | Best Test: %.4f/%.4f/%.4f' % (
        train_mae, torch.sqrt(train_mse), train_pc,
        #valid_mae, torch.sqrt(valid_mse),
        test_mae, torch.sqrt(test_mse), test_pc)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)


########## SAVE PREDICTIONS ######
all_pred = []
all_probas = []
with torch.set_grad_enabled(False):
    for batch_idx, (features, targets, levels, x) in enumerate(test_loader):
        
        features = features.to(DEVICE)
        logits, probas, score = model(features)
        all_probas.append(score)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        lst = [str(int(i)) for i in predicted_labels]
        all_pred.extend(lst)

torch.save(torch.cat(all_probas).to(torch.device('cpu')), TEST_ALLPROBAS)
with open(TEST_PREDICTIONS, 'w') as f:
    all_pred = ','.join(all_pred)
    f.write(all_pred)
