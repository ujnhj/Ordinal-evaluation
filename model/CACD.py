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
from model.build import build_model
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image

torch.backends.cudnn.deterministic = True

TRAIN_CSV_PATH = './dataset/cacd_train.csv'
VALID_CSV_PATH = './dataset/cacd_valid.csv'
TEST_CSV_PATH = './dataset/cacd_test.csv'
IMAGE_PATH = './CACD2000'


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
learning_rate = 0.0001
num_epochs = 200

# Architecture
NUM_CLASSES = 49
#NUM_CLASSES1 = 11
BATCH_SIZE = 32
GRAYSCALE = False

df = pd.read_csv(TRAIN_CSV_PATH, index_col=0)
ages = df['age'].values
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

        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['filename'].values
        self.y = df['score'].values
        self.age = df['class'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.age[index]
        levels = [1]*label + [0]*(NUM_CLASSES - 1 - label)
        levels = torch.tensor(levels, dtype=torch.float32)
        age = self.age[index]

        return img, label, levels, age

    def __len__(self):
        return self.y.shape[0]


custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.CenterCrop((120, 120)),
                                       transforms.Resize((224, 224)),
                                       transforms.ToTensor()])

train_dataset = CACDDataset(csv_path=TRAIN_CSV_PATH,
                            img_dir=IMAGE_PATH,
                            transform=custom_transform)


custom_transform2 = transforms.Compose([transforms.Resize((128, 128)),
                                        transforms.CenterCrop((120, 120)),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor()])

test_dataset = CACDDataset(csv_path=TEST_CSV_PATH,
                           img_dir=IMAGE_PATH,
                           transform=custom_transform2)

valid_dataset = CACDDataset(csv_path=VALID_CSV_PATH,
                            img_dir=IMAGE_PATH,
                            transform=custom_transform2)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=NUM_WORKERS)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)



##########################
# MODEL
##########################


###########################################
# Initialize Cost, Model, and Optimizer
###########################################

def cost_fn(logits, levels, imp):
    val = (-torch.sum((F.logsigmoid(logits)*levels
                      + (F.logsigmoid(logits) - logits)*(1-levels))*imp,
           dim=1))
    return torch.mean(val)


torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
model = build_model()

model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 


def compute_mae_and_mse(model, data_loader, device):
    mae, mse, num_examples = 0, 0, 0
    for i, (features, targets, levels, x) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)
        x = x.to(device)

        logits, probas = model(features)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        num_examples += targets.size(0)
        mae += torch.sum(torch.abs(predicted_labels - x))
        mse += torch.sum((predicted_labels - x)**2)
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    return mae, mse

loss = torch.nn.MSELoss()
start_time = time.time()

best_mae, best_rmse, best_epoch = 999, 999, -1
for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, targets, levels, x) in enumerate(train_loader):

        features = features.to(DEVICE)
        targets = targets
        targets = targets.to(DEVICE)
        levels = levels.to(DEVICE) 
        x = x.to(DEVICE)

        # FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = cost_fn(probas, levels, imp)
        #cost2 = loss(logits.float(), x.float())
        #cost = cost1 + cost2
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
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
        valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader,
                                                   device=DEVICE)

    if valid_mae < best_mae:
        best_mae, best_rmse, best_epoch = valid_mae, torch.sqrt(valid_mse), epoch
        ########## SAVE MODEL #############
        torch.save(model.state_dict(), os.path.join(PATH, 'best_model.pt'))


    s = 'MAE/RMSE: | Current Valid: %.4f/%.4f Ep. %d | Best Valid : %.4f/%.4f Ep. %d' % (
        valid_mae, torch.sqrt(valid_mse), epoch, best_mae, best_rmse, best_epoch)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

    s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

model.eval()
with torch.set_grad_enabled(False):  # save memory during inference

    train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                               device=DEVICE)
    valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader,
                                               device=DEVICE)
    test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                             device=DEVICE)

    s = 'MAE/RMSE: | Train: %.4f/%.4f | Valid: %.4f/%.4f | Test: %.4f/%.4f' % (
        train_mae, torch.sqrt(train_mse),
        valid_mae, torch.sqrt(valid_mse),
        test_mae, torch.sqrt(test_mse))
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
    train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                               device=DEVICE)
    valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader,
                                               device=DEVICE)
    test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                             device=DEVICE)

    s = 'MAE/RMSE: | Best Train: %.4f/%.4f | Best Valid: %.4f/%.4f | Best Test: %.4f/%.4f' % (
        train_mae, torch.sqrt(train_mse),
        valid_mae, torch.sqrt(valid_mse),
        test_mae, torch.sqrt(test_mse))
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)


########## SAVE PREDICTIONS ######
all_pred = []
all_probas = []
with torch.set_grad_enabled(False):
    for batch_idx, (features, targets, levels) in enumerate(test_loader):
        
        features = features.to(DEVICE)
        logits, probas = model(features)
        all_probas.append(probas)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        lst = [str(int(i)) for i in predicted_labels]
        all_pred.extend(lst)

torch.save(torch.cat(all_probas).to(torch.device('cpu')), TEST_ALLPROBAS)
with open(TEST_PREDICTIONS, 'w') as f:
    all_pred = ','.join(all_pred)
    f.write(all_pred)
