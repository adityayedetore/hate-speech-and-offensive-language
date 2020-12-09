import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models.LSTM import LSTMClassifier
import argparse
import pandas as pd
import ipdb

TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not 32):
            # One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1

        #ipdb.set_trace()
        
        if steps % 100 == 0:
            print(f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f} percent')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter):
    # adding for error analysis
    ex_list = []
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] is not 32):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()
            
            #error analysis:
            for i in range(len(batch)):
                ex = {"text" : " ".join([TEXT.vocab.itos[x] for x in batch.text[0][i] if x != 1]),
                    "length" : len([TEXT.vocab.itos[x] for x in batch.text[0][i] if x != 1]),
                    "pred" : torch.max(prediction, 1)[1][i].item(),
                    "target" : target[i].item(), 
                    "correct" : 1 if torch.max(prediction, 1)[1][i].item() == target[i].item() else 0
                }
                ex_list.append(ex)
            #ipdb.set_trace()
            

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter), ex_list
	
# TODO: turn these into arguments / manually change parameters
# learning rate: 2e-3, 2e-4, 2e-5
# hidden_size: 128, 256, 512
# embedding_length: 150, 300, 600
# 3x3x3 = 27 models (don't worry about random seeds for now)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--learning-rate', '-l', help="learning rate", type=float, default=2e-5)
parser.add_argument('--batch-size', '-b', help="batch size", type=int, default=32)
parser.add_argument('--output-size', '-o', type=int, default=3)
parser.add_argument('--hidden-size', '-h', type=int, default=256)
parser.add_argument('--embedding-length', '-e', type=int, default=300)

args = parser.parse_args()

learning_rate = args.learning_rate
batch_size = args.batch_size
output_size = args.output_size
hidden_size = args.hidden_size
embedding_length = args.embedding_length

print("Hyperparams:", args)

model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
loss_fn = F.cross_entropy

for epoch in range(10):
    train_loss, train_acc = train_model(model, train_iter, epoch)
    val_loss, val_acc, _ = eval_model(model, valid_iter)
    
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
    
test_loss, test_acc, ex_list = eval_model(model, test_iter)
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

# TODO: find a way to save models/results?
filename = f"results_{args.learning_rate}_{args.hidden_size}_{args.embedding_length}.csv"
df = pd.DataFrame(ex_list)
df.to_csv(filename)


# TODO: edit output: esp. fix the sample predictions code below (instead of pos/neg, hate/offensive/neither)
   
# ''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
# test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
# test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."
# 
# test_sen1 = TEXT.preprocess(test_sen1)
# test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]
# 
# test_sen2 = TEXT.preprocess(test_sen2)
# test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]
# 
# test_sen = np.asarray(test_sen1)
# test_sen = torch.LongTensor(test_sen)
# test_tensor = Variable(test_sen, volatile=True)
# test_tensor = test_tensor.cuda()
# model.eval()
# output = model(test_tensor, 1)
# out = F.softmax(output, 1)
# if (torch.argmax(out[0]) == 1):
#     print ("Sentiment: Positive")
# else:
#     print ("Sentiment: Negative")
