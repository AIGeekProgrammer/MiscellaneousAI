# File: From Bi-Gram to GPT-2 - final version
# Author: Szymon Manduk
# Date: Feb 23, 2023
# Description: A journey through language models - part 7
# Work based on the ideas presented by A. Karpathy: https://www.youtube.com/watch?v=kCc8FmEb1nY 
# Part 7 includes: 
# - training on Polish book "Potop" by Henryk Sienkiewicz

import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from timeit import default_timer as timer

# # if run from Google Colab, uncomment and mount the drive
# # from google.colab import drive
# drive.mount('/gdrive')
# # Also we need to reach out to diferent folder
# with open(r'/gdrive/My Drive/Colab Notebooks/Data/Shakespeare.txt', 'r', encoding='utf-8' ) as f:

# import data from a text file 
with open('Data\Potop.txt', 'r', encoding='utf-8' ) as f:
    data = f.read()

# create vocabulary 
vocabulary = sorted(list(set(data)))

# helper dictionaries to convert from string to character and vice versa
char2idx = {ch:i for i,ch in enumerate(vocabulary)}
idx2char = {i:ch for i,ch in enumerate(vocabulary)}

# convert data to integers
dataset = torch.tensor([char2idx[ch] for ch in data])

# split data into training and validations sets
train_size = int(len(dataset) * 0.9)
train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:]

# Hyperparameters for the model and training
TRAINING = True  # set to False if you want to load the model from the checkpoint
SAVE_MODEL = True  # set to True if you want to save the model
LOAD_MODEL = False  # set to True if you want to load the model from the checkpoint
GENERATE_TEXT = True  # set to True if you want to generate text
batch_size = 256  # how many block_size elements are we looking at a time
block_size = 128  # how many characters are we looking at a time
learning_rate = 0.01
epochs = 2 # 0000
# lr_scheduler_step = epochs / 100
# lr_scheduler_gamma = 0.9
vocab_size = len(vocabulary)
embedding_dim = 64  # previously, with the Bi-Gram model embedding was equal to the vocabulary size. Right now we may set it differently
eval_iters = 1000  # How often we evaluate the model (e.g. every 1000 iterations or whatever value is set here)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# function to get a batch of data from the training or validation dataset
def get_batch(source, batch_size=5, block_size=8):
    # get the data from the training or validation set
    dataset = train_dataset if source == 'train' else val_dataset

    # initialize empty tensors for the data and target
    data = torch.empty((batch_size, block_size), dtype=torch.long)
    target = torch.empty((batch_size, block_size), dtype=torch.long)

    for i in range(batch_size):
        # randomly select a starting point
        idx = torch.randint(len(dataset) - block_size, (1,)).item()

        # get the data from the starting point to the block size
        data[i] = dataset[idx:idx+block_size]

        # get the target from the starting point + 1 to the block size + 1
        target[i] = dataset[idx+1:idx+block_size+1]

    return idx, data, target

# class for FFNN
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1   )
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# class for a single Head of Attention (SingleAttentionHead)
class SingleAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.Q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.K = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.V = nn.Linear(embedding_dim, embedding_dim, bias=False)
    
    def forward(self, x):
        # x is [Batch, Time, Embedding]
        QX = self.Q(x)
        KX = self.K(x)
        VX = self.V(x)

        # calculate score
        score = QX @ torch.transpose(KX, 1, 2) / torch.sqrt(torch.tensor(embedding_dim, dtype=torch.float32))

        # mask the score
        mask = torch.tril(torch.ones(score.shape, device=device), diagonal=0)
        score = score.masked_fill(mask == 0, float('-inf'))

        # softmax over the last dimension
        attention = F.softmax(score, dim=-1)

        # multiply the attention with the value
        out = attention @ VX

        return out

# class for multiple heads of attention (MultiHeadAttention)
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([SingleAttentionHead() for _ in range(num_heads)])
        # matrtix for combining the heads
        self.W = nn.Linear(embedding_dim * num_heads, embedding_dim, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x is [Batch, Time, Embedding]
        stacked_attention = torch.empty_like(x)

        for i, head in enumerate(self.heads):
            if i == 0:
                stacked_attention = head(x)
            else:
                stacked_attention = torch.cat((stacked_attention, head(x)), dim=2)

        attention = self.dropout(self.W(stacked_attention))

        return attention       

# DecoderBlock class with skip connection
class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffnn = FFNN(embedding_dim, embedding_dim*4, embedding_dim)  # inner dimension is 4 times the embedding dim - as in the original paper
        self.attention = MultiHeadAttention(4)  # 4 heads of attention
        self.ln1 = nn.LayerNorm(embedding_dim)  # layer normalization
        self.ln2 = nn.LayerNorm(embedding_dim)  # layer normalization

    def forward(self, x):
        y = x + self.attention(self.ln1(x))  # including a skip connection
        y = y + self.ffnn(self.ln2(y))       # including a skip connection
        return y

# class containing only an embedding layer for bi-gram model
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # word embedding layer [B, T, C]
        self.pos_embedding = nn.Embedding(block_size, embedding_dim) # positional embedding layer [B, T, C]
        self.decoder_blocks = nn.Sequential(
            DecoderBlock(),
            DecoderBlock(),
            DecoderBlock(),
            nn.LayerNorm(embedding_dim),
        ) # 2 decoder blocks
        self.lm_head = nn.Linear(embedding_dim, vocab_size) # linear layer for the output 

    def forward(self, idx, target=None):
        # standard and postional embedding
        token_emb = self.embedding(idx) # [B=batch_size, T=block_size, embedding_dim]
        pos_emb = self.pos_embedding(torch.arange(block_size).expand(idx.shape[0], -1).to(device)) # [B, T, embedding_dim]  
        token = token_emb + pos_emb # [B, T, embedding_dim]

        # forward pass through the decoder blocks
        decod = self.decoder_blocks(token) 
        logits = self.lm_head(decod)

        # loss calculation
        if target == None:
            loss = None
        else:
            # y_hat is [B, T, C] and y is [B, T]. Crossentropy expects [B, C, T] for multidimentional outputs (larger than 2-d) 
            # so we need to permute y_hat to be able to calculate the loss
            loss = F.cross_entropy(torch.permute(logits, (0,2,1)), target)
        return logits, loss
    
    # generate text using the model by drawing from multinomial distribution
    # Please note: idx is the starting state with a dimension of [1, T], e.g. 8 spaces or new_lines or letters 'A'
    @torch.no_grad()
    def generate(self, idx, length=100):
        self.eval()
        result = idx

        # loop through the length of the expected predictions
        for _ in range(length):
            # get the embedding for the current index
            logits, _ = self(idx) # [B, T, C]
            
            # focus only on the last time step
            logits = logits[:, -1, :] # [B, C]
            
            # get the prediction from multinomial distribution
            idx_next = torch.multinomial(torch.softmax(logits, dim=1), num_samples=1) # [B, 1]
            
            # concatenate the predicted tensor to the result
            result = torch.cat([result, idx_next], dim=1) # [B, T]

            # concatenate the predicted tensor to the input tensor and remove the first element
            idx = torch.cat([idx, idx_next], dim=1)[:,1:] # [B, T]

        self.train()
        return result

# Create model and get the output from embedding table corresponding to a given vacabulary character
# As we implement bi-gram here, the embedding dimension is the same as the vocabulary size
model = LanguageModel().to(device)

# Print number of parameters in the model
print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

if TRAINING:
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4000,8000,12000,16000], gamma=0.1)

    # Variables to collect time elapsed during training and validation
    training_time, validation_time = 0., 0.

    # main training loop
    for epoch in range(1, epochs+1):
        start = timer()
        
        # get batch of data
        idx, x, y = get_batch('train', batch_size=batch_size, block_size=block_size)
        x, y = x.to(device), y.to(device)

        # get the output from the model
        y_hat, loss = model(x, y)

        # calculate the gradients
        loss.backward()

        # update the model parameters
        optimizer.step()

        # reset the gradients
        optimizer.zero_grad()

        # schedulrer step - LR decay  
        scheduler.step()

        end = timer()
        training_time += end - start

        if epoch == 1 or epoch % (epochs / 100) == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()}, time elapsed: {training_time:.2f}s')

        # every eval_iters iterations print the loss and validate model
        if epoch % eval_iters == 0:
            start = timer()
            # get loss on the validation dataset
            with torch.no_grad():
                model.eval()
                idx, x, y = get_batch('valid', batch_size=len(val_dataset) // 10, block_size=block_size)
                x, y = x.to(device), y.to(device)
                y_hat, loss = model(x, y)
                print(f'Validation loss: {loss.item()}')
                model.train()    
            end = timer()
            validation_time += end - start

    print(f'Training time: {training_time:.2f}s. Validation time: {validation_time:.2f}s.')

    if SAVE_MODEL:
        # Saving the model (getting current time to use it in the name of the model file)
        current_time = datetime.now().strftime("%d.%m.%Y-%H_%M_%S")
        torch.save(model.to('cpu').state_dict(), '.\Models\part-4-model-2000-epochs-'+current_time+'.pt')

else:  # if TRAINING is False then load the model from a file
    if LOAD_MODEL:
        file_name = '.\Models\part-4-model-XXXXXXXXX.pt'
        # print(f'Loading model from {file_name}...')
        model = model.to('cpu')  # move model to cpu
        model.load_state_dict(torch.load(file_name))

if GENERATE_TEXT:
    ##### generate text using the trained model #####
    # as an input we set 8 characters - all set to a "new line" (\n) character. Its index is 0.
    start = torch.tensor([0]*block_size, device=device).view(1, block_size)
    output = model.generate(start, length=1000)
    output = output.detach().cpu().tolist()[0]
    prediction = ''.join([idx2char[i] for i in output])

    # we started with block_size '\n' characters, so we need to remove them from the output
    prediction = prediction[block_size:]
    print(f'Text prediction:\n{prediction}') 

### OUTPUT ###
# block_size = 128 from 64
# batch_size = 256 from 128
# embedding_dim = 64 from 32
# number of parameters = 313,345
# number of epochs = 20000
# training dataset: Polish book "Potop" by Henryk Sienkiewicz
# RESULTS: