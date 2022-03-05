%matplotlib inline
import numpy as np
import torch
import pandas as pd 
import torch.nn as nn
from collections import OrderedDict
 
# Data Visualisation 
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

# In[2]:
col_list = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
df = pd.read_csv("/Housing.csv", usecols=col_list)
x1 = df["area"]
x2 = df["bedrooms"]
x3 = df["bathrooms"]
x4 = df["stories"]
x5 = df["parking"]

torch.set_printoptions(edgeitems=2)
t_c = df["price"]
t_u = [x1, x2, x3, x4, x5]
t_c = torch.tensor(t_c).unsqueeze(1) # actual
t_u = torch.tensor(t_u).unsqueeze(1) # unknown

t_un = 0.1 * t_u

def model(t_u, w1, w2, w3, w4, w5, b): # w-weight, b-bias
    return (w1*t_u[0]) + (w2*t_u[1]) + (w3*t_u[2]) + (w4*t_u[3]) + (w5*t_u[4]) + b
def loss_fn(t_p, t_c):
    abs_diffs = abs(t_p - t_c)
    return abs_diffs.mean()

model = nn.Sequential(
    nn.Linear(3072, 512),
    nn.Tanh(),
    nn.Linear(512, 2),
    nn.LogSoftmax(dim=1)
)

learning_rate = 1e-2

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_fn = nn.NLLLoss()

n_epochs = 100

for epoch in range(n_epochs):
  for img, label in cifar2:
    out = model(img.view(-1).unsqueeze(0)
    loss = loss_fn(out, torch.tensor([label]))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

# In[6]:
#linear_model.weight

# In[7]:
#linear_model.bias

# In[9]:
#x = torch.ones(10, 1)
#linear_model(x)

# In[10]:
#linear_model = nn.Linear(1, 1)
#optimizer = torch.optim.SGD(
    #linear_model.parameters(),
    #lr=1e-2)

# In[11]:
#linear_model.parameters()

# In[12]:
#list(linear_model.parameters())

# In[12]:
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)
shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]
train_indices, val_indices

# In[13]:
train_t_u = t_u[train_indices]
t_c_train = t_c[train_indices]
val_t_u = t_u[val_indices]
t_c_val = t_c[val_indices]
t_un_train = 0.1 * train_t_u
t_un_val = 0.1 * val_t_u

# In[13]:
def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val, t_c_train, t_c_val):
    for epoch in range(1, n_epochs + 1):
        t_p_train = model(t_u_train)
        loss_train = loss_fn(t_p_train, t_c_train)

        t_p_val = model(t_u_val)
        loss_val = loss_fn(t_p_val, t_c_val)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f},"
            f" Validation loss {loss_val.item():.4f}")

# In[15]:
linear_model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(linear_model.parameters(), lr=1e-2)
training_loop(
    n_epochs = 300,
    optimizer = optimizer,
    model = linear_model,
    loss_fn = nn.MSELoss(),
    t_u_train = t_un_train,
    t_u_val = t_un_val,
    t_c_train = t_c_train,
    t_c_val = t_c_val)

print()
print(linear_model.weight)
print(linear_model.bias)

# In[16]:
seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1))
seq_model

# In[17]:
[param.shape for param in seq_model.parameters()]

# In[18]:
for name, param in seq_model.named_parameters():
    print(name, param.shape)

# In[19]:
seq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 8)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(8, 1))
]))

seq_model

# In[20]:
for name, param in seq_model.named_parameters():
    print(name, param.shape)

# In[21]:
seq_model.output_linear.bias

# In[22]:
optimizer = torch.optim.SGD(seq_model.parameters(), lr=1e-3)
training_loop(
    n_epochs = 5000,
    optimizer = optimizer,
    model = seq_model,
    loss_fn = nn.MSELoss(),
    t_u_train = t_un_train,
    t_u_val = t_un_val,
    t_c_train = t_c_train,
    t_c_val = t_c_val)
print('output', seq_model(t_un_val))
print('answer', t_c_val)
print('hidden', seq_model.hidden_linear.weight.grad)
