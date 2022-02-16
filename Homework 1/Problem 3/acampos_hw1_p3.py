%matplotlib inline
import numpy as np
import torch
import pandas as pd 
 
# Data Visualisation 
import matplotlib.pyplot as plt 

col_list = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
df = pd.read_csv("/Housing.csv", usecols=col_list)
x1 = df["area"]
x2 = df["bedrooms"]
x3 = df["bathrooms"]
x4 = df["stories"]
x5 = df["parking"]

torch.set_printoptions(edgeitems=2)
t_c = torch.tensor(df["price"]) # actual price
t_u = torch.tensor([x1, x2, x3, x4, x5])

t_un = 0.1 * t_u
def model(t_u, w1, w2, w3, w4, w5, b):
    return (w1*t_u[0]) + (w2*t_u[1]) + (w3*t_u[2]) + (w4*t_u[3]) + (w5*t_u[4]) + b
def loss_fn(t_p, t_c):
    abs_diffs = abs(t_p - t_c)
    return abs_diffs.mean()

def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None: 
            params.grad.zero_()
        
        t_p = model(t_u, *params) #predicted
        loss = loss_fn(t_p, t_c)
        loss.backward()

        with torch.no_grad():
            params -= learning_rate * params.grad

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            
    return params
training_loop(
    n_epochs = 5000, 
    learning_rate = 1e-4, 
    params = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], requires_grad=True),
    t_u = t_un,
    t_c = t_c)

plt.plot(t_c, color = 'orange')
plt.show()
