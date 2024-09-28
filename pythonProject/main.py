from cProfile import label

import matplotlib.pyplot as plt
import numpy as np
import openpyxl as ox
import pandas as pd
# a simple implementation of a linear regression model, we first ensure that the data we have can be converted into a numpy array,

#a dataframe is a 2D datastructure that comes with a lot of different functions that help us a lot
df = pd.read_excel('test.xlsx')
da_array = df.to_numpy()
#in order to be able to graph it in matplotlib we want to be able to convert this data into two arrays one for x and another for y
transpose_da_array = da_array.transpose()
sorted_array = transpose_da_array[:, np.argsort(transpose_da_array[1])]
print(sorted_array)

x = np.array(transpose_da_array[0])
y = np.array(transpose_da_array[1])
x_n = np.array([[8],[3],[7],[9],[0]])
def linearregression(x_m,y_m):
    #find the mean for dependent and independent variable
    x_mean = (float)(np.sum(x_m))/(np.prod(x_m.shape)) # mean for independent variable
    y_mean = (float)(np.sum(y_m))/(np.prod(y_m.shape)) # mean for dependent variable
    x_diff = []
    y_diff = []
    xy_prod = []
    xx_prod = []
    print(x_mean)
    print(y_mean)
    #find the mean-value difference for dependent and independent variable
    for n in x_m: #mean value difference for independent variable
        x_diff.append(n - x_mean)
    x_diff_np = np.array(x_diff)
    for m in y_m: #mean value difference for dependent variable
        y_diff.append(m - y_mean)
    y_diff_np = np.array(y_diff)

    print(x_diff_np)
    print(y_diff_np)
     #product of the two differences
    temp = 0
    for p in x_diff_np:
        xy_prod.append(x_diff_np[temp] * y_diff_np[temp])
        xx_prod.append(x_diff_np[temp] * x_diff_np[temp])
        temp += 1
    xy_prod_np = np.array(xy_prod)
    xx_prod_np = np.array(xx_prod)
    xy_prod_np_sum = np.sum(xy_prod_np)
    xx_prod_np_sum = np.sum(xx_prod_np)
    print(f"{xy_prod_np} Their sum is {xy_prod_np_sum}")
    print(f"{xx_prod_np} Their sum is {xx_prod_np_sum}")
    #the linear regression equation is y = B0 + B1x + e
    B_one = (float)(xy_prod_np_sum/xx_prod_np_sum)
    B_o = (float)(y_mean - (B_one *  x_mean))
    print(f"b1 is {B_one} and b0 is {B_o}")
    return [B_o,B_one,x_mean]

def predictionmodel(x_mTest,x_independent,y_dependent):
    y_pred = []
    val = linearregression(x_independent,y_dependent)
    for n in x_mTest:
        y_pred.append((float)(val[0] + (val[1] * n)))
    y_pred_np = np.array(y_pred)
    print(y_pred_np, f"b0 is {val[0]} and b1 is {val[1]} while x_mean is {val[2]}")
    return y_pred_np

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(5, 2))
axes[0].plot(x,y,color='red',marker='o',label='Given')
axes[1].plot(x,predictionmodel(x,x,y),color='green',marker='o',label='Predicted')
plt.plot(x,y,color='red',marker='o',label='Given')
plt.plot(x,predictionmodel(x,x,y),color='green',marker='o',label='Predicted')
plt.xlabel('LPA')
plt.ylabel('Happiness Index')
plt.show()
fig.tight_layout()