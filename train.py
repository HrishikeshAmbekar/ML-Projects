import numpy as np
import csv

def import_data():
    X=np.genfromtxt("train_X_lr.csv",delimiter=",",dtype=np.float64,skip_header=1)
    Y=np.genfromtxt("train_Y_lr.csv",delimiter=",",dtype=np.float64)
    return X,Y

def compute_cost(X, Y, W):
    Y_pred=np.dot(X,W)
    difference=Y_pred-Y
    sq_difference=np.square(difference)
    sq_dif=(np.sum(sq_difference))
    cost=sq_dif/(2*len(Y))
    cost=float(cost)
    
    return cost

def compute_gradient_of_cost_function(X, Y, W):
    m = len(X)
    Y_pred = np.dot(X, W)
    difference =  Y_pred - Y
    dW = (1/m) * (np.dot(difference.T, X))
    dW = dW.T
    return dW

def optimize_weights_using_gradient_descent(X, Y, W, learning_rate):
    decreasing=True
    old_cost=float(10000000)
    i=0
    while decreasing:
        dW = compute_gradient_of_cost_function(X, Y, W)
        W=W - np.dot(learning_rate,dW)
        new_cost=compute_cost(X,Y,W)
        print(i,new_cost)
        i+=1
        if old_cost-new_cost<=0.000000001:
            decreasing=False
        old_cost=new_cost
    print(W)
    return W

def train_model(X,Y):
    X=np.insert(X,0,1,axis=1)
    Y=Y.reshape(len(X),1)
    W=np.zeros((X.shape[1],1))
    W=optimize_weights_using_gradient_descent(X, Y, W,0.0002)
    return W

def save_model(weights,weights_file_name):
    with open(weights_file_name,'w') as weights_file:
        wr=csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()

if __name__=="__main__":
    X,Y=import_data()
    weights=train_model(X,Y)
    save_model(weights,"WEIGHTS_FILE.csv")
    

