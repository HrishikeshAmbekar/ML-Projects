import math
import numpy as np
def scan_1d_array():
    Z = [int(i) for i in input().split()]
    return Z


def scan_2d_array(N):
    X = []
    for i in range(0, N):
      Xi = [float(j) for j in input().split()]
      X.append(Xi)
    return X


def compute_ln_norm_distance(vector1, vector2, n):
    """
    Arguments:
    vector1 -- A 1-dimensional array of size > 0.
    vector2 -- A 1-dimensional array of size equal to size of vector1
    n       -- n in Ln norm distance (>0)
    """
    vector_len = len(vector1)
    diff_vector = []

    for i in range(0, vector_len):
      abs_diff = abs(vector1[i] - vector2[i])
      diff_vector.append(abs_diff ** n)
    ln_norm_distance = (sum(diff_vector))**(1.0/n)
    return ln_norm_distance


def find_k_nearest_neighbors(train_X, test_example, k, n_in_ln_norm_distance):
    """
    Returns indices of 1st k - nearest neighbors in train_X, in order with nearest first.
    """
    indices_dist_pairs = []
    index= 0
    for train_elem_x in train_X:
      distance = compute_ln_norm_distance(train_elem_x, test_example,n_in_ln_norm_distance)
      indices_dist_pairs.append([index, distance])
      index += 1
    indices_dist_pairs.sort(key = lambda x: x[1])
    top_k_pairs = indices_dist_pairs[0:k]
    top_k_indices = [i[0] for i in top_k_pairs]
    return top_k_indices


def classify_points_using_knn(train_X, train_Y, test_X, n_in_ln_norm_distance, k):
    test_Y = []
    for test_elem_x in test_X:
      top_k_nn_indices = find_k_nearest_neighbors(train_X, test_elem_x, k,n_in_ln_norm_distance)
      top_knn_labels = []
      for i in top_k_nn_indices:
        top_knn_labels.append(train_Y[i])
      most_frequent_label = max(set(top_knn_labels), key = top_knn_labels.count)
      test_Y.append(most_frequent_label)
    return test_Y

def calculate_accuracy(predicted_Y, actual_Y):
    total_num_of_observations = len(predicted_Y)
    num_of_values_matched = 0
    for i in range(0,total_num_of_observations):
        if(predicted_Y[i] == actual_Y[i]):
            num_of_values_matched +=1
    return float(num_of_values_matched)/total_num_of_observations


def get_best_k_using_validation_set(train_X, train_Y, validation_split_percent,n_in_ln_norm_distance):
    """
    Returns best value of k which gives best accuracy
    """
    split_index=math.floor((100- validation_split_percent)*len(train_X)/100)
    training_Xset=[]
    training_Yset=[]
    validation_Xset=[]
    validation_Yset=[]
    for i in range(split_index):
        training_Yset.append(train_Y[i])
        training_Xset.append(train_X[i])
    for i in range(split_index,len(train_X)):
        validation_Yset.append(train_Y[i])
        validation_Xset.append(train_X[i])
    predicted_Ys=[]
    for k in range(1,len(training_Xset)+1):
        test_Y_k=classify_points_using_knn(training_Xset,training_Yset,validation_Xset, n_in_ln_norm_distance, k)
        predicted_Ys.append(test_Y_k)
    
    accuracies=[]
    for i in range(len(predicted_Ys)):
        accuracy_i=calculate_accuracy(predicted_Ys[i],validation_Yset)
        accuracies.append(accuracy_i)
    required_k=accuracies.index(max(accuracies))+1
    return required_k


def import_data():
    X=np.genfromtxt("train_X_knn.csv",delimiter=",",dtype=np.float64,skip_header=1)
    Y=np.genfromtxt("train_Y_knn.csv",delimiter=",",dtype=np.float64)
    return X,Y


if __name__== "__main__":
    train_X,train_Y=import_data()
    k=get_best_k_using_validation_set(train_X, train_Y,25 ,2)
    print(k)

