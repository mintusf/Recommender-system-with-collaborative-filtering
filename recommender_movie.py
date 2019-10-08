#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import random as random
import matplotlib.pyplot as plt

#A function calculating cost function
def rec_cost(
    X,
    theta,
    Y,
    reg_lambda,
    ):
    m = np.sum(Y != 0)
    J = 1 / 2 * np.sum(np.multiply((np.matmul(X, np.transpose(theta))
                       - Y) ** 2, Y != 0))
    J = J + reg_lambda / 2 * np.sum(X ** 2) + reg_lambda / 2 \
        * np.sum(theta ** 2)
    return (J / m) ** 0.5

# A function updating features of recommender system with collaborative filtering 
def X_update(
    X,
    theta,
    Y,
    l_rate,
    reg_lambda,
    ):

    # X_grad = ((X*theta'-Y).*(Y==0))*theta + lambda * X;

    X_grad = np.matmul(np.multiply(np.matmul(X, np.transpose(theta))
                       - Y, Y != 0), theta) + reg_lambda * X
    X -= l_rate * X_grad
    return X

# A function updating weights of recommender system with collaborative filtering 
def weights_update(
    X,
    theta,
    Y,
    reg_lambda,
    l_rate,
    ):

    # theta_grad =((X*theta'-Y).*(Y==0))'*X + lambda*theta;

    theta_grad = np.matmul(np.transpose(np.multiply(np.matmul(X,
                           np.transpose(theta)) - Y, Y != 0)), X) \
        + reg_lambda * theta
    theta -= l_rate * theta_grad
    return theta




# A function splitting dataset into train and test sets remembering that not all ratings are known

def my_train_test_split(Y, ratio):
    m = np.sum(Y != 0)  # No. of ratings
    (movie_no, users_no) = Y.shape
    temp = np.arange(movie_no * users_no) + 1
    rated_idx = np.multiply(temp[:, None], np.reshape(Y != 0, (-1, 1)))
    rated_idx = rated_idx[rated_idx != 0]
    rated_idx -= 1
    Y_train = Y
    Y_test = np.zeros(Y.shape)
    test_idx = np.array(random.sample(range(m), int(m * ratio)))
    test_idx = rated_idx[test_idx]
    for i in range(movie_no * users_no):

        # check if there is a rating

        if Y[i // users_no, i - i // users_no * users_no] != 0 and i \
            in test_idx:

            # copy value to Y_test and set the value in Y_train to 0

            Y_test[i // users_no, i - i // users_no * users_no] = Y[i
                    // users_no, i - i // users_no * users_no]
            Y[i // users_no, i - i // users_no * users_no] = 0
    (Y_train, mean) = Y_mean(Y_train)
    Y_test = normalize(Y_test, mean)
    return (Y_train, Y_test, test_idx)
        
    

# A function calculating mean and normalizing rows with at least one rating

def Y_mean(Y):
    mean = []
    for i in range(Y.shape[0]):
        if Y[i, Y[i] != 0].size == 0:
            mean.append(0)
        else:
            mean.append(Y[i, Y[i] != 0].mean())
    result = Y - np.array(mean)[:, None]
    result = np.multiply(result, Y != 0)
    return (result, mean)


# A function for mean normalization with provided mean

def normalize(Y, mean):
    result = Y - np.array(mean)[:, None]
    result = np.multiply(result, Y != 0)
    return result


# A function validating model's performance

def model_test(
    Y_test,
    X,
    theta,
    movies_no,
    users_no,
    ):
    m = np.sum(Y_test != 0)
    Y_predict = np.zeros((movies_no, users_no))
    Y_predict = np.multiply(np.matmul(X, np.transpose(theta)), Y_test
                            != 0)
    result = (Y_predict - Y_test) ** 2
    return ((np.sum(result) / m) ** 0.5, Y_predict)


# A function loading dataset from files

def load_data():
    print ('start!')
    print ('Loading files')
    ratings = pd.read_csv('ratings.dat', sep='::', header=None,
                          names=['User', 'Movie ID', 'Rating'],
                          index_col=False)
    movies = pd.read_csv('movies.dat', sep='::', header=None,
                         names=['ID', 'Title', 'Type'], index_col=False)

    # ratings["Movie title"]=ratings["Movie ID"].apply(lambda x: movies[movies["ID"]==x]["Title"].values)

    most_rated = pd.DataFrame(ratings['Movie ID'].value_counts()[0:300])
    most_rated.columns = ['Count']
    most_rated['Movie ID'] = most_rated.index
    most_rated['Movie title'] = most_rated['Movie ID'].apply(lambda x: \
            movies[movies['ID'] == x]['Title'].values)
    most_rated['Average rating'] = most_rated['Movie ID'
            ].apply(lambda x: ratings[ratings['Movie ID'] == x]['Rating'
                    ].mean())
    most_rated.reset_index(drop=True)
    return (ratings, most_rated)

# A function for Y matrix creating

def create_Y(ratings):
    users_no = ratings['User'].max()
    movies_no = ratings['Movie ID'].max()
    Y = np.zeros((movies_no, users_no))
    for (index, row) in ratings.iterrows():
        Y[row['Movie ID'] - 1, row['User'] - 1] = row['Rating']
    return Y

# A function for weights and X matrices initialization

def init_model(Y, features_no):
    (movies_no, users_no) = Y.shape
    theta = np.random.uniform(-6 ** 0.5 / (features_no + users_no), 6
                              ** 0.5 / (features_no + users_no),
                              (users_no, features_no))
    X = np.random.uniform(-6 ** 0.5 / (features_no + movies_no), 6
                          ** 0.5 / (features_no + movies_no),
                          (movies_no, features_no))
    return (theta, X)

# A function for a whole model training

def train_model(
    X,
    theta,
    Y_train,
    Y_test,
    reg_lambda,
    steps,
    l_rate,
    ):
    print ('Starting learning')
    (movies_no, users_no) = Y_train.shape
    cost_axis = []
    val_cost_axis = []
    for i in range(steps):
        X = X_update(X, theta, Y_train, l_rate, reg_lambda)
        theta = weights_update(X, theta, Y_train, reg_lambda, l_rate)
        print (rec_cost(X, theta, Y_train, reg_lambda))
        cost_axis.append(rec_cost(X, theta, Y_train, reg_lambda))
        val_cost_axis.append(rec_cost(X, theta, Y_test, reg_lambda))
        if not i % 10:
            print(cost_axis)
            plt.plot(cost_axis)
            plt.plot(val_cost_axis)
            plt.show()
    (val_test, Y_predict) = model_test(Y_test, X, theta, movies_no,
            users_no)
    print (val_test)
    (val_train, Y_train_predict) = model_test(Y_train, X, theta,
            movies_no, users_no)
    print (val_train)
    return (theta, X, cost_axis, val_cost_axis)

# A function for building model for all data

def all_data(reg_lambda, steps, l_rate):
    (ratings, most_rated) = load_data()
    Y = create_Y(ratings)
    (theta_init, X_init) = init_model(Y, features_no=4)
    (Y_train, Y_test, test_idx) = my_train_test_split(Y, 0.2)
    (theta_final, X_final, cost_axis, val_cost_axis) = train_model(
        X_init,
        theta_init,
        Y_train,
        Y_test,
        reg_lambda,
        steps,
        l_rate,
        )
    

# A function utilizing already learnt weights for making recommendations for a new user

def train_newuser(
    X,
    Y,
    steps,
    l_rate=0.02,
    reg_lambda=0.1,
    ):
    (theta, X_init) = init_model(Y, X.shape[1])
    movies_no = X.shape[0]
    for i in range(steps):
        X = X_update(X, theta, Y, l_rate, reg_lambda)
        theta = weights_update(X, theta, Y, reg_lambda, l_rate)
    (val_test, Y_predict) = model_test(Y, X, theta, movies_no, 1)
    return theta

def load_model():
    features = np.loadtxt('Features.txt', delimiter=';')
    mean = np.loadtxt('Mean.txt', delimiter=';')
    movies_no = features.shape[0]
    Y_new = pd.read_csv('Movies_fornewuser.txt', delimiter=',',
                        header=None)
    Y_new.fillna(0, inplace=True)
    movies_id = pd.read_csv('Movies_ID.txt', header=None)
    Y_new = pd.concat([movies_id[0], Y_new[1]], keys=['Movie ID',
                      'Rating'], axis=1)
    Y_oneuser = np.zeros(movies_no)
    for (index, row) in Y_new.iterrows():
        Y_oneuser[int(row['Movie ID']) - 1] = row['Rating']
    Y_oneuser = normalize(Y_oneuser[:, None], mean)
    return (features, mean, movies_no, Y_oneuser)


# A function finding recommendations

def find_recommendations(features, theta, mean):
    predictions = np.matmul(features, np.transpose(theta))
    predictions = normalize(predictions, -mean)
    movies = pd.read_csv('movies.dat', sep='::', header=None,
                         names=['ID', 'Title', 'Type'], index_col=False)
    temp = predictions
    best_id = []
    for i in range(25):
        idx = temp.argmax()
        best_id.append(idx)
        temp[idx] = 0
    recommendations = []
    for idx in best_id:
        recommendations.append(movies[movies['ID'] == idx]['Title'
                               ].values)
    for movie in recommendations:
        if movie != '':
            print (movie)
    return predictions
    
    
# A function recommending movies for a new user

def new_user():

    # Load the learnt model

    (features, mean, movies_no, Y_oneuser) = load_model()

    # Find weights associated with the new user using existing recommender system

    theta = train_newuser(features, Y_oneuser, steps=3000)

    # Determine and print predictions

    predictions = find_recommendations(features, theta, mean)

    return predictions


def main():
    while True:
        option = \
            input("""Choose the option:
1: Training whole model (Warning: it mights take long time (up to 15 min.))
2: Use trained model to get recommendations for yourself (Warning: in this case, you should fill some ratings in the file 'Movies_fornewuser')
3: Finish
""")
        if option == '1':
            all_data(reg_lambda=0.1, steps=15, l_rate=0.002)
            break
        elif option == '2':
            new_user()
            break
        elif option == '3':
            break
        else:
            print ('''
Input again, this time corretly...
''')
    

if __name__=="__main__":
    main()