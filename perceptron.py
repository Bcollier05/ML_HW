#-------------------------------------------------------------------------
# AUTHOR: Britney COllier
# FILENAME: perceptron.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

# Initialize variables to track highest accuracies
highest_perceptron_acc = 0
highest_mlp_acc = 0
best_perceptron_params = {}
best_mlp_params = {}

for learning_rate in n: #iterates over n

    for shuffle in r: #iterates over r

        #iterates over both algorithms
        for algorithm in ['Perceptron', 'MLP']: #iterates over the algorithms

            #Create a Neural Network classifier
            if algorithm == 'Perceptron':
                clf = Perceptron(eta0=learning_rate, shuffle=shuffle, max_iter=1000)
            else:
                clf = MLPClassifier(activation='logistic', 
                                   learning_rate_init=learning_rate,
                                   hidden_layer_sizes=(25,),  # 1 hidden layer with 25 neurons
                                   shuffle=shuffle, 
                                   max_iter=1000)

            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            #make the classifier prediction for each test sample and compute accuracy
            correct_predictions = 0
            total_samples = len(X_test)
            
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])[0]
                if prediction == y_testSample:
                    correct_predictions += 1
            
            accuracy = correct_predictions / total_samples

            #check if the calculated accuracy is higher than the previously one calculated
            if algorithm == 'Perceptron':
                if accuracy > highest_perceptron_acc:
                    highest_perceptron_acc = accuracy
                    best_perceptron_params = {'learning_rate': learning_rate, 'shuffle': shuffle}
                    print(f"Highest Perceptron accuracy so far: {highest_perceptron_acc:.4f}, Parameters: learning rate={learning_rate}, shuffle={shuffle}")
            else:
                if accuracy > highest_mlp_acc:
                    highest_mlp_acc = accuracy
                    best_mlp_params = {'learning_rate': learning_rate, 'shuffle': shuffle}
                    print(f"Highest MLP accuracy so far: {highest_mlp_acc:.4f}, Parameters: learning rate={learning_rate}, shuffle={shuffle}")

# Print final best results
print("\nFinal Results:")
print(f"Best Perceptron Accuracy: {highest_perceptron_acc:.4f} with Parameters: {best_perceptron_params}")
print(f"Best MLP Accuracy: {highest_mlp_acc:.4f} with Parameters: {best_mlp_params}")
