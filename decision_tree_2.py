#-------------------------------------------------------------------------
# AUTHOR: Britney Collier
# FILENAME: decision_tree_2.py
# SPECIFICATION: Decision Tree Classifier for contact lens dataset
# FOR: CS 4210 - Assignment #2
# TIME SPENT: 
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY SUCH AS numpy OR pandas.
# You must only use standard Python lists, dictionaries, and arrays.

# Importing necessary Python libraries
from sklearn import tree
import csv

# Training datasets
dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

# Define categorical mappings
age = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3}
spectacle = {"Myope": 1, "Hypermetrope": 2}
astigmatism = {"No": 1, "Yes": 2}
tear = {"Reduced": 1, "Normal": 2}
recommendation = {"Yes": 1, "No": 2}

# Loop through each dataset
for ds in dataSets:
    dbTraining = []
    X = []
    Y = []

    # Read training data from CSV file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            X.append([age[row[0]], spectacle[row[1]], astigmatism[row[2]], tear[row[3]]])
            Y.append(recommendation[row[4]])

    # Loop through training and testing 10 times
    total_accuracy = 0
    for _ in range(10):

        # Train the Decision Tree classifier with max_depth=5
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf.fit(X, Y)

        # Read test data
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header
            for row in reader:
                dbTest.append(row)

        # Evaluate classifier
        correct_predictions = 0
        total_tests = len(dbTest)

        for row in dbTest:
            # Transform test data
            test_X = [[age[row[0]], spectacle[row[1]], astigmatism[row[2]], tear[row[3]]]]
            test_Y = recommendation[row[4]]

            # Make prediction
            prediction = clf.predict(test_X)[0]

            # Compare with actual label
            if prediction == test_Y:
                correct_predictions += 1

        # Compute accuracy for this iteration
        accuracy = correct_predictions / total_tests
        total_accuracy += accuracy

    # Compute final average accuracy
    final_accuracy = total_accuracy / 10
    print(f"Final accuracy when training on {ds}: {final_accuracy:.2f}")
