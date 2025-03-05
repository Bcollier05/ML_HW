#-------------------------------------------------------------------------
# AUTHOR: Britney Collier
# FILENAME: knn.py
# SPECIFICATION: Implement 1-Nearest Neighbor (1NN) for email classification using Leave-One-Out Cross-Validation (LOO-CV)
# FOR: CS 4210 - Assignment #2
# TIME SPENT: <fill this in>
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas.
# You have to work here only with standard Python lists.

# Importing necessary Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

# Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # Skipping the header
            db.append(row)

# Label encoding dictionary
label_mapping = {"ham": 0, "spam": 1}

# Initialize error counter
error_count = 0
total_instances = len(db)

# Loop your data to allow each instance to be your test set
for i in range(total_instances):

    # Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    # For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    # Convert each feature value to float to avoid warning messages
    X = []
    Y = []
    
    for j in range(total_instances):
        if i != j:  # Exclude the test instance
            X.append([float(value) for value in db[j][:-1]])  # Convert features to float
            Y.append(label_mapping[db[j][-1]])  # Convert label using dictionary

    # Transform the original training classes to numbers and add them to the vector Y.
    # Do not forget to remove the instance that will be used for testing in this iteration.
    # For instance, Y = [1, 2, ,...].
    # Convert each feature value to float to avoid warning messages

    # Store the test sample of this iteration in the vector testSample
    testSample = [float(value) for value in db[i][:-1]]
    true_label = label_mapping[db[i][-1]]  # Get the actual class label

    # Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)  # 1NN using Euclidean distance (p=2)
    clf.fit(X, Y)

    # Use your test sample in this iteration to make the class prediction. For instance:
    # class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    class_predicted = clf.predict([testSample])[0]

    # Compare the prediction with the true label of the test instance to start calculating the error rate.
    if class_predicted != true_label:
        error_count += 1

# Print the error rate
error_rate = error_count / total_instances
print(f"LOO-CV Error Rate: {error_rate:.2f}")
