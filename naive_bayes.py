#-------------------------------------------------------------------------
# AUTHOR: Britney Collier
# FILENAME: naive_bayes.py
# SPECIFICATION: Implement a Naïve Bayes classifier to predict weather-based play decisions.
# FOR: CS 4210- Assignment #2
# TIME SPENT: <fill this in>
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas.
# You have to work here only with standard dictionaries, lists, and arrays.

# Importing necessary Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

# Reading the training data in a csv file
# --> add your Python code here
dbTraining = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        dbTraining.append(row)

# Transform the original training features to numbers and add them to the 4D array X.
# For instance, Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
# --> add your Python code here
outlook = {"Sunny": 1, "Overcast": 2, "Rain": 3}
temperature = {"Hot": 1, "Mild": 2, "Cool": 3}
humidity = {"High": 1, "Normal": 2}
wind = {"Weak": 1, "Strong": 2}

X = []
Y = []

for row in dbTraining:
    X.append([outlook[row[1]], temperature[row[2]], humidity[row[3]], wind[row[4]]])
    
# Transform the original training classes to numbers and add them to the vector Y.
# For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
# --> add your Python code here
play_tennis = {"Yes": 1, "No": 2}
for row in dbTraining:
    Y.append(play_tennis[row[5]])

# Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

# Reading the test data in a csv file
# --> add your Python code here
dbTest = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        dbTest.append(row)

# Printing the header of the solution
# --> add your Python code here
print("Day\tOutlook\tTemperature\tHumidity\tWind\tPlayTennis\tConfidence")

# Use your test samples to make probabilistic predictions.
# For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
# --> add your Python code here
for row in dbTest:
    test_X = [[outlook[row[1]], temperature[row[2]], humidity[row[3]], wind[row[4]]]]
    prob = clf.predict_proba(test_X)[0]
    predicted_class = clf.predict(test_X)[0]

    confidence = max(prob)  # Get the highest confidence value
    if confidence >= 0.75:
        predicted_label = "Yes" if predicted_class == 1 else "No"
        print(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t{predicted_label}\t{confidence:.2f}")
