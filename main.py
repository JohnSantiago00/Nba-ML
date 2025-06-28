import pandas as pd

# Small manually created NBA player dataset
data = {
    "PTS": [27.1, 15.3, 22.5, 12.0, 18.1],
    "REB": [7.4, 4.1, 5.3, 3.2, 6.0],
    "AST": [7.0, 2.8, 4.0, 1.9, 3.4],
    "STL": [1.6, 0.8, 1.1, 0.5, 0.9],
    "BLK": [0.5, 0.2, 0.6, 0.1, 0.3],
    "AllStar": [1, 0, 1, 0, 0]
}

# Create DataFrame
df = pd.DataFrame(data)
print(df)

# Correct casing: "AllStar"
X = df.drop("AllStar", axis=1)
y = df["AllStar"]

print("\nX:")
print(X)
print("\nY:")
print(y)

from sklearn.model_selection import train_test_split

#Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#print the results
print("\nTraining data (X_train):")
print(X_train)

print("\nTraining labels (y_train):")
print(y_train)

print("\nTesting data (X_test):")
print(X_test)

print("\yTesting labels (y_test):")
print(y_test)

from sklearn.linear_model import LogisticRegression 

#Create and train the model
model = LogisticRegression()
model.fit(X_train,y_train)

#Predict on the test data
predictions = model.predict(X_test)

print("\nPredicted label(s):")
print(predictions)

print("\nActual label(s):")
print(y_test.values)



