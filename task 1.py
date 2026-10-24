Task 1 

from sklearn.tree import DecisionTreeClassifier

# Sample data: [Age, Fare, Passenger Class]
X = [
    [22, 7.25, 3],
    [38, 71.28, 1],
    [26, 7.92, 3],
    [35, 53.10, 1],
    [19, 8.05, 3],
    [40, 30.00, 2]
]

# 0 = Did not survive, 1 = Survived
y = [0, 1, 1, 1, 0, 1]

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Predict for a new passenger
new_passenger = [[25, 50.00, 2]]
prediction = model.predict(new_passenger)

print("Survived" if prediction[0] == 1
      else "Did not survive")