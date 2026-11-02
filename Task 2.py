Task 2

from sklearn.linear_model import LinearRegression

# Sample features: [Release Year, Duration in minutes, Votes]
X = [
    [2020, 120, 1000],
    [2021, 100, 500],
    [2019, 150, 3000],
    [2022, 110, 1500],
    [2018, 140, 2500],
    [2023, 125, 2000]
]

# Example ratings out of 10
y = [8.0, 6.5, 9.0, 7.0, 8.5, 7.8]

# Train the regression model
model = LinearRegression()
model.fit(X, y)

# Predict rating for a new movie
new_movie = [[2024, 130, 1800]]
rating = model.predict(new_movie)

print("Predicted movie rating:", round(rating[0], 2))