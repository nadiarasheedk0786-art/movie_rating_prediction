import pandas as pd

# Load the dataset
ratings = pd.read_csv("ratings.csv")
# Drop useless index column
ratings = ratings.drop(columns=["Unnamed: 0"])
# Fill missing metascore with mean
ratings["metascore"] = ratings["metascore"].fillna(ratings["metascore"].mean())

X = ratings[["year", "metascore", "votes"]]
y = ratings["imdb"]
# Check first 5 rows
print(ratings.head())

# Check basic info
print(ratings.info())


# Step 3: Train ML Model (Linear Regression)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)
import matplotlib.pyplot as plt

# Scatter plot: Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Movie Ratings")
plt.show()



