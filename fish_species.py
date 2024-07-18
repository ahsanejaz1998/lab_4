import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load the dataset
df = pd.read_csv('fish.csv')  # Adjust the path to your dataset

# Preprocess the data
# Assuming 'Species' is the target variable and the rest are features
X = df.drop(columns=['Species'])
y = df['Species']

# Encode categorical variables if any (example code, adjust as needed)
X = pd.get_dummies(X, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(model, 'fish_species_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
