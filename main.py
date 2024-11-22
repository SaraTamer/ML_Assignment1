import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import SGDClassifier


# a) Load dataset
def load_dataset(filename):
    data = pd.read_csv(filename)
    print(data.shape)
    data.info()
    print("Number of duplicate rows:", data.duplicated().sum())
    print(data.describe())
    print(data.describe(include="object"))
    return data


# b) Dataset analysis
def analyze_dataset(data):
    # i) Check for missing values
    if data.isna().sum().sum() == 0:
        print("There are no missing values in the dataset.")
    else:
        print(data.isna().sum())

    # ii) Check numeric feature scale
    print(data.describe())
    for column in data.select_dtypes(include=[np.number]):
        print(f"{column}: Min = {data[column].min()}, Max = {data[column].max()}")

    # iii) Visualize pairplot
    sns.pairplot(data, diag_kind='hist')
    plt.show()

    # iv) Visualize correlation heatmap
    numeric_data = data.select_dtypes(include=[np.number])
    sns.heatmap(numeric_data.corr(), annot=True, cbar=True)
    plt.show()


def preprocess_data(data):
    # i) Separate features and targets
    features = data.drop(columns=['CO2 Emissions(g/km)', 'Emission Class'])
    co2_amount = data['CO2 Emissions(g/km)']  # Target for linear regression
    emission_class = data['Emission Class']  # Target for logistic regression

    # ii) Encode categorical data for emission_class
    enc = OrdinalEncoder(categories=[['VERY LOW', 'LOW', 'MODERATE', 'HIGH']])
    emission_class_encoded = enc.fit_transform(emission_class.values.reshape(-1, 1))

    # using frequency encoding for the other categorical features
    data['Make'] = data['Make'].map(data['Make'].value_counts())
    data['Model'] = data['Model'].map(data['Model'].value_counts())
    data['Vehicle Class'] = data['Vehicle Class'].map(data['Vehicle Class'].value_counts())
    data['Transmission'] = data['Transmission'].map(data['Transmission'].value_counts())


    # iii) Split data into training and testing sets
    X_train, X_test, y_train_co2, y_test_co2, y_train_class, y_test_class = train_test_split(
        features, co2_amount, emission_class_encoded, test_size=0.2, random_state=0)

    # iv) Scale numeric features
    numeric_features = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled = scaler.transform(X_test[numeric_features])

    # v) Scale CO2 targets (optional for linear regression)
    y_train_co2 = y_train_co2.values.reshape(-1, 1)
    y_test_co2 = y_test_co2.values.reshape(-1, 1)
    y_train_co2 = scaler.fit_transform(y_train_co2)
    y_test_co2 = scaler.transform(y_test_co2)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train_co2,
        y_test_co2,
        y_train_class.flatten(),
        y_test_class.flatten(),
        scaler,
        numeric_features
    )


# d) Linear regression with gradient descent
def linear_regression_gradient_descent(X_train, y_train, learning_rate, iterations):
    m = X_train.shape[0]
    w1, w2, b = 0, 0, 0
    cost_history = []

    for i in range(iterations):
        w1_gradient, w2_gradient, b_gradient = 0, 0, 0
        sum_square_error = 0

        for count in range(m):
            x1, x2 = X_train[count]
            y = y_train[count]
            f = w1 * x1 + w2 * x2 + b
            distance = f - y
            w1_gradient += distance * x1
            w2_gradient += distance * x2
            b_gradient += distance
            sum_square_error += distance ** 2

        w1 -= (learning_rate / m) * w1_gradient
        w2 -= (learning_rate / m) * w2_gradient
        b -= (learning_rate / m) * b_gradient

        cost = (1 / (2 * m)) * sum_square_error
        cost_history.append(cost)

        if i % 10 == 0:
            print(f"Iteration {i}: Cost = {cost}")

    return w1, w2, b, cost_history


def prepare_emission(data):

    emission_class = data['Emission Class']
    y_train, y_test = train_test_split(
        emission_class, test_size=0.2, random_state=0)

    return y_train, y_test


# e) Fit a logistic regression model
def logistic_regression_gradient_descent(X_train, y_train, X_test, y_test):
    logModel = SGDClassifier(loss='log_loss', random_state=0, max_iter=1000, tol=1e-3)
    logModel.fit(X_train, y_train)
    y_predict = logModel.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    return accuracy


def main():
    # a) Load the dataset
    data = load_dataset("co2_emissions_data.csv")

    # b) Analyze the dataset
    analyze_dataset(data)

    # c) Preprocess the data
    X_train, X_test, y_train_1, y_test_1, y_train_2, y_test_2, scaler, numeric_features = preprocess_data(data)

    # Select two features based on correlation heatmap
    X_train = pd.DataFrame(X_train, columns=numeric_features)
    X_test = pd.DataFrame(X_test, columns=numeric_features)
    selected_features = ['Engine Size(L)', 'Fuel Consumption Comb (L/100 km)']
    X_train = X_train[selected_features].values
    X_test = X_test[selected_features].values
    y_train_1 = y_train_1.flatten()

    # d) Train the linear regression model
    learning_rate = 0.1
    iterations = 100
    w1, w2, b, cost_history = linear_regression_gradient_descent(
        X_train, y_train_1, learning_rate, iterations
    )

    # Plot cost reduction
    plt.plot(range(iterations), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Reduction over Time")
    plt.show()

    # Evaluate on the test set
    y_test_pred = np.dot(X_test, [w1, w2]) + b
    r2 = r2_score(y_test_1, y_test_pred)

    print(f"R2 Score on Test Set: {r2}")

    # e) Fit a logistic regression model
    print(f"Logistic Regression Accuracy: {logistic_regression_gradient_descent(X_train, y_train_2, X_test, y_test_2)}")


if __name__ == "__main__":
    main()
