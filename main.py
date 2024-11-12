import pandas
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# ---------------------------Load the "co2_emissions_data.csv" dataset.------------------------------------
co2_emissions_data = pandas.read_csv("co2_emissions_data.csv")
# print(co2_emissions_data.shape)


# co2_emissions_data.info()
# print("number of dublicate rows:",co2_emissions_data.duplicated().sum())

# print(co2_emissions_data.describe())

# print(co2_emissions_data.describe(include="object"))


# -------------------------------b)  Perform analysis on the dataset to:----------------------:
# # i) check whether there are missing values
# if (co2_emissions_data.isna().sum().sum() == 0):
#     print("There are no missing values in the dataset.")
# print(co2_emissions_data.isna().sum())
#  ii) check whether numeric features have the same scale
# print(co2_emissions_data.describe())

# for column in co2_emissions_data.select_dtypes(include=[np.number]):
#     print(
#         f"{column}: Min = {co2_emissions_data[column].min()}, Max = {co2_emissions_data[column].max()}")


# iii) visualize a pairplot in which diagonal subplots are histograms
# sns.pairplot(co2_emissions_data, diag_kind='hist')
# plt.show()
# #  iv) visualize a correlation heatmap between numeric columns
#                  # Select only numeric columns
# numeric_data = co2_emissions_data.select_dtypes(include=[np.number])
#               #correlation matrix for the numeric columns.
# sns.heatmap(numeric_data.corr(), annot=True, cbar=True, )
# plt.show()
# ---------------------------c)Preprocess the data such that---------------------------:
# don't modify the original dataset
co2_emissions_data_pre = co2_emissions_data.copy()
# i)  the features and targets are separated
features = co2_emissions_data_pre.drop(
    columns=['CO2 Emissions(g/km)', 'Emission Class'])

# Target for linear regression
co2_amount = co2_emissions_data_pre['CO2 Emissions(g/km)']
emission_class = co2_emissions_data_pre['Emission Class']
# ii) categorical features and targets are encoded
enc = OrdinalEncoder(categories=[['VERY LOW', 'LOW', 'MODERATE', 'HIGH']])
# Target is encoded into [0, 1, 2, 3]
encoded_emission_class = co2_emissions_data_pre
encoded_emission_class[['Emission Class']] = enc.fit_transform(
    co2_emissions_data_pre[['Emission Class']])

# print(encoded_emission_class)
# iii)the data is shuffled and split into training and testing sets
# split dataset into 80% training, 20% testing
# for linear regression
X_train, X_test, y_train, y_test = train_test_split(
    features, co2_amount, test_size=0.2, random_state=0)
# iv)numeric features are scaled

# make standardization for data

#get all numeric features  to apply scaling
numericFeatures = X_train.select_dtypes(include=[np.number]).columns
Stdscaler = StandardScaler()
X_train = Stdscaler.fit_transform(X_train[numericFeatures])
X_test = Stdscaler.transform(X_test[numericFeatures])

# Reshape y_train and y_test to 2D arrays for scaling
y_train = y_train.values.reshape(-1, 1)  # Reshape to 2D (N, 1)
y_test = y_test.values.reshape(-1, 1)  # Reshape to 2D (N, 1)

# Apply scaling to y_train and y_test
y_train = Stdscaler.fit_transform(y_train)  # Fit and transform y_train
y_test = Stdscaler.transform(y_test)  # Use the same scaler to transform y_test

# get the needed data as dataframes

# Convert the scaled NumPy arrays back to DataFrames with the original column names
X_train_scaled = pandas.DataFrame(X_train, columns=numericFeatures)
X_test_scaled = pandas.DataFrame(X_test, columns=numericFeatures)



# #after scaling std=>close to 1 or equal 1, mean =>close to or equal 0


# print("X_Training data mean:", X_train.mean())
# print("X_Training data std:", X_train.std())
# print("X_Testing data mean:", X_test.mean())
# print("X_Testing data std:", X_test.std())


# iii)the data is shuffled and split into training and testing sets
# iv)numeric features are scaled


# feat = X_train[['Engine Size(L)', 'Fuel Consumption Comb (L/100 km)']]

# targ = y_train[["CO2 Emissions(g/km)"]]

# print(feat)
# print(targ)




# Selecting two features (e.g., 'Engine Size' and 'Fuel Consumption Comb (L/100 km)') and target 'CO2 Emissions'
X_train = X_train_scaled[['Engine Size(L)', 'Fuel Consumption Comb (L/100 km)']].values
y_train = y_train.flatten()  # Flatten y_train to a 1D array for proper calculations

# Initializing parameters
m = X_train.shape[0]  # number of training examples
w1 = 0
w2 = 0
b = 0 
learning_rate = 0.1  # Learning rate
iterations = 100  # Number of iterations for gradient descent
cost_history = []  # To store the cost values for plotting

# gradient descent algorithm
for i in range(iterations):

    # initial values for sumition
    w1_gradient = 0
    w2_gradient = 0
    b_gradient = 0
    sumSquareError = 0

    # loop to calculate sumition
    for count in range(m):
        # extract x1, x2 from the 2 features table for each data row
        x1,x2 = X_train[count]
        # extract CO2 Emissions value from the target table for each data row
        y = y_train[count]
        # calculate function of leanier regression model
        f = w1 * x1 + w2 * x2 + b
        # calculate distance between predected data and actual data points
        distance = f-y
        # sum the distance
        w1_gradient += distance * x1
        w2_gradient += distance * x2
        b_gradient += distance

        # sum square distance for cost calculation
        sumSquareError += distance ** 2


    # calcualte the new w1,w2,b
    w1 = w1  - (learning_rate / m) * w1_gradient
    w2 = w2 - (learning_rate / m) * w2_gradient
    b  = b - (learning_rate / m) * b_gradient

    # Compute cost for each iteration
    cost = (1 / (2 * m)) * sumSquareError
    cost_history.append(cost)
    
    # Optional: Print cost every 100 iterations for monitoring
    if i % 10 == 0:
        print(f"Iteration {i}: Cost = {cost}")

# Plot cost over iterations
plt.plot(range(iterations), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction over Time")
plt.show()

# Evaluate the model on the test set
X_test = X_test_scaled[['Engine Size(L)', 'Fuel Consumption Comb (L/100 km)']].values
y_test_pred = np.dot(X_test, [w1,w2]) + b
r2 = r2_score(y_test, y_test_pred)
print(f"R2 Score on Test Set: {r2}")