import pandas
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

# ---------------------------Load the "co2_emissions_data.csv" dataset.------------------------------------
co2_emissions_data = pandas.read_csv("co2_emissions_data.csv")
print(co2_emissions_data)

# -------------------------------b)  Perform analysis on the dataset to:----------------------:
# i) check whether there are missing values
if (co2_emissions_data.isna().sum().sum() == 0):
    print("There are no missing values in the dataset.")
print(co2_emissions_data.isna().sum())
#  ii) check whether numeric features have the same scale
print(co2_emissions_data.describe())

for column in co2_emissions_data.select_dtypes(include=[np.number]):
    print(
        f"{column}: Min = {co2_emissions_data[column].min()}, Max = {co2_emissions_data[column].max()}")


# iii) visualize a pairplot in which diagonal subplots are histograms
sns.pairplot(co2_emissions_data, diag_kind='hist')
plt.show()
# #  iv) visualize a correlation heatmap between numeric columns
#                  # Select only numeric columns
numeric_data = co2_emissions_data.select_dtypes(include=[np.number])
#               #correlation matrix for the numeric columns.
sns.heatmap(numeric_data.corr(), annot=True, cbar=True, )
plt.show()
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

print(encoded_emission_class)
# iii)the data is shuffled and split into training and testing sets
# split dataset into 80% training, 20% testing
# for linear regression
X_train, X_test, y_train, y_test = train_test_split(
    features, co2_amount, test_size=0.2, random_state=0)
# iv)numeric features are scaled
# iii)the data is shuffled and split into training and testing sets
# iv)numeric features are scaled
