'''Perform regression over californian house prices'''

from pathlib import Path
import sys
import tarfile
import urllib.request
from zlib import crc32

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

assert sys.version_info >= (3, 7)

def load_housing_data():
    '''Retrieves and extracts the dataset'''
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

def shuffle_and_split_data(data, test_ratio):
    '''Splits the supplied data into training and a test set of data in accordance
    with the supplied ratio. This approach has downsides as it is random, meaning
    that each time this split is performed different records will appear in the
    training or test sets.'''
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def is_id_in_test_set(identifier, test_ratio):
    '''Checks to see if an id is in the test set'''
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    '''Splits data into training and test sets by hashing the id'''
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[-in_test_set], data.loc[in_test_set]

housing = load_housing_data()

# train_set, test_set = shuffle_and_split_data(housing, 0.2)

# Add an index column - fails if new data is not always appended to the end of the dataset
#housing_with_id = housing.reset_index()
#train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

# to counter the above, create a unique id by joining the long and lat values
#housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
#train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")

 # Leverage scikit-learn
train_set, test_set = train_test_split(housing,
                                       test_size=0.2,
                                       random_state=42)

print(f'{len(train_set)} houses in the training set, {len(test_set)} houses in the test set.')

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.savefig("Histogram of income categories")

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits=[]
for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]

strat_train_set, strat_test_set = train_test_split(housing,
                                                     test_size=0.2,
                                                     stratify=housing["income_cat"],
                                                     random_state=42)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    set_.drop("ocean_proximity", axis=1, inplace=True)

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
plt.savefig("A geographical scatterplot of the data")

housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.savefig("California housing prices")

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.savefig("Scatter matrix")

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)
plt.savefig("Median income versus median house value")
