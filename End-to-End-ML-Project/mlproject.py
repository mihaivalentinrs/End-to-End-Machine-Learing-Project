import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from scipy import stats
import os
import tarfile
import urllib
import joblib

#Modelul va trebui sa prezica un rezultat pentru "median housing price"
#=> Vom aborda un model de regresie in functie de plot ul initial pe care il realizam
#Creez un set de date de test

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("../datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

#functie care imi permite la fiecare apel sa downloadez dataset ul in directorul proiectului
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path) #se poate mai sigur si fara filter = "data"
    housing_tgz.close()
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
#functie alternativa pentru download local de date
def download_csv_housing_data():
    dataset = pd.read_csv('housing_dataset.csv')
    print("\n", dataset.head())
    return dataset

def split_train_test(data, test_ratio):

    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#Functie pentru stratificarea datelor impartite in test si antrenament
def stratified_data(housing):
    #random_state = 42 reseteaza split-ul la acelasi rezultat - nu e random
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    return strat_train_set, strat_test_set

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
#constante globale initializate cu 3 si la care se itereaza +1 la fiecare element secvential
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix]/X[:,households_ix]
        population_per_household = X[:, population_ix]/X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())

def main():
    #Folosesc functiile implementate pentru a descarca setul de date
    fetch_housing_data()
    dataset = load_housing_data()
    print("\nVerifying the integrity of the data...")
    print("\nStatistical calculations of the downloaded datasaet: ")
    print(dataset.describe())
    longitude = np.array(dataset.iloc[:,0])
    latitude = np.array(dataset.iloc[:,1])
    data_matrix = np.column_stack((longitude,latitude))
    print("\n-----Longitutude-----||-----Latitude-----")
    print(data_matrix)
    print("\nThe dataset has been saved successfully!")


    #Observam setul de date ales pentru calculul regresiei
    plt.title("| Plotting the desired variables for regression observations |")
    plt.scatter(longitude, latitude, label="Longitude")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    #Display-ul histogramelor tuturor variabilelor din setul de date
    dataset.hist(bins = 50, figsize = (20, 15))
    plt.suptitle("Histograms for data understanding", fontstyle='italic')
    plt.show()


    #Seturi de antrenament si de test stratificate
    dataset["income_cat"] = pd.cut(dataset["median_income"],
                                   bins = [0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels = [1, 2, 3, 4, 5])
    strat_train_set, strat_test_set = stratified_data(dataset)



    #Pregatirea datasetului pentru algortimi de machine learning
    dataset = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy() #creez o copie a coloanei de date cu valori: median_house_value
    #practic variabila x (cauza) din modelul de regresie

    imputer = SimpleImputer(strategy = "median")
    dataset_num = dataset.drop("ocean_proximity", axis = 1)
    imputer.fit(dataset_num)

    #Gestionare datelor categorice (non-numerice)
    #Atribui valorilor categorice valori numerice pentru a putea crea o distributie palpabila
    dataset_cat = dataset[["ocean_proximity"]]
    ordinal_encoder = OrdinalEncoder()
    dataset_cat_encoded = ordinal_encoder.fit_transform(dataset_cat)
    print("\nOcean proximity encoded with OrdinalEncoder: ", dataset_cat_encoded)
    cat_encoder = OneHotEncoder()
    dataset_cat_1hot = cat_encoder.fit_transform(dataset_cat)
    print("\nOcean proximity encoded with OneHotEncoder: ", dataset_cat_1hot)

    attributes_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
    dataset_extra_attributes = attributes_adder.transform(dataset.values)
    #Standardizarea datelor => aducerea la distributia normala
    num_pipeline = Pipeline([
                            ('imputer', SimpleImputer(strategy = "median")),
                             ('attributes_adder', CombinedAttributesAdder()),
                             ('standard_scaler', StandardScaler())
    ])
    dataset_num_tr = num_pipeline.fit_transform(dataset_num)
    num_attributes = list(dataset_num)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ("cat", OneHotEncoder(), cat_attribs)
    ])
    housing_prepared = full_pipeline.fit_transform(dataset)

    #Select and train a model
    #Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    joblib.dump(lin_reg, "Linear_Regression_Model.joblib")
    lin_reg_loaded = joblib.load("Linear_Regression_Model.joblib")
    housing_predictions = lin_reg.predict(housing_prepared)
    linear_mse = mean_squared_error(housing_labels, housing_predictions)
    root_mse = np.sqrt(linear_mse)
    print("\nPrediction error: ", root_mse)
    print("\nVerifying the equality of the saved data. Are the coefficients identical?\n Answer:", np.allclose(lin_reg.coef_, lin_reg_loaded.coef_))
    #Decision Tree
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)
    joblib.dump(tree_reg, "../DecisionTREE_Regression_Model.joblib")
    tree_reg_loaded = joblib.load("../DecisionTREE_Regression_Model.joblib")
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print(tree_rmse)
    #Cross evaluation scores
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring = "neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    display_scores(tree_rmse_scores)
     #Grid Search
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2,4,6,8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, param_grid, cv = 5,
                               scoring='neg_mean_squared_error',
                               return_train_score = True)
    grid_search.fit(housing_prepared, housing_labels)
    joblib.dump(grid_search, "GridSearch_Model.joblib")
    grid_search_loaded = joblib.load("GridSearch_Model.joblib")
    print("\nBest parameters: ", grid_search.best_params_)
    print("\nBest model found: ", grid_search.best_estimator_)

    #evaluarea sistemului pe setul de date
    final_model = grid_search.best_estimator_

    X_test = strat_test_set.drop("median_house_value", axis=1)
    Y_test = strat_test_set["median_house_value"].copy()

    X_test_prepared = full_pipeline.transform(X_test)

    final_predictions = final_model.predict(X_test_prepared)

    final_mse = mean_squared_error(Y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)

    #Interval de incredere pentru probabilitate de 95%
    confidence = 0.95
    squared_errors = (final_predictions - Y_test)**2
    #(np.sqrt(stats.t.interval(confidence, len(squared_errors)-1,
    #                      loc = squared_errors.mean(),
    #                         scale =stats.sem(squared_errors)
    #                        ))
    #      )




    

if __name__ == '__main__':
    main()









