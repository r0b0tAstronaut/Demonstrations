import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    cross_val_score,
    cross_val_predict
)
from sklearn.metrics import root_mean_squared_error, make_scorer, r2_score
import matplotlib.pyplot as plt

# This dataset was obtained from Kaggle. The output is formatted for
#   submisssion to the Kaggle competition.
# Anna Montoya, DataCanary. (2016). House Prices - Advanced Regression
# Techniques. Kaggle.
# https://kaggle.com/competitions/house-prices-advanced-regression-techniques

# Set the path to inputs and outputs
PATH_TO_HOUSE_TRAIN = "./data/house_prices/train.csv"
PATH_TO_HOUSE_TEST  = "./data/house_prices/test.csv"
SAVE_DIR = "./results/housing"

# Create the output path
os.makedirs(SAVE_DIR, exist_ok=True)

# Read in the house prices data
housing = pd.DataFrame(pd.read_csv(PATH_TO_HOUSE_TRAIN))

# Before processing, we need to clean the data.
#   Entries that don't make sense need to be fixed. Any data with missing
#       entries needs to be addressed. Most NaNs really mean "not applicable",
#       rather than "unknown". It is all categorical data, so we can replace
#       np.nan with "Na"
#   Any categorical data that is not of object type needs to be converted to
#       object type and vice versa. Ex: MSSubClass is read as an int, but that
#       each int corresponds to a category, and those categories are not
#       ordinal.
#   Any categorical data into one-hot matrices and merge with our dataset to
#       process with sklearn GradientBoostingRegressor

# First up, data that doesn't make sense. Normally a SME (subject matter
#   expert) would be consulted, but I don't have access to one for this data.
#   Presumably, this data was pulled from MLS (multiple listing service) and
#   anything that seems off is an error. I'm sure if I dug deeper, and
#   especially if I spoke with an SME then I'd find more anomalies.

# How can a house have no masonry veneer (MasVnrType is category NA), but have
#   a non-zero masonry veneer area (MasVnrArea > 0). There's only 5 entries
#   where this a problem.
housing["MasVnrArea"] = np.where(
    pd.isnull(housing["MasVnrType"]), 0.0, housing["MasVnrArea"]
)

# Presumably, houses with NaN lot frontage have 0 lot frontage. But there does
#   not seem to be an easy relation such as being agricultural land, or an
#   apartment. With ~1/6 of the data having NaN for LotFrontage, I decided to
#   simply drop this feature for now.
housing.drop("LotFrontage", axis=1, inplace=True)

# This is the only entry where Electrical is NaN. There's not supposed to be
#   a None category for electrical. 90% of the entries have SBrkr
#   (Standard Breaker), so we can guess this property has that type.
housing.loc[1379, "Electrical"] = "SBrkr"

# There's one entry (332) where BsmtFinType2 indicates no basement, but all
#   the other basement data indicates there is a basement. Similar for 
#   BsmtExposure and another entry (948). I decided to just drop these entries
# All other entries have all NaNs or no NaNs for basement related features
housing.drop([332, 948], axis=0, inplace=True)

# In this next step, I assume that for these data types, "nan" means none or
#   not applicable as per the data description file. It is possible that these
#   are actually unknown, which is bad data practice on the part of the data
#   aggregator. If a feature like BsmtQual is NaN, all other basement features
#   should be NaN, I checked all the related features and entries for such
#   cases before setting them to "Na".
nan_features = ["Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure",
                "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType",
                "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond",
                "PoolQC", "Fence", "MiscFeature"]
for col in nan_features:
    housing[col] = np.where(pd.isnull(housing[col]), "Na", housing[col])

# In this final step, any data that is supposed to be categorical is converted
#   to such. Similarly if the data is int/float but read as an object.
# For the GarageYrBlt, if there is no garage I'll enter the year the house was
#   built that way we can make GarageYrBlt an int
# For MSSubClass, this is not directional or numerical so I'll just convert
#   to a string
housing["GarageYrBlt"] = np.where(
    housing["GarageYrBlt"] == "Na",
    housing["YearBuilt"],
    housing["GarageYrBlt"]
).astype(str)
housing["GarageYrBlt"] = housing["GarageYrBlt"].astype(float).astype(int)
housing["MSSubClass"] = housing["MSSubClass"].astype(str)

# Convert categorical data to one-hot vectors
housing = pd.get_dummies(
    housing,
    columns=housing.select_dtypes(include=["object"]).columns,
    dtype='bool'
)

# Then "150" category for MSSubClass appears in the test data, but never
#   appeared in the training data. Effectively, we need to add a category for
#   "150" to the one-hot encodings for training to be able to use the test data
housing.insert(37, "MSSubClass_150", False)

# Split the data into features and labels (drop Id since it isn't predictive)
housing_prices = housing['SalePrice']
housing.drop(['Id', 'SalePrice'], axis=1, inplace=True)

# Set the parameter search space
param_grid = {
    "learning_rate": (0.0075, 0.01, 0.02, 0.04),
    "n_estimators": (750, 1000, 2000, 3000),
    "max_depth": (2, 3, 4),
    "min_samples_split": (2, 3, 4),
}

# This is the (negative) of the error function used by Kaggle. GridSearchCV
#   will find the set of parameters with the maximum score in the
#   search space
score_func = make_scorer(
    lambda x, y : -root_mean_squared_error(np.log(x), np.log(y))
)

# Perform the grid search over the given parameters with 5-fold cross
#   validation. "Best" is defined by score_func.
# The subsample rate is set to 0.5, which adds bagging to the training.
#   Varying the subsample rate between 0.5-1.0 did not seem to change
#   performance by much, but did reduce training time since the trees only
#   train on half the data. In general this should reduce model variance.
gs = GridSearchCV(
    GradientBoostingRegressor(loss="squared_error", subsample=0.5),
    param_grid,
    scoring=score_func,
    cv = 5,
    n_jobs = 4,
    pre_dispatch='n_jobs',
)
gs.fit(housing, housing_prices)
print("Best Params:", gs.best_params_)

# Get the Kaggle score. From Kaggle: "Submissions are evaluated on Root-Mean-
#   Squared-Error (RMSE) between the logarithm of the predicted value and the
#   logarithm of the observed sales price."
# This is done so the error is relative, and more equally weights the error of
#   cheap and expensive houses. Notably though, this metric also incurs a
#   higher penalty for underestimation than overestimation.
# A more potentially more intuitive metric is given below
print("Kaggle Error:", -gs.best_score_)

# To evaluate, we can get the average performance over a cross validation
housing_pred = cross_val_predict(
    GradientBoostingRegressor(**gs.best_params_),
    housing,
    housing_prices,
    cv=5,
    n_jobs=-1
)

# A more intuitive metric than the RMSE of log difference is correlation.
# In my trial I get a Pearson coefficient of ~89%, which is great performance.
# That translates to ~89% of the the sale price variance can be explained by
#   this model.
print("Validation Coefficient of Determination:",
      r2_score(housing_prices, housing_pred)
)

# Plot the true vs predicted sale price on the validation data
plt.figure("true_vs_pred")
plt.scatter(housing_prices/1000, housing_pred/1000, alpha=0.2, linewidths=0)
plt.xlabel("True House Price ($1,000)")
plt.ylabel("Predicted House Price ($1,000)")
one2one = np.array([min(min(housing_prices), min(housing_pred)),
                    max(max(housing_prices), max(housing_pred))])/1000
plt.plot(one2one, one2one, c='g', linestyle="--")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.savefig(
    os.path.join(SAVE_DIR, "house_price_validation_performance.png"),
    bbox_inches="tight"
)
plt.close("true_vs_pred")

# Now that set of well performing parameters is knowsn, a model with those
#   parameters can be trained on all the data. This model will be used for
#   the test data.
reg = GradientBoostingRegressor(**gs.best_params_)
reg.fit(housing, housing_prices)

# For the test data, we can't remove entries. Only modified the data similarly
#   to how we did for the input data. For entries that can't be removed, we can
#   set them to the average, or the most common value
test = pd.DataFrame(pd.read_csv(PATH_TO_HOUSE_TEST))

# We didn't use the LotFrontage for training, so we can't use it during testing
test.drop("LotFrontage", axis=1, inplace=True)

# Simlar to the training data, we set the masonry veneer area to zero where
#   there is no masonry veneer
test["MasVnrArea"] = np.where(
    pd.isnull(test["MasVnrType"]),
    0.0,
    test["MasVnrArea"]
)

# For the basement, there was some funny business for two entries where all but
#   one basement features indicate a basement, and the one would indicate no
#   basement. For the rest, instead of NULL, use Na for "No Basement"
test.loc[[757, 758], "BsmtQual"]  = "TA"
bsmt = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
test.loc[pd.isnull(test["BsmtQual"]), bsmt]  = "Na"

# Similar to the basements, there are multiple features for the garage that we
#   can use ot build a consensus if one feature is "wrong". GarageType seems to
#   be "correct" in all cases, so we can set Na for "No Garage" based on that
garage = ["GarageCond", "GarageQual", "GarageFinish", "GarageType"]
test.loc[pd.isnull(test["GarageType"]), garage] = "Na"

# The following entries have NaN when that shouldn't be possible. We can set
#   them to the average or the mode as applicable. The test data seems to be
#   much worse than the training data in this regard

# Similar to the train data, nans becomes 0.0, the mode, or "Na"
zero_features = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", 
                 "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea"]
for col in zero_features:
    test.loc[pd.isnull(test[col]), col] = 0.0

mode_features = ["Utilities", "MSZoning", "Exterior1st", "Exterior2nd",
                 "BsmtCond", "BsmtExposure", "Functional", "KitchenQual",
                 "SaleType"]
for col in mode_features:
    test.loc[pd.isnull(test[col]), col] = test[col].mode()[0]

# Set all remaining features
nan_features = ["Alley", "MasVnrType", "FireplaceQu", "GarageYrBlt",
                "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence",
                "MiscFeature", ]
for col in nan_features:
    test[col] = np.where(pd.isnull(test[col]), "Na", test[col])

# For Na GarageYrBlt, set it to YearBuilt then convert feature to an integer
test["GarageYrBlt"] = np.where(
    test["GarageCond"] == "Na", test["YearBuilt"], test["GarageYrBlt"]
).astype(str)
test["GarageYrBlt"] = test["GarageYrBlt"].astype(float).astype(int)

# Convert MSSubClass to categorical
test["MSSubClass"] = test["MSSubClass"].astype(str)

# Convert categorical entries to one-hot vectors
test = pd.get_dummies(
    test,
    columns=test.select_dtypes(include=["object"]).columns,
    dtype='bool'
)
test.drop(['Id'], axis=1, inplace=True)

# Some categories of some features appeared in the train data, but never in
#   the test data, so the one-hot encodings would differ. We need to fix that
#   to use the regression model. Insert all False for those columns.
test_columns_set = set(test.columns)
for i, column in enumerate(housing.columns):
    if column not in test_columns_set:
        test.insert(i, column, False)

# Get the prices of the testing data houses
test_prices = reg.predict(test)

# Output the results in the format accepted by Kaggle
output = pd.DataFrame({"Id":range(1461, 2920), "SalePrice":test_prices})
output.to_csv(
    os.path.join(SAVE_DIR, "house_price_submission.csv"),
    index=False
)
