import HelperMethods;

X, Y = HelperMethods.ReadTrainingData("../Data/train.csv");

# Generate plots

HelperMethods.Generate2DPlot(X, Y);
HelperMethods.Generate3DPlot(X, Y);
HelperMethods.GenerateFeatureWise2DPlot(X, Y);

#import pandas;
#from sklearn.linear_model import LinearRegression;
#from sklearn.cross_validation import KFold;
#import numpy as np;
#
## Load and process training data first
#
#trainData = pandas.read_csv("../Data/train.csv");
#
## Fill missing values of age with median
#trainData["Age"] = trainData["Age"].fillna(trainData["Age"].median());
#
## Convert Sex to int
#trainData.loc[trainData["Sex"] == "male", "Sex"] = 0;
#trainData.loc[trainData["Sex"] == "female", "Sex"] = 1;
#
## Convert Embarked to int
#trainData["Embarked"] = trainData["Embarked"].fillna("S");
#trainData.loc[trainData["Embarked"] == "S", "Embarked"] = 0;
#trainData.loc[trainData["Embarked"] == "C", "Embarked"] = 0;
#trainData.loc[trainData["Embarked"] == "Q", "Embarked"] = 0;
#
## Features
#predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"];
#
#algm = LinearRegression();
#kf = KFold(trainData.shape[0], n_folds=3, random_state=1);
#
#predictions = []
#for train,cv in kf:
#    train_predictors = (trainData[predictors].iloc[train,:]);
#    train_target = trainData["Survived"].iloc[train];
#    algm.fit(train_predictors, train_target);
#    test_predictions = algm.predict(trainData[predictors].iloc[cv,:]);
#    predictions.append(test_predictions);
#
#predictions = np.concatenate(predictions, axis=0);
#
#predictions[predictions > 0.5] = 1
#predictions[predictions <= 0.5] = 1
#
#accuracy = sum(predictions[predictions == trainData["Survived"]]) / len(predictions);
#print accuracy;
#
## Process Test data and create a results file.
