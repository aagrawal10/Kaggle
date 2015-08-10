# HelperModule
# Python helper module

import pandas;
import numpy as np;
from sklearn.decomposition import PCA;
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;
from mpl_toolkits.mplot3d import proj3d;

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"];

# This function reads the data from a file and returns it
# in a data frame after some basic processing
# The metadata of file is as follows
# PassengerId, Survived, PClass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
# Name, Ticket, Cabin, Embarked definitely looks useless
def ReadTrainingData(fileName):
	
	# Read the data from the passed file
	data = pandas.read_csv(fileName);

	# Delete useless features
	del data["Name"];
	del data["Ticket"];
	del data["Cabin"];
	del data["Embarked"];

	# Fill missing values of age with median
	ageMedian = data["Age"].median();
	data["Age"] = data["Age"].fillna(ageMedian);

	# Convert sex to int
	data.loc[data["Sex"] == "male", "Sex"] = 0;
	data.loc[data["Sex"] == "female", "Sex"] = 1;

	# Separate features and outcomes from the data
	X = data[features];
	Y = data[["Survived"]];
	X = (X - X.mean()) / (X.max() - X.min());

	return X, Y;

def ApplyPCA(X, n):
	pca = PCA(n_components=n);
	pca.fit(X.T);
	return pca;

# This method transforms the data frame X to n dimensions 
# with PLA and generates a plot to visulalize the data.
def Generate2DPlot(X, Y) :

	# Apply PLA on X to reduce to 2 features
	pca = ApplyPCA(X, 2);
	colormap = np.array(['r', 'b']);
	cat = np.array(Y["Survived"]);

	fig = plt.figure();
	ax = fig.add_subplot(1, 1, 1);
	ax.scatter(pca.components_[0], pca.components_[1], color=colormap[cat]);
	plt.savefig("../Plots/features_2.png");

def Generate3DPlot(X, Y) :

	pca = ApplyPCA(X, 3);
	colormap = np.array(['r', 'b']);
	cat = np.array(Y["Survived"]);

	fig = plt.figure(figsize=(8,8));
	ax = fig.add_subplot(111, projection='3d');
	plt.rcParams['legend.fontsize'] = 10;
	ax.scatter(pca.components_[0], pca.components_[1], pca.components_[2], color=colormap[cat]);
	plt.savefig("../Plots/features_3.png");

def GenerateFeatureWise2DPlot(X, Y) :
	
	# For each 2 features in X generate a plot
	for index1 in range(1,len(features)) :
		for index2 in range(index1,len(features)) :
			if index2 <= index1 :
				continue;
			item1 = features[index1];
			item2 = features[index2];
			x1 = np.array(X[item1]);
			x2 = np.array(X[item2]);
			colormap = np.array(['r', 'b']);
			cat = np.array(Y["Survived"]);

			fig = plt.figure();
			ax = fig.add_subplot(1, 1, 1);
			ax.scatter(x1, x2, color=colormap[cat]);

			plt.xlabel(item1);
			plt.ylabel(item2);
			plt.title(item1 + "vs" + item2);
			plt.savefig("../Plots/" + item1 + "_" + item2 + ".png");
