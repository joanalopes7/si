# Evaluation: exercise 10

from typing import Literal

import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier(Model):
    """
    RandomForestClassifier is an implementation of the Random Forest algorithm, which is an ensemble 
    method that builds multiple decision trees and aggregates their results for classification.

    Parameters
    ----------
    n_estimators: int
        Number of decision trees to use.
    max_features:int
        Maximum number of features to use per tree.
    min_sample_split : int
        Minimum samples allowed in a split.
    max_depth: int
        Maximum depth of the trees.
    mode: Literal['gini', 'entropy']
        The mode to use for calculating the information gain.
    seed: None
        Random seed to use to assure reproducibility.
    
    
    Attributes
    ----------
    trees: list
        The trees and features used for training (initialized as an empty list)

    """

    def __init__(self, min_sample_split: int = 2, max_depth = None, mode: Literal['gini', 'entropy'] = 'gini', n_estimators: int = 100, max_features = None, seed = None,  **kwargs):
        """
        Initializes the RandomForestClassifier with the specified parameters.

        """
        # parameters
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed=seed

        #estimated parameters
        self.trees = []
    

    def _fit (self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Train the decision trees of the random forest.

        """
        if self.seed:
            np.random.seed(self.seed)

        # If max_features is not provided, set it to sqrt(n_features)
        m, n = dataset.X.shape

        if self.max_features is None:
            self.max_features = int(np.sqrt(n))

        #Train 'n_estimators' trees
        for e in range(self.n_estimators):         # e will not be used, just to iteration cicle
            
            #picking n_samples random samples with replacement 
            n_samples = np.random.choice(np.arange(m), size=m, replace=True)
            bootstrap_X = dataset.X[n_samples]
            bootstrap_y = dataset.y[n_samples]

            #picking self.max_features random features without replacement
            n_features = np.random.choice(np.arange(n), size=self.max_features, replace=False)
            bootstrap_X = bootstrap_X[:, n_features]

            # Create a new dataset (bootstrape dataset) with only the selected features
            bootstrap_dataset = Dataset(X=bootstrap_X, y=bootstrap_y)

            # Create and train a decision tree with the bootstrap dataset
            tree = DecisionTreeClassifier(min_sample_split= self.min_sample_split, max_depth=self.max_depth, mode=self.mode)
            tree._fit(bootstrap_dataset)


            # Store the trained tree and the selected features 
            self.trees.append((n_features, tree))

        return self
    

        
    def _predict (self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the labels using the ensemble models.

        """
        # Initialize a list to store predictions from each tree
        tree_predictions = []

        # Iterate through each tree and make predictions using the respective set of features
        for n_features, tree in self.trees:

            # Extract the relevant features
            selected_features = []
            for i in n_features:  # Aqui substituímos "feature_indices" por "n_features"
                selected_features.append(dataset.features[i])
            
            # Use only the features relevant to the tree
            tree_dataset = Dataset(X=dataset.X[:, n_features], 
                            y=dataset.y, 
                            features=selected_features, 
                            label=dataset.label)
            
            #  Get predictions for each tree
            predictions = tree.predict(tree_dataset)
            tree_predictions.append(predictions)

        # Convert tree predictions to a NumPy array  (Before the transposition the predictions are stored by tree ; after the transposition the predictions are organized by example, which facilitates majority voting.)
        tree_predictions = np.array(tree_predictions).T

        # Get the most common predicted class for each sample
        def majority_vote(preds):
            values, counts = np.unique(preds, return_counts=True)
            return values[np.argmax(counts)]

        predictions = np.apply_along_axis(majority_vote, axis=1, arr=tree_predictions)

        return predictions
        

    def _score (self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Computes the accuracy between predicted and real labels.

        """   
        # Compute the accuracy between predicted and real values
        return accuracy(dataset.y, predictions)
    

if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split

    filename= r'C:\Users\Asus\Desktop\MESTRADO BI 2324\2ANO\SIB\portfólio\si\datasets\iris\iris.csv'  
    iris_data = read_csv(filename, sep=",", features=True, label=True)
    train, test = train_test_split(iris_data, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
            n_estimators=100, 
            max_features=4, 
            min_sample_split=2, 
            max_depth=5, 
            mode='gini', 
            random_seed=42
        )
    model.fit(train)
    testing=model.score(test)
    print(f"Model score: {testing:.2f}")
        