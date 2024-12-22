from unittest import TestCase

from datasets import DATASETS_PATH

import os

from si.ensemble.stacking_classifier import StackingClassifier
from si.io.data_file import read_data_file
from si.metrics.accuracy import accuracy
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression

class TestVotingClassifier(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

        decision_tree = DecisionTreeClassifier()
        knn = KNNClassifier()
        logistic_regression = LogisticRegression()
        final_model = KNNClassifier()

        self.stacking = StackingClassifier(models=[decision_tree, knn, logistic_regression], final_model=final_model)

    def test_fit(self):

        self.stacking.fit(self.train_dataset)

        #Check if the dimension of the predictions dataset is correct
        self.assertEqual(self.stacking.predictions_dataset.X.shape[0], self.train_dataset.X.shape[0])
        self.assertEqual(self.stacking.predictions_dataset.X.shape[1], len(self.stacking.models))


    def test_predict(self):
        # Train the stacking model
        self.stacking.fit(self.train_dataset)
        
        # Make predictions on the test dataset
        predictions = self.stacking.predict(self.test_dataset)

        # Check if the number of predictions matches the number of test instances
        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0])
    
    def test_score(self):
        # Train the stacking model
        self.stacking.fit(self.train_dataset)
        
        # Calculate the accuracy of the model
        accuracy_ = self.stacking.score(self.test_dataset)
        
        # Expected accuracy based on actual labels and predictions
        expected_accuracy = accuracy(self.test_dataset.y, self.stacking.predict(self.test_dataset))

        # Compare the predicted accuracy and expected accuracy rounded to 2 decimal places
        self.assertEqual(round(accuracy_, 2), round(expected_accuracy, 2))