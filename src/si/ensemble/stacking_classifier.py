import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class StackingClassifier(Model):

    """
    Ensemble classifier that 

    Parameters
    ----------
    models : array-like, shape = [n_models]
        Different models for the ensemble.

    Attributes
    ----------
    """
    def __init__(self, models, final_model, **kwargs):
        """
        Initialize the ensemble classifier.

        Parameters
        ----------
        models: array-like, shape = [n_models]
            Initial set of models
        final_model: 
            The model to make the final predictions.

        """
        # parameters
        super().__init__(**kwargs)
        self.models = models
        self.final_model = final_model
        self.predictions_dataset = None

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Train the ensemble models and the final model with the predictions of the initial set of models.

        Parameters
        ----------
        dataset : Dataset
            The training data.

        Returns
        -------
        self : StackingClassifier
            The fitted model.
        """
        for model in self.models:
            model.fit(dataset)
        
        #Generate predictions from the base models
        #Create an empty list to store predictions of the initial set of models
        base_predictions_list = []

        #Iterate over each model and generate predictions
        for model in self.models:
            predictions = model.predict(dataset)
            base_predictions_list.append(predictions)

        #Stack the predictions column-wise to create a 2D array
        base_predictions = np.column_stack(base_predictions_list)

        #Create a dataset out of the base models' predictions
        self.predictions_dataset = Dataset(X=base_predictions, y=dataset.y, label=dataset.label)

        #Train the final model on the predictions of the base models
        self.final_model.fit(self.predictions_dataset)

        return self


    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        dataset : Dataset
            The test data.

        Returns
        -------
        final_predictions : np.ndarray
            The predicted labels.
        """
        base_predictions_list = []
        for model in self.models:
            predictions = model.predict(dataset)
            base_predictions_list.append(predictions)

        base_predictions = np.column_stack(base_predictions_list)

        # Predict using the final model
        final_predictions = self.final_model.predict(Dataset(X=base_predictions, y=None))

        return final_predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        dataset : Dataset
            The test data.
        predictions: np.ndarray
            Predictions

        Returns
        -------
        score : float
            Mean accuracy
        """
        return accuracy(dataset.y, predictions)
    

if __name__ == '__main__':
    # Import required modules
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    from si.models.knn_classifier import KNNClassifier
    from si.models.logistic_regression import LogisticRegression
    from si.models.decision_tree_classifier import DecisionTreeClassifier
    from si.ensemble.stacking_classifier import StackingClassifier  
    from si.io.csv_file import read_csv

    # Load and split the dataset
    filename= r'C:\Users\Asus\Desktop\MESTRADO BI 2324\2ANO\SIB\portfólio\si\datasets\breast_bin\breast-bin.csv'  
    dataset = read_csv(filename, label=True, sep=",", features=True)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Initialize the base models
    knn1 = KNNClassifier(k=3)
    logreg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    dtree = DecisionTreeClassifier()

    # Initialize the final model
    knn2 = KNNClassifier(k=3)

    # Initialize the StackingClassifier with the base models and final model
    stacking = StackingClassifier(models=[knn1, logreg, dtree], final_model=knn2)

    # Train the StackingClassifier model
    stacking.fit(train_dataset)

    # Compute the score on the test dataset
    score = stacking.score(test_dataset)
    print(f"StackingClassifier Score: {score}")

    # Get predictions on the test dataset
    predictions = stacking.predict(test_dataset)
    print("Predictions:", predictions)