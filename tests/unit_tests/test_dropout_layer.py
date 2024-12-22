from unittest import TestCase
import numpy as np
from si.neural_networks.layers import Dropout


class TestDropoutLayer(TestCase):

    def setUp(self):
        self.input_data = np.random.randn(3, 3)  # Entrada aleatória (3x3)
        self.probability = 0.5  
        self.dropout_layer = Dropout(probability=self.probability)

    def test_forward_propagation_training(self):
        output_train = self.dropout_layer.forward_propagation(self.input_data, training=True)
        mask = self.dropout_layer.mask
        self.assertEqual(mask.shape, self.input_data.shape, "Máscara tem que ter o mesmo formato da entrada.")
        expected_output = self.input_data * mask / (1 - self.probability)
        np.testing.assert_almost_equal(output_train, expected_output, decimal=5, err_msg="A saída não está correta.")

    def test_forward_propagation_inference(self):
        output_infer = self.dropout_layer.forward_propagation(self.input_data, training=False)
        np.testing.assert_almost_equal(output_infer, self.input_data, decimal=5, err_msg="A saída durante a inferência deve ser igual à entrada.")

    def test_mask_is_binary(self):
        self.dropout_layer.forward_propagation(self.input_data, training=True)
        mask = self.dropout_layer.mask
        unique_values = np.unique(mask)
        self.assertTrue(np.array_equal(unique_values, [0, 1]), "A máscara deve conter apenas 0s e 1s.")
