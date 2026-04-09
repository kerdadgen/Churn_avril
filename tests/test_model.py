import pytest
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


class TestModel:
    """Tests simples pour le modèle de churn."""
    
    @pytest.fixture
    def sample_model(self):
        """Crée un modèle LogisticRegression simple pour les tests."""
        X, y = make_classification(n_samples=100, n_features=4, n_informative=4, 
                                  n_redundant=0, random_state=101)
        model = LogisticRegression(random_state=101, solver='liblinear', class_weight='balanced')
        model.fit(X, y)
        return model

    def test_model_training(self, sample_model):
        """Vérifie que le modèle est bien entraîné."""
        assert sample_model is not None
        assert hasattr(sample_model, 'predict')
        assert hasattr(sample_model, 'predict_proba')

    def test_model_prediction(self, sample_model):
        """Teste les prédictions du modèle."""
        X_test = np.array([[1.0, 2.0, 3.0, 4.0]])
        prediction = sample_model.predict(X_test)
        assert prediction in [0, 1]
        assert len(prediction) == 1

    def test_model_predict_proba(self, sample_model):
        """Teste que le modèle retourne des probabilités valides."""
        X_test = np.array([[1.0, 2.0, 3.0, 4.0]])
        proba = sample_model.predict_proba(X_test)
        assert proba.shape == (1, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_model_pickle_serialization(self, sample_model):
        """Teste que le modèle peut être sérialisé et désérialisé."""
        serialized = pickle.dumps(sample_model)
        loaded_model = pickle.loads(serialized)
        
        X_test = np.array([[1.0, 2.0, 3.0, 4.0]])
        original_pred = sample_model.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)
        
        assert np.array_equal(original_pred, loaded_pred)

    def test_model_coefficients_exist(self, sample_model):
        """Vérifie que le modèle a des coefficients."""
        assert hasattr(sample_model, 'coef_')
        assert sample_model.coef_.shape == (1, 4)
