import pytest
import json
from unittest.mock import patch, MagicMock
from app import app, validate_input


@pytest.fixture
def client():
    """Crée un client de test pour l'API."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestHealthCheck:
    """Tests pour l'endpoint /health."""
    
    def test_health_check_success(self, client):
        """Vérifie que l'endpoint health retourne un statut ok."""
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'ok'
        assert 'model_loaded' in data


class TestPredict:
    """Tests pour l'endpoint /predict."""
    
    @patch('app.model')
    def test_predict_success(self, mock_model, client):
        """Test une prédiction réussie avec données valides."""
        # Configuration du mock
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        
        data = {
            "Age": 30,
            "Total_Purchase": 5000,
            "Years": 5,
            "Num_Sites": 3
        }
        
        response = client.post('/predict', 
                              data=json.dumps(data),
                              content_type='application/json')
        
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result['prediction'] == 1
        assert result['churn_probability'] == 0.7
        assert result['features']['Age'] == 30.0

    def test_predict_missing_fields(self, client):
        """Test que la prédiction échoue avec des champs manquants."""
        data = {"Age": 30, "Total_Purchase": 5000}
        
        response = client.post('/predict',
                              data=json.dumps(data),
                              content_type='application/json')
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'error' in result
        assert 'manquants' in result['error'].lower()

    def test_predict_invalid_data_type(self, client):
        """Test que la prédiction échoue avec des types invalides."""
        data = {
            "Age": "invalid",
            "Total_Purchase": 5000,
            "Years": 5,
            "Num_Sites": 3
        }
        
        response = client.post('/predict',
                              data=json.dumps(data),
                              content_type='application/json')
        
        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'error' in result


class TestValidateInput:
    """Tests pour la fonction validate_input."""
    
    def test_validate_input_valid(self):
        """Test la validation avec des données valides."""
        data = {
            "Age": 30,
            "Total_Purchase": 5000,
            "Years": 5,
            "Num_Sites": 3
        }
        is_valid, result = validate_input(data)
        assert is_valid is True
        assert result == [30.0, 5000.0, 5.0, 3.0]

    def test_validate_input_missing_field(self):
        """Test la validation avec des champs manquants."""
        data = {"Age": 30, "Total_Purchase": 5000}
        is_valid, result = validate_input(data)
        assert is_valid is False
        assert 'manquants' in result.lower()

    def test_validate_input_invalid_type(self):
        """Test la validation avec un type invalide."""
        data = {
            "Age": "not_a_number",
            "Total_Purchase": 5000,
            "Years": 5,
            "Num_Sites": 3
        }
        is_valid, result = validate_input(data)
        assert is_valid is False
        assert 'numérique' in result.lower()
