import pytest
from src.sentiment_model_roberta import RoBERTaSentimentModel, SentimentAnalyzer


class TestRoBERTaSentimentModel:
    
    @pytest.fixture
    def model(self):
        """Fixture: Modello RoBERTa inizializzato."""
        return RoBERTaSentimentModel()
    
    def test_model_initialization(self, model):
        """Test: Modello inizializzato correttamente."""
        assert model.model is not None
        assert model.tokenizer is not None
        assert model.pipeline is not None
        
        info = model.get_model_info()
        assert info['num_labels'] == 3
        assert len(info['labels']) == 3
    
    def test_model_info(self, model):
        """Test: Informazioni modello."""
        info = model.get_model_info()
        
        assert 'model_name' in info
        assert 'labels' in info
        assert info['model_name'] == 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        assert info['model_type'] == 'RoBERTa (preaddestrato)'
    
    def test_inference_single_text(self, model):
        """Test: Inference su un singolo testo."""
        texts = ["This is amazing!"]
        predictions = model.inference(texts)
        
        assert len(predictions) == 1
        pred = predictions[0]
        
        assert 'text' in pred
        assert 'label' in pred
        assert 'confidence' in pred
        assert pred['label'] in ['positive', 'neutral', 'negative']
        assert 0 <= pred['confidence'] <= 1
    
    def test_inference_batch(self, model):
        """Test: Inference batch."""
        texts = [
            "I love this product",
            "It is okay",
            "I hate this"
        ]
        predictions = model.inference(texts)
        
        assert len(predictions) == 3
        for pred in predictions:
            assert 'label' in pred
            assert pred['label'] in ['positive', 'neutral', 'negative']


class TestSentimentAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        """Fixture: Analyzer inizializzato."""
        return SentimentAnalyzer()
    
    def test_analyzer_initialization(self, analyzer):
        """Test: Analyzer inizializzato."""
        assert analyzer.model is not None
    
    def test_analyze_texts(self, analyzer):
        """Test: Analisi di testi."""
        texts = [
            "Amazing product!",
            "Not satisfied",
            "It is fine"
        ]
        
        result = analyzer.analyze(texts)
        
        assert result['total_texts'] == 3
        assert 'sentiment_distribution' in result
        assert 'sentiment_percentages' in result
        assert 'average_confidence' in result