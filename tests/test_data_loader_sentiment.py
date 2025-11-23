import pytest
from src.data_loader_sentiment import SentimentDataLoader, DatasetAnalyzer


class TestSentimentDataLoader:
    
    @pytest.fixture
    def loader(self):
        return SentimentDataLoader()
    
    def test_initialization(self, loader):
        """Test: DataLoader inizializzato correttamente."""
        assert loader.label_mapping[0] == 'negative'
        assert loader.label_mapping[1] == 'neutral'
        assert loader.label_mapping[2] == 'positive'
    
    def test_preprocess_text(self, loader):
        """Test: Preprocessing del testo."""
        text_with_url = "Check this http://example.com amazing!"
        processed = loader.preprocess_text(text_with_url)
        assert "http" not in processed
        assert "amazing" in processed
        
        text_uppercase = "THIS IS AMAZING"
        processed = loader.preprocess_text(text_uppercase)
        assert processed == "this is amazing"
    
    def test_prepare_for_inference(self, loader):
        """Test: Preparazione testi per inference."""
        texts = ["Hello WORLD", "Check http://test.com"]
        prepared = loader.prepare_for_inference(texts)
        
        assert len(prepared) == 2
        assert prepared[0] == "hello world"
        assert "http" not in prepared[1]
    
    def test_load_dataset(self, loader):
        """Test: Caricamento dataset."""
        dataset = loader.load_public_dataset()
        
        assert 'train' in dataset
        assert 'validation' in dataset
        assert len(dataset['train']) > 0
        assert len(dataset['validation']) > 0
    
    def test_label_info(self, loader):
        """Test: Informazioni etichette."""
        info = loader.get_label_info()
        
        assert info['num_classes'] == 3
        assert 'negative' in info['class_names']
        assert 'neutral' in info['class_names']
        assert 'positive' in info['class_names']


class TestDatasetAnalyzer:
    
    def test_analyze_fallback_data(self):
        """Test: Analisi dataset di fallback."""
        loader = SentimentDataLoader()
        dataset = loader.load_public_dataset()
        
        analysis = DatasetAnalyzer.analyze_dataset(dataset['train'])
        
        assert 'total_samples' in analysis
        assert 'label_distribution' in analysis
        assert 'avg_text_length' in analysis
        assert analysis['total_samples'] > 0