"""
TEST DATA_LOADER FASTTEXT
==========================
Unit test per il modulo data_loader_fasttext.
Verifica il caricamento, preprocessing e conversione in formato FastText.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.data_loader_fasttext import SentimentDataLoader


class TestSentimentDataLoader:
    """Test suite per la classe SentimentDataLoader."""
    
    @pytest.fixture
    def data_loader(self):
        """Fixture che fornisce un'istanza di SentimentDataLoader."""
        return SentimentDataLoader()
    
    def test_initialization(self, data_loader):
        """
        Test: Verifica che il DataLoader sia inizializzato correttamente.
        
        Controlla:
        - Label mapping configurato
        - Reverse mapping disponibile
        """
        assert data_loader.label_mapping[0] == 'negative'
        assert data_loader.label_mapping[1] == 'neutral'
        assert data_loader.label_mapping[2] == 'positive'
        
        assert data_loader.reverse_mapping['negative'] == 0
        assert data_loader.reverse_mapping['neutral'] == 1
        assert data_loader.reverse_mapping['positive'] == 2
    
    def test_preprocess_text(self, data_loader):
        """
        Test: Verifica il preprocessing del testo.
        
        Controlla:
        - Conversione a minuscolo
        - Rimozione URL
        - Rimozione menzioni
        - Pulizia spazi multipli
        """
        # Test URL removal
        text_with_url = "Check this out http://example.com amazing!"
        processed = data_loader.preprocess_text(text_with_url)
        assert "http" not in processed
        assert "amazing" in processed
        
        # Test mention removal
        text_with_mention = "Hello @user this is great!"
        processed = data_loader.preprocess_text(text_with_mention)
        assert "@" not in processed
        
        # Test lowercase conversion
        text_uppercase = "THIS IS AMAZING!"
        processed = data_loader.preprocess_text(text_uppercase)
        assert processed == "this is amazing!"
        
        # Test multiple spaces removal
        text_spaces = "Hello   world    test"
        processed = data_loader.preprocess_text(text_spaces)
        assert "   " not in processed
    
    def test_fasttext_format_conversion(self, data_loader):
        """
        Test: Verifica la conversione nel formato FastText.
        
        Controlla:
        - Prefisso __label__ presente
        - Label corretto
        - Testo preservato
        """
        text = "questo prodotto è fantastico"
        
        # Converti in formato FastText
        line_positive = data_loader.convert_to_fasttext_format(text, 2)
        line_neutral = data_loader.convert_to_fasttext_format(text, 1)
        line_negative = data_loader.convert_to_fasttext_format(text, 0)
        
        # Verifica formato
        assert line_positive.startswith("__label__positive")
        assert line_neutral.startswith("__label__neutral")
        assert line_negative.startswith("__label__negative")
        
        # Verifica testo preservato
        assert text in line_positive
        assert text in line_neutral
        assert text in line_negative
    
    def test_label_names(self, data_loader):
        """
        Test: Verifica il mapping tra indici e nomi di sentiment.
        
        Controlla:
        - Mapping completo per le tre classi
        - Valori stringhe corretti
        """
        label_mapping = data_loader.get_label_names()
        
        assert label_mapping[0] == 'negative'
        assert label_mapping[1] == 'neutral'
        assert label_mapping[2] == 'positive'
        assert len(label_mapping) == 3
    
    def test_reverse_label_mapping(self, data_loader):
        """
        Test: Verifica il mapping inverso.
        
        Controlla:
        - Conversione da nomi a indici
        """
        reverse_map = data_loader.get_reverse_label_mapping()
        
        assert reverse_map['negative'] == 0
        assert reverse_map['neutral'] == 1
        assert reverse_map['positive'] == 2


class TestTextPreprocessing:
    """Test per la pipeline di preprocessing dei testi."""
    
    @pytest.fixture
    def sample_texts(self):
        """Fixture con testi di esempio."""
        return [
            "Love this product! https://example.com",
            "@user Great quality!",
            "Not satisfied with the service",
            "AMAZING EXPERIENCE!!!   multiple   spaces"
        ]
    
    def test_batch_preprocessing(self):
        """
        Test: Verifica il preprocessing batch di testi.
        
        Controlla:
        - Processamento di multipli testi
        - Applicazione corretta della pulizia
        - Output coerente
        """
        loader = SentimentDataLoader()
        
        texts = [
            "This is GREAT http://test.com",
            "@person Nice job!"
        ]
        
        processed_texts = [loader.preprocess_text(t) for t in texts]
        
        assert len(processed_texts) == 2
        assert all(isinstance(t, str) for t in processed_texts)
        assert all(t == t.lower() for t in processed_texts)
        assert "http" not in processed_texts[0]
        assert "@" not in processed_texts[1]


class TestFastTextFormatGeneration:
    """Test per la generazione del formato FastText."""
    
    def test_format_consistency(self):
        """
        Test: Verifica che il formato FastText sia coerente.
        
        Controlla:
        - Ogni riga ha un label
        - Label è valido
        - Testo non è vuoto
        """
        loader = SentimentDataLoader()
        
        test_cases = [
            ("positive text", 2),
            ("neutral opinion", 1),
            ("negative experience", 0)
        ]
        
        for text, label in test_cases:
            line = loader.convert_to_fasttext_format(text, label)
            
            # Verifica formato
            assert line.startswith("__label__")
            
            # Estrai label e testo
            parts = line.split(' ', 1)
            extracted_label = parts[0].replace("__label__", "")
            extracted_text = parts[1] if len(parts) > 1 else ""
            
            # Verifica contenuto
            assert extracted_label in ['negative', 'neutral', 'positive']
            assert extracted_text == text
