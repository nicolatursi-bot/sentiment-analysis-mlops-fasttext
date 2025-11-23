"""
TEST MODEL FASTTEXT
===================
Unit test per il modulo model_fasttext.
Verifica il training, inference e salvataggio del modello FastText.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.model_fasttext import FastTextSentimentModel, ModelTrainer


class TestFastTextSentimentModel:
    """Test suite per la classe FastTextSentimentModel."""
    
    @pytest.fixture
    def temp_dir(self):
        """Crea una directory temporanea per i test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def sample_training_file(self, temp_dir):
        """Crea un file di training di esempio."""
        train_file = os.path.join(temp_dir, "train.txt")
        
        # Crea dati di training di esempio
        samples = [
            "__label__positive questo prodotto è fantastico",
            "__label__positive adoro assolutamente",
            "__label__positive eccellente qualità",
            "__label__negative non sono soddisfatto",
            "__label__negative pessima esperienza",
            "__label__negative molto deludente",
            "__label__neutral è un prodotto normale",
            "__label__neutral niente di speciale",
            "__label__neutral abbastanza ok",
        ]
        
        with open(train_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(samples))
        
        return train_file
    
    def test_model_initialization(self):
        """
        Test: Verifica l'inizializzazione del modello.
        
        Controlla:
        - Modello inizializzato correttamente
        - Label mapping configurato
        """
        model = FastTextSentimentModel()
        
        assert model.model is None  # Non ancora addestrato
        assert model.label_mapping[0] == 'negative'
        assert model.label_mapping[1] == 'neutral'
        assert model.label_mapping[2] == 'positive'
    
    def test_training(self, sample_training_file, temp_dir):
        """
        Test: Verifica il training del modello.
        
        Controlla:
        - Training completato senza errori
        - Modello non è None
        - Statistiche restituite
        """
        model = FastTextSentimentModel()
        
        # Esegui training
        stats = model.train(
            sample_training_file,
            lr=0.5,
            epoch=10,
            wordNgrams=2,
            dim=50,
            loss='softmax'
        )
        
        # Verifica
        assert model.model is not None
        assert 'training_time' in stats
        assert stats['loss'] == 'softmax'
        assert stats['epoch'] == 10
    
    def test_inference_single(self, sample_training_file):
        """
        Test: Verifica l'inference su un singolo testo.
        
        Controlla:
        - Predizione generata
        - Formato dell'output
        - Confidence score valido
        """
        # Train modello
        model = FastTextSentimentModel()
        model.train(sample_training_file, epoch=10, dim=50)
        
        # Test inference
        texts = ["questo è fantastico"]
        predictions = model.inference(texts)
        
        assert len(predictions) == 1
        pred = predictions[0]
        
        assert 'text' in pred
        assert 'label' in pred
        assert 'confidence' in pred
        assert 'all_predictions' in pred
        
        assert pred['label'] in ['negative', 'neutral', 'positive']
        assert 0 <= pred['confidence'] <= 1
        assert len(pred['all_predictions']) == 3
    
    def test_inference_batch(self, sample_training_file):
        """
        Test: Verifica l'inference su batch multipli.
        
        Controlla:
        - Batch processing corretto
        - Predizioni per ogni testo
        - Formato consistente
        """
        # Train modello
        model = FastTextSentimentModel()
        model.train(sample_training_file, epoch=10, dim=50)
        
        # Test batch inference
        texts = [
            "fantastico prodotto",
            "non mi piace",
            "è ok"
        ]
        
        predictions = model.inference(texts)
        
        assert len(predictions) == 3
        
        for pred in predictions:
            assert 'label' in pred
            assert 'confidence' in pred
            assert isinstance(pred['confidence'], float)
            assert 0 <= pred['confidence'] <= 1
    
    def test_model_save_load(self, sample_training_file, temp_dir):
        """
        Test: Verifica il salvataggio e caricamento del modello.
        
        Controlla:
        - Modello salvato correttamente
        - Modello caricato da disco
        - Inference funziona dopo il caricamento
        """
        # Train e salva modello
        model1 = FastTextSentimentModel()
        model1.train(sample_training_file, epoch=10, dim=50)
        
        model_path = os.path.join(temp_dir, "test_model")
        model1.save_model(model_path)
        
        # Verifica che i file siano stati creati
        assert os.path.exists(f"{model_path}.bin")
        
        # Carica modello
        model2 = FastTextSentimentModel()
        model2.load_model(f"{model_path}.bin")
        
        # Verifica che l'inference funzioni
        text = "test text"
        predictions = model2.inference([text])
        
        assert len(predictions) == 1
        assert 'label' in predictions[0]
    
    def test_get_model_info(self, sample_training_file):
        """
        Test: Verifica le informazioni del modello.
        
        Controlla:
        - Info modello restituite
        - Campi corretti
        """
        model = FastTextSentimentModel()
        model.train(sample_training_file, epoch=10, dim=50)
        
        info = model.get_model_info()
        
        assert 'model_type' in info
        assert info['model_type'] == 'FastText'
        assert 'labels' in info
        assert 'num_labels' in info
        assert info['num_labels'] == 3


class TestModelTrainer:
    """Test suite per la classe ModelTrainer."""
    
    @pytest.fixture
    def temp_dir(self):
        """Crea una directory temporanea."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def train_val_files(self, temp_dir):
        """Crea file di training e validation."""
        train_file = os.path.join(temp_dir, "train.txt")
        val_file = os.path.join(temp_dir, "val.txt")
        
        train_data = [
            "__label__positive fantastico",
            "__label__negative terribile",
            "__label__neutral ok",
            "__label__positive bellissimo",
            "__label__negative orribile"
        ]
        
        val_data = [
            "__label__positive meravigliosi",
            "__label__negative pessimo",
            "__label__neutral normale"
        ]
        
        with open(train_file, 'w') as f:
            f.write('\n'.join(train_data))
        
        with open(val_file, 'w') as f:
            f.write('\n'.join(val_data))
        
        return train_file, val_file
    
    def test_trainer_initialization(self):
        """
        Test: Verifica l'inizializzazione del trainer.
        
        Controlla:
        - Trainer creato
        - History inizializzato
        """
        trainer = ModelTrainer()
        
        assert trainer.best_model is None
        assert trainer.best_metrics is None
        assert len(trainer.training_history) == 0
    
    def test_train_and_evaluate(self, train_val_files):
        """
        Test: Verifica il training e la valutazione.
        
        Controlla:
        - Training completato
        - Metriche calcolate
        - Risultati salvati
        """
        train_file, val_file = train_val_files
        
        trainer = ModelTrainer()
        results = trainer.train_and_evaluate(train_file, val_file)
        
        assert 'train_stats' in results
        assert 'val_metrics' in results
        assert 'hyperparameters' in results
        
        # Verifica metriche
        metrics = results['val_metrics']
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_get_training_history(self, train_val_files):
        """
        Test: Verifica l'accesso allo storico di training.
        
        Controlla:
        - History è recuperabile
        - Formato corretto
        """
        train_file, val_file = train_val_files
        
        trainer = ModelTrainer()
        trainer.train_and_evaluate(train_file, val_file)
        
        history = trainer.get_training_history()
        
        assert isinstance(history, list)
        assert len(history) > 0
        assert 'train_stats' in history[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```