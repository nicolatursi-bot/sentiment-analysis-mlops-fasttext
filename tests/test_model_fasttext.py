"""
TEST MODEL FASTTEXT
===================
Unit test per il modulo model_fasttext.
Verifica il training, inference e salvataggio del modello FastText.
"""

import pytest
import tempfile
import os
from src.model_fasttext import FastTextSentimentModel, ModelTrainer


class TestFastTextSentimentModel:
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def sample_training_file(self, temp_dir):
        train_file = os.path.join(temp_dir, "train.txt")
        samples = [
            "__label__positive questo prodotto è fantastico",
            "__label__positive adoro assolutamente",
            "__label__negative non sono soddisfatto",
            "__label__negative pessima esperienza",
            "__label__neutral è un prodotto normale",
        ]
        with open(train_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(samples))
        return train_file
    
    def test_model_initialization(self):
        model = FastTextSentimentModel()
        assert model.model is None
        assert model.label_mapping[0] == 'negative'
        assert model.label_mapping[1] == 'neutral'
        assert model.label_mapping[2] == 'positive'
    
    def test_training(self, sample_training_file, temp_dir):
        model = FastTextSentimentModel()
        stats = model.train(sample_training_file, lr=0.5, epoch=5, dim=50)
        assert model.model is not None
        assert 'training_time' in stats
        assert stats['loss'] == 'softmax'


class TestModelTrainer:
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def train_val_files(self, temp_dir):
        train_file = os.path.join(temp_dir, "train.txt")
        val_file = os.path.join(temp_dir, "val.txt")
        
        train_data = [
            "__label__positive fantastico",
            "__label__negative terribile",
            "__label__neutral ok",
        ]
        val_data = [
            "__label__positive meravigliosi",
            "__label__negative pessimo",
            "__label__neutral normale",
        ]
        
        with open(train_file, 'w') as f:
            f.write('\n'.join(train_data))
        with open(val_file, 'w') as f:
            f.write('\n'.join(val_data))
        
        return train_file, val_file
    
    def test_trainer_initialization(self):
        trainer = ModelTrainer()
        assert trainer.best_model is None
        assert len(trainer.training_history) == 0