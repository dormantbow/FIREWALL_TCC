"""
Módulo principal para integração e execução do firewall baseado em rede neural.
"""

import os
import logging
import argparse
import numpy as np
import pandas as pd
import time
from datetime import datetime

from .preprocessing.dataset_loader import dataset_loader
from .preprocessing.preprocessor import DataPreprocessor
from .models.neural_models import AutoencoderModel, CNNLSTMModel
from .detection.decision_system import DecisionSystem
from .detection.intrusion_detector import IntrusionDetector
from .detection.response_mechanisms import ResponseSelector, ResponseExecutor
from .utils.logging_utils import performance_monitor
from .utils.config import MODELS_DIR

logger = logging.getLogger(__name__)

class NeuralFirewall:
    """
    Classe principal que integra todos os componentes do firewall baseado em rede neural.
    """
    
    def __init__(self):
        """
        Inicializa o firewall neural.
        """
        self.preprocessor = None
        self.autoencoder = None
        self.classifier = None
        self.decision_system = None
        self.intrusion_detector = None
        self.response_selector = None
        self.response_executor = None
        self.is_trained = False
    
    def train(self, dataset_type='nsl_kdd', save_models=True):
        """
        Treina os modelos do firewall neural.
        
        Args:
            dataset_type: Tipo de dataset ('nsl_kdd' ou 'cicids_2018')
            save_models: Se True, salva os modelos treinados
            
        Returns:
            dict: Resultados do treinamento
        """
        logger.info(f"Iniciando treinamento do firewall neural com dataset {dataset_type}")
        
        # Iniciar timer
        start_time = performance_monitor.start_timer()
        
        try:
            # 1. Carregar dataset
            if dataset_type == 'nsl_kdd':
                X_train, y_train = dataset_loader.load_nsl_kdd('train')
                X_test, y_test = dataset_loader.load_nsl_kdd('test')
            elif dataset_type == 'cicids_2018':
                # Para o CICIDS, dividimos manualmente em treino e teste
                X, y = dataset_loader.load_cicids()
                
                # Dividir em treino (80%) e teste (20%)
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                raise ValueError(f"Tipo de dataset desconhecido: {dataset_type}")
            
            # 2. Pré-processar dados
            self.preprocessor = DataPreprocessor(dataset_type)
            X_train_processed, y_train_processed = self.preprocessor.fit_transform(X_train, y_train)
            X_test_processed = self.preprocessor.transform(X_test)
            
            # 3. Treinar modelo de detecção de anomalias (autoencoder)
            # Usar apenas dados normais para treinar o autoencoder
            normal_indices = np.where(y_train_processed == 0)[0]
            X_train_normal = X_train_processed[normal_indices]
            
            input_dim = X_train_processed.shape[1]
            self.autoencoder = AutoencoderModel(input_dim)
            self.autoencoder.build()
            self.autoencoder.train(X_train_normal)
            
            # 4. Treinar modelo de classificação (CNN+LSTM)
            num_classes = len(np.unique(y_train_processed))
            self.classifier = CNNLSTMModel(input_dim, num_classes)
            self.classifier.build()
            
            # Preparar dados de sequência
            X_train_seq = self.classifier.prepare_sequence_data(X_train_processed)
            y_train_seq = y_train_processed[-len(X_train_seq):]  # Ajustar rótulos para sequências
            
            self.classifier.train(X_train_seq, y_train_seq)
            
            # 5. Inicializar sistema de decisão
            self.decision_system = DecisionSystem()
            self.decision_system.anomaly_threshold = self.autoencoder.threshold
            
            # 6. Inicializar detector de intrusão
            self.intrusion_detector = IntrusionDetector(
                self.autoencoder, self.classifier, self.decision_system
            )
            
            # 7. Inicializar seletor e executor de respostas
            self.response_selector = ResponseSelector()
            self.response_executor = ResponseExecutor()
            
            # 8. Avaliar modelos
            # Avaliar autoencoder
            anomalies = self.autoencoder.detect_anomalies(X_test_processed)
            anomaly_accuracy = np.mean((anomalies == (y_test > 0)).astype(int))
            
            # Avaliar classificador
            X_test_seq = self.classifier.prepare_sequence_data(X_test_processed)
            y_test_seq = y_test[-len(X_test_seq):]  # Ajustar rótulos para sequências
            
            classifier_metrics = self.classifier.evaluate(X_test_seq, y_test_seq)
            classifier_accuracy = classifier_metrics[1]  # Assumindo que accuracy é a segunda métrica
            
            # 9. Salvar modelos
            if save_models:
                self._save_models()
            
            self.is_trained = True
            
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('neural_firewall', 'train', 
                                          elapsed_time * 1000, 'ms', 
                                          {'dataset': dataset_type,
                                           'anomaly_accuracy': anomaly_accuracy,
                                           'classifier_accuracy': classifier_accuracy})
            
            logger.info(f"Treinamento concluído em {elapsed_time:.2f}s")
            
            return {
                'status': 'success',
                'training_time': elapsed_time,
                'anomaly_accuracy': anomaly_accuracy,
                'classifier_accuracy': classifier_accuracy,
                'dataset': dataset_type,
                'input_dim': input_dim,
                'num_classes': num_classes
            }
        
        except Exception as e:
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('neural_firewall', 'train', 
                                          elapsed_time * 1000, 'ms', 
                                          {'error': str(e)})
            logger.error(f"Erro no treinamento: {str(e)}")
            raise
    
    def detect(self, data, context=None):
        """
        Detecta ameaças nos dados fornecidos.
        
        Args:
            data: Dados a serem analisados
            context: Informações contextuais (opcional)
            
        Returns:
            dict: Resultado da detecção e resposta
        """
        if not self.is_trained:
            raise ValueError("O firewall não foi treinado. Chame train() primeiro.")
        
        logger.info("Iniciando detecção de ameaças")
        
        # Iniciar timer
        start_time = performance_monitor.start_timer()
        
        try:
            # 1. Pré-processar dados
            if isinstance(data, pd.DataFrame):
                processed_data = self.preprocessor.transform(data).values[0]
            else:
                processed_data = self.preprocessor.transform(pd.DataFrame([data])).values[0]
            
            # 2. Detectar ameaças
            decision, attack_type, confidence, explanation = self.intrusion_detector.detect(
                processed_data, context=context
            )
            
            # 3. Avaliar ameaça
            if decision == 1:  # Ataque detectado
                threat_level = confidence * 10  # Escala 0-10
                impact_assessment = f"Impacto potencial de ataque tipo {attack_type}"
                priority = 5 - int(confidence * 4)  # Escala 1-5 (1 é mais prioritário)
                
                # 4. Selecionar resposta
                response_actions, response_params, response_justification = self.response_selector.select_response(
                    (decision, attack_type, confidence),
                    (threat_level, impact_assessment, priority),
                    context
                )
                
                # 5. Executar resposta
                if response_actions:
                    execution_results = self.response_executor.execute_response(
                        response_actions, response_params, 
                        {
                            'decision': decision,
                            'attack_type': attack_type,
                            'confidence': confidence,
                            'context': context
                        }
                    )
                else:
                    execution_results = {}
            else:
                threat_level = 0
                impact_assessment = "Tráfego normal, sem impacto"
                priority = 5
                response_actions = []
                response_params = {}
                response_justification = "Nenhuma ação necessária para tráfego normal"
                execution_results = {}
            
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('neural_firewall', 'detect', 
                                          elapsed_time * 1000, 'ms', 
                                          {'decision': decision, 'attack_type': attack_type, 'confidence': confidence})
            
            logger.info(f"Detecção concluída em {elapsed_time:.2f}s: {'Ataque' if decision == 1 else 'Normal'}")
            
            return {
                'decision': decision,
                'attack_type': attack_type,
                'confidence': confidence,
                'explanation': explanation,
                'threat_assessment': {
                    'level': threat_level,
                    'impact': impact_assessment,
                    'priority': priority
                },
                'response': {
                    'actions': response_actions,
                    'justification': response_justification,
                    'execution_results': execution_results
                },
                'timestamp': datetime.now().isoformat(),
                'processing_time': elapsed_time
            }
        
        except Exception as e:
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('neural_firewall', 'detect', 
                                          elapsed_time * 1000, 'ms', 
                                          {'error': str(e)})
            logger.error(f"Erro na detecção: {str(e)}")
            raise
    
    def process_feedback(self, detection_id, is_correct, true_label=None, comments=None):
        """
        Processa feedback do usuário sobre uma detecção.
        
        Args:
            detection_id: ID da detecção
            is_correct: Se a detecção está correta
            true_label: Rótulo verdadeiro (se conhecido)
            comments: Comentários adicionais
            
        Returns:
            dict: Resultado do processamento de feedback
        """
        if not self.is_trained or not self.intrusion_detector:
            raise ValueError("O firewall não foi treinado ou o detector de intrusão não está inicializado.")
        
        logger.info(f"Processando feedback para detecção {detection_id}")
        
        try:
            feedback_result = self.intrusion_detector.process_feedback(
                detection_id, is_correct, true_label, comments
            )
            
            logger.info(f"Feedback processado: {'Correto' if is_correct else 'Incorreto'}")
            return feedback_result
        
        except Exception as e:
            logger.error(f"Erro ao processar feedback: {str(e)}")
            raise
    
    def _save_models(self, base_dir=None):
        """
        Salva os modelos treinados.
        
        Args:
            base_dir: Diretório base para salvar os modelos
        """
        if base_dir is None:
            base_dir = MODELS_DIR
        
        # Criar diretório se não existir
        os.makedirs(base_dir, exist_ok=True)
        
        # Salvar preprocessador
        preprocessor_path = os.path.join(base_dir, 'preprocessor.pkl')
        self.preprocessor.save(preprocessor_path)
        
        # Salvar autoencoder
        autoencoder_path = os.path.join(base_dir, 'autoencoder')
        self.autoencoder.save(autoencoder_path)
        
        # Salvar classificador
        classifier_path = os.path.join(base_dir, 'classifier')
        self.classifier.save(classifier_path)
        
        # Salvar sistema de decisão
        decision_system_path = os.path.join(base_dir, 'decision_system.json')
        self.decision_system.save(decision_system_path)
        
        logger.info(f"Modelos salvos em {base_dir}")
    
    @classmethod
    def load(cls, base_dir=None, input_dim=None, num_classes=None):
        """
        Carrega um firewall neural treinado.
        
        Args:
            base_dir: Diretório base onde os modelos estão salvos
            input_dim: Dimensão de entrada para os modelos (necessário se não puder ser inferido)
            num_classes: Número de classes para o classificador (necessário se não puder ser inferido)
            
        Returns:
            NeuralFirewall: Firewall neural carregado
        """
        if base_dir is None:
            base_dir = MODELS_DIR
        
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Diretório {base_dir} não encontrado")
        
        # Criar instância
        firewall = cls()
        
        try:
            # Carregar preprocessador
            preprocessor_path = os.path.join(base_dir, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                from .preprocessing.preprocessor import DataPreprocessor
                firewall.preprocessor = DataPreprocessor.load(preprocessor_path)
            else:
                raise FileNotFoundError(f"Preprocessador não encontrado em {preprocessor_path}")
            
            # Carregar detector de intrusão (que carrega autoencoder, classificador e sistema de decisão)
            firewall.intrusion_detector = IntrusionDetector.load(base_dir, input_dim, num_classes)
            
            # Extrair componentes do detector
            firewall.autoencoder = firewall.intrusion_detector.autoencoder
            firewall.classifier = firewall.intrusion_detector.classifier
            firewall.decision_system = firewall.intrusion_detector.decision_system
            
            # Inicializar seletor e executor de respostas
            firewall.response_selector = ResponseSelector()
            firewall.response_executor = ResponseExecutor()
            
            firewall.is_trained = True
            
            logger.info(f"Firewall neural carregado de {base_dir}")
            return firewall
        
        except Exception as e:
            logger.error(f"Erro ao carregar firewall neural: {str(e)}")
            raise
    
    def get_performance_summary(self):
        """
        Retorna um resumo do desempenho do firewall.
        
        Returns:
            dict: Resumo do desempenho
        """
        return performance_monitor.get_summary()

def main():
    """
    Função principal para execução do firewall neural via linha de comando.
    """
    parser = argparse.ArgumentParser(description='Firewall baseado em rede neural para prevenção de ataques zero-day')
    parser.add_argument('--train', action='store_true', help='Treinar o firewall')
    parser.add_argument('--dataset', choices=['nsl_kdd', 'cicids_2018'], default='nsl_kdd', help='Dataset para treinamento')
    parser.add_argument('--save', action='store_true', help='Salvar m
(Content truncated due to size limit. Use line ranges to read in chunks)