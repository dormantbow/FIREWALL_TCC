"""
Módulo para implementação do detector de intrusão que integra os modelos
de detecção de anomalias, classificação de ataques e o sistema de decisão.
"""

import os
import logging
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime

from models.neural_models import AutoencoderModel, CNNLSTMModel
from detection.decision_system import DecisionSystem
from utils.logging_utils import performance_monitor, AlertManager
from utils.config import MODELS_DIR

logger = logging.getLogger(__name__)

class IntrusionDetector:
    """
    Detector de intrusão que integra os modelos de detecção de anomalias,
    classificação de ataques e o sistema de decisão.
    """
    
    def __init__(self, autoencoder=None, classifier=None, decision_system=None):
        """
        Inicializa o detector de intrusão.
        
        Args:
            autoencoder: Modelo de autoencoder para detecção de anomalias
            classifier: Modelo CNN+LSTM para classificação de ataques
            decision_system: Sistema de decisão
        """
        self.autoencoder = autoencoder
        self.classifier = classifier
        self.decision_system = decision_system or DecisionSystem()
        self.alert_manager = AlertManager()
        self.detection_log = []
        
        # Mapeamento de tipos de ataque para nomes
        self.attack_names = {
            0: "Desconhecido (possível zero-day)",
            1: "DoS/DDoS",
            2: "Probe/Scan",
            3: "R2L (Remote to Local)",
            4: "U2R (User to Root)",
            5: "Botnet"
        }
    
    def detect(self, processed_data, sequence_data=None, context=None):
        """
        Detecta ameaças nos dados processados.
        
        Args:
            processed_data: Dados pré-processados para análise
            sequence_data: Dados em formato de sequência para o classificador (opcional)
            context: Informações contextuais da rede (opcional)
            
        Returns:
            detection_result: Resultado da detecção (0: normal, 1: ataque)
            attack_type: Tipo de ataque identificado (se houver)
            confidence: Nível de confiança na detecção
            explanation: Explicação da decisão
        """
        logger.info(f"Iniciando detecção em dados processados")
        
        # Verificar se os modelos estão disponíveis
        if self.autoencoder is None:
            raise ValueError("Modelo de autoencoder não disponível")
        
        if self.classifier is None:
            raise ValueError("Modelo de classificação não disponível")
        
        # Iniciar timer
        start_time = performance_monitor.start_timer()
        
        try:
            # Detecção de anomalias
            anomalies, anomaly_scores = self.autoencoder.detect_anomalies(
                np.array([processed_data]), return_scores=True
            )
            anomaly_score = anomaly_scores[0]
            
            # Classificação de ataques
            if sequence_data is None and hasattr(self.classifier, 'prepare_sequence_data'):
                # Criar sequência a partir dos dados atuais (simplificado)
                # Nota: Em um cenário real, seria necessário manter um buffer de dados históricos
                sequence_data = np.array([processed_data]).reshape(1, 1, -1)
                logger.warning("Sequência de dados não fornecida, usando dados atuais como sequência")
            
            classification_result = self.classifier.predict(sequence_data)[0]
            
            # Atualizar threshold de anomalia no sistema de decisão
            if self.decision_system.anomaly_threshold is None and hasattr(self.autoencoder, 'threshold'):
                self.decision_system.anomaly_threshold = self.autoencoder.threshold
            
            # Combinar resultados através do sistema de decisão
            decision, attack_type, confidence = self.decision_system.decide(
                anomaly_score, classification_result
            )
            
            # Gerar explicação
            explanation = self._generate_explanation(
                processed_data,
                anomaly_score,
                classification_result,
                (decision, attack_type, confidence)
            )
            
            # Registrar detecção
            detection_info = {
                'timestamp': time.time(),
                'decision': decision,
                'attack_type': attack_type,
                'attack_name': self.attack_names.get(attack_type, f"Tipo {attack_type}") if attack_type is not None else None,
                'confidence': confidence,
                'anomaly_score': float(anomaly_score),
                'classification_result': classification_result.tolist() if hasattr(classification_result, 'tolist') else classification_result,
                'context': context
            }
            self.detection_log.append(detection_info)
            
            # Gerar alerta se for um ataque
            if decision == 1:
                severity = 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low'
                attack_name = self.attack_names.get(attack_type, f"Tipo {attack_type}") if attack_type is not None else "Desconhecido"
                
                message = f"Ataque detectado: {attack_name} com confiança {confidence:.2f}"
                if context and 'source_ip' in context:
                    message += f"\nOrigem: {context['source_ip']}"
                if context and 'destination_ip' in context:
                    message += f"\nDestino: {context['destination_ip']}"
                
                self.alert_manager.send_alert(severity, message, detection_info)
            
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('intrusion_detector', 'detect', 
                                          elapsed_time * 1000, 'ms', 
                                          {'decision': decision, 'attack_type': attack_type, 'confidence': confidence})
            
            logger.info(f"Detecção concluída: {'Ataque' if decision == 1 else 'Normal'}, " +
                       f"Tipo: {self.attack_names.get(attack_type, str(attack_type)) if attack_type is not None else 'N/A'}, " +
                       f"Confiança: {confidence:.4f}")
            
            return decision, attack_type, confidence, explanation
        
        except Exception as e:
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('intrusion_detector', 'detect', 
                                          elapsed_time * 1000, 'ms', 
                                          {'error': str(e)})
            logger.error(f"Erro na detecção: {str(e)}")
            raise
    
    def _generate_explanation(self, input_data, anomaly_score, classification_result, decision_result):
        """
        Gera explicação para a decisão tomada.
        
        Args:
            input_data: Dados de entrada
            anomaly_score: Pontuação de anomalia
            classification_result: Resultado da classificação
            decision_result: Resultado da decisão (decision, attack_type, confidence)
            
        Returns:
            dict: Explicação da decisão
        """
        decision, attack_type, confidence = decision_result
        
        explanation = {
            'decision': 'attack' if decision == 1 else 'normal',
            'attack_type': self.attack_names.get(attack_type, str(attack_type)) if attack_type is not None else None,
            'confidence': float(confidence),
            'anomaly': {
                'score': float(anomaly_score),
                'threshold': float(self.decision_system.anomaly_threshold) if self.decision_system.anomaly_threshold else None,
                'is_anomaly': float(anomaly_score) > (self.decision_system.anomaly_threshold or 0)
            },
            'classification': {
                'result': classification_result.tolist() if hasattr(classification_result, 'tolist') else classification_result,
                'predicted_class': int(np.argmax(classification_result)),
                'max_probability': float(np.max(classification_result)),
                'threshold': float(self.decision_system.classification_threshold)
            },
            'decision_weights': self.decision_system.weights,
            'timestamp': datetime.now().isoformat()
        }
        
        # Adicionar informações sobre características mais importantes
        try:
            if hasattr(self.autoencoder, 'model') and self.autoencoder.model is not None:
                # Identificar características que mais contribuíram para anomalia
                reconstruction = self.autoencoder.model.predict(np.array([input_data]))[0]
                reconstruction_errors = np.power(input_data - reconstruction, 2)
                
                # Top 5 características com maior erro de reconstrução
                top_features_idx = np.argsort(reconstruction_errors)[-5:]
                
                explanation['contributing_features'] = {
                    'anomaly': [
                        {
                            'feature_index': int(idx),
                            'error_contribution': float(reconstruction_errors[idx]),
                            'original_value': float(input_data[idx]),
                            'reconstructed_value': float(reconstruction[idx])
                        } for idx in top_features_idx
                    ]
                }
        except Exception as e:
            logger.warning(f"Erro ao gerar explicação detalhada para anomalia: {str(e)}")
        
        return explanation
    
    def process_feedback(self, detection_id, is_correct, true_label=None, comments=None):
        """
        Processa feedback do usuário sobre uma detecção.
        
        Args:
            detection_id: ID da detecção (índice no log)
            is_correct: Se a detecção está correta
            true_label: Rótulo verdadeiro (se conhecido)
            comments: Comentários adicionais
            
        Returns:
            dict: Resultado do processamento de feedback
        """
        logger.info(f"Processando feedback para detecção {detection_id}")
        
        try:
            # Verificar se a detecção existe
            if detection_id < 0 or detection_id >= len(self.detection_log):
                raise ValueError(f"ID de detecção inválido: {detection_id}")
            
            # Obter detecção
            detection = self.detection_log[detection_id]
            
            # Criar dados de feedback
            feedback_data = {
                'detection_id': detection_id,
                'timestamp': time.time(),
                'is_correct': is_correct,
                'true_label': true_label,
                'comments': comments,
                'anomaly_score': detection['anomaly_score'],
                'classification_result': detection['classification_result'],
                'original_decision': detection['decision'],
                'original_attack_type': detection['attack_type']
            }
            
            # Atualizar sistema de decisão com feedback
            if self.decision_system is not None:
                updates = self.decision_system.update_from_feedback(feedback_data)
                feedback_data['system_updates'] = updates
            
            logger.info(f"Feedback processado: {'Correto' if is_correct else 'Incorreto'}")
            return feedback_data
        
        except Exception as e:
            logger.error(f"Erro ao processar feedback: {str(e)}")
            raise
    
    def get_detection_log(self, count=None, filter_attacks=None):
        """
        Retorna o log de detecções.
        
        Args:
            count: Número máximo de detecções a retornar
            filter_attacks: Se True, retorna apenas ataques; se False, apenas tráfego normal
            
        Returns:
            list: Log de detecções
        """
        if filter_attacks is not None:
            filtered_log = [d for d in self.detection_log if d['decision'] == (1 if filter_attacks else 0)]
        else:
            filtered_log = self.detection_log
        
        if count is not None:
            return filtered_log[-count:]
        
        return filtered_log
    
    def save(self, base_dir=None):
        """
        Salva o detector de intrusão.
        
        Args:
            base_dir: Diretório base para salvar os modelos
            
        Returns:
            dict: Caminhos dos arquivos salvos
        """
        if base_dir is None:
            base_dir = MODELS_DIR
        
        # Criar diretório se não existir
        os.makedirs(base_dir, exist_ok=True)
        
        saved_paths = {}
        
        # Salvar autoencoder
        if self.autoencoder is not None:
            autoencoder_path = os.path.join(base_dir, 'autoencoder')
            self.autoencoder.save(autoencoder_path)
            saved_paths['autoencoder'] = autoencoder_path
        
        # Salvar classificador
        if self.classifier is not None:
            classifier_path = os.path.join(base_dir, 'classifier')
            self.classifier.save(classifier_path)
            saved_paths['classifier'] = classifier_path
        
        # Salvar sistema de decisão
        if self.decision_system is not None:
            decision_system_path = os.path.join(base_dir, 'decision_system.json')
            self.decision_system.save(decision_system_path)
            saved_paths['decision_system'] = decision_system_path
        
        # Salvar log de detecções
        if self.detection_log:
            log_path = os.path.join(base_dir, 'detection_log.json')
            with open(log_path, 'w') as f:
                json.dump(self.detection_log, f, indent=4)
            saved_paths['detection_log'] = log_path
        
        logger.info(f"Detector de intrusão salvo em {base_dir}")
        return saved_paths
    
    @classmethod
    def load(cls, base_dir=None, input_dim=None, num_classes=None):
        """
        Carrega um detector de intrusão.
        
        Args:
            base_dir: Diretório base onde os modelos estão salvos
            input_dim: Dimensão de entrada para os modelos (necessário se não puder ser inferido)
            num_classes: Número de classes para o classificador (necessário se não puder ser inferido)
            
        Returns:
            IntrusionDetector: Detector de intrusão carregado
        """
        if base_dir is None:
            base_dir = MODELS_DIR
        
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Diretório {base_dir} não encontrado")
        
        # Carregar autoencoder
        autoencoder_path = os.path.join(base_dir, 'autoencoder')
        if os.path.exists(autoencoder_path):
            threshold_path = os.path.join(base_dir, 'autoencoder', 'threshold.npy')
            autoencoder = AutoencoderModel.load(autoencoder_path, threshold_path)
        else:
            autoencoder = None
            logger.warning(f"Modelo de autoencoder não encontrado em {autoencoder_path}")
        
        # Carregar classificador
        classifier_path = os.path.join(base_dir, 'classifier')
        if os.path.exists(classifier_path):
            if input_dim is None or num_classes is None:
                raise ValueError("input_dim e num_classes são necessários para carregar o classificador")
            classifier = CNNLSTMModel.load(classifier_path, input_dim, num_classes)
        else:
            classifier = None
            logger.warning(f"Modelo de classificação não encontrado em {classifier_path}")
        
        # Carregar sistema de decisão
        decision_system_path = os.path.join(base_dir, 'decision_system.json')
        if os.path.exists(decision_system_path):
            # Carregar o sistema de decisão
            with open(decision_system_path, 'r') as f:
                decision_system_data = json.load(f)
