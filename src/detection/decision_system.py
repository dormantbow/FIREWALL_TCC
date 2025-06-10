"""
Módulo para implementação do sistema de decisão que combina os resultados
dos modelos de detecção de anomalias e classificação de ataques.
"""

import os
import logging
import numpy as np
import json
from datetime import datetime

from utils.config import DECISION_SYSTEM_CONFIG
from utils.logging_utils import performance_monitor

logger = logging.getLogger(__name__)

class DecisionSystem:
    """
    Sistema de decisão que combina os resultados dos modelos de detecção
    de anomalias e classificação de ataques para determinar a resposta final.
    """
    
    def __init__(self, config=None):
        """
        Inicializa o sistema de decisão.
        
        Args:
            config: Configuração do sistema de decisão (opcional)
        """
        self.config = config or DECISION_SYSTEM_CONFIG
        self.weights = self.config.get('weights', {'anomaly': 0.6, 'classification': 0.4})
        self.classification_threshold = self.config.get('classification_threshold', 0.7)
        self.anomaly_threshold = None  # Será definido pelo modelo de anomalia
        self.feedback_history = []
    
    def decide(self, anomaly_score, classification_result):
        """
        Toma decisão combinando resultados de anomalia e classificação.
        
        Args:
            anomaly_score: Pontuação de anomalia (erro de reconstrução)
            classification_result: Probabilidades de classes de ataque
            
        Returns:
            decision: Decisão final (0: normal, 1: ataque)
            attack_type: Tipo de ataque identificado (se houver)
            confidence: Nível de confiança na decisão
        """
        logger.debug(f"Tomando decisão com score de anomalia {anomaly_score:.6f} e classificação {classification_result}")
        
        # Iniciar timer
        start_time = performance_monitor.start_timer()
        
        try:
            # Verificar anomalia
            is_anomaly = anomaly_score > self.anomaly_threshold if self.anomaly_threshold else False
            
            # Verificar classificação
            max_class_prob = np.max(classification_result)
            predicted_class = np.argmax(classification_result)
            is_attack_classified = max_class_prob > self.classification_threshold and predicted_class > 0
            
            # Calcular confiança combinada
            if is_anomaly and is_attack_classified:
                # Ambos modelos indicam ataque
                confidence = self.weights['anomaly'] * (anomaly_score / self.anomaly_threshold) + \
                             self.weights['classification'] * max_class_prob
                decision = 1
                attack_type = predicted_class
            elif is_anomaly:
                # Apenas detector de anomalias indica ataque
                confidence = self.weights['anomaly'] * (anomaly_score / self.anomaly_threshold)
                decision = 1
                attack_type = 0  # Ataque desconhecido (possível zero-day)
            elif is_attack_classified:
                # Apenas classificador indica ataque
                confidence = self.weights['classification'] * max_class_prob
                decision = 1
                attack_type = predicted_class
            else:
                # Nenhum modelo indica ataque
                confidence = 1 - (self.weights['anomaly'] * (anomaly_score / (self.anomaly_threshold or 1)) + \
                                 self.weights['classification'] * max_class_prob)
                decision = 0
                attack_type = None
            
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('decision_system', 'decide', 
                                          elapsed_time * 1000, 'ms', 
                                          {'decision': decision, 'attack_type': attack_type, 'confidence': confidence})
            
            logger.debug(f"Decisão: {decision}, Tipo de ataque: {attack_type}, Confiança: {confidence:.4f}")
            return decision, attack_type, confidence
        
        except Exception as e:
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('decision_system', 'decide', 
                                          elapsed_time * 1000, 'ms', 
                                          {'error': str(e)})
            logger.error(f"Erro ao tomar decisão: {str(e)}")
            raise
    
    def update_from_feedback(self, feedback_data):
        """
        Atualiza parâmetros do sistema baseado em feedback.
        
        Args:
            feedback_data: Dados de feedback (detecção, classificação correta, etc.)
            
        Returns:
            updates: Atualizações realizadas no sistema
        """
        logger.info(f"Atualizando sistema de decisão com feedback")
        
        # Adicionar feedback ao histórico
        self.feedback_history.append(feedback_data)
        updates = {}
        
        # Verificar se é hora de atualizar com base no feedback acumulado
        if len(self.feedback_history) >= self.config.get('update_frequency', 10):
            # Atualizar threshold de anomalia
            if self.anomaly_threshold is not None:
                new_threshold = self._optimize_anomaly_threshold()
                if new_threshold:
                    old_threshold = self.anomaly_threshold
                    self.anomaly_threshold = new_threshold
                    updates['anomaly_threshold'] = {
                        'old': old_threshold,
                        'new': new_threshold
                    }
                    logger.info(f"Threshold de anomalia atualizado: {old_threshold:.6f} -> {new_threshold:.6f}")
            
            # Atualizar pesos do sistema de decisão
            new_weights = self._optimize_decision_weights()
            if new_weights:
                old_weights = self.weights.copy()
                self.weights = new_weights
                updates['decision_weights'] = {
                    'old': old_weights,
                    'new': new_weights
                }
                logger.info(f"Pesos atualizados: {old_weights} -> {new_weights}")
            
            # Limitar histórico para evitar crescimento infinito
            self.feedback_history = self.feedback_history[-100:]
        
        return updates
    
    def _optimize_anomaly_threshold(self):
        """
        Otimiza o threshold de anomalia com base no feedback.
        
        Returns:
            float: Novo threshold otimizado ou None se não for possível otimizar
        """
        # Extrair dados relevantes do feedback
        relevant_feedback = [f for f in self.feedback_history 
                            if 'anomaly_score' in f and 'true_label' in f]
        
        if len(relevant_feedback) < 5:  # Mínimo de amostras para otimização
            logger.debug("Feedback insuficiente para otimizar threshold de anomalia")
            return None
        
        # Extrair scores e labels
        scores = np.array([f['anomaly_score'] for f in relevant_feedback])
        labels = np.array([f['true_label'] for f in relevant_feedback])
        
        # Encontrar threshold ótimo
        best_f1 = 0
        best_threshold = None
        
        # Calcular precisão e recall para diferentes thresholds
        for threshold in np.linspace(np.min(scores), np.max(scores), 20):
            predictions = (scores > threshold).astype(int)
            
            # Calcular métricas
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            
            # Evitar divisão por zero
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Calcular F1-score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Aplicar suavização com threshold atual
        if best_threshold is not None:
            learning_rate = self.config.get('learning_rate', 0.1)
            new_threshold = self.anomaly_threshold * (1 - learning_rate) + best_threshold * learning_rate
            logger.debug(f"Threshold otimizado: {best_threshold:.6f}, F1: {best_f1:.4f}, Novo threshold (suavizado): {new_threshold:.6f}")
            return new_threshold
        
        return None
    
    def _optimize_decision_weights(self):
        """
        Otimiza os pesos do sistema de decisão com base no feedback.
        
        Returns:
            dict: Novos pesos otimizados ou None se não for possível otimizar
        """
        # Extrair dados relevantes do feedback
        relevant_feedback = [f for f in self.feedback_history 
                            if 'anomaly_score' in f and 'classification_result' in f and 'true_label' in f]
        
        if len(relevant_feedback) < 5:  # Mínimo de amostras para otimização
            logger.debug("Feedback insuficiente para otimizar pesos")
            return None
        
        # Calcular acurácia para diferentes combinações de pesos
        best_accuracy = 0
        best_weights = self.weights.copy()
        
        # Testar diferentes combinações de pesos
        for anomaly_weight in np.linspace(0.2, 0.8, 7):
            classification_weight = 1 - anomaly_weight
            weights = {'anomaly': anomaly_weight, 'classification': classification_weight}
            
            correct_predictions = 0
            
            for feedback in relevant_feedback:
                # Simular decisão com estes pesos
                anomaly_score = feedback['anomaly_score']
                classification_result = np.array(feedback['classification_result'])
                true_label = feedback['true_label']
                
                # Simplificação do processo de decisão
                is_anomaly = anomaly_score > self.anomaly_threshold if self.anomaly_threshold else False
                
                max_class_prob = np.max(classification_result)
                predicted_class = np.argmax(classification_result)
                is_attack_classified = max_class_prob > self.classification_threshold and predicted_class > 0
                
                # Decisão ponderada
                if (is_anomaly and weights['anomaly'] > 0.5) or (is_attack_classified and weights['classification'] > 0.5):
                    decision = 1
                else:
                    decision = 0
                
                if decision == (true_label > 0):
                    correct_predictions += 1
            
            accuracy = correct_predictions / len(relevant_feedback)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weights
        
        # Aplicar suavização com pesos atuais
        if best_accuracy > 0:
            learning_rate = self.config.get('learning_rate', 0.1)
            new_weights = {}
            for key in self.weights:
                new_weights[key] = self.weights[key] * (1 - learning_rate) + best_weights[key] * learning_rate
            
            logger.debug(f"Pesos otimizados: {best_weights}, Acurácia: {best_accuracy:.4f}, Novos pesos (suavizados): {new_weights}")
            return new_weights
        
        return None
    
    def save(self, filepath):
        """
        Salva o estado do sistema de decisão.
        
        Args:
            filepath: Caminho do arquivo
        """
        state = {
            'weights': self.weights,
            'classification_threshold': self.classification_threshold,
            'anomaly_threshold': self.anomaly_threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Salvar estado
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=4)
        
        logger.info(f"Estado do sistema de decisão salvo em {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Carrega o estado do sistema de decisão.
        
        Args:
            filepath: Caminho do arquivo
            
        Returns:
            DecisionSystem: Sistema de decisão carregado
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo {filepath} não encontrado")
        
        # Carregar estado
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Criar instância
        system = cls()
        
        # Restaurar estado
        system.weights = state.get('weights', system.weights)
        system.classification_threshold = state.get('classification_threshold', system.classification_threshold)
        system.anomaly_threshold = state.get('anomaly_threshold', system.anomaly_threshold)
        
        logger.info(f"Estado do sistema de decisão carregado de {filepath}")
        return system
