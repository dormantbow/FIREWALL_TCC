"""
Script para testar o sistema de detecção de intrusão do firewall neural.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Adicionar diretório pai ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.dataset_loader import dataset_loader
from src.preprocessing.preprocessor import DataPreprocessor
from src.models.neural_models import AutoencoderModel, CNNLSTMModel
from src.detection.decision_system import DecisionSystem
from src.detection.intrusion_detector import IntrusionDetector
from src.utils.logging_utils import performance_monitor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

def prepare_models(dataset_type='nsl_kdd'):
    """
    Prepara os modelos necessários para o sistema de detecção de intrusão.
    
    Args:
        dataset_type: Tipo de dataset ('nsl_kdd' ou 'cicids_2018')
        
    Returns:
        tuple: (autoencoder, classifier, decision_system, X_test, y_test)
    """
    logger.info(f"Preparando modelos para teste do sistema de detecção de intrusão")
    
    try:
        # Carregar e pré-processar dados
        if dataset_type == 'nsl_kdd':
            X_train, y_train = dataset_loader.load_nsl_kdd('train')
            X_test, y_test = dataset_loader.load_nsl_kdd('test')
        elif dataset_type == 'cicids_2018':
            X, y = dataset_loader.load_cicids()
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            raise ValueError(f"Tipo de dataset desconhecido: {dataset_type}")
        
        # Pré-processar dados
        preprocessor = DataPreprocessor(dataset_type)
        X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Obter dimensões
        input_dim = X_train_processed.shape[1]
        num_classes = len(np.unique(y_train_processed))
        
        # Treinar autoencoder
        logger.info("Treinando modelo de autoencoder")
        normal_indices = np.where(y_train_processed == 0)[0]
        X_train_normal = X_train_processed[normal_indices]
        
        autoencoder = AutoencoderModel(input_dim)
        autoencoder.build()
        autoencoder.train(X_train_normal, epochs=5, batch_size=32, validation_split=0.2)
        autoencoder.calculate_threshold(X_train_normal)
        
        # Treinar classificador
        logger.info("Treinando modelo CNN+LSTM")
        classifier = CNNLSTMModel(input_dim, num_classes)
        classifier.build()
        
        X_train_seq = classifier.prepare_sequence_data(X_train_processed)
        y_train_seq = y_train_processed[-len(X_train_seq):]
        
        classifier.train(X_train_seq, y_train_seq, epochs=5, batch_size=32, validation_split=0.2)
        
        # Inicializar sistema de decisão
        logger.info("Inicializando sistema de decisão")
        decision_system = DecisionSystem()
        decision_system.anomaly_threshold = autoencoder.threshold
        
        logger.info("Modelos preparados com sucesso")
        return autoencoder, classifier, decision_system, X_test_processed, y_test
        
    except Exception as e:
        logger.error(f"Erro ao preparar modelos: {str(e)}")
        raise

def test_intrusion_detector(autoencoder, classifier, decision_system, X_test, y_test):
    """
    Testa o detector de intrusão.
    
    Args:
        autoencoder: Modelo de autoencoder treinado
        classifier: Modelo CNN+LSTM treinado
        decision_system: Sistema de decisão
        X_test: Dados de teste processados
        y_test: Rótulos de teste
        
    Returns:
        IntrusionDetector: Detector de intrusão testado
    """
    logger.info("Testando detector de intrusão")
    
    try:
        # Criar diretório para visualizações
        os.makedirs('visualizations', exist_ok=True)
        
        # Inicializar detector de intrusão
        intrusion_detector = IntrusionDetector(autoencoder, classifier, decision_system)
        
        # Selecionar amostras para teste
        num_samples = min(100, len(X_test))
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        # Preparar dados para classificador
        X_test_seq = classifier.prepare_sequence_data(X_test)
        
        # Resultados de detecção
        detection_results = []
        
        # Testar detector
        logger.info(f"Testando detector com {num_samples} amostras")
        
        for i, idx in enumerate(indices):
            # Criar contexto simulado
            context = {
                'source_ip': f'192.168.1.{np.random.randint(1, 255)}',
                'destination_ip': f'10.0.0.{np.random.randint(1, 255)}',
                'timestamp': datetime.now().isoformat()
            }
            
            # Detectar intrusão
            try:
                # Encontrar índice correspondente em X_test_seq
                seq_idx = min(idx, len(X_test_seq) - 1)
                
                decision, attack_type, confidence, explanation = intrusion_detector.detect(
                    X_test[idx], 
                    sequence_data=np.array([X_test_seq[seq_idx]]), 
                    context=context
                )
                
                # Registrar resultado
                result = {
                    'sample_idx': idx,
                    'true_label': int(y_test.iloc[idx]) if hasattr(y_test, 'iloc') else int(y_test[idx]),
                    'decision': decision,
                    'attack_type': attack_type,
                    'confidence': float(confidence),
                    'context': context
                }
                
                detection_results.append(result)
                
                if i % 10 == 0:
                    logger.info(f"Progresso: {i+1}/{num_samples} amostras processadas")
                
            except Exception as e:
                logger.error(f"Erro ao processar amostra {idx}: {str(e)}")
        
        # Salvar resultados
        with open('visualizations/detection_results.json', 'w') as f:
            json.dump(detection_results, f, indent=4)
        
        # Analisar resultados
        logger.info("Analisando resultados da detecção")
        
        # Converter para arrays
        y_true = np.array([r['true_label'] > 0 for r in detection_results], dtype=int)
        y_pred = np.array([r['decision'] for r in detection_results], dtype=int)
        
        # Calcular métricas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        logger.info(f"Métricas de desempenho do detector de intrusão (binário):")
        logger.info(f"  - Acurácia: {accuracy:.4f}")
        logger.info(f"  - Precisão: {precision:.4f}")
        logger.info(f"  - Recall: {recall:.4f}")
        logger.info(f"  - F1-score: {f1:.4f}")
        
        # Plotar matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de confusão - Detector de intrusão')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Normal', 'Ataque'])
        plt.yticks(tick_marks, ['Normal', 'Ataque'])
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('Classe real')
        plt.xlabel('Classe prevista')
        plt.tight_layout()
        plt.savefig('visualizations/intrusion_detector_confusion_matrix.png')
        plt.close()
        
        # Analisar distribuição de confiança
        confidences = [r['confidence'] for r in detection_results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20)
        plt.xlabel('Nível de confiança')
        plt.ylabel('Frequência')
        plt.title('Distribuição de níveis de confiança')
        plt.grid(True)
        plt.savefig('visualizations/confidence_distribution.png')
        plt.close()
        
        # Analisar confiança por tipo de decisão
        conf_correct = [r['confidence'] for r in detection_results if (r['decision'] == 1 and r['true_label'] > 0) or (r['decision'] == 0 and r['true_label'] == 0)]
        conf_incorrect = [r['confidence'] for r in detection_results if (r['decision'] == 1 and r['true_label'] == 0) or (r['decision'] == 0 and r['true_label'] > 0)]
        
        plt.figure(figsize=(10, 6))
        plt.hist(conf_correct, bins=20, alpha=0.5, label='Decisões corretas')
        plt.hist(conf_incorrect, bins=20, alpha=0.5, label='Decisões incorretas')
        plt.xlabel('Nível de confiança')
        plt.ylabel('Frequência')
        plt.title('Distribuição de confiança por tipo de decisão')
        plt.legend()
        plt.grid(True)
        plt.savefig('visualizations/confidence_by_decision_type.png')
        plt.close()
        
        # Testar processamento de feedback
        logger.info("Testando processamento de feedback")
        
        # Simular feedback para algumas detecções
        num_feedback = min(10, len(detection_results))
        feedback_indices = np.random.choice(len(detection_results), num_feedback, replace=False)
        
        for i, idx in enumerate(feedback_indices):
            result = detection_results[idx]
            is_correct = (result['decision'] == 1 and result['true_label'] > 0) or (result['decision'] == 0 and result['true_label'] == 0)
            
            feedback = intrusion_detector.process_feedback(
                idx, 
                is_correct, 
                true_label=result['true_label'],
                comments=f"Feedback simulado {i+1}/{num_feedback}"
            )
            
            logger.info(f"Feedback {i+1}/{num_feedback} processado: {'Correto' if is_correct else 'Incorreto'}")
        
        return intrusion_detector
        
    except Exception as e:
        logger.error(f"Erro ao testar detector de intrusão: {str(e)}")
        raise

def test_zero_day_detection(intrusion_detector, X_test, y_test):
    """
    Testa a capacidade de detecção de ataques zero-day.
    
    Args:
        intrusion_detector: Detector de intrusão
        X_test: Dados de teste processados
        y_test: Rótulos de teste
    """
    logger.info("Testando capacidade de detecção de ataques zero-day")
    
    try:
        # Criar diretório para visualizações
        os.makedirs('visualizations', exist_ok=True)
        
        # Simular ataques zero-day modificando amostras normais
        logger.info("Simulando ataques zero-day")
        
        # Selecionar amostras normais
        normal_indices = np.where(y_test == 0)[0]
        num_samples = min(50, len(normal_indices))
        selected_indices = np.random.choice(normal_indices, num_samples, replace=False)
        
        # Resultados de detecção
        zero_day_results = []
        
        # Modificar características para simular ataques zero-day
        for i, idx in enumerate(selected_indices):
            # Copiar amostra original
            original_sample = X_test[idx].copy()
            
            # Modificar algumas características aleatoriamente
            modified_sample = original_sample.copy()
            
            # Selecionar características para modificar
            num_features = min(5, len(modified_sample))
            feature_indices = np.random.choice(len(modified_sample), num_features, replace=False)
            
            # Aplicar modificações
            for feat_idx in feature_indices:
                # Aumentar valor em 50-200%
                modified_sample[feat_idx] *= (1.5 + np.random.random())
            
            # Criar contexto simulado
            context = {
                'source_ip': f'192.168.1.{np.random.randint(1, 255)}',
                'destination_ip': f'10.0.0.{np.random.randint(1, 255)}',
                'timestamp': datetime.now().isoformat(),
                'simulated_zero_day': True
            }
            
            # Detectar intrusão na amostra original
            decision_orig, attack_type_orig, confidence_orig, _ = intrusion_detector.detect(
                original_sample, context=context.copy()
            )
            
            # Detectar intrusão na amostra modificada
            decision_mod, attack_type_mod, confidence_mod, explanation = intrusion_detector.detect(
                modified_sample, context=context
            )
            
            # Registrar resultado
            result = {
                'sample_idx': int(idx),
                'original_decision': int(decision_orig),
                'original_confidence': float(confidence_orig),
                'modified_decision': int(decision_mod),
                'modified_confidence': float(confidence_mod),
                'attack_type': attack_type_mod,
                'features_modified': [int(idx) for idx in feature_indices],
                'detected_as_zero_day': attack_type_mod == 0 and decision_mod == 1
            }
            
            zero_day_results.append(result)
            
            if i % 10 == 0:
                logger.info(f"Progresso: {i+1}/{num_samples} amostras processadas")
        
        # Salvar resultados
        with open('visualizations/zero_day_detection_results.json', 'w') as f:
            json.dump(zero_day_results, f, indent=4)
        
        # Analisar resultados
        logger.info("Analisando resultados da detecção de ataques zero-day")
        
        # Calcular taxa de detecção
        detection_rate = sum(1 for r in zero_day_results if r['modified_decision'] == 1) / len(zero_day_results)
        zero_day_detection_rate = sum(1 for r in zero_day_results if r['detected_as_zero_day']) / len(zero_day_results)
        
        logger.info(f"Taxa de detecção de ataques simulados: {detection_rate:.4f}")
        logger.info(f"Taxa de detecção como zero-day: {zero_day_detection_rate:.4f}")
        
        # Plotar comparação de confiança
        original_conf = [r['original_confidence'] for r in zero_day_results]
        modified_conf = [r['modified_confidence'] for r in zero_day_results]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(zero_day_results)), original_conf, label='Amostra original', alpha=0.7)
        plt.scatter(range(len(zero_day_results)), modified_conf, label='Amostra modificada (zero-day)', alpha=0.7)
        plt.xlabel('Índice da amostra')
        plt.ylabel('Nível de confiança')
        plt.title('Comparação de confiança: amostras originais vs. modificadas')
        plt.legend()
        plt.grid(True)
        plt.savefig('visualizations/zero_day_confidence_comparison.png')
        plt.close()
        
        # Plotar distribuição de diferenças de confiança
        conf_diff = [m - o for m, o in zip(modified_conf, original_conf)]
        
        plt.figure(figsize=(10, 6))
        plt.hist(conf_diff, bins=20)
        plt.xlabel('Diferença de confiança (modificada - original)')
        plt.ylabel('Frequência')
        plt.title('Distribuição de diferenças de confiança')
 
(Content truncated due to size limit. Use line ranges to read in chunks)