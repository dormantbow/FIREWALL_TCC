"""
Script para testar o sistema completo do firewall neural.
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

from src.main import NeuralFirewall
from src.utils.logging_utils import performance_monitor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

def test_firewall_training(dataset_type='nsl_kdd'):
    """
    Testa o treinamento do firewall neural.
    
    Args:
        dataset_type: Tipo de dataset ('nsl_kdd' ou 'cicids_2018')
        
    Returns:
        NeuralFirewall: Firewall neural treinado
    """
    logger.info(f"Testando treinamento do firewall neural com dataset {dataset_type}")
    
    try:
        # Criar diretório para visualizações
        os.makedirs('visualizations', exist_ok=True)
        
        # Inicializar firewall
        firewall = NeuralFirewall()
        
        # Treinar firewall
        logger.info(f"Iniciando treinamento...")
        result = firewall.train(dataset_type, save_models=True)
        
        # Verificar resultado
        logger.info(f"Treinamento concluído em {result['training_time']:.2f}s")
        logger.info(f"Acurácia do detector de anomalias: {result['anomaly_accuracy']:.4f}")
        logger.info(f"Acurácia do classificador: {result['classifier_accuracy']:.4f}")
        
        # Salvar resultado
        with open('visualizations/training_result.json', 'w') as f:
            # Converter valores numpy para tipos Python nativos
            result_json = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                          for k, v in result.items()}
            json.dump(result_json, f, indent=4)
        
        return firewall
        
    except Exception as e:
        logger.error(f"Erro no treinamento do firewall: {str(e)}")
        raise

def test_firewall_detection(firewall, dataset_type='nsl_kdd', num_samples=50):
    """
    Testa a detecção de ameaças do firewall neural.
    
    Args:
        firewall: Firewall neural treinado
        dataset_type: Tipo de dataset ('nsl_kdd' ou 'cicids_2018')
        num_samples: Número de amostras para testar
    """
    logger.info(f"Testando detecção de ameaças do firewall neural")
    
    try:
        # Criar diretório para visualizações
        os.makedirs('visualizations', exist_ok=True)
        
        # Carregar dados de teste
        if dataset_type == 'nsl_kdd':
            from src.preprocessing.dataset_loader import dataset_loader
            X_test, y_test = dataset_loader.load_nsl_kdd('test')
        elif dataset_type == 'cicids_2018':
            from src.preprocessing.dataset_loader import dataset_loader
            X, y = dataset_loader.load_cicids()
            
            from sklearn.model_selection import train_test_split
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            raise ValueError(f"Tipo de dataset desconhecido: {dataset_type}")
        
        # Selecionar amostras para teste
        num_samples = min(num_samples, len(X_test))
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        # Resultados de detecção
        detection_results = []
        
        # Testar detecção
        logger.info(f"Testando detecção com {num_samples} amostras")
        
        for i, idx in enumerate(indices):
            # Obter amostra
            if isinstance(X_test, pd.DataFrame):
                sample = X_test.iloc[idx].to_dict()
                true_label = int(y_test.iloc[idx])
            else:
                sample = {f'feature_{j}': X_test[idx, j] for j in range(X_test.shape[1])}
                true_label = int(y_test[idx])
            
            # Criar contexto simulado
            context = {
                'source_ip': f'192.168.1.{np.random.randint(1, 255)}',
                'destination_ip': f'10.0.0.{np.random.randint(1, 255)}',
                'timestamp': datetime.now().isoformat()
            }
            
            # Detectar ameaças
            try:
                result = firewall.detect(sample)
                
                # Adicionar informações adicionais
                result['sample_idx'] = int(idx)
                result['true_label'] = true_label
                
                # Converter valores numpy para tipos Python nativos
                result_clean = {}
                for k, v in result.items():
                    if isinstance(v, (np.float32, np.float64)):
                        result_clean[k] = float(v)
                    elif isinstance(v, np.int64):
                        result_clean[k] = int(v)
                    else:
                        result_clean[k] = v
                
                detection_results.append(result_clean)
                
                if i % 10 == 0:
                    logger.info(f"Progresso: {i+1}/{num_samples} amostras processadas")
                
            except Exception as e:
                logger.error(f"Erro ao processar amostra {idx}: {str(e)}")
        
        # Salvar resultados
        with open('visualizations/firewall_detection_results.json', 'w') as f:
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
        
        logger.info(f"Métricas de desempenho do firewall neural (binário):")
        logger.info(f"  - Acurácia: {accuracy:.4f}")
        logger.info(f"  - Precisão: {precision:.4f}")
        logger.info(f"  - Recall: {recall:.4f}")
        logger.info(f"  - F1-score: {f1:.4f}")
        
        # Plotar matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de confusão - Firewall Neural')
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
        plt.savefig('visualizations/firewall_confusion_matrix.png')
        plt.close()
        
        # Analisar distribuição de confiança
        confidences = [r['confidence'] for r in detection_results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20)
        plt.xlabel('Nível de confiança')
        plt.ylabel('Frequência')
        plt.title('Distribuição de níveis de confiança')
        plt.grid(True)
        plt.savefig('visualizations/firewall_confidence_distribution.png')
        plt.close()
        
        # Analisar tempo de processamento
        processing_times = [r['processing_time'] for r in detection_results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(processing_times, bins=20)
        plt.xlabel('Tempo de processamento (s)')
        plt.ylabel('Frequência')
        plt.title('Distribuição de tempos de processamento')
        plt.grid(True)
        plt.savefig('visualizations/firewall_processing_time_distribution.png')
        plt.close()
        
        # Analisar ações de resposta
        response_actions = {}
        for r in detection_results:
            if r['decision'] == 1:  # Ataque detectado
                actions = r['response']['actions']
                for action in actions:
                    response_actions[action] = response_actions.get(action, 0) + 1
        
        if response_actions:
            plt.figure(figsize=(10, 6))
            plt.bar(response_actions.keys(), response_actions.values())
            plt.xlabel('Ação de resposta')
            plt.ylabel('Frequência')
            plt.title('Distribuição de ações de resposta')
            plt.grid(True)
            plt.savefig('visualizations/firewall_response_actions.png')
            plt.close()
        
        # Analisar tipos de ataque detectados
        attack_types = {}
        for r in detection_results:
            if r['decision'] == 1:  # Ataque detectado
                attack_type = r['attack_type']
                attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
        
        if attack_types:
            plt.figure(figsize=(10, 6))
            plt.bar(attack_types.keys(), attack_types.values())
            plt.xlabel('Tipo de ataque')
            plt.ylabel('Frequência')
            plt.title('Distribuição de tipos de ataque detectados')
            plt.grid(True)
            plt.savefig('visualizations/firewall_attack_types.png')
            plt.close()
        
    except Exception as e:
        logger.error(f"Erro ao testar detecção do firewall: {str(e)}")
        raise

def test_firewall_feedback(firewall, num_samples=10):
    """
    Testa o processamento de feedback do firewall neural.
    
    Args:
        firewall: Firewall neural treinado
        num_samples: Número de amostras para testar
    """
    logger.info(f"Testando processamento de feedback do firewall neural")
    
    try:
        # Verificar se há log de detecção
        if not hasattr(firewall, 'intrusion_detector') or not firewall.intrusion_detector.detection_log:
            logger.warning("Nenhum log de detecção disponível para processar feedback")
            return
        
        # Selecionar amostras para feedback
        detection_log = firewall.intrusion_detector.detection_log
        num_samples = min(num_samples, len(detection_log))
        indices = np.random.choice(len(detection_log), num_samples, replace=False)
        
        # Resultados de feedback
        feedback_results = []
        
        # Processar feedback
        logger.info(f"Processando feedback para {num_samples} detecções")
        
        for i, idx in enumerate(indices):
            # Obter detecção
            detection = detection_log[idx]
            
            # Simular feedback (correto ou incorreto)
            is_correct = np.random.choice([True, False], p=[0.7, 0.3])  # 70% correto, 30% incorreto
            true_label = detection.get('decision', 0) if is_correct else (1 - detection.get('decision', 0))
            
            # Processar feedback
            try:
                result = firewall.process_feedback(
                    idx, 
                    is_correct, 
                    true_label=true_label,
                    comments=f"Feedback simulado {i+1}/{num_samples}"
                )
                
                # Adicionar informações adicionais
                result['detection_idx'] = idx
                result['is_correct'] = is_correct
                result['true_label'] = true_label
                
                feedback_results.append(result)
                
                logger.info(f"Feedback {i+1}/{num_samples} processado: {'Correto' if is_correct else 'Incorreto'}")
                
            except Exception as e:
                logger.error(f"Erro ao processar feedback para detecção {idx}: {str(e)}")
        
        # Salvar resultados
        with open('visualizations/firewall_feedback_results.json', 'w') as f:
            json.dump(feedback_results, f, indent=4)
        
        logger.info(f"Processamento de feedback concluído para {len(feedback_results)} detecções")
        
    except Exception as e:
        logger.error(f"Erro ao testar processamento de feedback: {str(e)}")
        raise

def test_firewall_performance():
    """
    Testa o desempenho do firewall neural.
    """
    logger.info("Testando desempenho do firewall neural")
    
    try:
        # Obter resumo de desempenho
        performance_summary = performance_monitor.get_summary()
        
        # Salvar resumo
        with open('visualizations/firewall_performance_summary.json', 'w') as f:
            json.dump(performance_summary, f, indent=4)
        
        # Plotar tempos médios por operação
        operations = []
        avg_times = []
        
        for operation, metrics in performance_summary.items():
            operations.append(operation)
            avg_times.append(metrics['avg'])
        
        plt.figure(figsize=(12, 8))
        plt.barh(operations, avg_times)
        plt.xlabel('Tempo médio (ms)')
        plt.ylabel('Operação')
        plt.title('Tempo médio por operação')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('visualizations/firewall_performance_by_operation.png')
        plt.close()
        
        logger.info("Análise de desempenho concluída")
        
    except Exception as e:
        logger.error(f"Erro ao testar desempenho: {str(e)}")
        raise

def main():
    """
    Função principal para testar o sistema completo do firewall neural.
    """
    logger.info("Iniciando testes do sistema completo do firewall neural")
    
    # Testar treinamento
    logger.info("=== Testando treinamento do firewall ===")
    firewall = test_firewall_training('nsl_kdd')
    
    # Testar detecção
    logger.info("\n=== Testando detecção de ameaças ===")
    test_firewall_detection(firewall, 'nsl_kdd', num_samples=50)
    
    # Testar processamento de feedback
    logger.info("\n=== Testando processamento de feedback ===")
    test_firewall_feedback(firewall, num_samples=10)
    
    # Testar desempenho
    logger.info("\n=== Testando desempenho do firewall ===")
    test_firewall_performance()
    
    logger.info("\nTestes do sistema completo do firewall neural concluídos com sucesso!")

if __name__ == "__main__":
    main()
