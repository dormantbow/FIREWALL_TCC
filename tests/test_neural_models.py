"""
Script para testar os modelos de rede neural do firewall.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

# Adicionar diretório pai ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.dataset_loader import dataset_loader
from src.preprocessing.preprocessor import DataPreprocessor
from src.models.neural_models import AutoencoderModel, CNNLSTMModel
from src.utils.logging_utils import performance_monitor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

def prepare_data(dataset_type='nsl_kdd'):
    """
    Prepara os dados para teste dos modelos.
    
    Args:
        dataset_type: Tipo de dataset ('nsl_kdd' ou 'cicids_2018')
        
    Returns:
        tuple: (X_train_processed, y_train_processed, X_test_processed, y_test, input_dim, num_classes)
    """
    logger.info(f"Preparando dados do dataset {dataset_type} para teste dos modelos")
    
    try:
        if dataset_type == 'nsl_kdd':
            # Carregar dados de treino e teste
            X_train, y_train = dataset_loader.load_nsl_kdd('train')
            X_test, y_test = dataset_loader.load_nsl_kdd('test')
        elif dataset_type == 'cicids_2018':
            # Carregar dados
            X, y = dataset_loader.load_cicids()
            
            # Dividir em treino e teste
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
        
        logger.info(f"Dados preparados com sucesso:")
        logger.info(f"  - Dimensão de entrada: {input_dim}")
        logger.info(f"  - Número de classes: {num_classes}")
        logger.info(f"  - Amostras de treino: {X_train_processed.shape[0]}")
        logger.info(f"  - Amostras de teste: {X_test_processed.shape[0]}")
        
        return X_train_processed, y_train_processed, X_test_processed, y_test, input_dim, num_classes
        
    except Exception as e:
        logger.error(f"Erro ao preparar dados: {str(e)}")
        raise

def test_autoencoder_model(X_train, y_train, X_test, y_test, input_dim):
    """
    Testa o modelo de autoencoder para detecção de anomalias.
    
    Args:
        X_train: Dados de treino processados
        y_train: Rótulos de treino processados
        X_test: Dados de teste processados
        y_test: Rótulos de teste
        input_dim: Dimensão de entrada
        
    Returns:
        AutoencoderModel: Modelo treinado
    """
    logger.info("Testando modelo de autoencoder para detecção de anomalias")
    
    try:
        # Criar diretório para visualizações
        os.makedirs('visualizations', exist_ok=True)
        
        # Usar apenas dados normais para treinar o autoencoder
        normal_indices = np.where(y_train == 0)[0]
        X_train_normal = X_train[normal_indices]
        
        logger.info(f"Treinando autoencoder com {len(X_train_normal)} amostras normais")
        
        # Criar e treinar modelo
        autoencoder = AutoencoderModel(input_dim)
        autoencoder.build()
        history = autoencoder.train(X_train_normal, epochs=10, batch_size=32, validation_split=0.2)
        
        # Plotar histórico de treinamento
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Treino')
        plt.plot(history.history['val_loss'], label='Validação')
        plt.xlabel('Época')
        plt.ylabel('Erro de reconstrução (MSE)')
        plt.title('Histórico de treinamento do autoencoder')
        plt.legend()
        plt.grid(True)
        plt.savefig('visualizations/autoencoder_training_history.png')
        plt.close()
        
        # Calcular erros de reconstrução
        logger.info("Calculando erros de reconstrução")
        
        # Para dados normais (treino)
        reconstruction_errors_normal = autoencoder.get_reconstruction_errors(X_train_normal)
        
        # Para dados de teste
        reconstruction_errors_test = autoencoder.get_reconstruction_errors(X_test)
        
        # Plotar distribuição de erros de reconstrução
        plt.figure(figsize=(10, 6))
        plt.hist(reconstruction_errors_normal, bins=50, alpha=0.5, label='Normal (treino)')
        plt.hist(reconstruction_errors_test, bins=50, alpha=0.5, label='Teste (todos)')
        plt.xlabel('Erro de reconstrução')
        plt.ylabel('Frequência')
        plt.title('Distribuição de erros de reconstrução')
        plt.legend()
        plt.grid(True)
        plt.savefig('visualizations/reconstruction_errors_distribution.png')
        plt.close()
        
        # Plotar distribuição de erros por classe
        plt.figure(figsize=(10, 6))
        for cls in sorted(np.unique(y_test)):
            indices = np.where(y_test == cls)[0]
            if len(indices) > 0:
                errors = reconstruction_errors_test[indices]
                plt.hist(errors, bins=30, alpha=0.5, label=f'Classe {cls}')
        
        plt.xlabel('Erro de reconstrução')
        plt.ylabel('Frequência')
        plt.title('Distribuição de erros de reconstrução por classe')
        plt.legend()
        plt.grid(True)
        plt.savefig('visualizations/reconstruction_errors_by_class.png')
        plt.close()
        
        # Calcular threshold ótimo
        logger.info("Calculando threshold ótimo para detecção de anomalias")
        autoencoder.calculate_threshold(X_train_normal)
        logger.info(f"Threshold calculado: {autoencoder.threshold:.6f}")
        
        # Avaliar desempenho
        logger.info("Avaliando desempenho do detector de anomalias")
        anomalies = autoencoder.detect_anomalies(X_test)
        y_test_binary = (y_test > 0).astype(int)  # Converter para binário (0: normal, 1: ataque)
        
        # Calcular métricas
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test_binary, anomalies)
        precision = precision_score(y_test_binary, anomalies)
        recall = recall_score(y_test_binary, anomalies)
        f1 = f1_score(y_test_binary, anomalies)
        
        logger.info(f"Métricas de desempenho do detector de anomalias:")
        logger.info(f"  - Acurácia: {accuracy:.4f}")
        logger.info(f"  - Precisão: {precision:.4f}")
        logger.info(f"  - Recall: {recall:.4f}")
        logger.info(f"  - F1-score: {f1:.4f}")
        
        # Plotar matriz de confusão
        cm = confusion_matrix(y_test_binary, anomalies)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de confusão - Detector de anomalias')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Normal', 'Anomalia'])
        plt.yticks(tick_marks, ['Normal', 'Anomalia'])
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('Classe real')
        plt.xlabel('Classe prevista')
        plt.tight_layout()
        plt.savefig('visualizations/anomaly_detection_confusion_matrix.png')
        plt.close()
        
        # Plotar curva ROC
        fpr, tpr, thresholds = roc_curve(y_test_binary, reconstruction_errors_test)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de falsos positivos')
        plt.ylabel('Taxa de verdadeiros positivos')
        plt.title('Curva ROC - Detector de anomalias')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('visualizations/anomaly_detection_roc_curve.png')
        plt.close()
        
        # Plotar curva Precision-Recall
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_binary, reconstruction_errors_test)
        pr_auc = auc(recall_curve, precision_curve)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precisão')
        plt.title('Curva Precision-Recall - Detector de anomalias')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig('visualizations/anomaly_detection_pr_curve.png')
        plt.close()
        
        return autoencoder
        
    except Exception as e:
        logger.error(f"Erro ao testar modelo de autoencoder: {str(e)}")
        raise

def test_cnn_lstm_model(X_train, y_train, X_test, y_test, input_dim, num_classes):
    """
    Testa o modelo CNN+LSTM para classificação de ataques.
    
    Args:
        X_train: Dados de treino processados
        y_train: Rótulos de treino processados
        X_test: Dados de teste processados
        y_test: Rótulos de teste
        input_dim: Dimensão de entrada
        num_classes: Número de classes
        
    Returns:
        CNNLSTMModel: Modelo treinado
    """
    logger.info("Testando modelo CNN+LSTM para classificação de ataques")
    
    try:
        # Criar diretório para visualizações
        os.makedirs('visualizations', exist_ok=True)
        
        # Criar e treinar modelo
        classifier = CNNLSTMModel(input_dim, num_classes)
        classifier.build()
        
        # Preparar dados de sequência
        X_train_seq = classifier.prepare_sequence_data(X_train)
        y_train_seq = y_train[-len(X_train_seq):]  # Ajustar rótulos para sequências
        
        logger.info(f"Treinando classificador com {len(X_train_seq)} sequências")
        
        history = classifier.train(X_train_seq, y_train_seq, epochs=10, batch_size=32, validation_split=0.2)
        
        # Plotar histórico de treinamento
        plt.figure(figsize=(12, 5))
        
        # Plotar loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Treino')
        plt.plot(history.history['val_loss'], label='Validação')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.title('Histórico de loss')
        plt.legend()
        plt.grid(True)
        
        # Plotar acurácia
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Treino')
        plt.plot(history.history['val_accuracy'], label='Validação')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.title('Histórico de acurácia')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('visualizations/cnn_lstm_training_history.png')
        plt.close()
        
        # Avaliar modelo
        logger.info("Avaliando desempenho do classificador")
        
        # Preparar dados de teste
        X_test_seq = classifier.prepare_sequence_data(X_test)
        y_test_seq = y_test[-len(X_test_seq):]  # Ajustar rótulos para sequências
        
        # Avaliar modelo
        metrics = classifier.evaluate(X_test_seq, y_test_seq)
        
        logger.info(f"Métricas de avaliação:")
        logger.info(f"  - Loss: {metrics[0]:.4f}")
        logger.info(f"  - Acurácia: {metrics[1]:.4f}")
        
        # Fazer previsões
        y_pred = classifier.predict(X_test_seq)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Plotar matriz de confusão
        cm = confusion_matrix(y_test_seq, y_pred_classes)
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de confusão - Classificador CNN+LSTM')
        plt.colorbar()
        
        # Ajustar rótulos das classes
        classes = sorted(np.unique(y_test_seq))
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Adicionar valores à matriz
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('Classe real')
        plt.xlabel('Classe prevista')
        plt.tight_layout()
        plt.savefig('visualizations/cnn_lstm_confusion_matrix.png')
        plt.close()
        
        # Gerar relatório de classificação
        report = classification_report(y_test_seq, y_pred_classes)
        logger.info(f"Relatório de classificação:\n{report}")
        
        # Plotar curvas ROC para cada classe (one-vs-rest)
        plt.figure(figsize=(10, 8))
        
        for i in range(num_classes):
            # Preparar rótulos binários (classe atual vs resto)
            y_test_bin = (y_test_seq == i).astype(int)
            y_score = y_pred[:, i]
            
            # Calcular curva ROC
            fpr, tpr, _ = roc_curve(y_test_bin, y_score)
            roc_auc = auc(fpr, tpr)
            
            # Plotar curva
            plt.plot(fpr, tpr, lw=2, label=f'Classe {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de falsos positivos')
        plt.ylabel('Taxa de verdadeiros positivos')
        plt.title('Curvas ROC - Classificador CNN+LSTM (one-vs-rest)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('visualizations/cnn_lstm_roc_curves.png')
        plt.close()
        
        return classifier
        
    except Exception as e:
        logger.error(f"Erro ao testar modelo CNN+LSTM: {str(e)}")
        raise

def main():
    """
    Função principal para testar os modelos de rede neural.
    """
    logger.info("Iniciando testes dos modelos de rede neural")
    
    # Preparar dados
    logger.info("=== Preparando dados ===")
    X_train, y_train, X_test, y_test, input_dim, num_classes = prepare_data('nsl_kdd')
    
    # Testar modelo de autoencoder
    logger.info("\n=== Testando modelo de autoencoder ===")
    autoencoder = test_autoencoder_model(X_train, y_train, X_test, y_test, input_dim)
    
    # Testar modelo CNN+LSTM
    logger.info("\n=== Testando modelo CNN+LSTM ===")
    classifier = test_cnn_lstm_model(X_train, y_train, X_test, y_test, input_dim, num_classes)
    
    # Exibir resumo de desempenho
    logger.info("\n=== Resumo de desempenho ===")
    performance_summary = performance_monitor.get_summary()
    
    for operation, metrics in performance_summary.items():
        logger.info(f"Operação: {operation}")
        logger.info(f"  - Contagem: {metrics['count']}")
        logger.info(f"  - Tempo médio: {metrics['avg']:.2f} {metrics['unit']}")
        logger.info(f"  - Tempo mínimo: {metrics['min']:.2f} {metrics['unit']}")
(Content truncated due to size limit. Use line ranges to read in chunks)