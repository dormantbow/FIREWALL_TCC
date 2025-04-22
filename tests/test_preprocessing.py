"""
Script para testar o carregamento e pré-processamento de dados do firewall neural.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Adicionar diretório pai ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.dataset_loader import dataset_loader
from src.preprocessing.preprocessor import DataPreprocessor
from src.utils.logging_utils import performance_monitor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)

def test_dataset_loading(dataset_type='nsl_kdd'):
    """
    Testa o carregamento dos datasets.
    
    Args:
        dataset_type: Tipo de dataset ('nsl_kdd' ou 'cicids_2018')
    """
    logger.info(f"Testando carregamento do dataset {dataset_type}")
    
    try:
        if dataset_type == 'nsl_kdd':
            # Carregar dados de treino e teste
            X_train, y_train = dataset_loader.load_nsl_kdd('train')
            X_test, y_test = dataset_loader.load_nsl_kdd('test')
            
            logger.info(f"Dataset NSL-KDD carregado com sucesso:")
            logger.info(f"  - Treino: {X_train.shape[0]} amostras, {X_train.shape[1]} características")
            logger.info(f"  - Teste: {X_test.shape[0]} amostras, {X_test.shape[1]} características")
            
            # Verificar distribuição de classes
            train_class_dist = y_train.value_counts().sort_index()
            test_class_dist = y_test.value_counts().sort_index()
            
            logger.info("Distribuição de classes (treino):")
            for cls, count in train_class_dist.items():
                logger.info(f"  - Classe {cls}: {count} amostras ({count/len(y_train)*100:.2f}%)")
            
            logger.info("Distribuição de classes (teste):")
            for cls, count in test_class_dist.items():
                logger.info(f"  - Classe {cls}: {count} amostras ({count/len(y_test)*100:.2f}%)")
            
            return X_train, y_train, X_test, y_test
            
        elif dataset_type == 'cicids_2018':
            # Carregar dados
            X, y = dataset_loader.load_cicids()
            
            logger.info(f"Dataset CICIDS-2018 carregado com sucesso:")
            logger.info(f"  - Total: {X.shape[0]} amostras, {X.shape[1]} características")
            
            # Verificar distribuição de classes
            class_dist = y.value_counts().sort_index()
            
            logger.info("Distribuição de classes:")
            for cls, count in class_dist.items():
                logger.info(f"  - Classe {cls}: {count} amostras ({count/len(y)*100:.2f}%)")
            
            # Dividir em treino e teste
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Dados divididos em treino e teste:")
            logger.info(f"  - Treino: {X_train.shape[0]} amostras")
            logger.info(f"  - Teste: {X_test.shape[0]} amostras")
            
            return X_train, y_train, X_test, y_test
            
        else:
            logger.error(f"Tipo de dataset desconhecido: {dataset_type}")
            return None
            
    except Exception as e:
        logger.error(f"Erro ao carregar dataset {dataset_type}: {str(e)}")
        raise

def test_preprocessing(X_train, y_train, X_test, y_test, dataset_type='nsl_kdd'):
    """
    Testa o pré-processamento dos dados.
    
    Args:
        X_train: Dados de treino
        y_train: Rótulos de treino
        X_test: Dados de teste
        y_test: Rótulos de teste
        dataset_type: Tipo de dataset ('nsl_kdd' ou 'cicids_2018')
    """
    logger.info(f"Testando pré-processamento para dataset {dataset_type}")
    
    try:
        # Criar preprocessador
        preprocessor = DataPreprocessor(dataset_type)
        
        # Pré-processar dados de treino
        logger.info("Pré-processando dados de treino...")
        X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
        
        # Pré-processar dados de teste
        logger.info("Pré-processando dados de teste...")
        X_test_processed = preprocessor.transform(X_test)
        
        logger.info(f"Pré-processamento concluído:")
        logger.info(f"  - Treino: {X_train_processed.shape[0]} amostras, {X_train_processed.shape[1]} características")
        logger.info(f"  - Teste: {X_test_processed.shape[0]} amostras, {X_test_processed.shape[1]} características")
        
        # Verificar se há valores ausentes
        train_missing = np.isnan(X_train_processed).sum().sum()
        test_missing = np.isnan(X_test_processed).sum().sum()
        
        logger.info(f"Valores ausentes após pré-processamento:")
        logger.info(f"  - Treino: {train_missing}")
        logger.info(f"  - Teste: {test_missing}")
        
        # Verificar distribuição de características
        logger.info("Analisando distribuição de características...")
        
        # Selecionar algumas características para visualização
        n_features = min(5, X_train_processed.shape[1])
        feature_indices = np.random.choice(X_train_processed.shape[1], n_features, replace=False)
        
        for i, idx in enumerate(feature_indices):
            logger.info(f"Característica {idx}:")
            logger.info(f"  - Média (treino): {X_train_processed[:, idx].mean():.4f}")
            logger.info(f"  - Desvio padrão (treino): {X_train_processed[:, idx].std():.4f}")
            logger.info(f"  - Mín (treino): {X_train_processed[:, idx].min():.4f}")
            logger.info(f"  - Máx (treino): {X_train_processed[:, idx].max():.4f}")
            logger.info(f"  - Média (teste): {X_test_processed[:, idx].mean():.4f}")
            logger.info(f"  - Desvio padrão (teste): {X_test_processed[:, idx].std():.4f}")
        
        return X_train_processed, y_train_processed, X_test_processed, y_test, preprocessor
        
    except Exception as e:
        logger.error(f"Erro no pré-processamento: {str(e)}")
        raise

def visualize_data(X_train, y_train, X_test, y_test, dataset_type='nsl_kdd'):
    """
    Visualiza os dados carregados.
    
    Args:
        X_train: Dados de treino
        y_train: Rótulos de treino
        X_test: Dados de teste
        y_test: Rótulos de teste
        dataset_type: Tipo de dataset ('nsl_kdd' ou 'cicids_2018')
    """
    logger.info(f"Gerando visualizações para dataset {dataset_type}")
    
    try:
        # Criar diretório para visualizações
        os.makedirs('visualizations', exist_ok=True)
        
        # 1. Distribuição de classes
        plt.figure(figsize=(10, 6))
        train_class_counts = pd.Series(y_train).value_counts().sort_index()
        test_class_counts = pd.Series(y_test).value_counts().sort_index()
        
        bar_width = 0.35
        index = np.arange(len(train_class_counts))
        
        plt.bar(index, train_class_counts, bar_width, label='Treino')
        plt.bar(index + bar_width, test_class_counts, bar_width, label='Teste')
        
        plt.xlabel('Classe')
        plt.ylabel('Número de amostras')
        plt.title(f'Distribuição de classes - {dataset_type}')
        plt.xticks(index + bar_width / 2, train_class_counts.index)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'visualizations/{dataset_type}_class_distribution.png')
        plt.close()
        
        # 2. Matriz de correlação (primeiras 20 características)
        if isinstance(X_train, pd.DataFrame):
            X_corr = X_train.iloc[:, :20]
        else:
            X_corr = pd.DataFrame(X_train[:, :20])
        
        plt.figure(figsize=(12, 10))
        corr_matrix = X_corr.corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title(f'Matriz de correlação (primeiras 20 características) - {dataset_type}')
        plt.tight_layout()
        plt.savefig(f'visualizations/{dataset_type}_correlation_matrix.png')
        plt.close()
        
        # 3. Distribuição de algumas características por classe
        if isinstance(X_train, pd.DataFrame):
            n_features = min(3, X_train.shape[1])
            feature_indices = np.random.choice(X_train.shape[1], n_features, replace=False)
            feature_names = X_train.columns[feature_indices]
            
            for i, feature in enumerate(feature_names):
                plt.figure(figsize=(10, 6))
                for cls in sorted(pd.Series(y_train).unique()):
                    sns.kdeplot(X_train.loc[y_train == cls, feature], label=f'Classe {cls}')
                
                plt.xlabel(feature)
                plt.ylabel('Densidade')
                plt.title(f'Distribuição de {feature} por classe - {dataset_type}')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'visualizations/{dataset_type}_{feature}_distribution.png')
                plt.close()
        
        logger.info(f"Visualizações salvas no diretório 'visualizations'")
        
    except Exception as e:
        logger.error(f"Erro ao gerar visualizações: {str(e)}")
        logger.info("Continuando sem visualizações...")

def main():
    """
    Função principal para testar o carregamento e pré-processamento de dados.
    """
    logger.info("Iniciando testes de carregamento e pré-processamento de dados")
    
    # Testar NSL-KDD
    logger.info("=== Testando dataset NSL-KDD ===")
    X_train, y_train, X_test, y_test = test_dataset_loading('nsl_kdd')
    X_train_processed, y_train_processed, X_test_processed, y_test_nsl, preprocessor_nsl = test_preprocessing(
        X_train, y_train, X_test, y_test, 'nsl_kdd'
    )
    visualize_data(X_train, y_train, X_test, y_test, 'nsl_kdd')
    
    # Testar CICIDS-2018 (com amostra menor para economizar tempo)
    logger.info("\n=== Testando dataset CICIDS-2018 ===")
    try:
        X_train, y_train, X_test, y_test = test_dataset_loading('cicids_2018')
        X_train_processed, y_train_processed, X_test_processed, y_test_cicids, preprocessor_cicids = test_preprocessing(
            X_train, y_train, X_test, y_test, 'cicids_2018'
        )
        visualize_data(X_train, y_train, X_test, y_test, 'cicids_2018')
    except Exception as e:
        logger.error(f"Erro ao testar CICIDS-2018: {str(e)}")
        logger.info("Continuando apenas com NSL-KDD")
    
    # Exibir resumo de desempenho
    logger.info("\n=== Resumo de desempenho ===")
    performance_summary = performance_monitor.get_summary()
    
    for operation, metrics in performance_summary.items():
        logger.info(f"Operação: {operation}")
        logger.info(f"  - Contagem: {metrics['count']}")
        logger.info(f"  - Tempo médio: {metrics['avg']:.2f} {metrics['unit']}")
        logger.info(f"  - Tempo mínimo: {metrics['min']:.2f} {metrics['unit']}")
        logger.info(f"  - Tempo máximo: {metrics['max']:.2f} {metrics['unit']}")
    
    logger.info("\nTestes de carregamento e pré-processamento concluídos com sucesso!")

if __name__ == "__main__":
    main()
