"""
Módulo de pré-processamento para o firewall baseado em rede neural.
Implementa a transformação e normalização dos dados para alimentar os modelos.
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

from ..utils.config import PREPROCESSING_CONFIG
from ..utils.logging_utils import performance_monitor

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Classe para pré-processamento dos dados para o firewall baseado em rede neural.
    """
    
    def __init__(self, dataset_type='nsl_kdd'):
        """
        Inicializa o pré-processador de dados.
        
        Args:
            dataset_type: Tipo de dataset ('nsl_kdd' ou 'cicids_2018')
        """
        self.dataset_type = dataset_type
        self.config = PREPROCESSING_CONFIG[dataset_type]
        
        # Inicializar transformadores
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.feature_selector = None
        
        # Flag para indicar se o pré-processador foi treinado
        self.is_fitted = False
    
    def fit_transform(self, X, y=None):
        """
        Ajusta os transformadores aos dados e transforma os dados.
        
        Args:
            X: DataFrame com os dados
            y: Series com os rótulos (opcional)
            
        Returns:
            tuple: (X_transformed, y) com dados transformados e rótulos
        """
        logger.info(f"Iniciando pré-processamento para dataset {self.dataset_type}...")
        
        start_time = performance_monitor.start_timer()
        
        try:
            # 1. Lidar com valores ausentes
            X = self._handle_missing_values(X, fit=True)
            
            # 2. Codificar variáveis categóricas
            X = self._encode_categorical(X, fit=True)
            
            # 3. Normalizar dados
            X = self._normalize_data(X, fit=True)
            
            # 4. Selecionar características
            if self.config.get('feature_selection', False) and y is not None:
                X = self._select_features(X, y, fit=True)
            
            # 5. Balancear classes (se necessário e se y for fornecido)
            if y is not None and self.config.get('balance_method', 'none') != 'none':
                X, y = self._balance_classes(X, y)
            
            self.is_fitted = True
            
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('preprocessor', 'fit_transform', 
                                          elapsed_time * 1000, 'ms', 
                                          {'rows': len(X), 'columns': X.shape[1]})
            
            logger.info(f"Pré-processamento concluído: {X.shape[1]} características, {len(X)} amostras")
            return X, y
        
        except Exception as e:
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('preprocessor', 'fit_transform', 
                                          elapsed_time * 1000, 'ms', 
                                          {'error': str(e)})
            logger.error(f"Erro no pré-processamento: {str(e)}")
            raise
    
    def transform(self, X):
        """
        Transforma novos dados usando os transformadores já ajustados.
        
        Args:
            X: DataFrame com os dados
            
        Returns:
            DataFrame: Dados transformados
        """
        if not self.is_fitted:
            raise ValueError("O pré-processador não foi treinado. Chame fit_transform primeiro.")
        
        logger.info(f"Transformando novos dados...")
        
        start_time = performance_monitor.start_timer()
        
        try:
            # 1. Lidar com valores ausentes
            X = self._handle_missing_values(X, fit=False)
            
            # 2. Codificar variáveis categóricas
            X = self._encode_categorical(X, fit=False)
            
            # 3. Normalizar dados
            X = self._normalize_data(X, fit=False)
            
            # 4. Selecionar características
            if self.config.get('feature_selection', False) and self.feature_selector is not None:
                X = self._select_features(X, fit=False)
            
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('preprocessor', 'transform', 
                                          elapsed_time * 1000, 'ms', 
                                          {'rows': len(X), 'columns': X.shape[1]})
            
            logger.info(f"Transformação concluída: {X.shape[1]} características, {len(X)} amostras")
            return X
        
        except Exception as e:
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('preprocessor', 'transform', 
                                          elapsed_time * 1000, 'ms', 
                                          {'error': str(e)})
            logger.error(f"Erro na transformação: {str(e)}")
            raise
    
    def _handle_missing_values(self, X, fit=True):
        """
        Lida com valores ausentes nos dados.
        
        Args:
            X: DataFrame com os dados
            fit: Se True, ajusta os imputers aos dados
            
        Returns:
            DataFrame: Dados sem valores ausentes
        """
        logger.debug("Tratando valores ausentes...")
        
        # Verificar se há valores ausentes
        missing_count = X.isnull().sum().sum()
        if missing_count == 0:
            logger.debug("Não há valores ausentes nos dados.")
            return X
        
        logger.debug(f"Encontrados {missing_count} valores ausentes.")
        
        # Estratégia de imputação
        strategy = self.config.get('handle_missing', 'mean')
        
        # Separar colunas numéricas e categóricas
        numeric_cols = X.select_dtypes(include=['number']).columns
        categorical_cols = X.select_dtypes(exclude=['number']).columns
        
        # Tratar colunas numéricas
        if len(numeric_cols) > 0:
            if fit:
                self.imputers['numeric'] = SimpleImputer(strategy=strategy)
                self.imputers['numeric'].fit(X[numeric_cols])
            
            if 'numeric' in self.imputers:
                X_numeric = pd.DataFrame(
                    self.imputers['numeric'].transform(X[numeric_cols]),
                    columns=numeric_cols,
                    index=X.index
                )
                X = X.copy()
                X[numeric_cols] = X_numeric
        
        # Tratar colunas categóricas
        if len(categorical_cols) > 0:
            if fit:
                self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
                self.imputers['categorical'].fit(X[categorical_cols])
            
            if 'categorical' in self.imputers:
                X_categorical = pd.DataFrame(
                    self.imputers['categorical'].transform(X[categorical_cols]),
                    columns=categorical_cols,
                    index=X.index
                )
                X = X.copy()
                X[categorical_cols] = X_categorical
        
        return X
    
    def _encode_categorical(self, X, fit=True):
        """
        Codifica variáveis categóricas.
        
        Args:
            X: DataFrame com os dados
            fit: Se True, ajusta os encoders aos dados
            
        Returns:
            DataFrame: Dados com variáveis categóricas codificadas
        """
        logger.debug("Codificando variáveis categóricas...")
        
        # Identificar colunas categóricas
        if self.dataset_type == 'nsl_kdd':
            categorical_cols = self.config.get('categorical_columns', [])
        else:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_cols:
            logger.debug("Não há variáveis categóricas para codificar.")
            return X
        
        logger.debug(f"Codificando {len(categorical_cols)} variáveis categóricas: {categorical_cols}")
        
        # Usar one-hot encoding
        X_encoded = X.copy()
        
        for col in categorical_cols:
            if col not in X.columns:
                logger.warning(f"Coluna {col} não encontrada nos dados.")
                continue
            
            # One-hot encoding
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=False)
            
            # Armazenar colunas para transformação futura
            if fit:
                self.encoders[col] = dummies.columns.tolist()
            
            # Adicionar colunas dummy
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            
            # Remover coluna original
            X_encoded = X_encoded.drop(col, axis=1)
        
        return X_encoded
    
    def _normalize_data(self, X, fit=True):
        """
        Normaliza os dados.
        
        Args:
            X: DataFrame com os dados
            fit: Se True, ajusta os scalers aos dados
            
        Returns:
            DataFrame: Dados normalizados
        """
        logger.debug("Normalizando dados...")
        
        # Selecionar método de normalização
        method = self.config.get('normalization', 'standard')
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Método de normalização desconhecido: {method}. Usando StandardScaler.")
            scaler = StandardScaler()
        
        # Selecionar apenas colunas numéricas
        numeric_cols = X.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            logger.warning("Não há colunas numéricas para normalizar.")
            return X
        
        # Ajustar e transformar
        if fit:
            self.scalers['main'] = scaler
            self.scalers['main'].fit(X[numeric_cols])
        
        if 'main' in self.scalers:
            X_scaled = X.copy()
            X_scaled[numeric_cols] = self.scalers['main'].transform(X[numeric_cols])
            return X_scaled
        
        return X
    
    def _select_features(self, X, y=None, fit=True):
        """
        Seleciona as características mais relevantes.
        
        Args:
            X: DataFrame com os dados
            y: Series com os rótulos (necessário se fit=True)
            fit: Se True, ajusta o seletor de características aos dados
            
        Returns:
            DataFrame: Dados com características selecionadas
        """
        logger.debug("Selecionando características...")
        
        # Verificar se a seleção de características está habilitada
        if not self.config.get('feature_selection', False):
            return X
        
        # Número de características a selecionar
        n_features = self.config.get('n_features', min(20, X.shape[1]))
        
        # Ajustar seletor
        if fit:
            if y is None:
                raise ValueError("Os rótulos (y) são necessários para ajustar o seletor de características.")
            
            self.feature_selector = SelectKBest(f_classif, k=n_features)
            self.feature_selector.fit(X, y)
            
            # Armazenar nomes das características selecionadas
            feature_mask = self.feature_selector.get_support()
            self.selected_features = X.columns[feature_mask].tolist()
            
            logger.debug(f"Características selecionadas: {self.selected_features}")
        
        # Transformar dados
        if self.feature_selector is not None:
            if fit:
                X_selected = pd.DataFrame(
                    self.feature_selector.transform(X),
                    columns=self.selected_features,
                    index=X.index
                )
            else:
                # Para novos dados, selecionar apenas as características já selecionadas
                missing_cols = set(self.selected_features) - set(X.columns)
                if missing_cols:
                    logger.warning(f"Colunas ausentes nos novos dados: {missing_cols}")
                    # Adicionar colunas ausentes com zeros
                    for col in missing_cols:
                        X[col] = 0
                
                X_selected = X[self.selected_features]
            
            return X_selected
        
        return X
    
    def _balance_classes(self, X, y):
        """
        Balanceia as classes usando técnicas de oversampling.
        
        Args:
            X: DataFrame com os dados
            y: Series com os rótulos
            
        Returns:
            tuple: (X_balanced, y_balanced) com dados balanceados
        """
        logger.debug("Balanceando classes...")
        
        # Verificar distribuição de classes
        class_counts = y.value_counts()
        logger.debug(f"Distribuição de classes original: {class_counts.to_dict()}")
        
        # Verificar se o balanceamento é necessário
        min_class = class_counts.min()
        max_class = class_counts.max()
        
        if min_class / max_class >= 0.2:  # Se a classe minoritária for pelo menos 20% da majoritária
            logger.debug("Classes já estão razoavelmente balanceadas. Pulando balanceamento.")
            return X, y
        
        # Selecionar método de balanceamento
        method = self.config.get('balance_method', 'smote')
        
        if method == 'smote':
            balancer = SMOTE(random_state=42)
        elif method == 'adasyn':
            balancer = ADASYN(random_state=42)
        elif method == 'smotetomek':
            balancer = SMOTETomek(random_state=42)
        else:
            logger.warning(f"Método de balanceamento desconhecido: {method}. Usando SMOTE.")
            balancer = SMOTE(random_state=42)
        
        # Aplicar balanceamento
        X_resampled, y_resampled = balancer.fit_resample(X, y)
        
        # Verificar nova distribuição
        new_class_counts = pd.Series(y_resampled).value_counts()
        logger.debug(f"Distribuição de classes após balanceamento: {new_class_counts.to_dict()}")
        
        return X_resampled, y_resampled
    
    def save(self, filepath):
        """
        Salva o pré-processador em um arquivo.
        
        Args:
            filepath: Caminho do arquivo
        """
        import joblib
        
        if not self.is_fitted:
            raise ValueError("O pré-processador não foi treinado. Não há nada para salvar.")
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Salvar o pré-processador
        joblib.dump(self, filepath)
        logger.info(f"Pré-processador salvo em {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Carrega um pré-processador de um arquivo.
        
        Args:
            filepath: Caminho do arquivo
            
        Returns:
            DataPreprocessor: Pré-processador carregado
        """
        import joblib
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo {filepath} não encontrado.")
        
        # Carregar o pré-processador
        preprocessor = joblib.load(filepath)
        logger.info(f"Pré-processador carregado de {filepath}")
        
        return preprocessor
