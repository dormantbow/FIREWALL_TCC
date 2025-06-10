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

from utils.config import PREPROCESSING_CONFIG
from utils.logging_utils import performance_monitor

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
        self.selected_features = None
        self.feature_selection = self.config.get('feature_selection', False)
        self.n_features = self.config.get('n_features', 20)
        
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
            # Converter NumPy array para DataFrame se necessário
            is_numpy = isinstance(X, np.ndarray)
            if is_numpy:
                X = pd.DataFrame(X)
                
            # 1. Lidar com valores ausentes
            X = self._handle_missing_values(X, fit=True)
            
            # 2. Codificar variáveis categóricas
            X = self._encode_categorical(X, fit=True)
            
            # 3. Normalizar dados
            X = self._normalize_data(X, fit=True)
            
            # 4. Selecionar características
            if self.config.get('feature_selection', False) and y is not None:
                X = self.select_features(X, y, fit=True)
            
            # 5. Balancear classes (se necessário e se y for fornecido)
            if y is not None and self.config.get('balance_method', 'none') != 'none':
                X, y = self._balance_classes(X, y)
            
            self.is_fitted = True
            
            # Converter de volta para NumPy array se necessário
            if is_numpy:
                X = X.values
                
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
            # Converter NumPy array para DataFrame se necessário
            is_numpy = isinstance(X, np.ndarray)
            if is_numpy:
                X = pd.DataFrame(X)
                
            # 1. Lidar com valores ausentes
            X = self._handle_missing_values(X, fit=False)
            
            # 2. Codificar variáveis categóricas
            X = self._encode_categorical(X, fit=False)
            
            # 3. Normalizar dados
            X = self._normalize_data(X, fit=False)
            
            # 4. Selecionar características
            if self.config.get('feature_selection', False) and self.feature_selector is not None:
                X = self.select_features(X, fit=False)
            
            # Converter de volta para NumPy array se necessário
            if is_numpy:
                X = X.values
                
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
        if hasattr(X, 'isnull'):
            # Se X for um DataFrame pandas
            missing_count = X.isnull().sum().sum()
        else:
            # Se X for um NumPy array
            missing_count = np.isnan(X).sum()

        if missing_count == 0:
            logger.debug("Não há valores ausentes nos dados.")
            return X
        
        logger.debug(f"Encontrados {missing_count} valores ausentes.")
        
        # Estratégia de imputação
        strategy = self.config.get('handle_missing', 'mean')
        
        # Separar colunas numéricas e categóricas
        if hasattr(X, 'select_dtypes'):
            numeric_cols = X.select_dtypes(include=['number']).columns
            categorical_cols = X.select_dtypes(exclude=['number']).columns
        else:
            numeric_cols = np.arange(X.shape[1])
            categorical_cols = []
        
        # Tratar colunas numéricas
        if len(numeric_cols) > 0:
            if fit:
                self.imputers['numeric'] = SimpleImputer(strategy=strategy)
                self.imputers['numeric'].fit(X[numeric_cols])
            
            if 'numeric' in self.imputers:
                X_numeric = pd.DataFrame(
                    self.imputers['numeric'].transform(X[numeric_cols]),
                    columns=numeric_cols,
                    index=X.index if hasattr(X, 'index') else None
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
                    index=X.index if hasattr(X, 'index') else None
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
            if hasattr(X, 'select_dtypes'):
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            else:
                categorical_cols = []
        
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
        if hasattr(X, 'select_dtypes'):
            # Se X for um DataFrame pandas
            numeric_cols = X.select_dtypes(include=['number']).columns
        else:
            # Se X for um NumPy array
            numeric_cols = np.arange(X.shape[1])  # Todos os índices de coluna

        
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
    
    # CORRIGIDO: Método select_features agora está dentro da classe DataPreprocessor
    def select_features(self, X, y=None, fit=True):
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
        if not self.feature_selection:
            return X
        
        # Número de características a selecionar
        n_features = self.n_features
        
        # Ajustar seletor
        if fit:
            if y is None:
                raise ValueError("Os rótulos (y) são necessários para ajustar o seletor de características.")
            
            self.feature_selector = SelectKBest(f_classif, k=n_features)
            self.feature_selector.fit(X, y)
            #columns_to_encode = ['protocol_type', 'service', 'flag']
            #missing_cols = [col for col in columns_to_encode if col not in X.columns]

            #if missing_cols:
            #    raise ValueError(f"Colunas categóricas ausentes no dataset: {missing_cols}")
            #else:
            #    X = pd.get_dummies(X, columns=columns_to_encode)
            #Verificando colunas existentes apos carregar dataset
            #print("Colunas disponíveis no dataset:", X.columns.tolist())
            #X = pd.get_dummies(X, columns=['protocol_type', 'service', 'flag'])
            
            # Armazenar nomes das características selecionadas
            feature_mask = self.feature_selector.get_support()
            
            if hasattr(X, 'columns'):
                # Se X for um DataFrame pandas
                self.selected_features = X.columns[feature_mask].tolist()
            else:
                # Se X for um NumPy array
                self.selected_features = np.where(feature_mask)[0].tolist()
            
            logger.debug(f"Características selecionadas: {self.selected_features}")
        
        # Transformar dados
        if hasattr(self, 'feature_selector') and self.feature_selector is not None:
            if fit:
                transformed = self.feature_selector.transform(X)
                
                # Verifica se X tem índice
                if hasattr(X, 'index'):
                    index = X.index
                else:
                    index = None
                
                # Criar DataFrame com as características selecionadas
                if hasattr(self, 'selected_features'):
                    columns = self.selected_features
                else:
                    columns = [f'feature_{i}' for i in range(transformed.shape[1])]
                
                X_selected = pd.DataFrame(
                    transformed,
                    columns=columns,
                    index=index
                )
            else:
                # Para novos dados, selecionar apenas as características já selecionadas
                if hasattr(X, 'columns'):
                    # Se X for um DataFrame pandas
                    missing_cols = set(self.selected_features) - set(X.columns)
                    if missing_cols:
                        logger.warning(f"Colunas ausentes nos novos dados: {missing_cols}")
                        for col in missing_cols:
                            X[col] = 0
                    X_selected = X[self.selected_features]
                else:
                    # Se X for um NumPy array
                    X_selected = self.feature_selector.transform(X)
                    X_selected = pd.DataFrame(X_selected)
            
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
        if hasattr(y, 'value_counts'):
            # Se y for uma Series pandas
            class_counts = y.value_counts()
        else:
            # Se y for um NumPy array
            unique_classes, class_counts = np.unique(y, return_counts=True)
            class_counts = pd.Series(class_counts, index=unique_classes)

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
        try:
            X_resampled, y_resampled = balancer.fit_resample(X, y)
            
            # Verificar nova distribuição
            if hasattr(y_resampled, 'value_counts'):
                new_counts = y_resampled.value_counts()
            else:
                unique_classes, new_counts_values = np.unique(y_resampled, return_counts=True)
                new_counts = pd.Series(new_counts_values, index=unique_classes)
            
            logger.debug(f"Distribuição de classes após balanceamento: {new_counts.to_dict()}")
            logger.debug(f"Dados balanceados: {X_resampled.shape[0]} amostras (original: {X.shape[0]})")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"Erro no balanceamento de classes: {str(e)}")
            logger.warning("Usando dados originais sem balanceamento.")
            return X, y
