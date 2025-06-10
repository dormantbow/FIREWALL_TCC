"""
Módulo de pré-processamento para o firewall baseado em rede neural.
Implementa a classe DataPreprocessor para preparação dos dados.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Configurar logging
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Classe para pré-processamento de dados para o firewall neural.
    """
    
    def __init__(self, handle_missing=True, encode_categorical=True, normalize=True, feature_selection=True, balance_classes=False):
        """
        Inicializa o preprocessador.
        
        Args:
            handle_missing: Se True, trata valores ausentes
            encode_categorical: Se True, codifica variáveis categóricas
            normalize: Se True, normaliza dados numéricos
            feature_selection: Se True, seleciona características mais relevantes
            balance_classes: Se True, equilibra as classes usando SMOTE
        """
        self.handle_missing = handle_missing
        self.encode_categorical = encode_categorical
        self.normalize = normalize
        self.feature_selection = feature_selection
        self.balance_classes = balance_classes
        
        self.imputer = None
        self.scaler = None
        self.encoders = {}
        self.feature_selector = None
        self.selected_features = None
        self.n_features = 20  # Número padrão de características a selecionar
    
    def fit_transform(self, X, y=None):
        """
        Aplica todas as transformações de pré-processamento aos dados.
        
        Args:
            X: Dados de entrada (NumPy array ou DataFrame)
            y: Rótulos (opcional)
            
        Returns:
            array: Dados pré-processados
        """
        try:
            logger.info(f"Iniciando pré-processamento para dataset...")
            
            # Converter NumPy array para DataFrame se necessário
            is_numpy = isinstance(X, np.ndarray)
            if is_numpy:
                X = pd.DataFrame(X)
            
            # Aplicar pré-processamento
            if self.handle_missing:
                X = self._handle_missing_values(X, fit=True)
            
            if self.encode_categorical:
                X = self._encode_categorical(X, fit=True)
            
            if self.normalize:
                X = self._normalize_data(X, fit=True)
            
            if self.feature_selection and y is not None:
                X = self.select_features(X, y, fit=True)
            
            if self.balance_classes and y is not None:
                X, y = self._balance_classes(X, y)
            
            # Converter de volta para NumPy array se necessário
            if is_numpy:
                X = X.values
            
            logger.info(f"Pré-processamento concluído. Shape final: {X.shape}")
            return X
            
        except Exception as e:
            logger.error(f"Erro no pré-processamento: {str(e)}")
            raise
    
    def transform(self, X):
        """
        Aplica transformações pré-ajustadas aos dados.
        
        Args:
            X: Dados de entrada (NumPy array ou DataFrame)
            
        Returns:
            array: Dados pré-processados
        """
        try:
            # Converter NumPy array para DataFrame se necessário
            is_numpy = isinstance(X, np.ndarray)
            if is_numpy:
                X = pd.DataFrame(X)
            
            # Aplicar pré-processamento
            if self.handle_missing:
                X = self._handle_missing_values(X, fit=False)
            
            if self.encode_categorical:
                X = self._encode_categorical(X, fit=False)
            
            if self.normalize:
                X = self._normalize_data(X, fit=False)
            
            if self.feature_selection:
                X = self.select_features(X, fit=False)
            
            # Converter de volta para NumPy array se necessário
            if is_numpy:
                X = X.values
            
            return X
            
        except Exception as e:
            logger.error(f"Erro na transformação: {str(e)}")
            raise
    
    def _handle_missing_values(self, X, fit=False):
        """
        Trata valores ausentes nos dados.
        
        Args:
            X: DataFrame com os dados
            fit: Se True, ajusta o imputer aos dados
            
        Returns:
            DataFrame: Dados sem valores ausentes
        """
        if hasattr(X, 'isnull'):
            # Se X for um DataFrame pandas
            missing_count = X.isnull().sum().sum()
        else:
            # Se X for um NumPy array
            missing_count = np.isnan(X).sum()
            
        if missing_count > 0:
            logger.debug(f"Tratando {missing_count} valores ausentes...")
            
            if fit:
                self.imputer = SimpleImputer(strategy='mean')
                self.imputer.fit(X)
            
            if self.imputer is not None:
                X_imputed = self.imputer.transform(X)
                
                # Manter índice e colunas se for DataFrame
                if hasattr(X, 'index') and hasattr(X, 'columns'):
                    X = pd.DataFrame(X_imputed, index=X.index, columns=X.columns)
                else:
                    X = X_imputed
        
        return X
    
    def _encode_categorical(self, X, fit=False):
        """
        Codifica variáveis categóricas.
        
        Args:
            X: DataFrame com os dados
            fit: Se True, ajusta os encoders aos dados
            
        Returns:
            DataFrame: Dados com variáveis categóricas codificadas
        """
        # Identificar colunas categóricas
        if hasattr(X, 'select_dtypes'):
            # Se X for um DataFrame pandas
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        else:
            # Se X for um NumPy array, assumir que não há colunas categóricas
            categorical_cols = []
        
        if len(categorical_cols) > 0:
            logger.debug(f"Codificando {len(categorical_cols)} variáveis categóricas...")
            
            X_encoded = X.copy()
            
            for col in categorical_cols:
                if fit:
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoder.fit(X[[col]])
                    self.encoders[col] = encoder
                
                if col in self.encoders:
                    # Codificar a coluna
                    encoded = self.encoders[col].transform(X[[col]])
                    
                    # Criar nomes para as novas colunas
                    feature_names = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0]]
                    
                    # Adicionar colunas codificadas
                    for i, name in enumerate(feature_names):
                        X_encoded[name] = encoded[:, i]
                    
                    # Remover coluna original
                    X_encoded = X_encoded.drop(col, axis=1)
            
            return X_encoded
        
        return X
    
    def _normalize_data(self, X, fit=False):
        """
        Normaliza dados numéricos.
        
        Args:
            X: DataFrame com os dados
            fit: Se True, ajusta o scaler aos dados
            
        Returns:
            DataFrame: Dados normalizados
        """
        # Identificar colunas numéricas
        if hasattr(X, 'select_dtypes'):
            # Se X for um DataFrame pandas
            numeric_cols = X.select_dtypes(include=['number']).columns
        else:
            # Se X for um NumPy array, assumir que todas as colunas são numéricas
            numeric_cols = np.arange(X.shape[1])
        
        if len(numeric_cols) > 0:
            logger.debug(f"Normalizando {len(numeric_cols)} variáveis numéricas...")
            
            if fit:
                self.scaler = StandardScaler()
                self.scaler.fit(X[numeric_cols])
            
            if self.scaler is not None:
                # Normalizar apenas colunas numéricas
                X_numeric = X[numeric_cols].copy()
                X_numeric_scaled = self.scaler.transform(X_numeric)
                
                # Substituir valores originais pelos normalizados
                X_scaled = X.copy()
                X_scaled[numeric_cols] = X_numeric_scaled
                
                return X_scaled
        
        return X
    
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
        
        # Ajustar seletor
        if fit:
            if y is None:
                raise ValueError("Os rótulos (y) são necessários para ajustar o seletor de características.")
            
            self.feature_selector = SelectKBest(f_classif, k=self.n_features)
            self.feature_selector.fit(X, y)
            #Verificando colunas existentes apos carregar dataset
            #print("Colunas disponíveis no dataset:", X.columns.tolist())
            X = pd.get_dummies(X, columns=['protocol_type', 'service', 'flag'])

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
        if self.feature_selector is not None:
            if fit:
                transformed = self.feature_selector.transform(X)
                
                # Verifica se X tem índice
                if hasattr(X, 'index'):
                    index = X.index
                else:
                    index = None
                
                X_selected = pd.DataFrame(
                    transformed,
                    columns=self.selected_features,
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
        Equilibra as classes usando SMOTE.
        
        Args:
            X: DataFrame com os dados
            y: Series com os rótulos
            
        Returns:
            tuple: (X_balanceado, y_balanceado)
        """
        if hasattr(y, 'value_counts'):
            # Se y for uma Series pandas
            class_counts = y.value_counts()
        else:
            # Se y for um NumPy array
            unique_classes, class_counts = np.unique(y, return_counts=True)
            class_counts = pd.Series(class_counts, index=unique_classes)
        
        # Verificar se há desbalanceamento
        min_count = class_counts.min()
        max_count = class_counts.max()
        
        if max_count / min_count > 1.5:  # Desbalanceamento significativo
            logger.debug(f"Equilibrando classes. Proporção atual: {max_count/min_count:.2f}")
            
            try:
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
                logger.debug(f"Classes equilibradas. Shape original: {X.shape}, novo shape: {X_resampled.shape}")
                return X_resampled, y_resampled
            except Exception as e:
                logger.warning(f"Erro ao aplicar SMOTE: {str(e)}. Usando dados originais.")
        
        return X, y
