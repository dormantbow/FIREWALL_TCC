"""
Módulo para implementação de modelos de rede neural para detecção de intrusão.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from ..utils.config import MODEL_CONFIG, MODELS_DIR
from ..utils.logging_utils import performance_monitor

logger = logging.getLogger(__name__)

class AutoencoderModel:
    """
    Implementação do modelo de autoencoder para detecção de anomalias.
    """
    
    def __init__(self, input_dim, config=None):
        """
        Inicializa o modelo de autoencoder.
        
        Args:
            input_dim: Dimensão de entrada (número de características)
            config: Configuração do modelo (opcional)
        """
        self.input_dim = input_dim
        self.config = config or MODEL_CONFIG['autoencoder']
        self.model = None
        self.threshold = None
        self.history = None
    
    def build(self):
        """
        Constrói a arquitetura do modelo de autoencoder.
        
        Returns:
            Model: Modelo de autoencoder
        """
        logger.info(f"Construindo modelo de autoencoder com dimensão de entrada {self.input_dim}...")
        
        # Definir a arquitetura
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder
        x = input_layer
        for i, layer_config in enumerate(self.config['architecture']):
            if i >= len(self.config['architecture']) // 2:
                break
            
            x = Dense(layer_config['units'], activation=layer_config['activation'])(x)
            x = BatchNormalization()(x)
        
        # Bottleneck (camada do meio)
        bottleneck_idx = len(self.config['architecture']) // 2
        bottleneck_config = self.config['architecture'][bottleneck_idx]
        bottleneck = Dense(bottleneck_config['units'], activation=bottleneck_config['activation'])(x)
        
        # Decoder
        x = bottleneck
        for i, layer_config in enumerate(self.config['architecture']):
            if i <= len(self.config['architecture']) // 2:
                continue
            
            x = Dense(layer_config['units'], activation=layer_config['activation'])(x)
            x = BatchNormalization()(x)
        
        # Camada de saída
        output_layer = Dense(self.input_dim, activation='linear')(x)
        
        # Criar modelo
        self.model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compilar modelo
        self.model.compile(
            optimizer=self.config['optimizer'],
            loss=self.config['loss']
        )
        
        logger.info(f"Modelo de autoencoder construído: {self.model.summary()}")
        return self.model
    
    def train(self, X_train, X_val=None, model_path=None):
        """
        Treina o modelo de autoencoder.
        
        Args:
            X_train: Dados de treinamento
            X_val: Dados de validação (opcional)
            model_path: Caminho para salvar o modelo (opcional)
            
        Returns:
            history: Histórico de treinamento
        """
        if self.model is None:
            self.build()
        
        logger.info("Iniciando treinamento do autoencoder...")
        
        # Preparar dados de validação
        if X_val is None and self.config['validation_split'] > 0:
            val_size = int(len(X_train) * self.config['validation_split'])
            indices = np.random.permutation(len(X_train))
            X_val = X_train[indices[:val_size]]
            X_train = X_train[indices[val_size:]]
        
        # Configurar callbacks
        callbacks = []
        
        if self.config['early_stopping']:
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=self.config['patience'],
                    restore_best_weights=True
                )
            )
        
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    model_path,
                    monitor='val_loss' if X_val is not None else 'loss',
                    save_best_only=True
                )
            )
        
        # Iniciar timer
        start_time = performance_monitor.start_timer()
        
        try:
            # Treinar modelo
            if X_val is not None:
                self.history = self.model.fit(
                    X_train, X_train,  # Autoencoder: input = output
                    epochs=self.config['epochs'],
                    batch_size=self.config['batch_size'],
                    validation_data=(X_val, X_val),
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                self.history = self.model.fit(
                    X_train, X_train,  # Autoencoder: input = output
                    epochs=self.config['epochs'],
                    batch_size=self.config['batch_size'],
                    validation_split=self.config['validation_split'],
                    callbacks=callbacks,
                    verbose=1
                )
            
            # Calcular threshold de anomalia
            self._calculate_threshold(X_train)
            
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('autoencoder', 'train', 
                                          elapsed_time * 1000, 'ms', 
                                          {'epochs': len(self.history.history['loss']),
                                           'final_loss': self.history.history['loss'][-1]})
            
            logger.info(f"Treinamento do autoencoder concluído em {elapsed_time:.2f}s")
            return self.history
        
        except Exception as e:
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('autoencoder', 'train', 
                                          elapsed_time * 1000, 'ms', 
                                          {'error': str(e)})
            logger.error(f"Erro no treinamento do autoencoder: {str(e)}")
            raise
    
    def _calculate_threshold(self, X_normal):
        """
        Calcula o threshold de anomalia baseado nos erros de reconstrução.
        
        Args:
            X_normal: Dados normais para calcular o threshold
        """
        logger.info("Calculando threshold de anomalia...")
        
        # Reconstruir os dados normais
        reconstructions = self.model.predict(X_normal)
        
        # Calcular erro de reconstrução (MSE)
        mse = np.mean(np.power(X_normal - reconstructions, 2), axis=1)
        
        # Definir threshold como percentil dos erros
        percentile = self.config['anomaly_threshold_percentile']
        self.threshold = np.percentile(mse, percentile)
        
        logger.info(f"Threshold de anomalia calculado: {self.threshold:.6f} (percentil {percentile})")
    
    def detect_anomalies(self, X, return_scores=False):
        """
        Detecta anomalias nos dados.
        
        Args:
            X: Dados para detecção
            return_scores: Se True, retorna também os scores de anomalia
            
        Returns:
            anomalies: Array booleano indicando anomalias (True) ou normal (False)
            scores: (opcional) Scores de anomalia (erro de reconstrução)
        """
        if self.model is None:
            raise ValueError("O modelo não foi treinado. Chame train() primeiro.")
        
        if self.threshold is None:
            raise ValueError("O threshold de anomalia não foi calculado.")
        
        logger.info(f"Detectando anomalias em {len(X)} amostras...")
        
        # Iniciar timer
        start_time = performance_monitor.start_timer()
        
        try:
            # Reconstruir os dados
            reconstructions = self.model.predict(X)
            
            # Calcular erro de reconstrução (MSE)
            mse = np.mean(np.power(X - reconstructions, 2), axis=1)
            
            # Classificar como anomalia se erro > threshold
            anomalies = mse > self.threshold
            
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('autoencoder', 'detect_anomalies', 
                                          elapsed_time * 1000, 'ms', 
                                          {'samples': len(X), 'anomalies': np.sum(anomalies)})
            
            logger.info(f"Detecção concluída: {np.sum(anomalies)} anomalias encontradas em {len(X)} amostras")
            
            if return_scores:
                return anomalies, mse
            return anomalies
        
        except Exception as e:
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('autoencoder', 'detect_anomalies', 
                                          elapsed_time * 1000, 'ms', 
                                          {'error': str(e)})
            logger.error(f"Erro na detecção de anomalias: {str(e)}")
            raise
    
    def save(self, model_path=None, include_threshold=True):
        """
        Salva o modelo treinado.
        
        Args:
            model_path: Caminho para salvar o modelo
            include_threshold: Se True, salva também o threshold de anomalia
        """
        if self.model is None:
            raise ValueError("O modelo não foi treinado. Não há nada para salvar.")
        
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, 'autoencoder')
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Salvar modelo
        self.model.save(model_path)
        logger.info(f"Modelo salvo em {model_path}")
        
        # Salvar threshold
        if include_threshold and self.threshold is not None:
            threshold_path = os.path.join(os.path.dirname(model_path), 'threshold.npy')
            np.save(threshold_path, self.threshold)
            logger.info(f"Threshold salvo em {threshold_path}")
    
    @classmethod
    def load(cls, model_path, threshold_path=None):
        """
        Carrega um modelo treinado.
        
        Args:
            model_path: Caminho do modelo
            threshold_path: Caminho do threshold (opcional)
            
        Returns:
            AutoencoderModel: Modelo carregado
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
        
        # Carregar modelo
        model = tf.keras.models.load_model(model_path)
        
        # Criar instância
        autoencoder = cls(input_dim=model.input_shape[1])
        autoencoder.model = model
        
        # Carregar threshold
        if threshold_path is None:
            threshold_path = os.path.join(os.path.dirname(model_path), 'threshold.npy')
        
        if os.path.exists(threshold_path):
            autoencoder.threshold = np.load(threshold_path)
            logger.info(f"Threshold carregado de {threshold_path}: {autoencoder.threshold:.6f}")
        
        logger.info(f"Modelo carregado de {model_path}")
        return autoencoder


class CNNLSTMModel:
    """
    Implementação do modelo CNN+LSTM para classificação de ataques.
    """
    
    def __init__(self, input_dim, num_classes, config=None):
        """
        Inicializa o modelo CNN+LSTM.
        
        Args:
            input_dim: Dimensão de entrada (número de características)
            num_classes: Número de classes
            config: Configuração do modelo (opcional)
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.config = config or MODEL_CONFIG['cnn_lstm']
        self.model = None
        self.history = None
    
    def build(self):
        """
        Constrói a arquitetura do modelo CNN+LSTM.
        
        Returns:
            Model: Modelo CNN+LSTM
        """
        logger.info(f"Construindo modelo CNN+LSTM com dimensão de entrada {self.input_dim} e {self.num_classes} classes...")
        
        # Definir a arquitetura
        sequence_length = self.config['sequence_length']
        
        # Criar modelo sequencial
        self.model = Sequential()
        
        # Camadas CNN
        for i, layer_config in enumerate(self.config['cnn_layers']):
            if i == 0:
                self.model.add(Conv1D(
                    filters=layer_config['filters'],
                    kernel_size=layer_config['kernel_size'],
                    activation=layer_config['activation'],
                    input_shape=(sequence_length, self.input_dim)
                ))
            else:
                self.model.add(Conv1D(
                    filters=layer_config['filters'],
                    kernel_size=layer_config['kernel_size'],
                    activation=layer_config['activation']
                ))
            
            self.model.add(MaxPooling1D(pool_size=2))
        
        # Camada LSTM
        self.model.add(LSTM(self.config['lstm_units']))
        
        # Camadas densas
        for layer_config in self.config['dense_layers']:
            self.model.add(Dense(layer_config['units'], activation=layer_config['activation']))
            if 'dropout' in layer_config and layer_config['dropout'] > 0:
                self.model.add(Dropout(layer_config['dropout']))
        
        # Camada de saída
        if self.num_classes == 2:
            self.model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            self.model.add(Dense(self.num_classes, activation='softmax'))
            loss = 'categorical_crossentropy'
        
        # Compilar modelo
        self.model.compile(
            optimizer=self.config['optimizer'],
            loss=loss,
            metrics=self.config['metrics']
        )
        
        logger.info(f"Modelo CNN+LSTM construído: {self.model.summary()}")
        return self.model
    
    def prepare_sequence_data(self, X, y=None):
        """
        Prepara os dados em formato de sequência para o modelo CNN+LSTM.
        
        Args:
            X: Dados de entrada
            y: Rótulos (opcional)
            
        Returns:
            X_seq: Dados em formato de sequência
            y: Rótulos (se fornecidos)
        """
        logger.info(f"Preparando dados em formato de sequência (tamanho {self.config['sequence_length']})...")
        
        sequence_length = self.config['sequence_length']
        
        # Verificar se há dados suficientes
        if len(X) < sequence_length:
            raise ValueError(f"Número insuficiente de amostras ({len(X)}) para criar sequências de tamanho {sequence_length}")
        
        # Criar sequências
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(len(X) - sequence_length + 1):
            X_seq.append(X[i:i+sequence_length])
            if y is not None:
                y_seq.append(y[i+sequence_length-1])  # Usar o rótulo da última amostra da sequência
        
        X_seq = np.array(X_seq)
        
        if y is not None:
            y_seq = np.array(y_seq)
            
            # Converter para one-hot encoding se necessário
            if self.num_classes > 2:
                y_seq = to_categorical(y_seq, num_classes=self.num_classes)
            
            logger.info(f"Dados preparados: {X_seq.shape} sequências, {y_seq.shape} rótulos")
            return
(Content truncated due to size limit. Use line ranges to read in chunks)