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
from tensorflow.keras.models import load_model


from tensorflow.keras.losses import MeanSquaredError

from utils.config import MODEL_CONFIG, MODELS_DIR
from utils.logging_utils import performance_monitor


logger = logging.getLogger(__name__)


class AutoencoderModel:
    """
    Implementação do modelo de autoencoder para detecção de anomalias.
    """
    
    def __init__(self, input_dim, config=None, input_shape=None):
        self.input_dim = input_dim
        self.config = config or MODEL_CONFIG['autoencoder']
        self.model = None
        self.threshold = None
        self.history = None
        self.input_shape = input_shape
    
    def predict(self, X):
        """Faz predição usando o modelo do autoencoder."""
        reconstructions = self.model.predict(X)
        reconstruction_errors = tf.reduce_mean(tf.square(reconstructions - X), axis=1).numpy()
        anomalies = reconstruction_errors > self.threshold
        return reconstructions, reconstruction_errors, anomalies
    def build(self):
        logger.info(f"Construindo modelo de autoencoder com dimensão de entrada {self.input_dim}...")
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder
        x = input_layer
        for i, layer_config in enumerate(self.config['architecture']):
            if i >= len(self.config['architecture']) // 2:
                break
            x = Dense(layer_config['units'], activation=layer_config['activation'])(x)
            x = BatchNormalization()(x)
        
        # Bottleneck
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
        
        output_layer = Dense(self.input_dim, activation='linear')(x)
        
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer=self.config['optimizer'], loss=self.config['loss'])
        
        logger.info(f"Modelo de autoencoder construído: {self.model.summary()}")
        return self.model

    def train(self, X_train, X_val=None, model_path=None, epochs=10, batch_size=32, validation_split=0.2):
        import numpy as np
        if self.model is None:
            self.build()

        logger.info("Iniciando treinamento do autoencoder...")

        if X_val is None and validation_split > 0:
            val_size = int(len(X_train) * validation_split)
            indices = np.random.permutation(len(X_train))
            X_val = X_train.iloc[indices[:val_size]]
            X_train = X_train.iloc[indices[val_size:]]


        callbacks = []
        if self.config['early_stopping']:
            callbacks.append(EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config['patience'],
                restore_best_weights=True
            ))
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            callbacks.append(ModelCheckpoint(
                model_path,
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True
            ))

        start_time = performance_monitor.start_timer()

        try:
            if X_val is not None:
                self.history = self.model.fit(
                    X_train, X_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, X_val),
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                self.history = self.model.fit(
                    X_train, X_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=callbacks,
                    verbose=1
                )
            
            self.calculate_threshold(X_train)
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('autoencoder', 'train', elapsed_time * 1000, 'ms', {
                'epochs': len(self.history.history['loss']),
                'final_loss': self.history.history['loss'][-1]
            })
            logger.info(f"Treinamento concluído em {elapsed_time:.2f}s")
            return self.history

        except Exception as e:
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('autoencoder', 'train', elapsed_time * 1000, 'ms', {'error': str(e)})
            logger.error(f"Erro no treinamento do autoencoder: {str(e)}")
            raise

    def calculate_threshold(self, X_normal):
        logger.info("Calculando threshold de anomalia...")
        reconstructions = self.model.predict(X_normal)
        mse = np.mean(np.power(X_normal - reconstructions, 2), axis=1)
        percentile = self.config['anomaly_threshold_percentile']
        self.threshold = np.percentile(mse, percentile)
        logger.info(f"Threshold de anomalia calculado: {self.threshold:.6f} (percentil {percentile})")

    def detect_anomalies(self, X, return_scores=False):
        if self.model is None:
            raise ValueError("O modelo não foi treinado.")
        if self.threshold is None:
            raise ValueError("Threshold não calculado.")

        logger.info(f"Detectando anomalias em {len(X)} amostras...")
        start_time = performance_monitor.start_timer()

        try:
            reconstructions = self.model.predict(X)
            mse = np.mean(np.power(X - reconstructions, 2), axis=1)
            anomalies = mse > self.threshold

            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('autoencoder', 'detect_anomalies', elapsed_time * 1000, 'ms', {
                'samples': len(X), 'anomalies': np.sum(anomalies)
            })
            logger.info(f"Detecção concluída: {np.sum(anomalies)} anomalias")
            if return_scores:
                return anomalies, mse
            return anomalies

        except Exception as e:
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('autoencoder', 'detect_anomalies', elapsed_time * 1000, 'ms', {'error': str(e)})
            logger.error(f"Erro na detecção de anomalias: {str(e)}")
            raise

    def save(self, model_path=None, include_threshold=True):
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, 'autoencoder')
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        logger.info(f"Modelo salvo em {model_path}")

        if include_threshold and self.threshold is not None:
            threshold_path = os.path.join(os.path.dirname(model_path), 'threshold.npy')
            np.save(threshold_path, self.threshold)
            logger.info(f"Threshold salvo em {threshold_path}")

    @classmethod
    def load(cls, model_path, threshold_path=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}")

        # Adicione o parâmetro `custom_objects` com 'mse'
        model = tf.keras.models.load_model(model_path, custom_objects={'mse': MeanSquaredError()})
        
        autoencoder = cls(input_dim=model.input_shape[1])
        autoencoder.model = model

        if threshold_path is None:
            threshold_path = os.path.join(os.path.dirname(model_path), 'threshold.npy')

        if os.path.exists(threshold_path):
            autoencoder.threshold = np.load(threshold_path)
            logger.info(f"Threshold carregado de {threshold_path}: {autoencoder.threshold:.6f}")
        
        logger.info(f"Modelo carregado de {model_path}")
        return autoencoder

    #@classmethod
    #def load(cls, model_path, threshold_path=None):
    #    if not os.path.exists(model_path):
    #        raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
    #
    #    model = tf.keras.models.load_model(model_path)
    #    autoencoder = cls(input_dim=model.input_shape[1])
    #    autoencoder.model = model
    #
    #    if threshold_path is None:
    #        threshold_path = os.path.join(os.path.dirname(model_path), 'threshold.npy')

    #    if os.path.exists(threshold_path):
    #        autoencoder.threshold = np.load(threshold_path)
    #        logger.info(f"Threshold carregado de {threshold_path}: {autoencoder.threshold:.6f}")
        
    #    logger.info(f"Modelo carregado de {model_path}")
    #    return autoencoder


class CNNLSTMModel:
    """
    Implementação do modelo CNN+LSTM para classificação de ataques.
    """
    
    def __init__(self, input_dim, num_classes, config=None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.config = config or MODEL_CONFIG['cnn_lstm']
        self.model = None
        self.history = None
    
    def build(self):
        logger.info(f"Construindo modelo CNN+LSTM com dimensão de entrada {self.input_dim} e {self.num_classes} classes...")
        
        sequence_length = self.config['sequence_length']
        self.model = Sequential()

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

        self.model.add(LSTM(self.config['lstm_units']))

        for layer_config in self.config['dense_layers']:
            self.model.add(Dense(layer_config['units'], activation=layer_config['activation']))
            if 'dropout' in layer_config and layer_config['dropout'] > 0:
                self.model.add(Dropout(layer_config['dropout']))

        if self.num_classes == 2:
            self.model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            self.model.add(Dense(self.num_classes, activation='softmax'))
            loss = 'categorical_crossentropy'

        self.model.compile(
            optimizer=self.config['optimizer'],
            loss=loss,
            metrics=self.config['metrics']
        )

        logger.info(f"Modelo CNN+LSTM construído: {self.model.summary()}")
        return self.model
    def train(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2):
        import numpy as np
        return self.model.fit(X_train, y_train, 
                              epochs=epochs, 
                              batch_size=batch_size, 
                              validation_split=validation_split)

    def evaluate(self, X_test, y_test):
        
        
        X_test = X_test.astype('float32')
        y_test = to_categorical(y_test, num_classes=5) 
        #classifier_metrics = self.classifier.evaluate(X_test_seq, y_test_seq)
        return self.model.evaluate(X_test, y_test)
    def prepare_sequence_data(self, X, y=None):
        logger.info(f"Preparando dados em formato de sequência (tamanho {self.config['sequence_length']})...")

        sequence_length = self.config['sequence_length']
        
        if len(X) < sequence_length:
            raise ValueError(f"Número insuficiente de amostras ({len(X)}) para criar sequências de tamanho {sequence_length}")

        X_seq = []
        y_seq = [] if y is not None else None

        for i in range(len(X) - sequence_length + 1):
            X_seq.append(X[i:i+sequence_length])
            if y is not None:
                y_seq.append(y[i+sequence_length-1])

        X_seq = np.array(X_seq)
        
        if y is not None:
            y_seq = np.array(y_seq)
            if self.num_classes > 2:
                y_seq = to_categorical(y_seq, num_classes=self.num_classes)
            logger.info(f"Dados preparados: {X_seq.shape} sequências, {y_seq.shape} rótulos")
            return X_seq, y_seq
        
        logger.info(f"Dados preparados: {X_seq.shape} sequências")
        return X_seq
    
    @classmethod
    def load(cls, model_path, config=None):
        """
        Carrega um modelo CNN+LSTM salvo em disco.
        
        Args:
            model_path (str): Caminho para o arquivo .h5
            config (dict): Configuração opcional do modelo (usado se necessário reconstruir algo)

        Returns:
            CNNLSTMModel: Instância do modelo carregado
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        # Carrega o modelo diretamente
        model = load_model(model_path)

        # Infere dimensões a partir do modelo carregado
        input_shape = model.input_shape  # (None, sequence_length, input_dim)
        sequence_length, input_dim = input_shape[1], input_shape[2]
        output_shape = model.output_shape  # (None, num_classes) ou (None, 1)
        num_classes = output_shape[1] if isinstance(output_shape, tuple) and len(output_shape) > 1 else 2

        instance = cls(input_dim=input_dim, num_classes=num_classes, config=config)
        instance.model = model

        logger.info(f"Modelo CNN+LSTM carregado de {model_path}")
        return instance
