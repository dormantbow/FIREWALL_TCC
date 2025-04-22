# Arquitetura da Rede Neural para Detecção de Intrusão

## 1. Visão Geral

A arquitetura da rede neural é o componente central do firewall baseado em rede neural para prevenção de ataques zero-day. Este documento detalha a estrutura, funcionamento e implementação dos modelos de rede neural utilizados para detecção de intrusão e identificação de ataques zero-day.

## 2. Objetivos da Arquitetura Neural

- Detectar eficientemente ataques conhecidos presentes nos datasets NSL-KDD e CICIDS-2018
- Identificar anomalias que podem representar ataques zero-day
- Minimizar falsos positivos e falsos negativos
- Operar com baixa latência para detecção em tempo real
- Fornecer explicabilidade para as decisões tomadas

## 3. Abordagem Híbrida

A arquitetura proposta utiliza uma abordagem híbrida, combinando:

1. **Detecção de Anomalias**: Modelo não supervisionado para identificar comportamentos que desviam do padrão normal
2. **Classificação de Ataques**: Modelo supervisionado para classificar tipos específicos de ataques
3. **Sistema de Decisão**: Mecanismo para combinar os resultados dos modelos anteriores

```
                      +-------------------+
                      | Autoencoder       |
                      | (Anomalia)        |
                      +-------------------+
                                |
Input -> Pré-processamento ->   +   -> Sistema de Decisão -> Resposta
                                |
                      +-------------------+
                      | CNN+LSTM          |
                      | (Classificação)   |
                      +-------------------+
```

## 4. Modelo de Detecção de Anomalias

### 4.1 Arquitetura do Autoencoder

O autoencoder é uma rede neural que aprende a reconstruir seus inputs, sendo treinado apenas com tráfego normal. Quando apresentado a um tráfego anômalo, o erro de reconstrução será maior, indicando uma possível intrusão.

```
Input Layer (n_features) -> 
  Encoder Layer 1 (128 neurons, ReLU) -> 
    Encoder Layer 2 (64 neurons, ReLU) -> 
      Bottleneck Layer (32 neurons, ReLU) -> 
        Decoder Layer 1 (64 neurons, ReLU) -> 
          Decoder Layer 2 (128 neurons, ReLU) -> 
            Output Layer (n_features, Linear)
```

### 4.2 Implementação do Autoencoder

```python
def create_autoencoder(input_dim):
    # Definir a arquitetura
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoder = Dense(128, activation='relu')(input_layer)
    encoder = BatchNormalization()(encoder)
    encoder = Dense(64, activation='relu')(encoder)
    encoder = BatchNormalization()(encoder)
    
    # Bottleneck
    bottleneck = Dense(32, activation='relu')(encoder)
    
    # Decoder
    decoder = Dense(64, activation='relu')(bottleneck)
    decoder = BatchNormalization()(decoder)
    decoder = Dense(128, activation='relu')(decoder)
    decoder = BatchNormalization()(decoder)
    
    # Output
    output_layer = Dense(input_dim, activation='linear')(decoder)
    
    # Criar modelo
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

def train_autoencoder(autoencoder, normal_data, epochs=50, batch_size=64):
    # Treinar apenas com dados normais
    history = autoencoder.fit(
        normal_data, normal_data,  # Input = Output (reconstrução)
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
    )
    return history

def detect_anomalies(autoencoder, data, threshold=None):
    # Reconstruir os dados
    reconstructions = autoencoder.predict(data)
    
    # Calcular erro de reconstrução (MSE)
    mse = np.mean(np.power(data - reconstructions, 2), axis=1)
    
    # Se threshold não for fornecido, calcular baseado nos dados de treinamento
    if threshold is None:
        threshold = np.percentile(mse, 95)  # 95º percentil como threshold
    
    # Classificar como anomalia se erro > threshold
    anomalies = mse > threshold
    
    return anomalies, mse, threshold
```

### 4.3 Treinamento e Validação

- Treinado apenas com tráfego normal dos datasets NSL-KDD e CICIDS-2018
- Validação com conjunto separado de tráfego normal e anômalo
- Ajuste de threshold baseado em métricas de desempenho (F1-score, AUC)
- Monitoramento de overfitting/underfitting

## 5. Modelo de Classificação de Ataques

### 5.1 Arquitetura CNN+LSTM

Esta arquitetura combina redes neurais convolucionais (CNN) para extração de características espaciais com redes LSTM para capturar dependências temporais no tráfego de rede.

```
Input Layer (n_features, sequence_length) -> 
  Conv1D Layer 1 (64 filters, kernel_size=3, ReLU) -> 
    MaxPooling1D -> 
      Conv1D Layer 2 (128 filters, kernel_size=3, ReLU) -> 
        MaxPooling1D -> 
          LSTM Layer (100 units) -> 
            Dense Layer 1 (64 neurons, ReLU) -> 
              Dropout (0.5) -> 
                Dense Layer 2 (32 neurons, ReLU) -> 
                  Output Layer (n_classes, Softmax)
```

### 5.2 Implementação do Modelo CNN+LSTM

```python
def create_cnn_lstm_model(input_shape, num_classes):
    # Definir a arquitetura
    input_layer = Input(shape=input_shape)
    
    # Camadas CNN
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    
    # Camada LSTM
    lstm = LSTM(100, return_sequences=False)(pool2)
    
    # Camadas densas
    dense1 = Dense(64, activation='relu')(lstm)
    dropout = Dropout(0.5)(dense1)
    dense2 = Dense(32, activation='relu')(dropout)
    
    # Camada de saída
    if num_classes == 2:
        output_layer = Dense(1, activation='sigmoid')(dense2)
        loss = 'binary_crossentropy'
    else:
        output_layer = Dense(num_classes, activation='softmax')(dense2)
        loss = 'categorical_crossentropy'
    
    # Criar modelo
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

def prepare_sequence_data(data, sequence_length=10):
    """Transforma dados em sequências para alimentar o modelo CNN+LSTM"""
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i+sequence_length])
    return np.array(sequences)

def train_cnn_lstm_model(model, X_train, y_train, epochs=50, batch_size=64):
    # Preparar dados de sequência
    X_train_seq = prepare_sequence_data(X_train)
    
    # Treinar modelo
    history = model.fit(
        X_train_seq, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
    )
    return history
```

### 5.3 Treinamento e Validação

- Treinado com dados rotulados dos datasets NSL-KDD e CICIDS-2018
- Validação com conjunto separado de teste
- Uso de técnicas para lidar com desbalanceamento de classes
- Avaliação com métricas específicas para cada tipo de ataque

## 6. Sistema de Decisão

### 6.1 Arquitetura do Sistema de Decisão

O sistema de decisão combina os resultados dos modelos de anomalia e classificação para determinar a resposta final.

```
                      +-------------------+
                      | Pontuação de      |
                      | Anomalia          |
                      +-------------------+
                                |
                                v
+-------------------+   +-------------------+   +-------------------+
| Classificação de  |-->| Mecanismo de     |-->| Nível de          |
| Ataque            |   | Decisão          |   | Confiança         |
+-------------------+   +-------------------+   +-------------------+
                                |
                                v
                      +-------------------+
                      | Feedback e        |
                      | Aprendizado       |
                      +-------------------+
```

### 6.2 Implementação do Sistema de Decisão

```python
class DecisionSystem:
    def __init__(self, anomaly_threshold=None, classification_threshold=0.7, weights=None):
        self.anomaly_threshold = anomaly_threshold
        self.classification_threshold = classification_threshold
        self.weights = weights or {'anomaly': 0.6, 'classification': 0.4}
        self.feedback_history = []
    
    def decide(self, anomaly_score, classification_result):
        """
        Toma decisão combinando resultados de anomalia e classificação
        
        Args:
            anomaly_score: Pontuação de anomalia (erro de reconstrução)
            classification_result: Probabilidades de classes de ataque
        
        Returns:
            decision: Decisão final (0: normal, 1: ataque)
            attack_type: Tipo de ataque identificado (se houver)
            confidence: Nível de confiança na decisão
        """
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
            confidence = 1 - (self.weights['anomaly'] * (anomaly_score / self.anomaly_threshold) + \
                             self.weights['classification'] * max_class_prob)
            decision = 0
            attack_type = None
        
        return decision, attack_type, confidence
    
    def update_from_feedback(self, feedback_data):
        """Atualiza parâmetros do sistema baseado em feedback"""
        self.feedback_history.append(feedback_data)
        
        # Recalcular thresholds baseado em feedback
        if len(self.feedback_history) >= 10:
            # Ajustar threshold de anomalia
            anomaly_scores = [f['anomaly_score'] for f in self.feedback_history]
            true_labels = [f['true_label'] for f in self.feedback_history]
            
            # Encontrar threshold ótimo baseado em feedback
            best_f1 = 0
            best_threshold = self.anomaly_threshold
            
            for threshold in np.linspace(min(anomaly_scores), max(anomaly_scores), 20):
                predictions = [1 if score > threshold else 0 for score in anomaly_scores]
                f1 = f1_score(true_labels, predictions)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            self.anomaly_threshold = best_threshold
            
            # Limitar histórico para evitar crescimento infinito
            self.feedback_history = self.feedback_history[-100:]
```

### 6.3 Explicabilidade e Interpretação

- Geração de explicações para decisões tomadas
- Visualização de características mais relevantes
- Rastreamento de decisões para auditoria

```python
def generate_explanation(anomaly_model, classification_model, input_data, decision_result):
    """Gera explicação para a decisão tomada"""
    explanation = {
        'decision': decision_result[0],
        'attack_type': decision_result[1],
        'confidence': decision_result[2],
        'contributing_factors': []
    }
    
    # Identificar características que mais contribuíram para anomalia
    if anomaly_model:
        reconstruction = anomaly_model.predict(np.array([input_data]))[0]
        reconstruction_errors = np.power(input_data - reconstruction, 2)
        
        # Top 5 características com maior erro de reconstrução
        top_features_idx = np.argsort(reconstruction_errors)[-5:]
        for idx in top_features_idx:
            explanation['contributing_factors'].append({
                'feature_index': idx,
                'error_contribution': float(reconstruction_errors[idx]),
                'original_value': float(input_data[idx]),
                'reconstructed_value': float(reconstruction[idx])
            })
    
    # Para classificação, usar técnicas como SHAP ou LIME
    # (Implementação simplificada aqui)
    if classification_model and decision_result[1] is not None:
        # Exemplo simples: usar gradientes para identificar características importantes
        input_tensor = tf.convert_to_tensor(np.array([input_data]))
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            prediction = classification_model(input_tensor)
            target_class = decision_result[1]
            if target_class > 0:  # Se não for classe normal
                target_prediction = prediction[0, target_class]
        
        gradients = tape.gradient(target_prediction, input_tensor)
        feature_importance = np.abs(gradients.numpy()[0])
        
        # Top 5 características mais importantes para classificação
        top_class_features_idx = np.argsort(feature_importance)[-5:]
        for idx in top_class_features_idx:
            explanation['contributing_factors'].append({
                'feature_index': idx,
                'classification_importance': float(feature_importance[idx]),
                'value': float(input_data[idx])
            })
    
    return explanation
```

## 7. Adaptação e Aprendizado Contínuo

### 7.1 Atualização Incremental de Modelos

```python
def update_models_incrementally(autoencoder, classifier, new_data, new_labels, update_frequency=1000):
    """Atualiza modelos incrementalmente com novos dados"""
    # Atualizar autoencoder com novos dados normais
    normal_indices = np.where(new_labels == 0)[0]
    if len(normal_indices) > 0:
        normal_data = new_data[normal_indices]
        autoencoder.fit(
            normal_data, normal_data,
            epochs=5,
            batch_size=32,
            verbose=0
        )
    
    # Atualizar classificador com todos os novos dados
    if len(new_data) > 0:
        new_data_seq = prepare_sequence_data(new_data)
        classifier.fit(
            new_data_seq, new_labels,
            epochs=5,
            batch_size=32,
            verbose=0
        )
    
    return autoencoder, classifier
```

### 7.2 Detecção de Drift de Conceito

```python
def detect_concept_drift(reference_data, current_data, threshold=0.05):
    """Detecta mudanças significativas na distribuição dos dados"""
    # Calcular estatísticas para cada característica
    ref_mean = np.mean(reference_data, axis=0)
    ref_std = np.std(reference_data, axis=0)
    
    current_mean = np.mean(current_data, axis=0)
    current_std = np.std(current_data, axis=0)
    
    # Calcular distância entre distribuições
    mean_distance = np.mean(np.abs(ref_mean - current_mean) / (ref_std + 1e-10))
    std_ratio = np.mean(np.abs(ref_std - current_std) / (ref_
(Content truncated due to size limit. Use line ranges to read in chunks)