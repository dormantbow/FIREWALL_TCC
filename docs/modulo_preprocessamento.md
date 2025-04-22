# Arquitetura do Módulo de Pré-processamento

## 1. Visão Geral

O módulo de pré-processamento é um componente crítico do firewall baseado em rede neural, responsável por transformar dados brutos de rede em características (features) adequadas para análise pelos algoritmos de aprendizado de máquina. Este documento detalha a arquitetura e funcionamento deste módulo.

## 2. Objetivos do Módulo

- Extrair características relevantes dos pacotes e fluxos de rede
- Normalizar e padronizar os dados para alimentar os modelos de rede neural
- Reduzir dimensionalidade mantendo informações discriminativas
- Lidar com dados desbalanceados (comum em datasets de segurança)
- Processar dados em tempo real com baixa latência

## 3. Arquitetura do Módulo

```
+----------------+     +----------------+     +----------------+
| Extração de    |---->| Transformação  |---->| Seleção de     |
| Características |     | de Dados       |     | Características |
+----------------+     +----------------+     +----------------+
                                                      |
                                                      v
+----------------+     +----------------+     +----------------+
| Saída para     |<----| Normalização   |<----| Balanceamento  |
| Detecção       |     | de Dados       |     | de Classes     |
+----------------+     +----------------+     +----------------+
```

## 4. Componentes do Módulo

### 4.1 Extração de Características

Este componente extrai informações relevantes dos pacotes e fluxos de rede.

**Características de Pacotes:**
- Informações de cabeçalho (IP, TCP/UDP, etc.)
- Tamanho e estrutura dos pacotes
- Flags e opções de protocolo
- Payload (quando relevante)

**Características de Fluxo:**
- Estatísticas de conexão (duração, bytes transferidos, etc.)
- Padrões temporais (intervalos entre pacotes)
- Comportamento bidirecional
- Relações entre conexões

**Implementação:**
```python
def extract_packet_features(packet):
    features = {}
    # Extrair informações de cabeçalho
    if IP in packet:
        features['ip_len'] = packet[IP].len
        features['ip_ttl'] = packet[IP].ttl
        # ...
    
    # Extrair informações TCP/UDP
    if TCP in packet:
        features['tcp_sport'] = packet[TCP].sport
        features['tcp_dport'] = packet[TCP].dport
        features['tcp_flags'] = packet[TCP].flags
        # ...
    
    return features

def extract_flow_features(flow):
    features = {}
    # Calcular estatísticas de fluxo
    features['duration'] = flow.end_time - flow.start_time
    features['packet_count'] = len(flow.packets)
    features['bytes_transferred'] = sum(len(p) for p in flow.packets)
    # ...
    
    return features
```

### 4.2 Transformação de Dados

Este componente transforma os dados brutos em formatos adequados para processamento.

**Operações:**
- Codificação de variáveis categóricas (one-hot, label encoding)
- Transformação de distribuições (log, raiz quadrada)
- Agregação temporal de características
- Tratamento de valores ausentes

**Implementação:**
```python
def transform_categorical_features(features, categorical_columns):
    transformed = features.copy()
    
    for col in categorical_columns:
        # One-hot encoding
        one_hot = pd.get_dummies(transformed[col], prefix=col)
        transformed = pd.concat([transformed, one_hot], axis=1)
        transformed.drop(col, axis=1, inplace=True)
    
    return transformed

def transform_numerical_features(features, skewed_columns):
    transformed = features.copy()
    
    for col in skewed_columns:
        # Log transformation para distribuições assimétricas
        transformed[col] = np.log1p(transformed[col])
    
    return transformed
```

### 4.3 Seleção de Características

Este componente identifica e seleciona as características mais relevantes para a detecção.

**Técnicas:**
- Análise de correlação
- Importância de características (feature importance)
- Métodos baseados em variância
- Seleção baseada em modelo (model-based selection)

**Implementação:**
```python
def select_features_correlation(features, threshold=0.95):
    corr_matrix = features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    return features.drop(to_drop, axis=1)

def select_features_importance(features, labels, n_features=20):
    model = RandomForestClassifier()
    model.fit(features, labels)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    selected_features = features.iloc[:, indices[:n_features]]
    return selected_features
```

### 4.4 Balanceamento de Classes

Este componente lida com o desbalanceamento de classes comum em dados de segurança.

**Técnicas:**
- Oversampling (SMOTE, ADASYN)
- Undersampling (Tomek links, NearMiss)
- Combinação de técnicas
- Pesos de classe

**Implementação:**
```python
def balance_classes(features, labels):
    # Verificar desbalanceamento
    class_counts = np.bincount(labels)
    if np.min(class_counts) / np.max(class_counts) < 0.2:
        # Aplicar SMOTE para classes minoritárias
        smote = SMOTE(random_state=42)
        features_balanced, labels_balanced = smote.fit_resample(features, labels)
        return features_balanced, labels_balanced
    
    return features, labels
```

### 4.5 Normalização de Dados

Este componente padroniza os dados para melhorar o desempenho dos algoritmos de aprendizado.

**Técnicas:**
- Padronização (Z-score)
- Min-Max scaling
- Robust scaling
- Normalização L2

**Implementação:**
```python
def normalize_features(features, method='standard'):
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    
    normalized = scaler.fit_transform(features)
    return normalized, scaler  # Retorna também o scaler para aplicar em dados futuros
```

## 5. Fluxo de Processamento

### 5.1 Fluxo de Treinamento

```
Datasets (NSL-KDD, CICIDS-2018) -> 
  Extração de Características -> 
    Transformação de Dados -> 
      Seleção de Características -> 
        Balanceamento de Classes -> 
          Normalização -> 
            Saída para Treinamento de Modelos
```

### 5.2 Fluxo em Tempo Real

```
Pacotes de Rede -> 
  Reconstrução de Fluxos -> 
    Extração de Características -> 
      Transformação de Dados -> 
        Aplicação de Seleção de Características (pré-definida) -> 
          Normalização (usando parâmetros pré-calculados) -> 
            Saída para Detecção
```

## 6. Considerações de Implementação

### 6.1 Otimização de Desempenho

- Implementação de processamento em lotes (batch processing)
- Utilização de estruturas de dados eficientes
- Paralelização de operações independentes
- Caching de resultados intermediários

### 6.2 Adaptação em Tempo Real

- Atualização incremental de estatísticas
- Detecção de drift de distribuição
- Ajuste dinâmico de parâmetros de normalização

### 6.3 Integração com Outros Módulos

- Interface padronizada para o módulo de detecção
- Feedback do módulo de detecção para ajuste de características
- Armazenamento de parâmetros de transformação na base de conhecimento

## 7. Implementação de Referência

```python
class PreprocessingModule:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.feature_selectors = {}
        self.categorical_columns = config.get('categorical_columns', [])
        self.numerical_columns = config.get('numerical_columns', [])
        
    def fit(self, data, labels=None):
        """Treina o módulo de pré-processamento com dados históricos"""
        # Extração e transformação
        features = self.extract_features(data)
        features = self.transform_features(features)
        
        # Seleção de características
        if self.config.get('feature_selection', 'correlation') == 'correlation':
            selected_features = select_features_correlation(features)
        else:
            selected_features = select_features_importance(features, labels)
        
        self.feature_selectors['columns'] = selected_features.columns
        
        # Balanceamento (apenas para treinamento)
        if labels is not None and self.config.get('balance_classes', True):
            selected_features, labels = balance_classes(selected_features, labels)
        
        # Normalização
        normalized, scaler = normalize_features(selected_features, 
                                               method=self.config.get('normalization', 'standard'))
        self.scalers['main'] = scaler
        
        return normalized, labels
    
    def transform(self, data):
        """Transforma novos dados usando parâmetros aprendidos"""
        # Extração e transformação
        features = self.extract_features(data)
        features = self.transform_features(features)
        
        # Aplicar seleção de características
        selected_features = features[self.feature_selectors['columns']]
        
        # Normalização
        normalized = self.scalers['main'].transform(selected_features)
        
        return normalized
    
    def extract_features(self, data):
        """Extrai características dos dados brutos"""
        if self.config.get('data_type') == 'packet':
            return pd.DataFrame([extract_packet_features(p) for p in data])
        else:
            return pd.DataFrame([extract_flow_features(f) for f in data])
    
    def transform_features(self, features):
        """Aplica transformações nas características"""
        features = transform_categorical_features(features, self.categorical_columns)
        features = transform_numerical_features(features, self.numerical_columns)
        return features
```

## 8. Conclusão

O módulo de pré-processamento é fundamental para o sucesso do firewall baseado em rede neural, pois a qualidade dos dados fornecidos aos algoritmos de aprendizado de máquina determina diretamente a eficácia da detecção. A arquitetura proposta é flexível, permitindo adaptação a diferentes tipos de dados e requisitos, e é otimizada para operação tanto em modo de treinamento quanto em tempo real.

A implementação seguirá uma abordagem modular, permitindo ajustes e melhorias incrementais em cada componente, com foco inicial no processamento eficiente dos datasets NSL-KDD e CICIDS-2018 para treinamento dos modelos de detecção.
