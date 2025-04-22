# Mecanismos de Detecção e Prevenção

## 1. Visão Geral

Os mecanismos de detecção e prevenção são componentes críticos do firewall baseado em rede neural, responsáveis por identificar ameaças e implementar contramedidas para proteger a rede. Este documento detalha a arquitetura e funcionamento destes mecanismos, com foco especial na detecção de ataques zero-day.

## 2. Objetivos dos Mecanismos

- Detectar ataques conhecidos e desconhecidos (zero-day) com alta precisão
- Implementar respostas apropriadas em tempo real
- Minimizar falsos positivos e falsos negativos
- Adaptar-se a novas ameaças através de aprendizado contínuo
- Fornecer informações detalhadas sobre ataques detectados

## 3. Arquitetura dos Mecanismos

```
+----------------+     +----------------+     +----------------+
| Módulo de      |---->| Avaliação de   |---->| Seleção de     |
| Detecção       |     | Ameaças        |     | Resposta       |
+----------------+     +----------------+     +----------------+
                                                      |
                                                      v
+----------------+     +----------------+     +----------------+
| Feedback e     |<----| Registro e     |<----| Execução de    |
| Aprendizado    |     | Auditoria      |     | Resposta       |
+----------------+     +----------------+     +----------------+
```

## 4. Componentes dos Mecanismos

### 4.1 Módulo de Detecção

Este componente integra os resultados dos modelos de rede neural para identificar atividades maliciosas.

**Funcionalidades:**
- Recebe dados processados do tráfego de rede
- Aplica modelos de detecção de anomalias e classificação
- Combina resultados através do sistema de decisão
- Gera alertas para atividades suspeitas

**Implementação:**
```python
class DetectionModule:
    def __init__(self, anomaly_model, classification_model, decision_system):
        self.anomaly_model = anomaly_model
        self.classification_model = classification_model
        self.decision_system = decision_system
        self.detection_log = []
    
    def detect(self, processed_data, sequence_data=None):
        """
        Detecta ameaças nos dados processados
        
        Args:
            processed_data: Dados pré-processados para análise
            sequence_data: Dados em formato de sequência para o classificador (opcional)
        
        Returns:
            detection_result: Resultado da detecção (0: normal, 1: ataque)
            attack_type: Tipo de ataque identificado (se houver)
            confidence: Nível de confiança na detecção
            explanation: Explicação da decisão
        """
        # Detecção de anomalias
        reconstructions = self.anomaly_model.predict(np.array([processed_data]))
        anomaly_score = np.mean(np.power(processed_data - reconstructions[0], 2))
        
        # Classificação de ataques
        if sequence_data is None:
            # Criar sequência a partir dos dados atuais (simplificado)
            sequence_data = np.array([processed_data]).reshape(1, 1, -1)
        
        classification_result = self.classification_model.predict(sequence_data)[0]
        
        # Combinar resultados através do sistema de decisão
        decision, attack_type, confidence = self.decision_system.decide(
            anomaly_score, classification_result
        )
        
        # Gerar explicação
        explanation = generate_explanation(
            self.anomaly_model, 
            self.classification_model,
            processed_data,
            (decision, attack_type, confidence)
        )
        
        # Registrar detecção
        self.detection_log.append({
            'timestamp': time.time(),
            'decision': decision,
            'attack_type': attack_type,
            'confidence': confidence,
            'anomaly_score': anomaly_score,
            'classification_result': classification_result.tolist()
        })
        
        return decision, attack_type, confidence, explanation
```

### 4.2 Avaliação de Ameaças

Este componente avalia a severidade e o impacto potencial das ameaças detectadas.

**Funcionalidades:**
- Atribui níveis de severidade às ameaças detectadas
- Avalia o impacto potencial na rede
- Considera o contexto da rede e sistemas protegidos
- Prioriza ameaças para resposta

**Implementação:**
```python
class ThreatEvaluator:
    def __init__(self, config):
        self.config = config
        self.asset_importance = config.get('asset_importance', {})
        self.attack_severity = config.get('attack_severity', {})
        self.context_rules = config.get('context_rules', [])
    
    def evaluate(self, detection_result, network_context):
        """
        Avalia a severidade e impacto da ameaça detectada
        
        Args:
            detection_result: Resultado da detecção
            network_context: Informações contextuais da rede
        
        Returns:
            threat_level: Nível de ameaça (0-10)
            impact_assessment: Avaliação de impacto
            priority: Prioridade para resposta (1-5)
        """
        decision, attack_type, confidence = detection_result[:3]
        
        if decision == 0:  # Não é ataque
            return 0, "Tráfego normal", 0
        
        # Obter severidade base do tipo de ataque
        base_severity = self.attack_severity.get(
            str(attack_type), 
            5  # Valor padrão médio para ataques desconhecidos
        )
        
        # Ajustar severidade com base na confiança
        adjusted_severity = base_severity * confidence
        
        # Avaliar impacto com base no contexto da rede
        impact = self._assess_impact(attack_type, network_context)
        
        # Calcular nível de ameaça combinado
        threat_level = min(10, (adjusted_severity + impact) / 2)
        
        # Determinar prioridade
        if threat_level >= 8:
            priority = 1  # Crítica
        elif threat_level >= 6:
            priority = 2  # Alta
        elif threat_level >= 4:
            priority = 3  # Média
        elif threat_level >= 2:
            priority = 4  # Baixa
        else:
            priority = 5  # Muito baixa
        
        # Gerar avaliação de impacto textual
        impact_assessment = self._generate_impact_assessment(
            attack_type, threat_level, network_context
        )
        
        return threat_level, impact_assessment, priority
    
    def _assess_impact(self, attack_type, network_context):
        """Avalia o impacto potencial com base no contexto da rede"""
        impact_score = 5  # Valor padrão médio
        
        # Considerar importância dos ativos afetados
        target_asset = network_context.get('target_asset')
        if target_asset and target_asset in self.asset_importance:
            impact_score = max(impact_score, self.asset_importance[target_asset])
        
        # Aplicar regras de contexto
        for rule in self.context_rules:
            if self._rule_matches(rule, attack_type, network_context):
                impact_score = max(impact_score, rule['impact_score'])
        
        return impact_score
    
    def _rule_matches(self, rule, attack_type, network_context):
        """Verifica se uma regra de contexto se aplica"""
        # Verificar tipo de ataque
        if 'attack_types' in rule and attack_type not in rule['attack_types']:
            return False
        
        # Verificar condições de contexto
        for key, value in rule.get('conditions', {}).items():
            if key not in network_context or network_context[key] != value:
                return False
        
        return True
    
    def _generate_impact_assessment(self, attack_type, threat_level, network_context):
        """Gera uma avaliação textual do impacto"""
        if attack_type == 0:  # Ataque desconhecido/zero-day
            assessment = "Possível ataque zero-day detectado. "
        else:
            attack_names = {
                1: "DoS/DDoS",
                2: "Probe/Scan",
                3: "R2L (Remote to Local)",
                4: "U2R (User to Root)",
                5: "Botnet",
                # Adicionar outros tipos conforme necessário
            }
            attack_name = attack_names.get(attack_type, f"Ataque tipo {attack_type}")
            assessment = f"Ataque {attack_name} detectado. "
        
        if threat_level >= 8:
            assessment += "Impacto potencialmente crítico para a rede. "
        elif threat_level >= 6:
            assessment += "Impacto potencialmente alto para a rede. "
        elif threat_level >= 4:
            assessment += "Impacto potencialmente moderado para a rede. "
        else:
            assessment += "Impacto potencialmente baixo para a rede. "
        
        # Adicionar informações específicas de contexto
        target_asset = network_context.get('target_asset')
        if target_asset:
            assessment += f"Alvo: {target_asset}. "
        
        return assessment
```

### 4.3 Seleção de Resposta

Este componente determina as ações apropriadas para mitigar as ameaças detectadas.

**Funcionalidades:**
- Seleciona respostas baseadas no tipo e severidade da ameaça
- Considera o impacto potencial das ações de mitigação
- Implementa políticas definidas pelo administrador
- Adapta respostas com base em feedback anterior

**Implementação:**
```python
class ResponseSelector:
    def __init__(self, config):
        self.config = config
        self.response_policies = config.get('response_policies', {})
        self.default_policy = config.get('default_policy', 'moderate')
        self.response_history = []
    
    def select_response(self, detection_result, threat_evaluation, network_context):
        """
        Seleciona a resposta apropriada para a ameaça detectada
        
        Args:
            detection_result: Resultado da detecção
            threat_evaluation: Avaliação da ameaça
            network_context: Informações contextuais da rede
        
        Returns:
            response_actions: Lista de ações a serem executadas
            response_params: Parâmetros para as ações
            response_justification: Justificativa para a resposta selecionada
        """
        decision, attack_type, confidence = detection_result[:3]
        threat_level, _, priority = threat_evaluation
        
        if decision == 0:  # Não é ataque
            return [], {}, "Nenhuma ação necessária para tráfego normal"
        
        # Determinar política de resposta aplicável
        policy_name = self._determine_policy(
            attack_type, threat_level, network_context
        )
        policy = self.response_policies.get(policy_name, self.response_policies[self.default_policy])
        
        # Selecionar ações baseadas na política e nível de ameaça
        response_actions = []
        response_params = {}
        
        for action_config in policy['actions']:
            min_threat = action_config.get('min_threat_level', 0)
            max_threat = action_config.get('max_threat_level', 10)
            
            if min_threat <= threat_level <= max_threat:
                action = action_config['action']
                response_actions.append(action)
                
                # Configurar parâmetros da ação
                if 'params' in action_config:
                    response_params[action] = self._configure_action_params(
                        action_config['params'],
                        detection_result,
                        threat_evaluation,
                        network_context
                    )
        
        # Gerar justificativa
        response_justification = self._generate_justification(
            policy_name, response_actions, threat_level, attack_type
        )
        
        # Registrar resposta selecionada
        self.response_history.append({
            'timestamp': time.time(),
            'detection': detection_result[:3],
            'threat_level': threat_level,
            'policy': policy_name,
            'actions': response_actions,
            'params': response_params
        })
        
        return response_actions, response_params, response_justification
    
    def _determine_policy(self, attack_type, threat_level, network_context):
        """Determina qual política de resposta aplicar"""
        # Verificar políticas específicas para o tipo de ataque
        attack_policies = {k: v for k, v in self.response_policies.items() 
                          if 'attack_types' in v and attack_type in v['attack_types']}
        
        if attack_policies:
            # Selecionar política mais específica
            for policy_name, policy in attack_policies.items():
                if 'conditions' in policy:
                    if all(network_context.get(k) == v for k, v in policy['conditions'].items()):
                        return policy_name
            
            # Se nenhuma condição específica corresponder, usar a primeira política para o tipo de ataque
            return list(attack_policies.keys())[0]
        
        # Selecionar política baseada no nível de ameaça
        if threat_level >= 8:
            return 'aggressive'
        elif threat_level >= 5:
            return 'moderate'
        else:
            return 'conservative'
    
    def _configure_action_params(self, param_template, detection_result, threat_evaluation, network_context):
        """Configura parâmetros específicos para uma ação"""
        params = {}
        
        for param_name, param_config in param_template.items():
            if isinstance(param_config, dict) and 'source' in param_config:
                # Parâmetro dinâmico baseado em contexto
                source = param_config['source']
                if source == 'detection':
                    params[param_name] = detection_result[param_config.get('index', 0)]
                elif source == 'threat':
                    params[param_name] = threat_evaluation[param_config.get('index', 0)]
                elif source == 'network':
                    params[param_name] = network_context.get(param_config.get('key', ''))
                elif source == 'formula':
                    # Avaliação de fórmula simples (exemplo)
                    if param_config.get('formula') == 'threat_level_seconds':
                        params[param_name] = int(threat_evaluation[0] * 60)  # Segundos baseados no nível de ameaça
            else:
                # Parâmetro estático
                params[param_name] = param_config
        
        return params
    
    def _generate_justification(self, policy_name, actions, threat_level, attack_type):
        """Gera justificativa para a resposta selecionada"""
        attack_names = {
            0: "desconhecido (possível zero-day)",
            1: "DoS/DDoS",
            2: "Probe/Scan",
            3: "R2L (Remote to Local)",
            4: "U2R (User to Root)",
            5: "Botnet",
            # Adicionar outros tipos conforme necessário
        }
        attack_name = attack_names.get(attack_type, f"tipo {attack_type}")
        
        justification = f"Resposta baseada na política '{policy_name}' para ataque {attack_name} "
        justification += f"com nível de ameaça {threat_level:.1f}/10. "
        
        if actions:
            justification += f"Ações selecionadas: {', '.join(actions)}."
        else:
            justification += "Nenhuma ação selecionada baseada nos critérios atuais."
        
        return justification
```

### 4.4 Execução de Resposta

Este componente implementa as ações de mitigação selecionadas.

**Funcionalidades:**
- Executa ações de bloqueio, limitação ou alerta
- Implementa regras de firewall dinâmicas
- Coordena respostas com outros sistemas de segurança
- Monitora a eficácia das ações tomadas

**Implementação:**
```python
class ResponseExecutor:
    def __init__(self, config):
        self.config = config
(Content truncated due to size limit. Use line ranges to read in chunks)