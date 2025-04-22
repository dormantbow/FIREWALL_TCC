# Manual de Uso do Firewall Baseado em Rede Neural

Este manual fornece instruções para instalação, configuração e uso do firewall baseado em rede neural para prevenção de ataques zero-day.

## Requisitos do Sistema

- Python 3.8 ou superior
- TensorFlow 2.x
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (para visualizações)
- Acesso root (para implementação real do firewall)

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/tcc-firewall-neural.git
   cd tcc-firewall-neural
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Baixe os datasets (opcional, se não estiverem incluídos):
   ```bash
   mkdir -p data
   # Para NSL-KDD
   wget -O data/KDDTrain+.txt http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
   wget -O data/KDDTest+.txt http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz
   # Para CICIDS-2018, baixe manualmente de https://www.unb.ca/cic/datasets/ids-2018.html
   ```

## Estrutura do Projeto

```
tcc_firewall/
├── data/                  # Diretório para armazenar datasets
├── docs/                  # Documentação
├── models/                # Modelos treinados
├── src/                   # Código-fonte
├── tests/                 # Scripts de teste
└── visualizations/        # Visualizações geradas pelos testes
```

## Uso Básico

### Treinamento do Firewall

Para treinar o firewall com o dataset NSL-KDD:

```bash
python -m src.main --train --dataset nsl_kdd --save
```

Para treinar com o dataset CICIDS-2018:

```bash
python -m src.main --train --dataset cicids_2018 --save
```

### Carregamento de um Modelo Treinado

Para carregar um modelo previamente treinado:

```bash
python -m src.main --load --model-dir models/
```

### Detecção de Ameaças

Para detectar ameaças em um arquivo de entrada:

```bash
python -m src.main --load --input-file data/test_samples.csv
```

### Execução de Testes

Para executar todos os testes:

```bash
./run_tests.sh
```

Para executar testes específicos:

```bash
python tests/test_preprocessing.py
python tests/test_neural_models.py
python tests/test_intrusion_detection.py
python tests/test_firewall_system.py
```

## Configuração Avançada

### Configuração do Sistema de Decisão

O sistema de decisão pode ser configurado editando o arquivo `src/utils/config.py`. Os principais parâmetros são:

- `weights`: Pesos para combinar os resultados dos modelos de anomalia e classificação
- `classification_threshold`: Limiar para considerar uma classificação como ataque
- `update_frequency`: Frequência de atualização com base em feedback

### Configuração das Políticas de Resposta

As políticas de resposta podem ser configuradas editando o arquivo `src/utils/config.py`. Cada política define:

- `actions`: Ações a serem tomadas (block_ip, limit_traffic, alert, etc.)
- `conditions`: Condições para aplicar a política
- `attack_types`: Tipos de ataque para os quais a política se aplica

Exemplo:
```python
'aggressive': {
    'actions': [
        {'action': 'block_ip', 'min_threat_level': 7.0, 'params': {...}},
        {'action': 'alert', 'min_threat_level': 5.0, 'params': {...}}
    ],
    'attack_types': [1, 2, 5],  # DoS/DDoS, Probe/Scan, Botnet
    'conditions': {'is_critical_asset': True}
}
```

### Integração com Firewall Real

Para integrar com um firewall real (iptables), é necessário modificar a classe `FirewallInterface` em `src/detection/response_mechanisms.py`. Os métodos `_implement_rule` e `_remove_rule_from_system` devem ser adaptados para executar comandos reais do iptables.

## Monitoramento e Logs

Os logs do sistema são armazenados no diretório especificado em `src/utils/config.py`. Por padrão, os logs são gravados em:

- `logs/firewall.log`: Log principal do firewall
- `logs/alerts.log`: Log de alertas
- `logs/performance.log`: Log de métricas de desempenho

Para visualizar os logs em tempo real:

```bash
tail -f logs/firewall.log
```

## Feedback e Aprendizado Contínuo

O sistema suporta aprendizado contínuo através de feedback. Para fornecer feedback sobre uma detecção:

```python
from src.main import NeuralFirewall

firewall = NeuralFirewall.load('models/')
# Supondo que a detecção com ID 5 foi incorreta e o tráfego era normal
firewall.process_feedback(5, is_correct=False, true_label=0, comments="Falso positivo")
```

## Solução de Problemas

### Problemas Comuns

1. **Erro ao carregar modelos**: Verifique se os caminhos estão corretos e se os modelos foram salvos corretamente.

2. **Desempenho lento**: Reduza a complexidade dos modelos ou utilize hardware mais potente.

3. **Falsos positivos frequentes**: Ajuste os thresholds no sistema de decisão ou forneça mais feedback para melhorar o aprendizado.

4. **Falha ao implementar regras de firewall**: Verifique as permissões e se o iptables está instalado e configurado corretamente.

### Logs de Depuração

Para ativar logs de depuração mais detalhados, edite o arquivo `src/utils/logging_utils.py` e altere o nível de log para DEBUG:

```python
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[...]
)
```

## Limitações Conhecidas

1. O sistema atual não foi otimizado para processamento em tempo real de grandes volumes de tráfego.

2. A detecção de ataques zero-day, embora promissora, ainda tem limitações para ataques completamente novos.

3. A interface com o firewall do sistema é simulada e precisa ser adaptada para um ambiente real.

4. Os datasets utilizados não contêm exemplos recentes de ataques, o que pode limitar a eficácia contra ameaças mais recentes.

## Suporte e Contribuições

Para relatar problemas ou contribuir com o projeto, entre em contato com o autor ou abra uma issue no repositório do projeto.

---

Este manual foi criado como parte do Trabalho de Conclusão de Curso em Engenharia da Computação.
