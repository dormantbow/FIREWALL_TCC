import json
from datetime import datetime

from ..utils.config import DECISION_SYSTEM_CONFIG
from ..utils.logging_utils import performance_monitor
from utils.config import DECISION_SYSTEM_CONFIG
from utils.logging_utils import performance_monitor

logger = logging.getLogger(__name__)

class DecisionSystem:
    """
    Sistema de decisão que combina os resultados dos modelos de detecção
    de anomalias e classificação de ataques para determinar a resposta final.
    """
    
    def __init__(self, config=None):
        """
        Inicializa o sistema de decisão.
        
        Args:
            config: Configuração do sistema de decisão (opcional)
        """
        self.config = config or DECISION_SYSTEM_CONFIG
        self.weights = self.config.get('weights', {'anomaly': 0.6, 'classification': 0.4})
        self.classification_threshold = self.config.get('classification_threshold', 0.7)
        self.anomaly_threshold = None  # Será definido pelo modelo de anomalia
        self.feedback_history = []
    
    def decide(self, anomaly_score, classification_result):
