import time
import json
from datetime import datetime

from ..models.neural_models import AutoencoderModel, CNNLSTMModel
from ..detection.decision_system import DecisionSystem
from ..utils.logging_utils import performance_monitor, AlertManager
from ..utils.config import MODELS_DIR
from models.neural_models import AutoencoderModel, CNNLSTMModel
from detection.decision_system import DecisionSystem
from utils.logging_utils import performance_monitor, AlertManager
from utils.config import MODELS_DIR

logger = logging.getLogger(__name__)

@ -368,5 +368,11 @@ class IntrusionDetector:
        
        # Carregar sistema de decisão
        decision_system_path = os.path.join(base_dir, 'decision_system.json')
        if os.path.exists(decision_system_pat
(Content truncated due to size limit. Use line ranges to read in chunks)
        if os.path.exists(decision_system_path):
            decision_system = DecisionSystem.load(decision_system_path)
        else:
            decision_system = None
            logger.warning(f"Sistema de decisão não encontrado em {decision_system_path}")
        
        # Retornar o detector de intrusão carregado
        return cls(autoencoder=autoencoder, classifier=classifier, decision_system=decision_system)
