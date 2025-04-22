"""
Módulo de utilidades para logging e monitoramento do firewall baseado em rede neural.
"""

import logging
import logging.config
import time
import os
import json
from datetime import datetime

from .config import LOGGING_CONFIG, LOGS_DIR

# Configurar logging
os.makedirs(LOGS_DIR, exist_ok=True)
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Classe para monitorar o desempenho do firewall neural.
    Registra métricas como tempo de processamento, uso de memória, etc.
    """
    
    def __init__(self, log_file=None):
        self.log_file = log_file or os.path.join(LOGS_DIR, 'performance.log')
        self.metrics = []
        
    def start_timer(self):
        """Inicia um timer para medir o tempo de execução"""
        self.start_time = time.time()
        return self.start_time
    
    def stop_timer(self):
        """Para o timer e retorna o tempo decorrido"""
        if not hasattr(self, 'start_time'):
            logger.warning("Timer não foi iniciado")
            return 0
        
        elapsed_time = time.time() - self.start_time
        return elapsed_time
    
    def log_metric(self, component, operation, value, unit='ms', additional_info=None):
        """
        Registra uma métrica de desempenho
        
        Args:
            component: Componente sendo monitorado (ex: 'preprocessor', 'model')
            operation: Operação realizada (ex: 'load_data', 'predict')
            value: Valor da métrica
            unit: Unidade da métrica (ex: 'ms', 'MB', '%')
            additional_info: Informações adicionais (opcional)
        """
        metric = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'operation': operation,
            'value': value,
            'unit': unit
        }
        
        if additional_info:
            metric['additional_info'] = additional_info
            
        self.metrics.append(metric)
        
        # Registrar no arquivo de log
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(metric) + '\n')
        except Exception as e:
            logger.error(f"Erro ao registrar métrica: {str(e)}")
    
    def measure_operation(self, component, operation, func, *args, **kwargs):
        """
        Mede o tempo de execução de uma operação
        
        Args:
            component: Componente sendo monitorado
            operation: Operação realizada
            func: Função a ser executada
            *args, **kwargs: Argumentos para a função
            
        Returns:
            Resultado da função
        """
        self.start_timer()
        try:
            result = func(*args, **kwargs)
            elapsed_time = self.stop_timer() * 1000  # Converter para ms
            self.log_metric(component, operation, elapsed_time, 'ms')
            return result
        except Exception as e:
            elapsed_time = self.stop_timer() * 1000
            self.log_metric(component, operation, elapsed_time, 'ms', 
                           {'error': str(e)})
            raise
    
    def get_summary(self):
        """Retorna um resumo das métricas coletadas"""
        if not self.metrics:
            return "Nenhuma métrica registrada"
        
        summary = {}
        for metric in self.metrics:
            key = f"{metric['component']}_{metric['operation']}"
            if key not in summary:
                summary[key] = {
                    'count': 0,
                    'total': 0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'unit': metric['unit']
                }
            
            summary[key]['count'] += 1
            summary[key]['total'] += metric['value']
            summary[key]['min'] = min(summary[key]['min'], metric['value'])
            summary[key]['max'] = max(summary[key]['max'], metric['value'])
        
        # Calcular médias
        for key in summary:
            summary[key]['avg'] = summary[key]['total'] / summary[key]['count']
            
        return summary

class AlertManager:
    """
    Gerencia alertas gerados pelo firewall neural.
    """
    
    def __init__(self, config=None):
        from .config import RESPONSE_CONFIG
        self.config = config or RESPONSE_CONFIG['alert_config']
        self.alert_log = self.config.get('log_file')
        self.console = self.config.get('console', True)
        self.email = self.config.get('email', False)
        self.alerts = []
        
    def send_alert(self, severity, message, details=None):
        """
        Envia um alerta
        
        Args:
            severity: Severidade do alerta ('low', 'medium', 'high', 'critical')
            message: Mensagem do alerta
            details: Detalhes adicionais (opcional)
            
        Returns:
            dict: Informações sobre o alerta enviado
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'message': message,
            'details': details or {}
        }
        
        # Adicionar ID único
        alert['alert_id'] = f"{int(time.time())}_{len(self.alerts)}"
        
        # Registrar alerta
        self.alerts.append(alert)
        
        # Registrar no arquivo de log
        if self.alert_log:
            try:
                with open(self.alert_log, 'a') as f:
                    f.write(json.dumps(alert) + '\n')
            except Exception as e:
                logger.error(f"Erro ao registrar alerta: {str(e)}")
        
        # Exibir no console
        if self.console:
            severity_colors = {
                'low': '\033[94m',  # Azul
                'medium': '\033[93m',  # Amarelo
                'high': '\033[91m',  # Vermelho
                'critical': '\033[1;91m'  # Vermelho brilhante
            }
            reset_color = '\033[0m'
            
            color = severity_colors.get(severity, '')
            print(f"{color}[ALERTA - {severity.upper()}] {message}{reset_color}")
        
        # Enviar por email (implementação simplificada)
        if self.email:
            self._send_email_alert(alert)
            
        return {
            'status': 'success',
            'alert_id': alert['alert_id']
        }
    
    def _send_email_alert(self, alert):
        """Envia alerta por email (implementação simplificada)"""
        # Esta é uma implementação simulada
        logger.info(f"Simulando envio de email para alerta: {alert['alert_id']}")
        # Em uma implementação real, usaria smtplib ou uma API de email
        
    def get_recent_alerts(self, count=10, severity=None):
        """
        Retorna os alertas mais recentes
        
        Args:
            count: Número de alertas a retornar
            severity: Filtrar por severidade (opcional)
            
        Returns:
            list: Alertas recentes
        """
        if severity:
            filtered_alerts = [a for a in self.alerts if a['severity'] == severity]
        else:
            filtered_alerts = self.alerts
            
        return sorted(filtered_alerts, key=lambda x: x['timestamp'], reverse=True)[:count]

# Instância global do monitor de desempenho
performance_monitor = PerformanceMonitor()
