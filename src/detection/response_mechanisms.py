"""
Módulo para implementação dos mecanismos de resposta a ataques detectados.
"""

import os
import logging
import time
import json
import subprocess
from datetime import datetime

from utils.config import RESPONSE_CONFIG
from utils.logging_utils import performance_monitor, AlertManager

logger = logging.getLogger(__name__)

class FirewallInterface:
    """
    Interface para interação com o firewall do sistema.
    Implementa regras de firewall dinâmicas usando iptables.
    """
    
    def __init__(self, config=None):
        """
        Inicializa a interface do firewall.
        
        Args:
            config: Configuração do firewall (opcional)
        """
        from utils.config import RESPONSE_CONFIG
        self.config = config or RESPONSE_CONFIG['firewall_config']
        self.interface = self.config.get('interface', 'eth0')
        self.rules_file = self.config.get('rules_file')
        self.rules = []
        
        # Carregar regras existentes
        if self.rules_file and os.path.exists(self.rules_file):
            try:
                with open(self.rules_file, 'r') as f:
                    self.rules = json.load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar regras do firewall: {str(e)}")
    
    def add_rule(self, rule):
        """
        Adiciona uma regra ao firewall.
        
        Args:
            rule: Dicionário com a regra a ser adicionada
            
        Returns:
            dict: Informações sobre a regra adicionada
        """
        logger.info(f"Adicionando regra de firewall: {rule}")
        
        # Iniciar timer
        start_time = performance_monitor.start_timer()
        
        try:
            # Validar regra
            if 'action' not in rule:
                raise ValueError("A regra deve conter uma ação (DROP, LIMIT, etc.)")
            
            if 'source' not in rule and 'destination' not in rule:
                raise ValueError("A regra deve conter origem ou destino")
            
            # Gerar ID único para a regra
            rule_id = f"rule_{int(time.time())}_{len(self.rules)}"
            
            # Adicionar informações adicionais
            rule_info = rule.copy()
            rule_info['rule_id'] = rule_id
            rule_info['created_at'] = datetime.now().isoformat()
            rule_info['expires_at'] = datetime.fromtimestamp(
                time.time() + rule.get('duration', 3600)
            ).isoformat() if 'duration' in rule else None
            
            # Implementar regra no sistema
            success = self._implement_rule(rule)
            
            if success:
                # Adicionar à lista de regras
                self.rules.append(rule_info)
                
                # Salvar regras
                if self.rules_file:
                    os.makedirs(os.path.dirname(self.rules_file), exist_ok=True)
                    with open(self.rules_file, 'w') as f:
                        json.dump(self.rules, f, indent=4)
                
                elapsed_time = performance_monitor.stop_timer()
                performance_monitor.log_metric('firewall', 'add_rule', 
                                              elapsed_time * 1000, 'ms', 
                                              {'rule_id': rule_id, 'action': rule['action']})
                
                logger.info(f"Regra adicionada com sucesso: {rule_id}")
                return {'status': 'success', 'rule_id': rule_id}
            else:
                elapsed_time = performance_monitor.stop_timer()
                performance_monitor.log_metric('firewall', 'add_rule', 
                                              elapsed_time * 1000, 'ms', 
                                              {'error': 'Falha ao implementar regra'})
                
                logger.error(f"Falha ao implementar regra: {rule}")
                return {'status': 'error', 'message': 'Falha ao implementar regra'}
        
        except Exception as e:
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('firewall', 'add_rule', 
                                          elapsed_time * 1000, 'ms', 
                                          {'error': str(e)})
            logger.error(f"Erro ao adicionar regra: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _implement_rule(self, rule):
        """
        Implementa uma regra no firewall do sistema.
        
        Args:
            rule: Dicionário com a regra a ser implementada
            
        Returns:
            bool: True se a regra foi implementada com sucesso, False caso contrário
        """
        # Nota: Esta é uma implementação simulada para fins de protótipo
        # Em um ambiente real, seria necessário usar comandos iptables reais
        
        logger.info(f"Simulando implementação de regra: {rule}")
        
        # Simular comando iptables
        action = rule['action']
        source = rule.get('source', '')
        destination = rule.get('destination', '')
        
        if action == 'DROP':
            # Simular bloqueio de IP
            command = f"iptables -A INPUT -s {source} -j DROP"
            logger.info(f"Simulando comando: {command}")
            # Em um ambiente real: subprocess.run(command, shell=True, check=True)
            return True
        
        elif action == 'LIMIT':
            # Simular limitação de taxa
            rate = rule.get('rate', '100kbps')
            command = f"iptables -A INPUT -s {source} -m limit --limit {rate} -j ACCEPT"
            logger.info(f"Simulando comando: {command}")
            # Em um ambiente real: subprocess.run(command, shell=True, check=True)
            return True
        
        elif action == 'REDIRECT':
            # Simular redirecionamento
            command = f"iptables -t nat -A PREROUTING -s {source} -j DNAT --to-destination {destination}"
            logger.info(f"Simulando comando: {command}")
            # Em um ambiente real: subprocess.run(command, shell=True, check=True)
            return True
        
        else:
            logger.warning(f"Ação desconhecida: {action}")
            return False
    
    def remove_rule(self, rule_id):
        """
        Remove uma regra do firewall.
        
        Args:
            rule_id: ID da regra a ser removida
            
        Returns:
            dict: Informações sobre a operação
        """
        logger.info(f"Removendo regra de firewall: {rule_id}")
        
        # Iniciar timer
        start_time = performance_monitor.start_timer()
        
        try:
            # Encontrar regra
            rule_index = None
            rule = None
            
            for i, r in enumerate(self.rules):
                if r.get('rule_id') == rule_id:
                    rule_index = i
                    rule = r
                    break
            
            if rule_index is None:
                logger.warning(f"Regra não encontrada: {rule_id}")
                return {'status': 'error', 'message': 'Regra não encontrada'}
            
            # Remover regra do sistema
            success = self._remove_rule_from_system(rule)
            
            if success:
                # Remover da lista de regras
                self.rules.pop(rule_index)
                
                # Salvar regras
                if self.rules_file:
                    with open(self.rules_file, 'w') as f:
                        json.dump(self.rules, f, indent=4)
                
                elapsed_time = performance_monitor.stop_timer()
                performance_monitor.log_metric('firewall', 'remove_rule', 
                                              elapsed_time * 1000, 'ms', 
                                              {'rule_id': rule_id})
                
                logger.info(f"Regra removida com sucesso: {rule_id}")
                return {'status': 'success', 'rule_id': rule_id}
            else:
                elapsed_time = performance_monitor.stop_timer()
                performance_monitor.log_metric('firewall', 'remove_rule', 
                                              elapsed_time * 1000, 'ms', 
                                              {'error': 'Falha ao remover regra do sistema'})
                
                logger.error(f"Falha ao remover regra do sistema: {rule_id}")
                return {'status': 'error', 'message': 'Falha ao remover regra do sistema'}
        
        except Exception as e:
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('firewall', 'remove_rule', 
                                          elapsed_time * 1000, 'ms', 
                                          {'error': str(e)})
            logger.error(f"Erro ao remover regra: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _remove_rule_from_system(self, rule):
        """
        Remove uma regra do firewall do sistema.
        
        Args:
            rule: Dicionário com a regra a ser removida
            
        Returns:
            bool: True se a regra foi removida com sucesso, False caso contrário
        """
        # Nota: Esta é uma implementação simulada para fins de protótipo
        
        logger.info(f"Simulando remoção de regra: {rule}")
        
        # Simular comando iptables
        action = rule['action']
        source = rule.get('source', '')
        
        if action == 'DROP':
            command = f"iptables -D INPUT -s {source} -j DROP"
            logger.info(f"Simulando comando: {command}")
            # Em um ambiente real: subprocess.run(command, shell=True, check=True)
            return True
        
        elif action == 'LIMIT':
            rate = rule.get('rate', '100kbps')
            command = f"iptables -D INPUT -s {source} -m limit --limit {rate} -j ACCEPT"
            logger.info(f"Simulando comando: {command}")
            # Em um ambiente real: subprocess.run(command, shell=True, check=True)
            return True
        
        elif action == 'REDIRECT':
            destination = rule.get('destination', '')
            command = f"iptables -t nat -D PREROUTING -s {source} -j DNAT --to-destination {destination}"
            logger.info(f"Simulando comando: {command}")
            # Em um ambiente real: subprocess.run(command, shell=True, check=True)
            return True
        
        else:
            logger.warning(f"Ação desconhecida: {action}")
            return False
    
    def cleanup_expired_rules(self):
        """
        Remove regras expiradas do firewall.
        
        Returns:
            int: Número de regras removidas
        """
        logger.info("Limpando regras expiradas do firewall")
        
        # Iniciar timer
        start_time = performance_monitor.start_timer()
        
        try:
            now = datetime.now().isoformat()
            expired_rules = []
            
            # Identificar regras expiradas
            for rule in self.rules:
                if rule.get('expires_at') and rule['expires_at'] < now:
                    expired_rules.append(rule)
            
            # Remover regras expiradas
            removed_count = 0
            for rule in expired_rules:
                result = self.remove_rule(rule['rule_id'])
                if result['status'] == 'success':
                    removed_count += 1
            
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('firewall', 'cleanup_expired_rules', 
                                          elapsed_time * 1000, 'ms', 
                                          {'removed_count': removed_count})
            
            logger.info(f"Limpeza concluída: {removed_count} regras removidas")
            return removed_count
        
        except Exception as e:
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('firewall', 'cleanup_expired_rules', 
                                          elapsed_time * 1000, 'ms', 
                                          {'error': str(e)})
            logger.error(f"Erro ao limpar regras expiradas: {str(e)}")
            return 0
    
    def enable_intensive_monitoring(self, ip, duration=1800):
        """
        Ativa monitoramento intensivo para um IP suspeito.
        
        Args:
            ip: Endereço IP a ser monitorado
            duration: Duração do monitoramento em segundos
            
        Returns:
            dict: Informações sobre a operação
        """
        logger.info(f"Ativando monitoramento intensivo para IP: {ip}")
        
        # Nota: Esta é uma implementação simulada para fins de protótipo
        
        # Em um ambiente real, isso poderia envolver:
        # 1. Configurar captura de pacotes detalhada
        # 2. Aumentar nível de logging
        # 3. Configurar alertas específicos
        
        # Simular ativação de monitoramento
        logger.info(f"Monitoramento intensivo ativado para {ip} por {duration} segundos")
        
        return {
            'status': 'success',
            'ip': ip,
            'duration': duration,
            'monitoring_id': f"mon_{int(time.time())}"
        }

class ResponseExecutor:
    """
    Executor de respostas a ataques detectados.
    """
    
    def __init__(self, config=None):
        """
        Inicializa o executor de respostas.
        
        Args:
            config: Configuração de respostas (opcional)
        """
        self.config = config or RESPONSE_CONFIG
        self.firewall = FirewallInterface(self.config.get('firewall_config'))
        self.alert_manager = AlertManager(self.config.get('alert_config'))
        self.execution_log = []
    
    def execute_response(self, response_actions, response_params, detection_data):
        """
        Executa as ações de resposta selecionadas.
        
        Args:
            response_actions: Lista de ações a serem executadas
            response_params: Parâmetros para as ações
            detection_data: Dados da detecção
            
        Returns:
            dict: Resultados da execução das ações
        """
        logger.info(f"Executando resposta: {response_actions}")
        
        # Iniciar timer
        start_time = performance_monitor.start_timer()
        
        try:
            execution_results = {}
            
            for action in response_actions:
                params = response_params.get(action, {})
                
                try:
                    if action == 'block_ip':
                        result = self._block_ip(params, detection_data)
                    elif action == 'limit_traffic':
                        result = self._limit_traffic(params, detection_data)
                    elif action == 'alert':
                        result = self._send_alert(params, detection_data)
                    elif action == 'log':
                        result = self._log_attack(params, detection_data)
                    elif action == 'quarantine':
                        result = self._quarantine_connection(params, detection_data)
                    else:
                        result = {'status': 'error', 'message': f'Ação desconhecida: {action}'}
                    
                    execution_results[action] = result
                except Exception as e:
                    execution_results[action] = {
                        'status': 'error',
                        'message': f'Erro ao executar ação: {str(e)}'
                    }
            
            # Registrar execução
            execution_info = {
                'timestamp': time.time(),
                'actions': response_actions,
                'params': response_params,
                'results': execution_results,
            }
        except Exception as e:
            execution_results[action] = {
                'status': 'error',
                'message': f'Erro ao executar ação {action}: {str(e)}'
            }
            logger.error(f"Erro ao executar a ação {action}: {e}")

class ResponseSelector:
    def __init__(self):
        print("[ResponseSelector] Inicializado")

    def select_response(self, traffic_info):
        print("[ResponseSelector] Dados recebidos:")
        print(traffic_info)

        # Aqui você pode colocar lógicas no futuro
        # Exemplo de estrutura esperada:
        # {'ip': '192.168.0.10', 'score': 0.87, 'features': [...], 'attack_type': 'DDoS'}

        # Por enquanto só retorna uma ação padrão
        return "LIMIT"
