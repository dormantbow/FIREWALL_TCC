"""
Módulo de configuração para o firewall baseado em rede neural.
Contém configurações globais e parâmetros para os diferentes módulos.
"""

import os
import json

# Diretórios base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Criar diretórios se não existirem
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configurações dos datasets
DATASET_CONFIG = {
    'nsl_kdd': {
        'train_path': os.path.join(DATA_DIR, 'NSL-KDD', 'KDDTrain+.txt'),
        'test_path': os.path.join(DATA_DIR, 'NSL-KDD', 'KDDTest+.txt'),
        'column_names': [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
            'num_access_files', 'num_outbound_cmds', 'is_host_login', 
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
            'dst_host_srv_count', 'dst_host_same_srv_rate', 
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
            'dst_host_srv_rerror_rate', 'label', 'difficulty'
        ],
        'categorical_columns': ['protocol_type', 'service', 'flag'],
        'label_column': 'label',
        'attack_map': {
            'normal': 0,
            'dos': 1,
            'probe': 2,
            'r2l': 3,
            'u2r': 4
        },
        'attack_categories': {
            'normal': ['normal'],
            'dos': ['neptune', 'back', 'land', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm', 'processtable', 'worm', 'mailbomb'],
            'probe': ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint'],
            'r2l': ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy', 'xlock', 'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named'],
            'u2r': ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']
        }
    },
    'cicids_2018': {
        'files': [
            os.path.join(DATA_DIR, 'CICIDS-2018', 'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv'),
            os.path.join(DATA_DIR, 'CICIDS-2018', 'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv'),
            os.path.join(DATA_DIR, 'CICIDS-2018', 'Friday-23-02-2018_TrafficForML_CICFlowMeter.csv'),
            os.path.join(DATA_DIR, 'CICIDS-2018', 'Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv'),
            os.path.join(DATA_DIR, 'CICIDS-2018', 'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv'),
            os.path.join(DATA_DIR, 'CICIDS-2018', 'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv'),
            os.path.join(DATA_DIR, 'CICIDS-2018', 'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv'),
            os.path.join(DATA_DIR, 'CICIDS-2018', 'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv'),
            os.path.join(DATA_DIR, 'CICIDS-2018', 'Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv')
        ],
        'label_column': 'Label',
        'attack_map': {
            'Benign': 0,
            'DoS': 1,
            'PortScan': 2,
            'BruteForce': 3,
            'WebAttack': 4,
            'Botnet': 5,
            'Infiltration': 6
        },
        'attack_categories': {
            'Benign': ['Benign'],
            'DoS': ['DoS attacks-GoldenEye', 'DoS attacks-Slowloris', 'DoS attacks-SlowHTTPTest', 'DoS attacks-Hulk', 'DDOS attack-LOIC-HTTP'],
            'PortScan': ['FTP-BruteForce', 'SSH-Bruteforce'],
            'BruteForce': ['Brute Force', 'Brute Force -Web', 'Brute Force -XSS'],
            'WebAttack': ['SQL Injection', 'XSS'],
            'Botnet': ['Bot'],
            'Infiltration': ['Infiltration']
        }
    }
}

# Configurações de pré-processamento
PREPROCESSING_CONFIG = {
    'nsl_kdd': {
        'numerical_columns': [
            'duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent',
            'hot', 'num_failed_logins', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ],
        'binary_columns': [
            'land', 'logged_in', 'is_host_login', 'is_guest_login'
        ],
        'normalization': 'standard',  # 'standard', 'minmax', ou 'robust'
        'handle_missing': 'mean',     # 'mean', 'median', ou 'most_frequent'
        'feature_selection': True,
        'n_features': 25,             # Número de características a selecionar
        'balance_method': 'smote'     # 'smote', 'adasyn', 'none'
    },
    'cicids_2018': {
        'drop_columns': ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'],
        'normalization': 'standard',
        'handle_missing': 'mean',
        'feature_selection': True,
        'n_features': 30,
        'balance_method': 'smote'
    }
}

# Configurações dos modelos
MODEL_CONFIG = {
    'autoencoder': {
        'architecture': [
            {'units': 128, 'activation': 'relu'},
            {'units': 64, 'activation': 'relu'},
            {'units': 32, 'activation': 'relu'},
            {'units': 16, 'activation': 'relu'},
            {'units': 8, 'activation': 'relu'},
            {'units': 16, 'activation': 'relu'},
            {'units': 32, 'activation': 'relu'},
            {'units': 64, 'activation': 'relu'},
            {'units': 128, 'activation': 'relu'}
        ],
        'optimizer': 'adam',
        'loss': 'mse',
        'batch_size': 64,
        'epochs': 50,
        'validation_split': 0.2,
        'early_stopping': True,
        'patience': 5,
        'anomaly_threshold_percentile': 95  # Percentil para definir o threshold de anomalia
    },
    'cnn_lstm': {
        'sequence_length': 10,
        'cnn_layers': [
            {'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
            {'filters': 128, 'kernel_size': 3, 'activation': 'relu'}
        ],
        'lstm_units': 100,
        'dense_layers': [
            {'units': 64, 'activation': 'relu', 'dropout': 0.5},
            {'units': 32, 'activation': 'relu', 'dropout': 0.3}
        ],
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
        'batch_size': 64,
        'epochs': 50,
        'validation_split': 0.2,
        'early_stopping': True,
        'patience': 5
    }
}

# Configurações do sistema de decisão
DECISION_SYSTEM_CONFIG = {
    'weights': {
        'anomaly': 0.6,
        'classification': 0.4
    },
    'classification_threshold': 0.7,
    'update_frequency': 10,
    'learning_rate': 0.1
}

# Configurações de resposta
RESPONSE_CONFIG = {
    'policies': {
        'conservative': {
            'description': 'Política conservadora com foco em minimizar falsos positivos',
            'actions': [
                {'action': 'log', 'min_threat_level': 0, 'max_threat_level': 10},
                {'action': 'alert', 'min_threat_level': 3, 'max_threat_level': 10, 
                 'params': {'severity': 'low'}},
                {'action': 'limit_traffic', 'min_threat_level': 6, 'max_threat_level': 10, 
                 'params': {'rate': '500kbps', 'duration': 300}}
            ]
        },
        'moderate': {
            'description': 'Política moderada com equilíbrio entre detecção e falsos positivos',
            'actions': [
                {'action': 'log', 'min_threat_level': 0, 'max_threat_level': 10},
                {'action': 'alert', 'min_threat_level': 2, 'max_threat_level': 10, 
                 'params': {'severity': 'medium'}},
                {'action': 'limit_traffic', 'min_threat_level': 5, 'max_threat_level': 10, 
                 'params': {'rate': '200kbps', 'duration': 600}},
                {'action': 'block_ip', 'min_threat_level': 7, 'max_threat_level': 10, 
                 'params': {'duration': 1800}}
            ]
        },
        'aggressive': {
            'description': 'Política agressiva com foco em máxima proteção',
            'actions': [
                {'action': 'log', 'min_threat_level': 0, 'max_threat_level': 10},
                {'action': 'alert', 'min_threat_level': 1, 'max_threat_level': 10, 
                 'params': {'severity': 'high'}},
                {'action': 'limit_traffic', 'min_threat_level': 3, 'max_threat_level': 6, 
                 'params': {'rate': '100kbps', 'duration': 900}},
                {'action': 'block_ip', 'min_threat_level': 5, 'max_threat_level': 10, 
                 'params': {'duration': 3600}},
                {'action': 'quarantine', 'min_threat_level': 8, 'max_threat_level': 10, 
                 'params': {'duration': 7200}}
            ]
        }
    },
    'default_policy': 'moderate',
    'alert_config': {
        'email': False,
        'log_file': os.path.join(LOGS_DIR, 'alerts.log'),
        'console': True
    },
    'firewall_config': {
        'interface': 'eth0',
        'rules_file': os.path.join(LOGS_DIR, 'firewall_rules.json')
    }
}

# Configurações de logging
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': os.path.join(LOGS_DIR, 'firewall.log'),
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}

def save_config(config_dict, filename):
    """Salva configuração em arquivo JSON"""
    with open(filename, 'w') as f:
        json.dump(config_dict, f, indent=4)

def load_config(filename):
    """Carrega configuração de arquivo JSON"""
    with open(filename, 'r') as f:
        return json.load(f)
