"""
Módulo para download e carregamento dos datasets NSL-KDD e CICIDS-2018.
"""

import os
import logging
import pandas as pd
import numpy as np
import requests
import zipfile
import tarfile
import gzip
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import DATA_DIR, DATASET_CONFIG

from utils.logging_utils import performance_monitor
#from tcc_firewall.src.utils.logging_utils import performance_monitor


logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Classe para download e carregamento dos datasets NSL-KDD e CICIDS-2018.
    """
    
    def __init__(self):
        self.nsl_kdd_config = DATASET_CONFIG['nsl_kdd']
        self.cicids_config = DATASET_CONFIG['cicids_2018']
        
        # Criar diretórios para os datasets
        os.makedirs(os.path.join(DATA_DIR, 'NSL-KDD'), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, 'CICIDS-2018'), exist_ok=True)
    
    def download_nsl_kdd(self, force=False):
        """
        Faz o download do dataset NSL-KDD se não estiver presente.
        
        Args:
            force: Se True, força o download mesmo se os arquivos já existirem
            
        Returns:
            bool: True se o download foi bem-sucedido, False caso contrário
        """
        train_path = self.nsl_kdd_config['train_path']
        test_path = self.nsl_kdd_config['test_path']
        
        # Verificar se os arquivos já existem
        if not force and os.path.exists(train_path) and os.path.exists(test_path):
            logger.info("Dataset NSL-KDD já existe. Pulando download.")
            return True
        
        # URLs para download
        train_url = "https://iscxdownloads.cs.unb.ca/iscxdownloads/NSL-KDD/KDDTrain+.txt"
        test_url = "https://iscxdownloads.cs.unb.ca/iscxdownloads/NSL-KDD/KDDTest+.txt"
        
        try:
            logger.info("Baixando dataset NSL-KDD...")
            
            # Download do conjunto de treinamento
            self._download_file(train_url, train_path)
            
            # Download do conjunto de teste
            self._download_file(test_url, test_path)
            
            logger.info("Download do dataset NSL-KDD concluído com sucesso.")
            return True
        
        except Exception as e:
            logger.error(f"Erro ao baixar dataset NSL-KDD: {str(e)}")
            return False
    
    def download_cicids(self, force=False):
        """
        Faz o download do dataset CICIDS-2018 se não estiver presente.
        
        Args:
            force: Se True, força o download mesmo se os arquivos já existirem
            
        Returns:
            bool: True se o download foi bem-sucedido, False caso contrário
        """
        # Verificar se os arquivos já existem
        files_exist = all(os.path.exists(file_path) for file_path in self.cicids_config['files'])
        
        if not force and files_exist:
            logger.info("Dataset CICIDS-2018 já existe. Pulando download.")
            return True
        
        # URL base para download
        base_url = "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2018/"
        
        # Arquivos a serem baixados
        files = [
            "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
            "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
            "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
            "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
            "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
            "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
            "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
            "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
            "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"
        ]
        
        try:
            logger.info("Baixando dataset CICIDS-2018...")
            
            for file in files:
                url = base_url + file
                file_path = os.path.join(DATA_DIR, 'CICIDS-2018', file)
                
                if not force and os.path.exists(file_path):
                    logger.info(f"Arquivo {file} já existe. Pulando download.")
                    continue
                
                self._download_file(url, file_path)
            
            logger.info("Download do dataset CICIDS-2018 concluído com sucesso.")
            return True
        
        except Exception as e:
            logger.error(f"Erro ao baixar dataset CICIDS-2018: {str(e)}")
            return False
    
    def _download_file(self, url, file_path):
        """
        Faz o download de um arquivo com barra de progresso.
        
        Args:
            url: URL do arquivo
            file_path: Caminho onde o arquivo será salvo
        """
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        with open(file_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(file_path)) as pbar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))
    
    def load_nsl_kdd(self, subset='train'):
        """
        Carrega o dataset NSL-KDD.
        
        Args:
            subset: 'train' ou 'test'
            
        Returns:
            tuple: (X, y) com dados e rótulos
        """
        logger.info(f"Carregando dataset NSL-KDD ({subset})...")
        
        # Selecionar caminho do arquivo
        if subset == 'train':
            file_path = self.nsl_kdd_config['train_path']
        elif subset == 'test':
            file_path = self.nsl_kdd_config['test_path']
        else:
            raise ValueError("O parâmetro 'subset' deve ser 'train' ou 'test'")
        
        # Verificar se o arquivo existe
        if not os.path.exists(file_path):
            logger.warning(f"Arquivo {file_path} não encontrado. Tentando fazer o download...")
            success = self.download_nsl_kdd()
            if not success:
                raise FileNotFoundError(f"Não foi possível baixar o dataset NSL-KDD")
        
        # Carregar o dataset
        start_time = performance_monitor.start_timer()
        
        try:    
            # Carregar o dataset com os nomes das colunas
            df = pd.read_csv(file_path, header=None, names=self.nsl_kdd_config['column_names'])
            
            # Mapear rótulos para categorias
            df['attack_category'] = df['label'].apply(self._map_nsl_kdd_attack_to_category)
            
            # Mapear categorias para números
            df['attack_class'] = df['attack_category'].map(self.nsl_kdd_config['attack_map'])
            
            # Separar features e rótulos
            X = df.drop(['label', 'difficulty', 'attack_category', 'attack_class'], axis=1)
            # Codificar colunas categóricas (ex: protocol_type, service, flag)
            categorical_cols = X.select_dtypes(include=['object']).columns
            X = pd.get_dummies(X, columns=categorical_cols)
            
            y = df['attack_class']
            
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('dataset_loader', f'load_nsl_kdd_{subset}', 
                                          elapsed_time * 1000, 'ms', 
                                          {'rows': len(df), 'columns': len(df.columns)})
            
            logger.info(f"Dataset NSL-KDD ({subset}) carregado com sucesso: {len(df)} amostras")
            return X, y
        
        except Exception as e:
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('dataset_loader', f'load_nsl_kdd_{subset}', 
                                          elapsed_time * 1000, 'ms', 
                                          {'error': str(e)})
            logger.error(f"Erro ao carregar dataset NSL-KDD ({subset}): {str(e)}")
            raise
    
    def _map_nsl_kdd_attack_to_category(self, attack_name):
        """
        Mapeia o nome do ataque para sua categoria.
        
        Args:
            attack_name: Nome do ataque
            
        Returns:
            str: Categoria do ataque
        """
        attack_categories = self.nsl_kdd_config['attack_categories']
        
        for category, attacks in attack_categories.items():
            if attack_name.lower() in [a.lower() for a in attacks]:
                return category
        
        # Se não encontrar, retorna 'unknown'
        logger.warning(f"Ataque desconhecido: {attack_name}")
        return 'unknown'
    
    def load_cicids(self, sample_size=None, file_names=None, max_files=None):
        """
        Carrega o dataset CICIDS-2018 a partir dos arquivos disponíveis no diretório.

        Args:
            sample_size: Tamanho da amostra a ser carregada por arquivo (opcional)
            file_names: Lista de nomes de arquivos específicos a carregar (opcional)
            max_files: Número máximo de arquivos a carregar (opcional)

        Returns:
            tuple: (X, y) com dados e rótulos
        """
        logger.info("Carregando dataset CICIDS-2018...")

        data_dir = os.path.join(DATA_DIR, 'CICIDS-2018')

        # Obter lista de arquivos CSV no diretório
        available_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                        if f.endswith('.csv') and os.path.isfile(os.path.join(data_dir, f))]

        if not available_files:
            logger.warning("Nenhum arquivo CSV encontrado no diretório CICIDS-2018.")
            raise FileNotFoundError("Nenhum arquivo CICIDS-2018 disponível para carregar.")

        # Selecionar arquivos com base nos parâmetros
        if file_names:
            selected_files = [os.path.join(data_dir, f) for f in file_names if os.path.join(data_dir, f) in available_files]
        elif max_files:
            selected_files = available_files[:max_files]
        else:
            selected_files = available_files

        logger.info(f"{len(selected_files)} arquivos serão carregados.")

        # Carregamento com temporizador
        start_time = performance_monitor.start_timer()

        try:
            dfs = []

            for file_path in selected_files:
                logger.info(f"Carregando arquivo {os.path.basename(file_path)}...")
                df = pd.read_csv(file_path, low_memory=False)

                if sample_size and len(df) > sample_size:
                    df = df.sample(sample_size, random_state=42)

                dfs.append(df)

            if not dfs:
                raise ValueError("Nenhum arquivo válido foi carregado.")

            df = pd.concat(dfs, ignore_index=True)

            # Mapear rótulos
            df['attack_category'] = df[self.cicids_config['label_column']].apply(self._map_cicids_attack_to_category)
            df['attack_class'] = df['attack_category'].map(self.cicids_config['attack_map'])

            drop_columns = self.cicids_config.get('drop_columns', []) + [self.cicids_config['label_column'], 'attack_category']
            X = df.drop(drop_columns, axis=1, errors='ignore')
            y = df['attack_class']

            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('dataset_loader', 'load_cicids', 
                                        elapsed_time * 1000, 'ms', 
                                        {'rows': len(df), 'columns': len(df.columns)})

            logger.info(f"Dataset CICIDS-2018 carregado com sucesso: {len(df)} amostras")
            return X, y

        except Exception as e:
            elapsed_time = performance_monitor.stop_timer()
            performance_monitor.log_metric('dataset_loader', 'load_cicids', 
                                        elapsed_time * 1000, 'ms', 
                                        {'error': str(e)})
            logger.error(f"Erro ao carregar dataset CICIDS-2018: {str(e)}")
            raise
    
    def _map_cicids_attack_to_category(self, attack_name):
        """
        Mapeia o nome do ataque para sua categoria.
        
        Args:
            attack_name: Nome do ataque
            
        Returns:
            str: Categoria do ataque
        """
        attack_categories = self.cicids_config['attack_categories']
        
        for category, attacks in attack_categories.items():
            if attack_name in attacks:
                return category
        
        # Se não encontrar, retorna 'unknown'
        logger.warning(f"Ataque desconhecido: {attack_name}")
        return 'unknown'

# Instância global do carregador de datasets
dataset_loader = DatasetLoader()
