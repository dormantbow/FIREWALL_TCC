#!/bin/bash
# Script para executar todos os testes do firewall neural

# Configurar ambiente
echo "Configurando ambiente de teste..."
mkdir -p logs
mkdir -p visualizations

# Executar testes de pré-processamento
echo "Executando testes de pré-processamento..."
python3 tests/test_preprocessing.py > logs/preprocessing_test.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Testes de pré-processamento concluídos com sucesso"
else
    echo "✗ Falha nos testes de pré-processamento. Verifique logs/preprocessing_test.log"
fi

# Executar testes de modelos neurais
echo "Executando testes de modelos neurais..."
python3 tests/test_neural_models.py > logs/neural_models_test.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Testes de modelos neurais concluídos com sucesso"
else
    echo "✗ Falha nos testes de modelos neurais. Verifique logs/neural_models_test.log"
fi

# Executar testes de detecção de intrusão
echo "Executando testes de detecção de intrusão..."
python3 tests/test_intrusion_detection.py > logs/intrusion_detection_test.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Testes de detecção de intrusão concluídos com sucesso"
else
    echo "✗ Falha nos testes de detecção de intrusão. Verifique logs/intrusion_detection_test.log"
fi

# Executar testes do sistema completo
echo "Executando testes do sistema completo..."
python3 tests/test_firewall_system.py > logs/firewall_system_test.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Testes do sistema completo concluídos com sucesso"
else
    echo "✗ Falha nos testes do sistema completo. Verifique logs/firewall_system_test.log"
fi

echo "Todos os testes foram executados. Verifique os logs e visualizações para análise detalhada."
