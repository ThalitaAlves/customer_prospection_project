# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from util import get_resampling
import pickle

from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth

# Antes das APIs
colunas = ['RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents', 'IncomePerPerson', 'NumOfPastDue', 'MonthlyDebt',
       'NumOfOpenCreditLines', 'MonthlyBalance', 'age_sqr']

# Criação de uma app
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'admin'
app.config['BASIC_AUTH_PASSWORD'] = '123'

#Habilitando autenticação na app
basic_auth = BasicAuth(app)

def load_model(file_name = 'xgboost_undersampling.pkl'):
    return pickle.load(open(file_name, "rb"))

modelo = load_model('../../models/xgboost_undersampling.pkl')

# Nova rota - receber um CPF como parâmetro
@app.route('/score/', methods=['POST'])
@basic_auth.required
def score_cpf():
    # Pegar o JSON da requisiç  ão
    dados = request.get_json()
    payload = [dados[col] for col in colunas]
    payload = xgb.DMatrix([payload], feature_names=colunas)
    score = np.float64(modelo.predict(payload)[0])
    status = 'APROVADO'
    if score <= 0.3:
        status = 'REPROVADO'
    elif score <= 0.6: 
        status = 'MESA DE AVALIACAO'

    return jsonify(cpf=dados['cpf'], score=score, status=status)

# Rota padrão
@app.route('/')
def home():
    return 'API de análise está funcionando\n'

# Rota padrão
@app.route('/auth/<cpf>', methods=['POST'])
@basic_auth.required
def score_by_cpf(cpf):
    return { 'cpf': cpf }

# Subir a API
if __name__ == '__main__':
    app.run(debug=True)