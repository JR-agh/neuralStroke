import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

#   wczytanie danych
C_train_df = pd.read_csv("rawdata/train.csv")
C_test_df = pd.read_csv("rawdata/test.csv")

pypl_df = yf.download("PYPL")

#   usunięcie niepotrzebnych kolumn
C_train_df = C_train_df.drop(columns=C_train_df.columns[0:2])
C_test_df = C_test_df.drop(columns=C_test_df.columns[0:2])
pypl_df.columns = pypl_df.columns.droplevel(1)

#   usunięcie braków
C_train_df = C_train_df.dropna()
C_test_df = C_test_df.dropna()
pypl_df = pypl_df.dropna()

#   zmiana zmiennej objaśnianej na wartości liczbowe
C_train_df['satisfaction'] = C_train_df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})
C_test_df['satisfaction'] = C_test_df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})

#   oddzielenie zmiennych objaśniających od zmiennej objaśnianej
C_X_train = C_train_df.drop('satisfaction', axis=1)
C_y_train = C_train_df['satisfaction']

C_X_test = C_test_df.drop('satisfaction', axis=1)
C_y_test = C_test_df['satisfaction']

pypl_df['target'] = pypl_df['Close'].shift(-1)
pypl_df = pypl_df.dropna()
R_X = pypl_df.drop(columns=['target'])
R_y = pypl_df['target']

#   podział na zbiór testowy oraz treningowy
R_X_train, R_X_test, R_y_train, R_y_test = train_test_split(R_X, R_y, test_size=0.2, shuffle=False)

#   zmiana tekstowych zmiennych objaśniających na zmienne 0-1
C_X_train = pd.get_dummies(C_X_train)
C_X_test = pd.get_dummies(C_X_test)

trees = [5,10,50,100]
depth = [3,5,10,None]
learning_rate = [0.001, 0.01, 0.1, 1.0]
wyniki = []

for t in trees:
    for d in depth:
        rf = RandomForestClassifier(n_estimators=t, max_depth = d, random_state=42)
        rf.fit(C_X_train, C_y_train)
        accuracy = rf.score(C_X_test, C_y_test)*100
        wyniki.append({
            'liczba drzew' : t,
            'max głębokość' : d,
            'dokładność[%]' : accuracy
        })

wyniki_df = pd.DataFrame(wyniki)
wyniki_df = wyniki_df.sort_values(by="dokładność[%]",ascending=False)
wyniki_df = wyniki_df.round({
    'dokładność[%]': 2
})
wyniki_df.to_csv('Python/wyniki/RandomForestClassification.csv', index=False)
    
wyniki = []

for t in trees:
    for lr in learning_rate:
        gb = GradientBoostingClassifier(n_estimators=t, learning_rate = lr,random_state=42)
        gb.fit(C_X_train, C_y_train)
        accuracy = gb.score(C_X_test, C_y_test)*100
        wyniki.append({
            'liczba drzew' : t,
            'szybkość uczenia' : lr,
            'dokładność[%]' : accuracy
        })

wyniki_df = pd.DataFrame(wyniki)
wyniki_df = wyniki_df.sort_values(by="dokładność[%]",ascending=False)
wyniki_df = wyniki_df.round({
    'dokładność[%]': 2
})
wyniki_df.to_csv('Python/wyniki/GradientBoosterClassification.csv', index=False)

wyniki = []

for t in trees:
    for d in depth:
        rf = RandomForestRegressor(n_estimators=t, max_depth = d, random_state=42)
        rf.fit(R_X_train, R_y_train)
        y_pred = rf.predict(R_X_test)
        mae = mean_absolute_error(R_y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(R_y_test, y_pred))
        mape = mean_absolute_percentage_error(R_y_test, y_pred)*100
        wyniki.append({
            'liczba drzew' : t,
            'max głębokość' : d,
            'mae' : mae,
            'rmse' : rmse,
            'mape[%]': mape
        })

wyniki_df = pd.DataFrame(wyniki)
wyniki_df = wyniki_df.sort_values(by="mape[%]",ascending=True)
wyniki_df = wyniki_df.round({
    'mae': 2,
    'rmse': 2,
    'mape[%]': 2
})
wyniki_df.to_csv('Python/wyniki/RandomForestRegression.csv', index=False)

wyniki = []

for t in trees:
    for lr in learning_rate:
        gb = GradientBoostingRegressor(n_estimators=t, learning_rate = lr, random_state=42)
        gb.fit(R_X_train, R_y_train)
        y_pred = gb.predict(R_X_test)
        mae = mean_absolute_error(R_y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(R_y_test, y_pred))
        mape = mean_absolute_percentage_error(R_y_test, y_pred)*100
        wyniki.append({
            'liczba drzew' : t,
            'szybkość uczenia' : lr,
            'mae' : mae,
            'rmse' : rmse,
            'mape[%]': mape
        })

wyniki_df = pd.DataFrame(wyniki)
wyniki_df = wyniki_df.sort_values(by="mape[%]",ascending=True)
wyniki_df = wyniki_df.round({
    'mae': 2,
    'rmse': 2,
    'mape[%]': 2
})
wyniki_df.to_csv('Python/wyniki/GradientBoosterRegression.csv', index=False)