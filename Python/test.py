# 1. n_estimators
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

#   Analizowane parametry - Random Forest
trees = [5,10,50,100]
depth = [3,5,10,None]
min_samples_split = [2, 5, 10, 20]
min_samples_leaf = [1, 2, 5, 10]

wyniki_trees = []
wyniki_depth = []
wyniki_split = []
wyniki_leaf = []

# 1. n_estimators
for t in trees:
    rf = RandomForestClassifier(n_estimators=t, random_state=42)
    rf.fit(C_X_train, C_y_train)
    wyniki_trees.append({
        'n_estimators': t,
        'max_depth': '-',
        'min_samples_split': '-',
        'min_samples_leaf': '-',
        'accuracy[%]': rf.score(C_X_test, C_y_test)*100
    })

df_trees = pd.DataFrame(wyniki_trees)
best_trees = int(df_trees.loc[df_trees['accuracy[%]'].idxmax(), 'n_estimators'])

# 2. max_depth
for d in depth:
    rf = RandomForestClassifier(n_estimators=best_trees, max_depth=d, random_state=42)
    rf.fit(C_X_train, C_y_train)
    wyniki_depth.append({
        'n_estimators': best_trees,
        'max_depth': str(d) if d is not None else 'None',
        'min_samples_split': '-',
        'min_samples_leaf': '-',
        'accuracy[%]': rf.score(C_X_test, C_y_test)*100
    })

df_depth = pd.DataFrame(wyniki_depth)
best_depth_val = df_depth.loc[df_depth['accuracy[%]'].idxmax(), 'max_depth']
best_depth = None if best_depth_val == 'None' else int(best_depth_val)

# 3. min_samples_split
for ss in min_samples_split:
    rf = RandomForestClassifier(n_estimators=best_trees, max_depth=best_depth, min_samples_split=ss, random_state=42)
    rf.fit(C_X_train, C_y_train)
    wyniki_split.append({
        'n_estimators': best_trees,
        'max_depth': best_depth_val,
        'min_samples_split': ss,
        'min_samples_leaf': '-',
        'accuracy[%]': rf.score(C_X_test, C_y_test)*100
    })

df_split = pd.DataFrame(wyniki_split)
best_split = int(df_split.loc[df_split['accuracy[%]'].idxmax(), 'min_samples_split'])

# 4. min_samples_leaf
for sl in min_samples_leaf:
    rf = RandomForestClassifier(n_estimators=best_trees, max_depth=best_depth, min_samples_split=best_split, min_samples_leaf=sl, random_state=42)
    rf.fit(C_X_train, C_y_train)
    wyniki_leaf.append({
        'n_estimators': best_trees,
        'max_depth': best_depth_val,
        'min_samples_split': best_split,
        'min_samples_leaf': sl,
        'accuracy[%]': rf.score(C_X_test, C_y_test)*100
    })

rf_c_df = pd.concat([
    pd.DataFrame(wyniki_trees),
    pd.DataFrame(wyniki_depth),
    pd.DataFrame(wyniki_split),
    pd.DataFrame(wyniki_leaf)
], ignore_index=True)
rf_c_df = rf_c_df.round({'accuracy[%]': 2})
rf_c_df.to_csv('Python/wyniki/TestRandomForestClassification.csv', index=False)