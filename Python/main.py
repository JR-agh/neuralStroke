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
rf_c_df.to_csv('Python/wyniki/RandomForestClassification.csv', index=False)

# Regresja - Random Forest

wyniki_trees = []
wyniki_depth = []
wyniki_split = []
wyniki_leaf  = []

wyniki_trees = []
wyniki_depth = []
wyniki_split = []
wyniki_leaf  = []

# 1. n_estimators
for t in trees:
    rf = RandomForestRegressor(n_estimators=t, random_state=42)
    rf.fit(R_X_train, R_y_train)
    y_pred = rf.predict(R_X_test)
    wyniki_trees.append({
        'n_estimators': t,
        'max_depth': '-',
        'min_samples_split': '-',
        'min_samples_leaf': '-',
        'mae': round(mean_absolute_error(R_y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(R_y_test, y_pred)), 2),
        'mape[%]': round(mean_absolute_percentage_error(R_y_test, y_pred)*100, 2)
    })

df_trees = pd.DataFrame(wyniki_trees)
best_trees = int(df_trees.loc[df_trees['mape[%]'].idxmin(), 'n_estimators'])

# 2. max_depth
for d in depth:
    rf = RandomForestRegressor(n_estimators=best_trees, max_depth=d, random_state=42)
    rf.fit(R_X_train, R_y_train)
    y_pred = rf.predict(R_X_test)
    wyniki_depth.append({
        'n_estimators': best_trees,
        'max_depth': str(d) if d is not None else 'None',
        'min_samples_split': '-',
        'min_samples_leaf': '-',
        'mae': round(mean_absolute_error(R_y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(R_y_test, y_pred)), 2),
        'mape[%]': round(mean_absolute_percentage_error(R_y_test, y_pred)*100, 2)
    })

df_depth = pd.DataFrame(wyniki_depth)
best_depth_val = df_depth.loc[df_depth['mape[%]'].idxmin(), 'max_depth']
best_depth = None if best_depth_val == 'None' else int(best_depth_val)

# 3. min_samples_split
for ss in min_samples_split:
    rf = RandomForestRegressor(n_estimators=best_trees, max_depth=best_depth, min_samples_split=ss, random_state=42)
    rf.fit(R_X_train, R_y_train)
    y_pred = rf.predict(R_X_test)
    wyniki_split.append({
        'n_estimators': best_trees,
        'max_depth': best_depth_val,
        'min_samples_split': ss,
        'min_samples_leaf': '-',
        'mae': round(mean_absolute_error(R_y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(R_y_test, y_pred)), 2),
        'mape[%]': round(mean_absolute_percentage_error(R_y_test, y_pred)*100, 2)
    })

df_split = pd.DataFrame(wyniki_split)
best_split = int(df_split.loc[df_split['mape[%]'].idxmin(), 'min_samples_split'])

# 4. min_samples_leaf
for sl in min_samples_leaf:
    rf = RandomForestRegressor(n_estimators=best_trees, max_depth=best_depth, min_samples_split=best_split, min_samples_leaf=sl, random_state=42)
    rf.fit(R_X_train, R_y_train)
    y_pred = rf.predict(R_X_test)
    wyniki_leaf.append({
        'n_estimators': best_trees,
        'max_depth': best_depth_val,
        'min_samples_split': best_split,
        'min_samples_leaf': sl,
        'mae': round(mean_absolute_error(R_y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(R_y_test, y_pred)), 2),
        'mape[%]': round(mean_absolute_percentage_error(R_y_test, y_pred)*100, 2)
    })

rf_r_df = pd.concat([
    pd.DataFrame(wyniki_trees),
    pd.DataFrame(wyniki_depth),
    pd.DataFrame(wyniki_split),
    pd.DataFrame(wyniki_leaf)
], ignore_index=True)
rf_r_df.to_csv('Python/wyniki/RandomForestRegression.csv', index=False)

# Analizowane parametry - Gradient Boosting

trees = [5,10,50,100]
depth = [3, 5, 10, 20]
learning_rate = [0.001, 0.01, 0.1, 1.0]
subsample =     [0.5, 0.7, 0.9, 1.0]

# Klasyfikacja - Gradient Boosting

wyniki_trees = []
wyniki_depth = []
wyniki_lr    = []
wyniki_sub   = []

# 1. n_estimators
for t in trees:
    gb = GradientBoostingClassifier(n_estimators=t, random_state=42)
    gb.fit(C_X_train, C_y_train)
    wyniki_trees.append({
        'n_estimators': t,
        'max_depth': '-',
        'learning_rate': '-',
        'subsample': '-',
        'accuracy[%]': round(gb.score(C_X_test, C_y_test)*100, 2)
    })

df_trees = pd.DataFrame(wyniki_trees)
best_trees = int(df_trees.loc[df_trees['accuracy[%]'].idxmax(), 'n_estimators'])

# 2. max_depth
for d in depth:
    gb = GradientBoostingClassifier(n_estimators=best_trees, max_depth=d, random_state=42)
    gb.fit(C_X_train, C_y_train)
    wyniki_depth.append({
        'n_estimators': best_trees,
        'max_depth': d,
        'learning_rate': '-',
        'subsample': '-',
        'accuracy[%]': round(gb.score(C_X_test, C_y_test)*100, 2)
    })

df_depth = pd.DataFrame(wyniki_depth)
best_depth = int(df_depth.loc[df_depth['accuracy[%]'].idxmax(), 'max_depth'])

# 3. learning_rate
for lr in learning_rate:
    gb = GradientBoostingClassifier(n_estimators=best_trees, max_depth=best_depth, learning_rate=lr, random_state=42)
    gb.fit(C_X_train, C_y_train)
    wyniki_lr.append({
        'n_estimators': best_trees,
        'max_depth': best_depth,
        'learning_rate': lr,
        'subsample': '-',
        'accuracy[%]': round(gb.score(C_X_test, C_y_test)*100, 2)
    })

df_lr = pd.DataFrame(wyniki_lr)
best_lr = float(df_lr.loc[df_lr['accuracy[%]'].idxmax(), 'learning_rate'])

# 4. subsample
for s in subsample:
    gb = GradientBoostingClassifier(n_estimators=best_trees, max_depth=best_depth, learning_rate=best_lr, subsample=s, random_state=42)
    gb.fit(C_X_train, C_y_train)
    wyniki_sub.append({
        'n_estimators': best_trees,
        'max_depth': best_depth,
        'learning_rate': best_lr,
        'subsample': s,
        'accuracy[%]': round(gb.score(C_X_test, C_y_test)*100, 2)
    })

gb_c_df = pd.concat([
    pd.DataFrame(wyniki_trees),
    pd.DataFrame(wyniki_depth),
    pd.DataFrame(wyniki_lr),
    pd.DataFrame(wyniki_sub)
], ignore_index=True)
gb_c_df.to_csv('Python/wyniki/GradientBoostingClassification.csv', index=False)

# Regresja - Gradient Boosting

wyniki_trees = []
wyniki_depth = []
wyniki_lr    = []
wyniki_sub   = []

# 1. n_estimators
for t in trees:
    gb = GradientBoostingRegressor(n_estimators=t, random_state=42)
    gb.fit(R_X_train, R_y_train)
    y_pred = gb.predict(R_X_test)
    wyniki_trees.append({
        'n_estimators': t,
        'max_depth': '-',
        'learning_rate': '-',
        'subsample': '-',
        'mae': round(mean_absolute_error(R_y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(R_y_test, y_pred)), 2),
        'mape[%]': round(mean_absolute_percentage_error(R_y_test, y_pred)*100, 2)
    })

df_trees = pd.DataFrame(wyniki_trees)
best_trees = int(df_trees.loc[df_trees['mape[%]'].idxmin(), 'n_estimators'])

# 2. max_depth
for d in depth:
    gb = GradientBoostingRegressor(n_estimators=best_trees, max_depth=d, random_state=42)
    gb.fit(R_X_train, R_y_train)
    y_pred = gb.predict(R_X_test)
    wyniki_depth.append({
        'n_estimators': best_trees,
        'max_depth': d,
        'learning_rate': '-',
        'subsample': '-',
        'mae': round(mean_absolute_error(R_y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(R_y_test, y_pred)), 2),
        'mape[%]': round(mean_absolute_percentage_error(R_y_test, y_pred)*100, 2)
    })

df_depth = pd.DataFrame(wyniki_depth)
best_depth = int(df_depth.loc[df_depth['mape[%]'].idxmin(), 'max_depth'])

# 3. learning_rate
for lr in learning_rate:
    gb = GradientBoostingRegressor(n_estimators=best_trees, max_depth=best_depth, learning_rate=lr, random_state=42)
    gb.fit(R_X_train, R_y_train)
    y_pred = gb.predict(R_X_test)
    wyniki_lr.append({
        'n_estimators': best_trees,
        'max_depth': best_depth,
        'learning_rate': lr,
        'subsample': '-',
        'mae': round(mean_absolute_error(R_y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(R_y_test, y_pred)), 2),
        'mape[%]': round(mean_absolute_percentage_error(R_y_test, y_pred)*100, 2)
    })

df_lr = pd.DataFrame(wyniki_lr)
best_lr = float(df_lr.loc[df_lr['mape[%]'].idxmin(), 'learning_rate'])

# 4. subsample
for s in subsample:
    gb = GradientBoostingRegressor(n_estimators=best_trees, max_depth=best_depth, learning_rate=best_lr, subsample=s, random_state=42)
    gb.fit(R_X_train, R_y_train)
    y_pred = gb.predict(R_X_test)
    wyniki_sub.append({
        'n_estimators': best_trees,
        'max_depth': best_depth,
        'learning_rate': best_lr,
        'subsample': s,
        'mae': round(mean_absolute_error(R_y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(R_y_test, y_pred)), 2),
        'mape[%]': round(mean_absolute_percentage_error(R_y_test, y_pred)*100, 2)
    })

gb_r_df = pd.concat([
    pd.DataFrame(wyniki_trees),
    pd.DataFrame(wyniki_depth),
    pd.DataFrame(wyniki_lr),
    pd.DataFrame(wyniki_sub)
], ignore_index=True)
gb_r_df.to_csv('Python/wyniki/GradientBoostingRegression.csv', index=False)