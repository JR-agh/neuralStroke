import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures

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


# --- KLASYFIKACJA: k-Nearest Neighbors ---

n_neighbors = [3, 5, 10, 20]
weights = ['uniform', 'distance']
p_metric = [1, 2, 3]
scalers = {
    'Standard': StandardScaler(), # Średnia 0, odchylenie 1
    'MinMax': MinMaxScaler(),     # Wszystko w przedziale 0-1
    'Robust': RobustScaler(),     # Odporny na wartości odstające 
    'None': None                  # Dane surowe
}

wyniki_k = []
wyniki_w = []
wyniki_p = []
wyniki_s = []

# 1. n_neighbors
for k in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(C_X_train, C_y_train)
    wyniki_k.append({
        'n_neighbors': k, 
        'weights': '-', 
        'p': '-', 
        'scaler': '-',
        'accuracy[%]': round(knn.score(C_X_test, C_y_test)*100, 2)
    })

df_k = pd.DataFrame(wyniki_k)
best_k = int(df_k.loc[df_k['accuracy[%]'].idxmax(), 'n_neighbors'])

# 2. weights
for w in weights:
    knn = KNeighborsClassifier(n_neighbors=best_k, weights=w)
    knn.fit(C_X_train, C_y_train)
    wyniki_w.append({
        'n_neighbors': best_k, 
        'weights': w, 
        'p': '-', 
        'scaler': '-',
        'accuracy[%]': round(knn.score(C_X_test, C_y_test)*100, 2)
    })

df_w = pd.DataFrame(wyniki_w)
best_w = df_w.loc[df_w['accuracy[%]'].idxmax(), 'weights']

# 3. p
for p_val in p_metric:
    knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_w, p=p_val)
    knn.fit(C_X_train, C_y_train)
    wyniki_p.append({
        'n_neighbors': best_k, 
        'weights': best_w, 
        'p': p_val, 
        'scaler': '-',
        'accuracy[%]': round(knn.score(C_X_test, C_y_test)*100, 2)
    })

df_p = pd.DataFrame(wyniki_p)
best_p = int(df_p.loc[df_p['accuracy[%]'].idxmax(), 'p'])

# 4. Scalers
for nazwa, scaler in scalers.items():
    # Kopiujemy dane, aby ich nie nadpisać na stałe
    X_train_scaled = C_X_train.copy()
    X_test_scaled = C_X_test.copy()
    
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(C_X_train)
        X_test_scaled = scaler.transform(C_X_test)
    
    # Trenujemy model z najlepszymi wypracowanymi wcześniej parametrami
    knn = KNeighborsClassifier(n_neighbors=best_k, weights=best_w, p=best_p)
    knn.fit(X_train_scaled, C_y_train)
    
    wyniki_s.append({
        'n_neighbors': best_k, 
        'weights': best_w, 
        'p': best_p, 
        'scaler': nazwa,
        'accuracy[%]': round(knn.score(X_test_scaled, C_y_test) * 100, 2)
    })

df_s = pd.DataFrame(wyniki_s)
best_scaler_name = df_s.loc[df_s['accuracy[%]'].idxmax(), 'scaler']


knn_c_df = pd.concat([
    pd.DataFrame(wyniki_k), 
    pd.DataFrame(wyniki_w), 
    pd.DataFrame(wyniki_p), 
    pd.DataFrame(wyniki_s)
    ], ignore_index=True)
knn_c_df.to_csv('Python/wyniki/KNN_Classification.csv', index=False)


# --- REGRESJA: k-Nearest Neighbors ---

wyniki_k = []
wyniki_w = []
wyniki_p = []
wyniki_s = []

# 1. n_neighbors
for k in n_neighbors:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(R_X_train, R_y_train)
    y_pred = knn.predict(R_X_test)
    wyniki_k.append({
        'n_neighbors': k, 
	'weights': '-', 
	'p': '-',
	'scaler': '-',
        'mae': round(mean_absolute_error(R_y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(R_y_test, y_pred)), 2),
        'mape[%]': round(mean_absolute_percentage_error(R_y_test, y_pred)*100, 2)
    })

df_k = pd.DataFrame(wyniki_k)
best_k = int(df_k.loc[df_k['mape[%]'].idxmin(), 'n_neighbors'])

# 2. weights
for w in weights:
    knn = KNeighborsRegressor(n_neighbors=best_k, weights=w)
    knn.fit(R_X_train, R_y_train)
    y_pred = knn.predict(R_X_test)
    wyniki_w.append({
        'n_neighbors': best_k, 
	'weights': w, 
	'p': '-',
	'scaler': '-',
        'mae': round(mean_absolute_error(R_y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(R_y_test, y_pred)), 2),
        'mape[%]': round(mean_absolute_percentage_error(R_y_test, y_pred)*100, 2)
    })

df_w = pd.DataFrame(wyniki_w)
best_w = df_w.loc[df_w['mape[%]'].idxmin(), 'weights']

# 3. p
for p_val in p_metric:
    knn = KNeighborsRegressor(n_neighbors=best_k, weights=best_w, p=p_val)
    knn.fit(R_X_train, R_y_train)
    y_pred = knn.predict(R_X_test)
    wyniki_p.append({
        'n_neighbors': best_k, 
	'weights': best_w,
	'p': p_val,
	'scaler': '-',
        'mae': round(mean_absolute_error(R_y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(R_y_test, y_pred)), 2),
        'mape[%]': round(mean_absolute_percentage_error(R_y_test, y_pred)*100, 2)
    })

df_p = pd.DataFrame(wyniki_p)
best_p = df_p.loc[df_p['mape[%]'].idxmin(), 'p']

# 4. Scalers dla Regresji
for nazwa, scaler in scalers.items():
    X_train_scaled = R_X_train.copy()
    X_test_scaled = R_X_test.copy()
    
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(R_X_train)
        X_test_scaled = scaler.transform(R_X_test)
    
    knn = KNeighborsRegressor(n_neighbors=best_k, weights=best_w, p=best_p)
    knn.fit(X_train_scaled, R_y_train)
    y_pred = knn.predict(X_test_scaled)
    
    wyniki_s.append({
        'n_neighbors': best_k, 
        'weights': best_w, 
        'p': best_p, 
        'scaler': nazwa,
        'mae': round(mean_absolute_error(R_y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(R_y_test, y_pred)), 2),
        'mape[%]': round(mean_absolute_percentage_error(R_y_test, y_pred)*100, 2)
    })

df_s = pd.DataFrame(wyniki_s)
best_s = df_s.loc[df_s['mape[%]'].idxmin(), 'scaler']

knn_r_df = pd.concat([
	pd.DataFrame(wyniki_k), 
	pd.DataFrame(wyniki_w), 
	pd.DataFrame(wyniki_p),
	pd.DataFrame(wyniki_s)
	], ignore_index=True)
knn_r_df.to_csv('Python/wyniki/KNN_Regression.csv', index=False)


# --- KLASYFIKACJA: Logistic Regression ---

C_param = [0.01, 0.1, 1.0, 10.0]
solvers = ['lbfgs', 'newton-cg', 'sag', 'saga']
poly_degrees = [1, 2]
log_scalers = {
    'Standard': StandardScaler(),
    'MinMax': MinMaxScaler(),
    'Robust': RobustScaler(),
    'None': None
}

wyniki_C = []
wyniki_s = []
wyniki_poly = []
wyniki_sc = []

# 1. C
for c in C_param:
    lr = LogisticRegression(C=c, max_iter=100000)
    lr.fit(C_X_train, C_y_train)
    wyniki_C.append({
        'C': c, 
	    'solver': '-', 
	    'max_iter': '-',
        'scaler': '-',
        'accuracy[%]': round(lr.score(C_X_test, C_y_test)*100, 2)
    })

df_C = pd.DataFrame(wyniki_C)
best_C = float(df_C.loc[df_C['accuracy[%]'].idxmax(), 'C'])

# 2. solver
for s in solvers:
    lr = LogisticRegression(C=best_C, solver=s, max_iter=100000)
    lr.fit(C_X_train, C_y_train)
    wyniki_s.append({
        'C': best_C, 
	    'solver': s, 
	    'max_iter': '-',
        'scaler': '-',
        'accuracy[%]': round(lr.score(C_X_test, C_y_test)*100, 2)
    })

df_s = pd.DataFrame(wyniki_s)
best_s = df_s.loc[df_s['accuracy[%]'].idxmax(), 'solver']

# 3. Wielomiany (Polynomial)
for d in poly_degrees:
    # Tworzymy tymczasowy scaler, żeby model się nie zawiesił
    temp_scaler = StandardScaler()
    X_train_p = temp_scaler.fit_transform(C_X_train)
    X_test_p = temp_scaler.transform(C_X_test)
    
    if d > 1:
        poly = PolynomialFeatures(degree=d, include_bias=False)
        X_train_p = poly.fit_transform(X_train_p)
        X_test_p = poly.transform(X_test_p)
        
    # Używamy solvera 'saga', który lepiej radzi sobie z dużymi zbiorami po wielomianach
    lr = LogisticRegression(C=best_C, solver='saga', max_iter=5000) 
    lr.fit(X_train_p, C_y_train)
    
    wyniki_poly.append({
        'C': best_C, 'solver': best_s, 'poly_degree': d, 'scaler': 'Temp-Standard',
        'accuracy[%]': round(lr.score(X_test_p, C_y_test)*100, 2)
    })

df_p = pd.DataFrame(wyniki_poly)
best_p = int(df_p.loc[df_p['accuracy[%]'].idxmax(), 'poly_degree'])


# 4. Scalers 
for nazwa, scaler in log_scalers.items():
    X_train_f = C_X_train.copy()
    X_test_f = C_X_test.copy()
    
    if scaler is not None:
        X_train_f = scaler.fit_transform(X_train_f)
        X_test_f = scaler.transform(X_test_f)
    
    if best_p > 1:
        poly = PolynomialFeatures(degree=best_p, include_bias=False)
        X_train_f = poly.fit_transform(X_train_f)
        X_test_f = poly.transform(X_test_f)
        
    lr = LogisticRegression(C=best_C, solver=best_s, max_iter=100000)
    lr.fit(X_train_f, C_y_train)
    
    wyniki_sc.append({
        'C': best_C, 'solver': best_s, 'poly_degree': best_p, 'scaler': nazwa,
        'accuracy[%]': round(lr.score(X_test_f, C_y_test)*100, 2)
    })

log_c_df = pd.concat([
	pd.DataFrame(wyniki_C), 
	pd.DataFrame(wyniki_s), 
	pd.DataFrame(wyniki_poly),
    pd.DataFrame(wyniki_sc)
	], ignore_index=True)
log_c_df.to_csv('Python/wyniki/LogisticRegression_Classification.csv', index=False)


# --- REGRESJA: Ridge (jako Linear Regression z parametrami) ---

alphas = [0.1, 1.0, 10.0, 100.0]
fit_intercept = [True, False]
ridge_scalers = {
    'Standard': StandardScaler(),
    'MinMax': MinMaxScaler(),
    'Robust': RobustScaler(),
    'None': None
}

wyniki_a = []
wyniki_fi = []
wyniki_sc = []

# 1. alpha
for a in alphas:
    model = Ridge(alpha=a)
    model.fit(R_X_train, R_y_train)
    y_pred = model.predict(R_X_test)
    wyniki_a.append({
        'alpha': a, 
	    'fit_intercept': '-',
        'scaler': '-',
        'mae': round(mean_absolute_error(R_y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(R_y_test, y_pred)), 2),
        'mape[%]': round(mean_absolute_percentage_error(R_y_test, y_pred)*100, 2)
    })

df_a = pd.DataFrame(wyniki_a)
best_a = float(df_a.loc[df_a['mape[%]'].idxmin(), 'alpha'])

# 2. fit_intercept
for fi in fit_intercept:
    model = Ridge(alpha=best_a, fit_intercept=fi)
    model.fit(R_X_train, R_y_train)
    y_pred = model.predict(R_X_test)
    wyniki_fi.append({
        'alpha': best_a, 
	    'fit_intercept': fi,
        'scaler': '-',
        'mae': round(mean_absolute_error(R_y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(R_y_test, y_pred)), 2),
        'mape[%]': round(mean_absolute_percentage_error(R_y_test, y_pred)*100, 2)
    })

df_fi = pd.DataFrame(wyniki_fi)
best_fi = df_fi.loc[df_fi['mape[%]'].idxmin(), 'fit_intercept']

# 3. Scalers
for nazwa, scaler in ridge_scalers.items():
    X_train_final = R_X_train.copy()
    X_test_final = R_X_test.copy()
    
    if scaler is not None:
        X_train_final = scaler.fit_transform(R_X_train)
        X_test_final = scaler.transform(R_X_test)
    
    model = Ridge(alpha=best_a, fit_intercept=best_fi)
    model.fit(X_train_final, R_y_train)
    y_pred = model.predict(X_test_final)
    
    wyniki_sc.append({
        'alpha': best_a, 
        'fit_intercept': best_fi,
        'scaler': nazwa,
        'mae': round(mean_absolute_error(R_y_test, y_pred), 2),
        'rmse': round(np.sqrt(mean_squared_error(R_y_test, y_pred)), 2),
        'mape[%]': round(mean_absolute_percentage_error(R_y_test, y_pred)*100, 2)
    })

ridge_r_df = pd.concat([
	pd.DataFrame(wyniki_a), 
	pd.DataFrame(wyniki_fi),
    pd.DataFrame(wyniki_sc),
	], ignore_index=True)
ridge_r_df.to_csv('Python/wyniki/RidgeRegression.csv', index=False)
