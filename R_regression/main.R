library(dplyr)
library(ggplot2)
library(tidyr)

# Wczytywanie wszystkich komponentów
source("./R_regression/utils.R")
source("./R_regression/clean_data_stock.R")
source("./R_regression/neural_learn.R")
source("./R_regression/neural_network.R")
source("./R_regression/neural_network_2.R")
source("./R_regression/predict_nn.R")

# 1. Przygotowanie danych PYPL
dane_raw <- clean_stock_data("R_regression/PYPL.csv")
X_raw <- as.matrix(dane_raw[, 1:5]) # Open, High, Low, Close, Volume
y_raw <- as.matrix(dane_raw$Next_Close)

# Funkcja do przeprowadzenia pojedynczego eksperymentu
do_test <- function(train_p, lays, h1, h2, ep, lr, act_f, act_d, init_r) {
	n <- nrow(X_raw)
	train_idx <- 1:round(n * train_p)
	test_idx <- (round(n * train_p) + 1):n

	# Skalowanie
	X_scaled <- apply(X_raw, 2, min_max_scale)
	y_min <- min(y_raw[train_idx])
	y_max <- max(y_raw[train_idx])
	y_scaled <- (y_raw - y_min) / (y_max - y_min)

	X_train <- X_scaled[train_idx,]
	y_train <- y_scaled[train_idx,]
	X_test <- X_scaled[test_idx,]
	y_test <- y_scaled[test_idx,]

	model <- neural_learn(X_train, y_train, lays, h1, h2, ep, TRUE, init_r, lr, 100,
						  act_f, act_d, train_p)

	pred_train <- predict_nn(model, X_train, lays, act_f)
	pred_test <- predict_nn(model, X_test, lays, act_f)

	mse_train <- mean((y_train - pred_train)^2)
	mse_test <- mean((y_test - pred_test)^2)

	# wartości rzeczywiste, nie znormalizowane

	pred_usd <- pred_test * (y_max - y_min) + y_min
	y_test_real <- y_raw[test_idx]

	mse_usd <- mean((y_test_real - pred_usd)^2)
	rmse_usd <- sqrt(mse_usd)
	mape <- mean(abs((y_test_real - pred_usd) / y_test_real)) * 100

	return(data.frame(
		Train_MSE_norm = mean((y_train - pred_train)^2),
		Test_MSE_norm  = mean((y_test - pred_test)^2),
		Test_RMSE_USD  = rmse_usd,
		Test_MAPE_pct  = mape
	))

}

results <- data.frame()

# --- ANALIZA 8 PARAMETRÓW (Po 4 wartości) ---

# 1. Liczba neuronów H1
for(v in c(4, 8, 12, 16)) {
	results <- rbind(results, cbind(Param="H1_Nodes", Val=v, do_test(0.8, 1, v, 4, 1000, 0.1, sigmoid, sigmoid_derivative, 0.5)))
}

# 2. Szybkość uczenia (Learning Rate)
for(v in c(0.01, 0.05, 0.1, 0.3)) {
	results <- rbind(results, cbind(Param="LR", Val=v, do_test(0.8, 1, 8, 4, 1000, v, sigmoid, sigmoid_derivative, 0.5)))
}

# 3. Liczba epok
for(v in c(500, 1000, 2000, 5000)) {
	results <- rbind(results, cbind(Param="Epochs", Val=v, do_test(0.8, 1, 8, 4, v, 0.1, sigmoid, sigmoid_derivative, 0.5)))
}

# 4. Zakres inicjalizacji wag
for(v in c(0.1, 0.3, 0.5, 1.0)) {
	results <- rbind(results, cbind(Param="Init_Range", Val=v, do_test(0.8, 1, 8, 4, 1000, 0.1, sigmoid, sigmoid_derivative, v)))
}

# 5. Podział zbioru (Train Split)
for(v in c(0.6, 0.7, 0.8, 0.9)) {
	results <- rbind(results, cbind(Param="Split", Val=v, do_test(v, 1, 8, 4, 1000, 0.1, sigmoid, sigmoid_derivative, 0.5)))
}

# 6. Funkcje aktywacji
acts <- list(
	list(sigmoid, sigmoid_derivative, "sigmoid"),
	list(tanh, tanh_derivative, "tanh"),
	list(relu, relu_derivative, "relu"),
	list(relu_leak, relu_leak_derivative, "leak")
)

for(a in acts) {
	cat("\n--- Testuję funkcję:", a[[3]], "---\n")
	# Wymuszamy parametry dla modelu 1-warstwowego, aby uniknąć konfliktów wymiarów
	res <- do_test(train_p = 0.8, lays = 1, h1 = 8, h2 = 4,
				   ep = 1000, lr = 0.1, act_f = a[[1]],
				   act_d = a[[2]], init_r = 0.5)
	results <- rbind(results, cbind(Param="Activation", Val=a[[3]], res))
}

# 7. Liczba warstw ukrytych
for(v in c(1, 2)) {
	results <- rbind(results, cbind(Param="Layers", Val=v, do_test(0.8, v, 8, 8, 2000, 0.1, sigmoid, sigmoid_derivative, 0.5)))
}

# 8. Liczba neuronów H2 (2. warstwa)
for(v in c(4, 8, 12, 16)) {
	results <- rbind(results, cbind(Param="H2_Nodes", Val=v, do_test(0.8, 2, 8, v, 2000, 0.1, sigmoid, sigmoid_derivative, 0.5)))
}

#print(results)
results %>% arrange(Test_MAPE_pct)

# WYKRESY
source("./R_regression/plot_prediction.R")

# 1. Liczba neuronów H1
plot_best_model("R_regression/data/weights_L1_H1-4_H2-4_E1000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L1_H1-12_H2-4_E1000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L1_H1-16_H2-4_E1000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)

# 2. Learning rate
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.01_I0.5_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.05_I0.5_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.3_I0.5_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)

# 3. Liczba epok
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E500_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E2000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E5000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)

# 4. Zakres inicjalizacji wag
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.1_I0.1_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.1_I0.3_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.1_I1_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)

# 5. Train split
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.1_I0.5_SP0.6_sigm.RData", dane_raw, 1, 0.6, sigmoid)
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.1_I0.5_SP0.7_sigm.RData", dane_raw, 1, 0.7, sigmoid)
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.1_I0.5_SP0.9_sigm.RData", dane_raw, 1, 0.9, sigmoid)

# 6. Funkcje aktywacji
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.1_I0.5_SP0.8_tanh.RData", dane_raw, 1, 0.8, tanh)
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.1_I0.5_SP0.8_relu.RData", dane_raw, 1, 0.8, relu)
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E1000_LR0.1_I0.5_SP0.8_leak.RData", dane_raw, 1, 0.8, relu_leak)

# 7. Liczba warstw ukrytych
plot_best_model("R_regression/data/weights_L1_H1-8_H2-4_E2000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 1, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L2_H1-8_H2-4_E2000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 2, 0.8, sigmoid)

# 8. Liczba neuronów H2
plot_best_model("R_regression/data/weights_L2_H1-8_H2-4_E2000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 2, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L2_H1-8_H2-8_E2000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 2, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L2_H1-8_H2-12_E2000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 2, 0.8, sigmoid)
plot_best_model("R_regression/data/weights_L2_H1-8_H2-16_E2000_LR0.1_I0.5_SP0.8_sigm.RData", dane_raw, 2, 0.8, sigmoid)

# więcej testów

results <- rbind(results, cbind(Param="many_params", Val=1, do_test(0.8, 1, 16, 4, 5000, 0.1, relu, relu_derivative, 1)))
results <- rbind(results, cbind(Param="many_params", Val=2, do_test(0.8, 1, 16, 4, 5000, 0.3, relu, relu_derivative, 1)))
plot_best_model("R_regression/data/weights_L1_H1-16_H2-4_E5000_LR0.1_I1_SP0.8_relu.RData", dane_raw, 1, 0.8, relu)
plot_best_model("R_regression/data/weights_L1_H1-16_H2-4_E5000_LR0.3_I1_SP0.8_relu.RData", dane_raw, 1, 0.8, relu)





