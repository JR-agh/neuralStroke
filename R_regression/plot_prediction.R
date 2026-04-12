plot_best_model <- function(model_path, dane_raw, layers, train_p, act_f) {
	load(model_path)

	X_raw <- as.matrix(dane_raw[, 1:5])
	y_raw <- as.matrix(dane_raw$Next_Close)

	n <- nrow(X_raw)

	train_idx <- 1:round(n * train_p)
	test_idx <- (round(n * train_p) + 1):n

	X_scaled <- apply(X_raw, 2, min_max_scale)

	y_min <- min(y_raw[train_idx])
	y_max <- max(y_raw[train_idx])

	y_scaled <- (y_raw - y_min) / (y_max - y_min)

	X_test <- X_scaled[test_idx, ]
	y_test_real <- y_raw[test_idx]

	pred_scaled <- predict_nn(weights_data, X_test, layers, act_f)

	pred_usd <- pred_scaled * (y_max - y_min) + y_min

	plot_df <- data.frame(
		Dzien = 1:length(test_idx),
		Rzeczywiste = as.vector(y_test_real),
		Przewidziane = as.vector(pred_usd)
	)

	ggplot(plot_df, aes(x = Dzien)) +
		geom_line(aes(y = Rzeczywiste, color = "Cena Rzeczywista"), size = 1) +
		geom_line(aes(y = Przewidziane, color = "Predykcja NN"), linetype = "dashed", size = 1) +
		scale_color_manual(values = c("Cena Rzeczywista" = "black", "Predykcja NN" = "red")) +
		theme_bw() +
		labs(title = "Porównanie cen rzeczywistych z predykcją (Zbiór Testowy)",
			 subtitle = paste("Model:", basename(model_path)),
			 x = "Kolejne dni (zbiór testowy)",
			 y = "Cena akcji PYPL (USD)",
			 color = "Legenda")
}
