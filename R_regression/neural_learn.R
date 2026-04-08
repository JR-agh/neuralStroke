neural_learn <- function(X, y, layers, h1_nodes, h2_nodes = 4, epochs, init = TRUE,
						 init_range = 0.5, learning_rate = 0.5, freq = 100,
						 activation_function, activation_function_derivative, train_p = 0.8) {

	# Generowanie unikalnej nazwy pliku na podstawie parametrów
	act_name <- get_act_name(activation_function)
	file_name <- paste0("weights_L", layers, "_H1-", h1_nodes, "_H2-", h2_nodes,
						"_E", epochs, "_LR", learning_rate, "_I", init_range,
						"_SP", train_p, "_", act_name, ".RData")
	dir_path <- "./R_regression/data/"
	if (!dir.exists(dir_path)) dir.create(dir_path, recursive = TRUE)
	full_path <- paste0(dir_path, file_name)

	if(layers == 1) {
		if(!init && file.exists(full_path)) load(full_path)
		weights_data <- neural_network(X, y, h1_nodes, epochs, init, init_range, learning_rate, freq, activation_function, activation_function_derivative)
	} else {
		if(!init && file.exists(full_path)) load(full_path)
		weights_data <- neural_network_2(X, y, h1_nodes, h2_nodes, epochs, init, init_range, learning_rate, freq, activation_function, activation_function_derivative)
	}

	save(weights_data, file = full_path)
	return(weights_data)
}
