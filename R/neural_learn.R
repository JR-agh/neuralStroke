neural_learn <- function(X, y, layers, h1_nodes, h2_nodes = 4, epochs, init = TRUE, learning_rate = 0.5, freq = 100, activation_function) {
	if(layers == 1) {
		if(!init)
			load(paste0("./data/weights_data", h1_nodes, ".RData"))
		weights_data <- neural_network(X = X,
					    y = y,
					    hidden_nodes = h1_nodes,
					    epochs = epochs,
					    init = init,
					    learning_rate = learning_rate,
					    freq = freq)
		save(weights_data, file = paste0("./data/weights_data", h1_nodes, ".RData"))
	}
	else {
		if(!init)
			load(paste0("./data/weights_data", h1_nodes, "_", h2_nodes, ".RData"))
		weights_data <- neural_network_2(X = X,
						y = y,
						h1_nodes = h1_nodes,
						h2_nodes = h2_nodes,
						epochs = epochs,
						init = init,
						learning_rate = learning_rate,
						freq = freq)
		save(weights_data, file = paste0("./data/weights_data", h1_nodes, "_", h2_nodes, ".RData"))
	}
	return(weights_data)
}
