neural_network <- function(X, y, hidden_nodes, epochs, init, init_range = 0.5,
						   learning_rate, freq, activation_function, activation_function_derivative) {
	input_nodes <- ncol(X)
	output_nodes <- 1
	n <- nrow(X)
	Vmse <- c()

	if(init == TRUE) {
		weights_0_1 <- matrix(runif(input_nodes * hidden_nodes, -init_range, init_range), nrow = input_nodes)
		weights_1_2 <- matrix(runif(hidden_nodes * output_nodes, -init_range, init_range), nrow = hidden_nodes)
		bias_hidden <- matrix(runif(hidden_nodes, -init_range, init_range), nrow = 1)
		bias_output <- matrix(runif(output_nodes, -init_range, init_range), nrow = 1)
	} else {
		weights_0_1 <- weights_data$weights_0_1
		weights_1_2 <- weights_data$weights_1_2
		bias_hidden <- weights_data$bias_hidden
		bias_output <- weights_data$bias_output
	}

	for (i in 1:epochs) {
		# Forward
		hidden_layer_output <- activation_function(X %*% weights_0_1 + matrix(rep(bias_hidden, n), byrow = TRUE, nrow = n))
		#predicted_output <- activation_function(hidden_layer_output %*% weights_1_2 + matrix(rep(bias_output, n), byrow = TRUE, nrow = n))
		predicted_output <- hidden_layer_output %*% weights_1_2 + matrix(rep(bias_output, n), byrow = TRUE, nrow = n)

		error <- y - predicted_output

		# Backprop
		#d_predicted_output <- error * activation_function_derivative(predicted_output)
		d_predicted_output <- error * 1
		d_hidden_layer <- (tcrossprod(d_predicted_output, weights_1_2)) * activation_function_derivative(hidden_layer_output)

		weights_1_2 <- weights_1_2 + (crossprod(hidden_layer_output, d_predicted_output) * (learning_rate / n))
		weights_0_1 <- weights_0_1 + (crossprod(X, d_hidden_layer) * (learning_rate / n))
		bias_output <- bias_output + colMeans(d_predicted_output) * learning_rate
		bias_hidden <- bias_hidden + colMeans(d_hidden_layer) * learning_rate

		if (i %% freq == 0) {
			current_mse <- mean(error^2)
			Vmse <- append(Vmse, current_mse)
			cat("Epoch:", i, "| MSE:", round(current_mse, 6), "\n")
		}
	}

	return(list(weights_0_1 = weights_0_1, weights_1_2 = weights_1_2,
				bias_hidden = bias_hidden, bias_output = bias_output, mse = Vmse))
}
