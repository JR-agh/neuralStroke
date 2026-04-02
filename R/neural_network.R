neural_network <- function(X, y, hidden_nodes, epochs, init, output_nodes = 1, learning_rate, freq, activation_function, activation_function_derivative) {
	input_nodes <- ncol(X)
	n <- nrow(X)
	Vmse <- c()

	if(init == TRUE) {
		weights_0_1 <- matrix(runif(input_nodes * hidden_nodes, -0.5, 0.5), nrow = input_nodes)
		weights_1_2 <- matrix(runif(hidden_nodes * output_nodes, -0.5, 0.5), nrow = hidden_nodes)

		bias_hidden <- matrix(runif(hidden_nodes, -0.5, 0.5), nrow = 1)
		bias_output <- matrix(runif(output_nodes, -0.5, 0.5), nrow = 1)

	} else {
		weights_0_1 <- weightsData$weights_0_1
		weights_1_2 <- weightsData$weights_1_2
		bias_hidden <- weightsData$bias_hidden
		bias_output <- weightsData$bias_output
	}

	for (i in 1:epochs) {
		# Forward
		hidden_layer_output <- activation_function(X %*% weights_0_1 + matrix(rep(bias_hidden, nrow(X)), byrow = TRUE, nrow = n))
		predicted_output <- activation_function(hidden_layer_output %*% weights_1_2 + matrix(rep(bias_output, nrow(X)), byrow = TRUE, nrow = n))

		# Error
		error <- y - predicted_output

		# Backprop
		d_predicted_output <- error * activation_function_derivative(predicted_output)
		d_hidden_layer <- (tcrossprod(d_predicted_output, weights_1_2)) * activation_function_derivative(hidden_layer_output)

		# Weights actualization
		weights_1_2 <- weights_1_2 + (crossprod(hidden_layer_output, d_predicted_output) * (learning_rate / n))
		weights_0_1 <- weights_0_1 + (crossprod(X, d_hidden_layer) * (learning_rate / n))

		bias_output <- bias_output + colMeans(d_predicted_output) * learning_rate
		bias_hidden <- bias_hidden + colMeans(d_hidden_layer) * learning_rate

		if (i %% freq == 0) {
			mse <- mean(error^2)
			Vmse <- append(Vmse, mse)
			cat("Epoch:", i, " MSE:", mse, "\n")
		}
	}

	final_predictions <- ifelse(predicted_output > 0.5, 1, 0)
	cat("Accuracy:", mean(final_predictions == y) * 100, "%\n")
	weights_data <- list(weights_1_2 = weights_1_2,
						weights_0_1 = weights_0_1,
						bias_output = bias_output,
						bias_hidden = bias_hidden,
						accuracy = (mean(final_predictions == y) * 100),
						mse = Vmse)
	return(weights_data)
}
