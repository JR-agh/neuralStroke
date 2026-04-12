neural_network_2 <- function(X, y, h1_nodes, h2_nodes, epochs, init, init_range = 0.5,
							 learning_rate, freq, activation_function, activation_function_derivative) {
	input_nodes <- ncol(X)
	output_nodes <- 1
	n <- nrow(X)
	Vmse <- c()

	if(init == TRUE) {
		weights_0_1 <- matrix(runif(input_nodes * h1_nodes, -init_range, init_range), nrow = input_nodes)
		weights_1_2 <- matrix(runif(h1_nodes * h2_nodes, -init_range, init_range), nrow = h1_nodes)
		weights_2_3 <- matrix(runif(h2_nodes * output_nodes, -init_range, init_range), nrow = h2_nodes)
		bias_h1 <- matrix(runif(h1_nodes, -init_range, init_range), nrow = 1)
		bias_h2 <- matrix(runif(h2_nodes, -init_range, init_range), nrow = 1)
		bias_output <- matrix(runif(output_nodes, -init_range, init_range), nrow = 1)
	}

	for (i in 1:epochs) {
		# Forward
		h1_output <- activation_function(X %*% weights_0_1 + matrix(rep(bias_h1, n), byrow = TRUE, nrow = n))
		h2_output <- activation_function(h1_output %*% weights_1_2 + matrix(rep(bias_h2, n), byrow = TRUE, nrow = n))
		#predicted_output <- activation_function(h2_output %*% weights_2_3 + matrix(rep(bias_output, n), byrow = TRUE, nrow = n))
		predicted_output <- h2_output %*% weights_2_3 + matrix(rep(bias_output, n), byrow = TRUE, nrow = n)

		error <- y - predicted_output

		# Backprop
		#d_out <- error * activation_function_derivative(predicted_output)
		d_out <- error * 1
		d_h2 <- (tcrossprod(d_out, weights_2_3)) * activation_function_derivative(h2_output)
		d_h1 <- (tcrossprod(d_h2, weights_1_2)) * activation_function_derivative(h1_output)

		weights_2_3 <- weights_2_3 + (crossprod(h2_output, d_out) * (learning_rate / n))
		weights_1_2 <- weights_1_2 + (crossprod(h1_output, d_h2) * (learning_rate / n))
		weights_0_1 <- weights_0_1 + (crossprod(X, d_h1) * (learning_rate / n))
		bias_output <- bias_output + colMeans(d_out) * learning_rate
		bias_h2 <- bias_h2 + colMeans(d_h2) * learning_rate
		bias_h1 <- bias_h1 + colMeans(d_h1) * learning_rate

		if (i %% freq == 0) {
			current_mse <- mean(error^2)
			Vmse <- append(Vmse, current_mse)
			cat("Epoch:", i, "| MSE:", round(current_mse, 6), "\n")
		}
	}

	return(list(weights_0_1=weights_0_1, weights_1_2=weights_1_2, weights_2_3=weights_2_3,
				bias_h1=bias_h1, bias_h2=bias_h2, bias_output=bias_output, mse=Vmse))
}
