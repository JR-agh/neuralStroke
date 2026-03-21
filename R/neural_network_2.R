neural_network_2 <- function(X, y, h1_nodes, h2_nodes, epochs, init = TRUE, output_nodes = 1, learning_rate = 0.5, freq = 1000) {
	input_nodes <- ncol(X)
	n <- nrow(X)
	Vmse <- c()

	if(init == TRUE) {
		# Inicjalizacja wag dla 3 połączeń (In -> H1, H1 -> H2, H2 -> Out)
		weights_0_1 <- matrix(runif(input_nodes * h1_nodes, -0.5, 0.5), nrow = input_nodes)
		weights_1_2 <- matrix(runif(h1_nodes * h2_nodes, -0.5, 0.5), nrow = h1_nodes)
		weights_2_3 <- matrix(runif(h2_nodes * output_nodes, -0.5, 0.5), nrow = h2_nodes)

		bias_h1 <- matrix(runif(h1_nodes, -0.5, 0.5), nrow = 1)
		bias_h2 <- matrix(runif(h2_nodes, -0.5, 0.5), nrow = 1)
		bias_output <- matrix(runif(output_nodes, -0.5, 0.5), nrow = 1)
	}

	for (i in 1:epochs) {
		# --- Forward Propagation ---
		# Warstwa ukryta 1
		h1_input <- X %*% weights_0_1 + matrix(rep(bias_h1, n), byrow = TRUE, nrow = n)
		h1_output <- sigmoid(h1_input)

		# Warstwa ukryta 2
		h2_input <- h1_output %*% weights_1_2 + matrix(rep(bias_h2, n), byrow = TRUE, nrow = n)
		h2_output <- sigmoid(h2_input)

		# Warstwa wyjściowa
		out_input <- h2_output %*% weights_2_3 + matrix(rep(bias_output, n), byrow = TRUE, nrow = n)
		predicted_output <- sigmoid(out_input)

		# --- Obliczanie błędu ---
		error <- y - predicted_output

		# --- Backpropagation ---
		# Gradient dla wyjścia
		d_predicted_output <- error * sigmoid_derivative(predicted_output)

		# Gradient dla warstwy ukrytej 2 (zależy od błędu wyjścia i wag w23)
		d_h2 <- (tcrossprod(d_predicted_output, weights_2_3)) * sigmoid_derivative(h2_output)

		# Gradient dla warstwy ukrytej 1 (zależy od gradientu h2 i wag w12)
		d_h1 <- (tcrossprod(d_h2, weights_1_2)) * sigmoid_derivative(h1_output)

		# --- Aktualizacja wag i biasów ---
		weights_2_3 <- weights_2_3 + (crossprod(h2_output, d_predicted_output) * (learning_rate / n))
		weights_1_2 <- weights_1_2 + (crossprod(h1_output, d_h2) * (learning_rate / n))
		weights_0_1 <- weights_0_1 + (crossprod(X, d_h1) * (learning_rate / n))

		bias_output <- bias_output + colMeans(d_predicted_output) * learning_rate
		bias_h2 <- bias_h2 + colMeans(d_h2) * learning_rate
		bias_h1 <- bias_h1 + colMeans(d_h1) * learning_rate

		if (i %% freq == 0) {
			mse <- mean(error^2)
			Vmse <- append(Vmse, mse)
			cat("Epoch:", i, " MSE:", mse, "\n")
		}
	}

	final_predictions <- ifelse(predicted_output > 0.5, 1, 0)
	cat("Accuracy:", mean(final_predictions == y) * 100, "%\n")
	weightsData <- list(weights_1_2 = weights_1_2,
						weights_0_1 = weights_0_1,
						weights_2_3 = weights_2_3,
						bias_output = bias_output,
						bias_h1 = bias_h1,
						bias_h2 = bias_h1,
						accuracy = (mean(final_predictions == y) * 100),
						mse = Vmse)
	return(weightsData)
}
