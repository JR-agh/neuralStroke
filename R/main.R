library(dplyr)
library(ggplot2)
library(tidyr)

#load raw training data
dane <- read.csv("./rawdata/train.csv")

#get functions from other files
file_sources <- list.files("R", full.names = TRUE)
file_sources <- file_sources[-2]
sapply(file_sources, source)

#clean data
dane <- clean_data(dane)

#chosing input and benchmark data
Xdf <- dane |>
	select(-c("satisfaction", "X", "id", "Customer.Type", "Type.of.Travel")) |>
	select(c("Online.boarding", "Flight.Distance", "Seat.comfort", "Cleanliness", "Class"))
Ydf <- dane$satisfaction

#scaling Xdf
Xdf <- as.data.frame(lapply(Xdf, min_max_scale))

#creating matrices based on created data frames
X <- as.matrix(Xdf)
y <- as.matrix(Ydf)

#machine learning
weights_data_l1_h10_e500_runif05 <- neural_learn(X, y, layers = 1, h1_nodes = 10, epochs = 500,
										 activation_function = sigmoid,
										 activation_function_derivative = sigmoid_derivative)
weights_data_l1_h6_e500_runif05 <- neural_learn(X, y, layers = 1, h1_nodes = 6, epochs = 500,
												 activation_function = sigmoid,
												 activation_function_derivative = sigmoid_derivative)
weights_data_l1_h8_e500_runif05 <- neural_learn(X, y, layers = 1, h1_nodes = 8, epochs = 500,
												 activation_function = sigmoid,
												 activation_function_derivative = sigmoid_derivative)
weights_data_l1_h12_e500_runif05 <- neural_learn(X, y, layers = 1, h1_nodes = 12, epochs = 500,
												 activation_function = sigmoid,
												 activation_function_derivative = sigmoid_derivative)
#2 layers
weights_data_l2_h6_h4_e500_runif05 <- neural_learn(X, y, layers = 2, h1_nodes = 6, h2_nodes = 4, epochs = 500,
										   activation_function = sigmoid,
										   activation_function_derivative = sigmoid_derivative)
weights_data_l2_h8_h8_e500_runif05 <- neural_learn(X, y, layers = 2, h1_nodes = 8, h2_nodes = 8, epochs = 500,
												   activation_function = sigmoid,
												   activation_function_derivative = sigmoid_derivative)
weights_data_l2_h10_h8_e500_runif05 <- neural_learn(X, y, layers = 2, h1_nodes = 10, h2_nodes = 8, epochs = 500,
												   activation_function = sigmoid,
												   activation_function_derivative = sigmoid_derivative)
weights_data_l2_h4_h4_e500_runif05 <- neural_learn(X, y, layers = 2, h1_nodes = 4, h2_nodes = 4, epochs = 500,
												   activation_function = sigmoid,
												   activation_function_derivative = sigmoid_derivative)

#different activation functions
weights_data_l1_h12_e500_runif05_tanh <- neural_learn(X, y, layers = 1, h1_nodes = 12, epochs = 500,
												 activation_function = tanh,
												 activation_function_derivative = tanh_derivative)

#different weights initialization
weights_data_l1_h12_e500_runif05_tanh <- neural_learn(X, y, layers = 1, h1_nodes = 12, epochs = 500,
													  activation_function = tanh,
													  activation_function_derivative = tanh_derivative)
weights_data_l1_h12_e500_rnormxavier_tanh <- neural_learn(X, y, layers = 1, h1_nodes = 12, epochs = 500,
													  activation_function = tanh,
													  activation_function_derivative = tanh_derivative)
weights_data_l1_h12_e500_rnormsd0.01_tanh <- neural_learn(X, y, layers = 1, h1_nodes = 12, epochs = 500,
														  activation_function = tanh,
														  activation_function_derivative = tanh_derivative)

#different ways of normalization
Xdf <- as.data.frame(lapply(Xdf, normalize))
weights_data_l1_h12_e500_runif05_tanh_dnorm <- neural_learn(X, y, layers = 1, h1_nodes = 12, epochs = 500,
														  activation_function = tanh,
														  activation_function_derivative = tanh_derivative)

#creating plot to display comparision between neural networks
MSEdf <- data.frame(mse_minmax = weights_data_l1_h12_e500_runif05_tanh$mse,
					mse_dnorm = weights_data_l1_h12_e500_runif05_tanh_dnorm$mse)
print(MSEdf)
MSEdf_long <- pivot_longer(MSEdf,
						cols = c("mse_minmax", "mse_dnorm"),
						names_to = "Input_normalization",
						values_to = "MSE")
MSEdf_long$Input_normalization = as.factor(rep(c("0-1 scale", "Normal distribution"), times = 5))
ep <- seq(from = 100, by = 100, length.out = 5)
MSEdf_long$Epochs <- rep(ep, each = 2)

ggplot(MSEdf_long, aes(x = Epochs, y = MSE, color = Input_normalization)) +
	labs(title = "MSE vs. Epochs by method of input normalization") +
	geom_line() +
	scale_colour_manual("Method", values = c("0-1 scale" = "red", "Normal distribution" = "blue")) +
	theme_minimal()
