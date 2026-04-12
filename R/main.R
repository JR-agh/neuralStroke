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
	select(c("Class", "Online.boarding", "Seat.comfort", "Inflight.entertainment", "On.board.service"))
Ydf <- dane$satisfaction

#scaling Xdf
Xdf <- as.data.frame(lapply(Xdf, normalize))

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
#Xdf <- as.data.frame(lapply(Xdf, normalize))
#Xdf <- as.data.frame(lapply(Xdf, min_max_scale))
weights_data_l1_h12_e500_runif05_tanh_dnorm <- neural_learn(X, y, layers = 1, h1_nodes = 12, epochs = 500,
														  activation_function = tanh,
														  activation_function_derivative = tanh_derivative)

#different inputs
Xdf <- dane |>
	select(c("Class", "Online.boarding", "Seat.comfort", "Inflight.entertainment", "On.board.service"))
weights_opt_C_Ob_S_I_Os <- neural_learn(X, y, layers = 1, h1_nodes = 12, epochs = 500,
										activation_function = tanh,
										activation_function_derivative = tanh_derivative)
Xdf <- dane |>
	select(c("Departure.Arrival.time.convenient", "Departure.Delay.in.Minutes", "Gender", "Age", "Gate.location"))
weights_opt_Da_Dd_G_A_Gl <- neural_learn(X, y, layers = 1, h1_nodes = 12, epochs = 500,
										activation_function = tanh,
										activation_function_derivative = tanh_derivative)
Xdf <- dane |>
	select(c("Gender", "Class", "Age", "Ease.of.Online.booking", "Food.and.drink"))
weights_opt_G_C_E_A_Fd <- neural_learn(X, y, layers = 1, h1_nodes = 12, epochs = 500,
									   activation_function = tanh,
									   activation_function_derivative = tanh_derivative)
Xdf <- dane |>
	select(c("Departure.Delay.in.Minutes", "Seat.comfort", "Food.and.drink", "Class", "Flight.Distance"))
weights_opt_Dd_S_Fo_C_Fd <- neural_learn(X, y, layers = 1, h1_nodes = 12, epochs = 500,
										 activation_function = tanh,
										 activation_function_derivative = tanh_derivative)

#best model
weights_data_best <- neural_learn(X, y, layers = 1, h1_nodes = 12, epochs = 10000,
								  activation_function = tanh,
								  activation_function_derivative = tanh_derivative,
								  init = F)

weights_data_03 <- neural_learn(X, y, layers = 1, h1_nodes = 12, epochs = 500,
								activation_function = tanh,
								activation_function_derivative = tanh_derivative)
weights_data_07 <- neural_learn(X, y, layers = 1, h1_nodes = 12, epochs = 500,
								activation_function = tanh,
								activation_function_derivative = tanh_derivative)
weights_data_05 <- neural_learn(X, y, layers = 1, h1_nodes = 12, epochs = 500,
								activation_function = tanh,
								activation_function_derivative = tanh_derivative)

#saved as weights_data12.RData

#creating plot to display comparision between neural networks
MSEdf <- data.frame(acc_1 = weights_data_03$accuracy,
					acc_2 = weights_data_05$accuracy,
					acc_3 = weights_data_07$accuracy)
print(MSEdf)
MSEdf_long <- pivot_longer(MSEdf,
						cols = c("acc_1", "acc_2", "acc_3"),
						names_to = "Cut",
						values_to = "Acc")
MSEdf_long$Inputs = as.factor(rep(1:3, times = 5))
ep <- seq(from = 100, by = 100, length.out = 5)
MSEdf_long$Epochs <- rep(ep, each = 3)

ggplot(MSEdf_long, aes(x = Epochs, y = MSE, color = Inputs)) +
	labs(title = "MSE vs. Epochs by combination of variables given as input") +
	geom_line() +
	scale_colour_manual("Inputs", values = c("1" = "red", "2" = "blue", "3" = "green", "4" = "black")) +
	theme_minimal()

dane_do_plota <- data.frame(acc = c(acc_1 = weights_data_03$accuracy,
									acc_2 = weights_data_05$accuracy,
									acc_3 = weights_data_07$accuracy),
							cut = factor(c(0.3, 0.5, 0.7)))

ggplot(dane_do_plota, aes(x = cut, y = acc, fill = cut)) +
	geom_bar(stat = "identity") +
	scale_fill_manual(values = c(
		"0.3" = "#FF5733",
		"0.5" = "#33FF57",
		"0.7" = "#FFDDA7"
	)) +
	guides(fill = "none") +
	theme_minimal()
