library(dplyr)
library(ggplot2)

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
weights_data <- neural_learn(X, y, layers = 1, h1_nodes = 10, epochs = 500)

#creating plot to display comparision between neural networks
MSEdf <- data.frame(mse = weights_data$mse)
print(MSEdf)
ggplot(MSEdf, aes(x = 1:5, y = mse)) +
	geom_line()
