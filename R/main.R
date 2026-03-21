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

#load model weights
load("./data/weightsData.RData")
#learning model
weightsData <- neural_learn(X, y, hidden_nodes = 12, epochs = 500, init = TRUE, freq = 100)
#save new model weights
save(weightsData, file = "./data/weightsData.RData")
#learning model with 2 hidden layers
weightsData2 <- neural_network_2(X, y, h1_nodes = 8, h2_nodes = 4, epochs = 500, freq = 100, init = TRUE)

#creating plot to display comparision between neural networks
MSEdf <- data.frame(mse1 = weightsData$mse,
					mse2 = weightsData2$mse)
print(MSEdf)
ggplot(MSEdf, aes(x = 1:5, y = mse1)) +
	geom_line() +
	geom_line(aes(x = 1:5, y = mse2, color = "red"))
