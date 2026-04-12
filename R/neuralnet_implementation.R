library(neuralnet)

dane <- read.csv("./rawdata/train.csv")

#get functions from other files
file_sources <- list.files("R", full.names = TRUE)
file_sources <- file_sources[-2]
file_sources <- file_sources[-5]
print(file_sources)
sapply(file_sources, source)

#clean data
dane <- clean_data(dane)

#chosing input and benchmark data
Xdf <- dane |>
	select(-c("X", "id", "Customer.Type", "Type.of.Travel")) |>
	select(c("Class", "Online.boarding", "Seat.comfort", "Inflight.entertainment", "On.board.service"))

Xdf <- as.data.frame(lapply(Xdf, normalize))
Xdf$Satisfaction <- dane$satisfaction

nn_model <- neuralnet(Satisfaction ~ Class+Online.boarding+Seat.comfort+Inflight.entertainment+On.board.service,
					  data = Xdf,
					  hidden = 12,
					  #lgorithm = "backprop",
					  #learningrate = 0.3,
					  linear.output = FALSE,
					  err.fct = "sse",
					  threshold = 0.1,
					  stepmax = 1e7)

pred_prob <- predict(nn_model, Xdf)
pred_class <- ifelse(pred_prob > 0.5, 1, 0)
acc <- ifelse(pred_class == Xdf$Satisfaction, 1, 0)
sum(acc)/length(acc)
load("./data/weights_data12.RData")
accuracy = c((sum(acc)/length(acc))*100, weights_data$accuracy)
print(accuracy)
dane_do_plota <- data.frame(accuracy = c((sum(acc)/length(acc))*100, weights_data$accuracy),
							typ = c("neuralnet", "własna implementacja"))
ggplot(dane_do_plota, aes(x = typ, y = accuracy, fill = typ)) +
	geom_bar(stat = "identity") +
	scale_fill_manual(values = c("#00AA00",
								 "#AA0000")) +
	guides(fill = "none") +
	theme_minimal()
