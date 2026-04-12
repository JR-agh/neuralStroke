library(xts)

clean_stock_data <- function(file_path) {
	raw_data <- read.csv(file_path, row.names = 1)

	data_xts <- as.xts(raw_data)

	# cena zamnkięcia z następnego dnia
	# k = -1 przesuwa przyszłość do dzisiaj
	data_xts$Next_Close <- stats::lag(data_xts[, "PYPL.Close"], k = -1)

	# usunięcie ostatniego wiersza (NA w Next_Close)
	data_cln <- na.omit(data_xts)
	return(as.data.frame(data_cln))
}
