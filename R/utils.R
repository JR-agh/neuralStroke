min_max_scale <- function(x) {
	return ((x - min(x)) / (max(x) - min(x)))
}

sigmoid <- function(x) {
	1 / (1 + exp(-x))
}

sigmoid_derivative <- function(x) {
	x * (1 - x)
}
