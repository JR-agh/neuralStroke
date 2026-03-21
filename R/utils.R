min_max_scale <- function(x) {
	(x - min(x)) / (max(x) - min(x))
}

sigmoid <- function(x) {
	1 / (1 + exp(-x))
}

sigmoid_derivative <- function(x) {
	x * (1 - x)
}

relu <- function(x) {
	if(x > 0)
		x
	else
		0
}

relu_derivative <- function(x) {
	if(x > 0)
		1
	else
		0
}

linear <- function(x) {
	x
}

linear_derivative <- function(x) {
	1
}

relu_leak <- function(x) {
	if(x >= 0)
		x
	else
		0.1*x
}

relu_leak_derivative <- function(x) {
	if(x >= 0)
		1
	else
		0.1
}
