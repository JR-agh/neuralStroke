library(microbenchmark)

XGPU <- gpu.matrix(X)

raw <- function(x1, x2) {
	return(t(x1) %*% x2)
}
raw(X, X)

microbenchmark(raw = raw(X, X),
			   gpu = raw(XGPU, XGPU),
			   times = 1000)

relu_max <- function(x) {
	return(max(0, x))
}

relu_if <- function(x) {
	if(x > 0)
		return(x)
	else
		return(0)
}

x <- -100:100
microbenchmark(max = sapply(x, relu_max),
			   if_ = sapply(x, relu_if),
			    times = 1e5)

microbenchmark(if_ = sapply(x, relu_if),
			   sigmoid = sapply(x, sigmoid),
			   times = 1e5)
