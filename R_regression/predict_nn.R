predict_nn <- function(model, X, layers, activation_function) {
	n <- nrow(X)

	if(layers == 1) {
		h_out <- activation_function(X %*% model$weights_0_1 + matrix(rep(model$bias_hidden, n), byrow = TRUE, nrow = n))

		#pred <- activation_function(h_out %*% model$weights_1_2 + matrix(rep(model$bias_output, n), byrow = TRUE, nrow = n))
		pred <- h_out %*% model$weights_1_2 + matrix(rep(model$bias_output, n), byrow = TRUE, nrow = n)
	} else {
		h1_out <- activation_function(X %*% model$weights_0_1 + matrix(rep(model$bias_h1, n), byrow = TRUE, nrow = n))
		h2_out <- activation_function(h1_out %*% model$weights_1_2 + matrix(rep(model$bias_h2, n), byrow = TRUE, nrow = n))
		#pred <- activation_function(h2_out %*% model$weights_2_3 + matrix(rep(model$bias_output, n), byrow = TRUE, nrow = n))
		pred <- h2_out %*% model$weights_2_3 + matrix(rep(model$bias_output, n), byrow = TRUE, nrow = n)
	}
	return(pred)
}
