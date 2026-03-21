clean_data <- function(dane) {
	cln_data <- dane |>
		mutate(Gender = case_match(Gender,
								   "Male" ~ 0,
								   "Female" ~ 1,
								   "Other" ~ 2),
			   satisfaction = case_match(satisfaction,
			   						  "satisfied" ~ 1,
			   						  "neutral or dissatisfied" ~ 0),
			   Class = case_match(Class,
			   				   "Eco" ~ 0,
			   				   "Eco Plus" ~ 0,
			   				   "Business" ~ 1))
	return (cln_data)
}
