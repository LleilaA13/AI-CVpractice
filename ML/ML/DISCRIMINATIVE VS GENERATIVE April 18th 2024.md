	Discriminative
	P(y == 1 | x)
	P(y == 0| x)
	the result of the dot product won't be a probability, you have to have a function that maps it into a prob
	inside the y vec: random data
	with Generative models:
	you arrive to this in a diff. way
	if i want P(Y|X):
	P(Y) prob of selecting class 1 or 0
	w/o prior? how many 1 , how many 0
	we have an estimate for Y
	the hard part is estimating P(X|Y):
	WHY? 
	instead of 2D, if Y is high dimensional data, its much harder to know X
	GDA == Gaussian discriminative analysis
		P(X|Y ) ~ N
		P(X|Y == 0) ~ N(mu, cov matrix)
			P(X| Y == 1) ~ N(mu1, cov matrix)
	distribute like a gaussian (?)
	you can generate data

		X_n -> Y == 0 or -> Y == 1
		you apply the numertor, let's see when Y == 0
		likelihood, prior , get number, bot a probsbility
		
INDEPENDENT RANDOM VARIABLE
	
Given data, learn the parameters:

	