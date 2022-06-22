#ifndef CPPLOGISTICREGRESSION
#define CPPLOGISTICREGRESSION

class CPPLogisticRegression{
	public:
		//Method for updating the weights and bias
	vector<double> updateWeightsAndBias(int noOfIterations, int noOfRows, int noOfColumns);
		//method for the prediction
		double predict(vector<double> vW, double* X_train_test);
};

#endif

