/* Logistic algorithm in C++
 * ------------------------
 *
 * To make hyperparameters be setup from python 
 *
 * */

#include<iostream>
#include<math.h>
#include<vector>
#include"logistic.hpp"

using namespace std;

vector<double> CPPLogisticRegression::updateWeightsAndBias(int noOfIterations, int noOfRows, int noOfColumns){
	double row_pred_diff = 0.0;
	double total_diff = 0.0;
	double feature_weight[noOfColumns] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
	double total_feature_weight[noOfColumns] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
	double weight_derivative[noOfColumns] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
	double bias_derivative = 0.0;
	double W[noOfColumns] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
	double bias = 0.0;
	vector<double> vWB;

	//Train set
	double X_train[noOfRows][noOfColumns] = {
		{57.0,0.0,0.0,140.0,241.0,0.0,1.0,123.0,1.0,0.2,1.0,0.0,3.0},
		{45.0,1.0,3.0,110.0,264.0,0.0,1.0,2.0,0.0,1.2,1.0,0.0,3.0},
		{68.0,1.0,0.0,144.0,13.0,1.0,1.0,141.0,0.0,3.4,1.0,2.0,3.0},
		{57.0,1.0,0.0,80.0,1.0,0.0,1.0,115.0,1.0,1.2,1.0,1.0,3.0},
		{57.0,0.0,1.0,0.0,236.0,0.0,0.0,174.0,0.0,0.0,1.0,1.0,2.0},
		{61.0,1.0,0.0,140.0,207.0,0.0,0.0,8.0,1.0,1.4,2.0,1.0,3.0},
		{46.0,1.0,0.0,140.0,311.0,0.0,1.0,120.0,1.0,1.8,1.0,2.0,3.0},
		{62.0,1.0,1.0,128.0,208.0,1.0,0.0,140.0,0.0,0.0,2.0,0.0,2.0},
		{62.0,1.0,1.0,128.0,208.0,1.0,0.0,140.0,0.0,0.0,2.0,0.0,2.0}};

	//labels
	double Y[noOfRows] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0};

	for (int l=0; l<noOfIterations; l++){
		for (int i=0; i<noOfRows; i++){
			double Wx = 0.0;
			for (int j=0; j<noOfColumns; j++){
				Wx += W[j] * X_train[i][j];
			}
			//computing (sigma(W.x)+b)-Y
			row_pred_diff = (1/(1 + exp(-(Wx + bias)))) - Y[i];
			for (int k=0; k<noOfColumns; k++){
				//computing (sigma(W.x)+b)-Yx(i)
				feature_weight[k] = row_pred_diff * X_train[i][k];
				//summation(figma) of each feature weight
				total_feature_weight[k] += feature_weight[k];
			}
			//summation of preds
			total_diff += row_pred_diff;
		}
		//updating the weights for eacg feature
		for (int z=0; z<noOfColumns; z++){
			//computing the average of the weights(1/m)
			weight_derivative[z] = total_feature_weight[z]/noOfRows;
			W[z] = W[z] - 0.1*weight_derivative[z];
			//storing the values in a vector
			vWB.push_back(W[z]);
		}
		//calculating the bias
		bias_derivative = total_diff/noOfRows;
		bias = bias - 0.1*bias_derivative;
		vWB.push_back(bias);
	}

	return vWB;
}

double CPPLogisticRegression::predict(vector<double> vW, double* X_train_test){
	static double predictions;
	double Wx_test = 0.0;
	//computing sigma(W.x)
	for (int j=0; j<13; j++){
		Wx_test += (vW[j] * X_train_test[j]);
	}
	//adding the bias term
	predictions = 1/(1 + exp(-(Wx_test + vW.back())));
	//making predictions
	if(predictions > 0.5){
		predictions = 1.0;
	}else{
		predictions = 0.0;
	}
	
	return predictions;
}

extern "C"{
	//vector to store the weights and bias gotten from the updateWeightAndBias() function
	vector<double> vX;
	CPPLogisticRegression* LogisticRegression(){
		CPPLogisticRegression* log_reg = new CPPLogisticRegression();
		return log_reg;
	}

	void fit(CPPLogisticRegression* log_reg){
		vX = log_reg -> updateWeightsAndBias(50,9,13);
	}

	double predict(CPPLogisticRegression* log_reg, double* array){
		return log_reg -> predict(vX, array);
	}
}









