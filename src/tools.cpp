#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    
    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if(estimations.size() != ground_truth.size()
       || estimations.size() == 0){
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }
    
    //accumulate squared residuals
    for(unsigned int i=0; i < estimations.size(); ++i){
        VectorXd residual = estimations[i] - ground_truth[i];
        
        //coefficient-wise multiplication
        residual = residual.array()*residual.array();
        rmse += residual;
    }
    
    //calculate the mean
    rmse = rmse/estimations.size();
    
    //calculate the squared root
    rmse = rmse.array().sqrt();
    
    //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    MatrixXd Hj(3,4);
    
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);
    
    //check division by zero
    float added_pow_px_py = pow(px, 2) + pow(py, 2);
    if(fabs(added_pow_px_py) < 0.0001) {
        std::cout << "CalculateJacobian() - Error - Division by zero" << std::endl;
        return Hj;
    }
    
    //compute the Jacobian matrix
    float sqrt_added_pow_px_py = sqrt(added_pow_px_py);
    float mul_vx_py = vx * py;
    float mul_vy_px = vy * px;
    float column2_dom = added_pow_px_py*sqrt_added_pow_px_py;
    
    Hj << (px/sqrt_added_pow_px_py), (py/sqrt_added_pow_px_py), 0, 0,
          (-py/added_pow_px_py), (px/added_pow_px_py), 0, 0,
          (py*(mul_vx_py-mul_vy_px)/column2_dom), (px*(mul_vy_px-mul_vx_py)/column2_dom), (px/sqrt_added_pow_px_py), (py/sqrt_added_pow_px_py);
    
    return Hj;
}
