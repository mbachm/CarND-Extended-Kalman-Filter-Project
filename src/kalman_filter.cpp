#include "kalman_filter.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * KF Measurement update step
   */
  VectorXd y = z - H_ * x_;
  Estimate(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * EKF Measurement update step
   */
  VectorXd y = z - h(x_);
  y(1) = normalize_phi(y(1));
  Estimate(y);
}

void KalmanFilter::Estimate(const VectorXd &y) {
  /**
   * Estamte the update for both EK and EKF as this part stays the 
   * same for both update functions
   */
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;
    
  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

VectorXd KalmanFilter::h(const VectorXd &x) {
  /**
   * Calculates h'(x) for the EKF Measurement update step
   */
  
  // extract position and velocity
  double px = x(0);
  double py = x(1);
  double vx = x(2);
  double vy = x(3);

  //Calculate rho, phi and rhodot with a check division by zero
  double rho = sqrt(px*px + py*py);
  if(rho < 0.0001){
    rho = 0.0001;
  }
  double phi = normalize_phi(atan2(py, px));
  double rho_dot = (px*vx + py*vy) / rho;
  
  //generate h'(x)
  VectorXd hx = VectorXd(3);
  hx << rho, phi, rho_dot;
  
  return hx;
}

double KalmanFilter::normalize_phi(double phi) {
  /**
   * Normalizes phi wirth atan2
   */
  return atan2(sin(phi), cos(phi));
}
