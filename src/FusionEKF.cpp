#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
  
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  ekf_ = KalmanFilter();
  tools = Tools();
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
    
    

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // Initialize the state ekf_.x_ with the first measurement.
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
    
    //Create the covariance matrix.
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ << 1, 0, 1, 0,
               0, 1, 0, 1,
               1, 0, 1, 0,
               0, 1, 0, 1;
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, 1, 0,
               0, 1, 0, 1,
               0, 0, 1, 0,
               0, 0, 0, 1;
    
    
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      //Conversion
      auto rho = measurement_pack.raw_measurements_[0];
      auto phi = measurement_pack.raw_measurements_[1];
      auto px = rho * cos(-phi);
      auto py = rho * sin(-phi);
      
      //initialize state
      ekf_.x_[0] = px;
      ekf_.x_[1] = py;
      ekf_.R_ = R_radar_;
      ekf_.H_ = Hj_;
      ekf_.H_ << 1, 1, 0, 0,
                 1, 1, 0, 0,
                 1, 1, 1, 1;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      //initialize state.
      ekf_.x_[0] = measurement_pack.raw_measurements_[0];
      ekf_.x_[1] = measurement_pack.raw_measurements_[1];
      ekf_.R_ = R_laser_;
      ekf_.H_ = H_laser_;
      ekf_.H_ << 1, 0, 0, 0,
                 0, 1, 0, 0;
    }
    
    long x_size = ekf_.x_.size();
    ekf_.I_ = MatrixXd::Identity(x_size, x_size);

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  //Time is measured in seconds.
  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
  
  //Update the state transition matrix F according to the new elapsed time.
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;
  
  //Update the process noise covariance matrix.
  ekf_.Q_(0,0) = noise_ax_ * pow(dt,4) / 4;
  ekf_.Q_(0,2) = noise_ax_ * pow(dt,3) / 2;
  ekf_.Q_(1,1) = noise_ay_ * pow(dt,4) / 4;
  ekf_.Q_(1,3) = noise_ay_ * pow(dt,3) / 2;
  ekf_.Q_(2,0) = noise_ax_ * pow(dt,3) / 2;
  ekf_.Q_(2,2) = noise_ax_ * pow(dt,2);
  ekf_.Q_(3,1) = noise_ay_ * pow(dt,3) / 2;
  ekf_.Q_(3,3) = noise_ay_ * pow(dt,2);
    
  ekf_.Predict();
  previous_timestamp_ = measurement_pack.timestamp_;

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    //Use the radar sensor type to perform the update step.
    ekf_.R_ = R_radar_;
    ekf_.H_ = Hj_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    
    VectorXd z(3);
    auto rho = measurement_pack.raw_measurements_[0];
    auto phi = measurement_pack.raw_measurements_[1];
    auto rhodot = measurement_pack.raw_measurements_[2];
    z << rho, phi, rhodot;
    
    //Update the state and covariance matrices.
    ekf_.UpdateEKF(z);
  } else {
    //Use the laser sensor type to perform the update step.
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    
    VectorXd z(2);
    auto px = measurement_pack.raw_measurements_[0];
    auto py = measurement_pack.raw_measurements_[1];
    z << px, py;
    
    //Update the state and covariance matrices.
    ekf_.Update(z);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
