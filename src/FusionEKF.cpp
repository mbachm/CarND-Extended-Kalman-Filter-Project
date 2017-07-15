#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
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

  VectorXd x_(4);
  x_ << 1, 1, 1, 1;
  
  MatrixXd P_ = MatrixXd(4, 4);
  P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;
  
  MatrixXd F_ = MatrixXd(4, 4);
  MatrixXd Q_ = MatrixXd(4, 4);
  
  ekf_.Init(x_, P_, F_, H_laser_, R_laser_, Q_);
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
    
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      double rho = measurement_pack.raw_measurements_[0];
      double phi = ekf_.normalize_phi(measurement_pack.raw_measurements_[1]);
      double px = rho * cos(phi);
      double py = rho * sin(phi);
      
      //initialize state
      ekf_.x_ << px, py, 0, 0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      //initialize state.
      ekf_.x_[0] = measurement_pack.raw_measurements_[0];
      ekf_.x_[1] = measurement_pack.raw_measurements_[1];
    }
    
    previous_timestamp_ = measurement_pack.timestamp_;

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
  ekf_.F_ << 1, 0, dt, 0,
             0, 1, 0, dt,
             0, 0, 1, 0,
             0, 0, 0, 1;
  
  //Update the process noise covariance matrix.
  double dt_2 = dt * dt;
  double dt_3 = dt_2 * dt;
  double dt_4 = dt_3 * dt;
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4/4*noise_ax_, 0, dt_3/2*noise_ax_, 0,
             0, dt_4/4*noise_ay_, 0, dt_3/2*noise_ay_,
             dt_3/2*noise_ax_, 0, dt_2*noise_ax_, 0,
             0, dt_3/2*noise_ay_, 0, dt_2*noise_ay_;
    
  ekf_.Predict();
  previous_timestamp_ = measurement_pack.timestamp_;

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    //Use the radar sensor type to perform the update step.
    ekf_.R_ = R_radar_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    
    VectorXd z(3);
    double rho = measurement_pack.raw_measurements_[0];
    double phi = ekf_.normalize_phi(measurement_pack.raw_measurements_[1]);
    double rhodot = measurement_pack.raw_measurements_[2];
    z << rho, phi, rhodot;
    
    //Update the state and covariance matrices.
    if(!ekf_.H_.isZero()) {
      ekf_.UpdateEKF(z);
    } else {
        cout << "Jacobian matrix is zero!" << endl;
    }
  } else {
    //Use the laser sensor type to perform the update step.
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    
    VectorXd z(2);
    double px = measurement_pack.raw_measurements_[0];
    double py = measurement_pack.raw_measurements_[1];
    z << px, py;
    
    //Update the state and covariance matrices.
    ekf_.Update(z);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
