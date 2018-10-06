#pragma once

#include <cmath>
#include <random>
#include <vector>

void AddNoiseToPose (const cv::Mat& _src, cv::Mat& _dst) {
  /*
    _src, _dst = [3 x 4]
  */

  std::vector<double> vd_noise(6);

  const double d_true_value = 0.0;
  double d_variance = 0.05;

  std::random_device seed;
  std::mt19937 engine(seed());

  double d_mu = d_true_value;
  double d_sig = sqrt(d_variance);
  std::normal_distribution<> dist_for_rotation(d_mu, d_sig);

  for (int i=0; i<3; ++i) {
    vd_noise[i] = dist_for_rotation(engine);
  }

  d_variance = 0.01;
  d_sig = sqrt(d_variance);
  std::normal_distribution<> dist_for_translation(d_mu, d_sig);

  for (int i=3; i<6; ++i) {
    vd_noise[i] = dist_for_translation(engine);
  }

  cv::Mat Rt = _src.clone();
  cv::Mat delta_Rt = cv::Mat::eye(4,4,CV_64F);

  cv::Mat delta_rot_x = (cv::Mat_<double>(3,3) << 1.f, 0.f, 0.f,
                                                  0.f, std::cos(vd_noise[0]*M_PI/180.0),std::sin(vd_noise[0]*M_PI/180.0),
                                                  0.f, -std::sin(vd_noise[0]*M_PI/180.0),std::cos(vd_noise[0]*M_PI/180.0));

  cv::Mat delta_rot_y = (cv::Mat_<double>(3,3) << std::cos(vd_noise[1]*M_PI/180.0), 0.f, -std::sin(vd_noise[1]*M_PI/180.0),
                                                  0.f, 1.f, 0.f,
                                                  std::sin(vd_noise[1]*M_PI/180.0), 0.f, std::cos(vd_noise[1]*M_PI/180.0));

  cv::Mat delta_rot_z = (cv::Mat_<double>(3,3) << std::cos(vd_noise[2]*M_PI/180.0), std::sin(vd_noise[2])*M_PI/180.0, 0.f,
                                                  -std::sin(vd_noise[2]*M_PI/180.0), std::cos(vd_noise[2]*M_PI/180.0), 0.f,
                                                  0.f, 0.f, 1.f);

  cv::Mat delta_rot = delta_rot_z * delta_rot_y * delta_rot_x;
  cv::Mat delta_trans = (cv::Mat_<double>(3,1) << vd_noise[3], vd_noise[4], vd_noise[5]);
  delta_rot.copyTo(delta_Rt.rowRange(0,3).colRange(0,3));
  delta_trans.copyTo(delta_Rt.rowRange(0,3).col(3));

  _dst = Rt * delta_Rt;

  return;
}


void AddNoiseToStructure (const cv::Mat& _src, cv::Mat& _dst) {
  /*
    _src, _dst = [3 x (point number)]
  */
  _dst = _src.clone();
  const int N = _dst.cols;

  const double d_true_value = 0.0;
  const double d_variance = 0.1;

  std::random_device seed;
  std::mt19937 engine(seed());

  double d_mu = d_true_value;
  double d_sig = sqrt(d_variance);
  std::normal_distribution<> dist(d_mu, d_sig);
  const int randon_num = 3 * _dst.cols;
  std::vector<double> vd_noise(randon_num);

  for (int i=0; i<N; ++i) {
    vd_noise[i] = dist(engine);
  }

  for(int i = 0; i < _dst.cols; i++) {
    _dst.at<double>(0,i) += vd_noise[3*i];
    _dst.at<double>(1,i) += vd_noise[3*i+1];
    _dst.at<double>(2,i) += vd_noise[3*i+2];
  }

  return;
}
