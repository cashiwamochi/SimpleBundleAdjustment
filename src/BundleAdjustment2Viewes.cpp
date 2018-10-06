#include "BundleAdjustment2Viewes.hpp"

#include <cmath>
#include <random>

namespace BA2Viewes {

  Optimizer::Optimizer(const PoseAndStructure _pose_and_structure, const BAMode _mode)
  : m_pose_and_structure(_pose_and_structure), me_mode(_mode)
  {
    mb_verbose = false;
    mpm_images = std::make_pair(cv::noArray().getMat(), cv::noArray().getMat());
  }

  cv::Mat Optimizer::ComputeJ(const std::vector<cv::Mat>& vm_data_for_process, const PoseAndStructure& _pose_and_structure) {
    cv::Mat J;
    const cv::Mat K = _pose_and_structure.m_Kd;

    switch(me_mode)
    {
      case BA2Viewes::POSE : {
        assert( vm_data_for_process.size() == 2 );
        const int N_points = _pose_and_structure.m_point3d.cols;
        const int N_cameras = 2;
        J = cv::Mat::zeros(2*N_points*N_cameras, 6*N_cameras, CV_64F);

        std::vector<cv::Mat> vm_poses{vm_data_for_process[0], vm_data_for_process[1]};
        cv::Mat point3d_homo = cv::Mat::ones(4, N_points, CV_64F);
        _pose_and_structure.m_point3d.copyTo(point3d_homo.rowRange(0,3));

        // Jacobian
        for(int i = 0; i < N_cameras; i++) {
          for(int j = 0; j < N_points; j++) {
            cv::Mat tmp = vm_poses[i]*point3d_homo;
            double x = tmp.at<double>(0,j);
            double y = tmp.at<double>(1,j);
            double z = tmp.at<double>(2,j);

            cv::Mat _j = (cv::Mat_<double>(2, 6)
              << K.at<double>(0,0)/z, 0.0, - K.at<double>(0,0)*x/(z*z), - K.at<double>(0,0)*x*y/(z*z), K.at<double>(0,0)*( 1.0 + ((x*x)/(z*z)) ), - K.at<double>(0,0)*y/z,
                 0.0, K.at<double>(1,1)/z, - K.at<double>(1,1)*y/(z*z), - K.at<double>(1,1)*( 1.0 + ((y*y)/(z*z)) ), K.at<double>(1,1)*((x*y)/(z*z)), K.at<double>(1,1)*x/z);

            _j.copyTo(J.rowRange(i*N_points*2 + 2*j, i*N_points*2 + 2*j + 2).colRange(6*i, 6*i + 6));
          }
        }
      } break;

      case BA2Viewes::STRUCTURE : {
        assert( vm_data_for_process.size() == 1 );
        //const cv::Mat point3d_homo = vm_data_for_process[0];
        cv::Mat point3d_homo = cv::Mat::ones(4, vm_data_for_process[0].cols, CV_64F);
        vm_data_for_process[0].copyTo(point3d_homo.rowRange(0,3));

        std::vector<cv::Mat> vm_poses{_pose_and_structure.vp_pose_and_structure[0].first, _pose_and_structure.vp_pose_and_structure[1].first};

        const int N_points = _pose_and_structure.m_point3d.cols;
        const int N_cameras = 2; // this is constant becaese this system is used for BA in 2-viewes.

        J = cv::Mat::zeros(2*N_points*N_cameras, 3*N_points, CV_64F);

        // Jacobian
        for(int i = 0; i < N_cameras; i++) {
          for(int j = 0; j < N_points; j++) {
            cv::Mat tmp = vm_poses[i]*point3d_homo;
            double x = tmp.at<double>(0,j);
            double y = tmp.at<double>(1,j);
            double z = tmp.at<double>(2,j);

            cv::Mat _j = (cv::Mat_<double>(2, 3)
              << K.at<double>(0,0)/z, 0.0, - K.at<double>(0,0)*x/(z*z),
                 0.0, K.at<double>(1,1)/z, - K.at<double>(1,1)*y/(z*z));

            _j *= vm_poses[i].colRange(0, 3);
            _j.copyTo(J.rowRange(i*N_points*2 + 2*j, i*N_points*2 + 2*j + 2).colRange(3*j, 3*j + 3));
          }
        }
      } break;

      case BA2Viewes::FULL : {
        assert( vm_data_for_process.size() == 3 );
        cv::Mat point3d_homo = cv::Mat::ones(4, vm_data_for_process[0].cols, CV_64F);
        vm_data_for_process[0].copyTo(point3d_homo.rowRange(0,3));
        std::vector<cv::Mat> vm_poses{vm_data_for_process[1], vm_data_for_process[2]};

        const int N_points = _pose_and_structure.m_point3d.cols;
        const int N_cameras = 2; // Only 1 camera is used in Optimization

        J = cv::Mat::zeros(2*N_points*N_cameras, 5 + 3*N_points, CV_64F);

        // Jacobian
        for(int i = 0; i < N_cameras; i++) {
          for(int j = 0; j < N_points; j++) {
            cv::Mat tmp = vm_poses[i]*point3d_homo;
            double x = tmp.at<double>(0,j);
            double y = tmp.at<double>(1,j);
            double z = tmp.at<double>(2,j);

            cv::Mat _j_structure = (cv::Mat_<double>(2, 3)
              << K.at<double>(0,0)/z, 0.0, - K.at<double>(0,0)*x/(z*z),
                 0.0, K.at<double>(1,1)/z, - K.at<double>(1,1)*y/(z*z));

            _j_structure *= vm_poses[i].colRange(0, 3);

            _j_structure.copyTo(J.rowRange(i*N_points*2 + 2*j, i*N_points*2 + 2*j + 2).colRange(5 + 3*j, 5 + 3*j + 3));

            /*
            へシアン行列がフルランクであるためには，自由度を下げる必要がある．
            つまり，第1視点を[I 0]に固定し，第2視点の6成分のうち一つを固定する．ここでは，第2視点の平行移動のxを固定した．
            ゆえに，第2視点の自由度は 5
            */
            if(i == 1) {
              cv::Mat _j_cam = (cv::Mat_<double>(2, 5)
                  << 0.0, - K.at<double>(0,0)*x/(z*z), - K.at<double>(0,0)*x*y/(z*z), K.at<double>(0,0)*( 1.0 + ((x*x)/(z*z)) ), - K.at<double>(0,0)*y/z,
                     K.at<double>(1,1)/z, - K.at<double>(1,1)*y/(z*z), - K.at<double>(1,1)*( 1.0 + ((y*y)/(z*z)) ), K.at<double>(1,1)*((x*y)/(z*z)), K.at<double>(1,1)*x/z);
              _j_cam.copyTo(J.rowRange(i*N_points*2 + 2*j, i*N_points*2 + 2*j + 2).colRange(0, 5));
            }

          }
        }

      } break;

    }
    return J;
  }

  double Optimizer::ComputeReprojectionError(cv::Mat& mat_reprojection_error, const std::vector<cv::Mat> vm_data_for_process, const PoseAndStructure& _pose_and_structure) {
    double error = -1.0;
    const cv::Mat K = _pose_and_structure.m_Kd;
    switch(me_mode)
    {
      case BA2Viewes::POSE : {
        assert( vm_data_for_process.size() == 2 );
        double reprojection_error = 0.0;
        const int N_points = _pose_and_structure.m_point3d.cols;
        const int N_cameras = 2;
        cv::Mat point3d_homo = cv::Mat::ones(4, N_points, CV_64F);
        _pose_and_structure.m_point3d.copyTo(point3d_homo.rowRange(0,3));

        // Reprojection Error (b)
        mat_reprojection_error = cv::Mat::zeros(2*N_cameras*N_points, 1, CV_64F);

        std::vector<cv::Mat> vm_point2d_noise(2);
        vm_point2d_noise[0] = K * vm_data_for_process[0] * point3d_homo;
        vm_point2d_noise[1] = K * vm_data_for_process[1] * point3d_homo;

        for(int i = 0; i < (int)vm_point2d_noise[0].cols; i++) {
          vm_point2d_noise[0].at<double>(0,i) = vm_point2d_noise[0].at<double>(0,i)/vm_point2d_noise[0].at<double>(2,i);
          vm_point2d_noise[0].at<double>(1,i) = vm_point2d_noise[0].at<double>(1,i)/vm_point2d_noise[0].at<double>(2,i);
          vm_point2d_noise[0].at<double>(2,i) = vm_point2d_noise[0].at<double>(2,i)/vm_point2d_noise[0].at<double>(2,i);

          vm_point2d_noise[1].at<double>(0,i) = vm_point2d_noise[1].at<double>(0,i)/vm_point2d_noise[1].at<double>(2,i);
          vm_point2d_noise[1].at<double>(1,i) = vm_point2d_noise[1].at<double>(1,i)/vm_point2d_noise[1].at<double>(2,i);
          vm_point2d_noise[1].at<double>(2,i) = vm_point2d_noise[1].at<double>(2,i)/vm_point2d_noise[1].at<double>(2,i);
        }

        for(int i = 0; i < N_cameras; i++) {
          for(int j = 0; j < N_points; j++) {

            double d_x = (vm_point2d_noise[i].at<double>(0,j) - _pose_and_structure.vp_pose_and_structure[i].second[j].x);
            double d_y = (vm_point2d_noise[i].at<double>(1,j) - _pose_and_structure.vp_pose_and_structure[i].second[j].y);

            mat_reprojection_error.at<double>(i*N_points*2 + 2*j, 0) = d_x;
            mat_reprojection_error.at<double>(i*N_points*2 + 2*j + 1, 0) = d_y;
            reprojection_error += sqrt(d_x*d_x + d_y*d_y);
          }
        }

        error = reprojection_error/(double)(N_cameras*N_points);
      } break;

      case BA2Viewes::STRUCTURE: {
        assert( vm_data_for_process.size() == 1 );
        double reprojection_error = 0.0;
        const int N_points = _pose_and_structure.m_point3d.cols;
        const int N_cameras = 2;
        cv::Mat point3d_homo = cv::Mat::ones(4,vm_data_for_process[0].cols,CV_64F);
        vm_data_for_process[0].copyTo(point3d_homo.rowRange(0,3));

        // Reprojection Error (b)
        mat_reprojection_error = cv::Mat::zeros(2*N_cameras*N_points, 1, CV_64F);

        std::vector<cv::Mat> vm_point2d_noise(2);
        vm_point2d_noise[0] = K * _pose_and_structure.vp_pose_and_structure[0].first * point3d_homo;
        vm_point2d_noise[1] = K * _pose_and_structure.vp_pose_and_structure[1].first * point3d_homo;

        for(int i = 0; i < (int)vm_point2d_noise[0].cols; i++) {
          vm_point2d_noise[0].at<double>(0,i) = vm_point2d_noise[0].at<double>(0,i)/vm_point2d_noise[0].at<double>(2,i);
          vm_point2d_noise[0].at<double>(1,i) = vm_point2d_noise[0].at<double>(1,i)/vm_point2d_noise[0].at<double>(2,i);
          vm_point2d_noise[0].at<double>(2,i) = vm_point2d_noise[0].at<double>(2,i)/vm_point2d_noise[0].at<double>(2,i);

          vm_point2d_noise[1].at<double>(0,i) = vm_point2d_noise[1].at<double>(0,i)/vm_point2d_noise[1].at<double>(2,i);
          vm_point2d_noise[1].at<double>(1,i) = vm_point2d_noise[1].at<double>(1,i)/vm_point2d_noise[1].at<double>(2,i);
          vm_point2d_noise[1].at<double>(2,i) = vm_point2d_noise[1].at<double>(2,i)/vm_point2d_noise[1].at<double>(2,i);
        }

        for(int i = 0; i < N_cameras; i++) {
          for(int j = 0; j < N_points; j++) {

            double d_x = (vm_point2d_noise[i].at<double>(0,j) - _pose_and_structure.vp_pose_and_structure[i].second[j].x);
            double d_y = (vm_point2d_noise[i].at<double>(1,j) - _pose_and_structure.vp_pose_and_structure[i].second[j].y);

            mat_reprojection_error.at<double>(i*N_points*2 + 2*j, 0) = d_x;
            mat_reprojection_error.at<double>(i*N_points*2 + 2*j + 1, 0) = d_y;
            reprojection_error += sqrt(d_x*d_x + d_y*d_y);
          }
        }

        error = reprojection_error/(double)(N_cameras*N_points);
      } break;

      case BA2Viewes::FULL: {
        assert( vm_data_for_process.size() == 3 );
        double reprojection_error = 0.0;
        const int N_points = _pose_and_structure.m_point3d.cols;
        const int N_cameras = 2;
        cv::Mat point3d_homo = cv::Mat::ones(4, N_points, CV_64F);
        vm_data_for_process[0].copyTo(point3d_homo.rowRange(0,3));

        // Reprojection Error (b)
        mat_reprojection_error = cv::Mat::zeros(2*N_cameras*N_points, 1, CV_64F);

        std::vector<cv::Mat> vm_point2d_noise(2);
        vm_point2d_noise[0] = K * vm_data_for_process[1] * point3d_homo;
        vm_point2d_noise[1] = K * vm_data_for_process[2] * point3d_homo;

        for(int i = 0; i < (int)vm_point2d_noise[0].cols; i++) {
          vm_point2d_noise[0].at<double>(0,i) = vm_point2d_noise[0].at<double>(0,i)/vm_point2d_noise[0].at<double>(2,i);
          vm_point2d_noise[0].at<double>(1,i) = vm_point2d_noise[0].at<double>(1,i)/vm_point2d_noise[0].at<double>(2,i);
          vm_point2d_noise[0].at<double>(2,i) = vm_point2d_noise[0].at<double>(2,i)/vm_point2d_noise[0].at<double>(2,i);

          vm_point2d_noise[1].at<double>(0,i) = vm_point2d_noise[1].at<double>(0,i)/vm_point2d_noise[1].at<double>(2,i);
          vm_point2d_noise[1].at<double>(1,i) = vm_point2d_noise[1].at<double>(1,i)/vm_point2d_noise[1].at<double>(2,i);
          vm_point2d_noise[1].at<double>(2,i) = vm_point2d_noise[1].at<double>(2,i)/vm_point2d_noise[1].at<double>(2,i);
        }

        for(int i = 0; i < N_cameras; i++) {
          for(int j = 0; j < N_points; j++) {

            double d_x = (vm_point2d_noise[i].at<double>(0,j) - _pose_and_structure.vp_pose_and_structure[i].second[j].x);
            double d_y = (vm_point2d_noise[i].at<double>(1,j) - _pose_and_structure.vp_pose_and_structure[i].second[j].y);

            mat_reprojection_error.at<double>(i*N_points*2 + 2*j, 0) = d_x;
            mat_reprojection_error.at<double>(i*N_points*2 + 2*j + 1, 0) = d_y;
            reprojection_error += sqrt(d_x*d_x + d_y*d_y);
          }
        }
        error = reprojection_error/(double)(N_cameras*N_points);
      } break;

    }

    return error;
  }

  cv::Mat Optimizer::ComputeUpdateParams(const cv::Mat& J, const cv::Mat& mat_reprojection_error) {
    cv::Mat delta_x;

    cv::Mat minus_b = -1.0 * mat_reprojection_error;
    cv::Mat H = J.t() * J;

    delta_x = H.inv() * J.t() * minus_b;

    return delta_x;
  }

  std::vector<cv::Mat> Optimizer::UpdateParams(const std::vector<cv::Mat>& vm_data_for_process, const cv::Mat& _delta_x) {
    switch(me_mode) {
      case BA2Viewes::POSE: {
        std::vector<cv::Mat> vm_poses = vm_data_for_process;
        const int N_cameras = (int)vm_data_for_process.size();

        for(int i = 0; i < N_cameras; i++) {
          // params on se3 are mapped to SE3
          const cv::Mat t = _delta_x.rowRange(i*6, i*6 + 3);
          const cv::Mat w = _delta_x.rowRange(i*6 + 3, i*6 + 6);
          cv::Mat w_x = (cv::Mat_<double>(3,3) << 0.0, -w.at<double>(2,0), w.at<double>(1,0),
                                                  w.at<double>(2,0), 0.0, -w.at<double>(0,0),
                                                  -w.at<double>(1,0), w.at<double>(0,0), 0.0);
          const double theta = sqrt(w.at<double>(0,0)*w.at<double>(0,0) + w.at<double>(1,0)*w.at<double>(1,0) + w.at<double>(2,0)*w.at<double>(2,0));

          cv::Mat e_w_x = cv::Mat::eye(3,3,CV_64F) + (std::sin(theta)/theta)*w_x + ((1.0-std::cos(theta))/(theta*theta))*w_x*w_x;
          cv::Mat V =  cv::Mat::eye(3,3,CV_64F) + ((1.0-std::cos(theta))/(theta*theta))*w_x + ((theta - std::sin(theta))/(theta*theta*theta))*w_x*w_x;
          cv::Mat Vt = V * t;
          cv::Mat delta_SE3 = cv::Mat::eye(4,4,CV_64F);
          e_w_x.copyTo(delta_SE3.rowRange(0,3).colRange(0,3));
          Vt.copyTo(delta_SE3.rowRange(0,3).col(3));

          // Update!
          vm_poses[i] = vm_poses[i] * delta_SE3;
        }

        return vm_poses;

      } break;
      case BA2Viewes::STRUCTURE: {
        cv::Mat _point3d = vm_data_for_process[0];
        const int N_points = _point3d.cols;
        cv::Mat new_point3d = _point3d.clone();

        for(int i = 0; i < N_points; i++) {
          new_point3d.at<double>(0,i) += _delta_x.at<double>(3*i,0);
          new_point3d.at<double>(1,i) += _delta_x.at<double>(3*i+1,0);
          new_point3d.at<double>(2,i) += _delta_x.at<double>(3*i+2,0);
        }

        return std::vector<cv::Mat>{new_point3d};

      } break;
      case BA2Viewes::FULL: {
        std::vector<cv::Mat> vm_poses{vm_data_for_process[1], vm_data_for_process[2]};

        // params on se3 are mapped to SE3
        const cv::Mat t = (cv::Mat_<double>(3,1) << 0.0, _delta_x.at<double>(0) , _delta_x.at<double>(1));
        const cv::Mat w = _delta_x.rowRange(2, 5);
        cv::Mat w_x = (cv::Mat_<double>(3,3) << 0.0, -w.at<double>(2,0), w.at<double>(1,0),
                                                w.at<double>(2,0), 0.0, -w.at<double>(0,0),
                                                -w.at<double>(1,0), w.at<double>(0,0), 0.0);
        const double theta = sqrt(w.at<double>(0,0)*w.at<double>(0,0) + w.at<double>(1,0)*w.at<double>(1,0) + w.at<double>(2,0)*w.at<double>(2,0));

        cv::Mat e_w_x = cv::Mat::eye(3,3,CV_64F) + (std::sin(theta)/theta)*w_x + ((1.0-std::cos(theta))/(theta*theta))*w_x*w_x;
        cv::Mat V =  cv::Mat::eye(3,3,CV_64F) + ((1.0-std::cos(theta))/(theta*theta))*w_x + ((theta - std::sin(theta))/(theta*theta*theta))*w_x*w_x;
        cv::Mat Vt = V * t;
        cv::Mat delta_SE3 = cv::Mat::eye(4,4,CV_64F);
        e_w_x.copyTo(delta_SE3.rowRange(0,3).colRange(0,3));
        Vt.copyTo(delta_SE3.rowRange(0,3).col(3));

        // Update!
        vm_poses[1] = vm_poses[1] * delta_SE3;

        cv::Mat _point3d = vm_data_for_process[0];
        const int N_points = _point3d.cols;
        cv::Mat new_point3d = _point3d.clone();

        // Upadate!
        for(int i = 0; i < N_points; i++) {
          new_point3d.at<double>(0,i) += _delta_x.at<double>(5 + 3*i,0);
          new_point3d.at<double>(1,i) += _delta_x.at<double>(5 + 3*i+1,0);
          new_point3d.at<double>(2,i) += _delta_x.at<double>(5 + 3*i+2,0);
        }

        return std::vector<cv::Mat>{new_point3d, vm_poses[0], vm_poses[1]};

      } break;
    }

    cv::Mat dummy;
    return std::vector<cv::Mat>{dummy};
  }

  // This returns |reprojection error|^2
  double Optimizer::Run() {
    if(mb_verbose) {
      cv::namedWindow(ms_window_name);
    }
    double error = -1.0;
    switch(me_mode)
    {
      case BA2Viewes::POSE : {
        error = OptimizeOnlyPose();
        } break;
      case BA2Viewes::STRUCTURE : {
        error = OptimizeOnlyStructure();
      } break;
      case BA2Viewes::FULL : {
        error = OptimizeAll();
      } break;

    }

    return error;
  }

  double Optimizer::OptimizeOnlyPose () {
    double error = -1.0;
    std::vector<cv::Mat> vm_poses_for_process{mpm_noised_poses.first, mpm_noised_poses.second};
    for(int iter = 0; iter < m_MAXITER; iter++) {
      cv::Mat J = ComputeJ(vm_poses_for_process, m_pose_and_structure);
      cv::Mat mat_reprojection_error;
      double current_error = ComputeReprojectionError(mat_reprojection_error, std::vector<cv::Mat>{mpm_noised_poses.first, mpm_noised_poses.second}, m_pose_and_structure);
      if(IsConverged(current_error, error)) {
        break;
      }
      error = current_error;
      std::cout << "Reprojection Error : " << error << " ( iter_num = " << iter << " )"<< std::endl;
      cv::Mat delta_x = ComputeUpdateParams(J, mat_reprojection_error);
      vm_poses_for_process = UpdateParams(vm_poses_for_process, delta_x);

      if(mb_verbose) {
        ShowProcess(vm_poses_for_process, m_pose_and_structure);
      }
    }

    return error;
  }

  double Optimizer::OptimizeOnlyStructure () {
    double error = -1.0;
    cv::Mat m_structure_for_process = mm_noised_structure.clone();

    for(int iter = 0; iter < m_MAXITER; iter++) {
      cv::Mat J = ComputeJ(std::vector<cv::Mat>{m_structure_for_process}, m_pose_and_structure);
      cv::Mat mat_reprojection_error;
      double current_error = ComputeReprojectionError(mat_reprojection_error, std::vector<cv::Mat>{m_structure_for_process}, m_pose_and_structure);
      if(IsConverged(current_error, error)) {
        break;
      }
      error = current_error;
      std::cout << "Reprojection Error : " << error << " ( iter_num = " << iter << " )"<< std::endl;
      cv::Mat delta_x = ComputeUpdateParams(J, mat_reprojection_error);
      m_structure_for_process = UpdateParams(std::vector<cv::Mat>{m_structure_for_process}, delta_x)[0];

      if(mb_verbose) {
        ShowProcess(std::vector<cv::Mat>{m_structure_for_process}, m_pose_and_structure);
      }
    }

    return error;
  }

  double Optimizer::OptimizeAll () {
    double error = -1.0;

    std::vector<cv::Mat> vm_data_for_process{mm_noised_structure, mpm_noised_poses.first, mpm_noised_poses.second};
    for(int iter = 0; iter < m_MAXITER; iter++) {
      cv::Mat J = ComputeJ(vm_data_for_process, m_pose_and_structure);
      cv::Mat mat_reprojection_error;
      double current_error = ComputeReprojectionError(mat_reprojection_error, vm_data_for_process, m_pose_and_structure);
      if(IsConverged(current_error, error)) {
        break;
      }
      error = current_error;
      std::cout << "Reprojection Error : " << error << " ( iter_num = " << iter << " )"<< std::endl;
      cv::Mat delta_x = ComputeUpdateParams(J, mat_reprojection_error);
      vm_data_for_process = UpdateParams(vm_data_for_process, delta_x);
      if(mb_verbose) {
        ShowProcess(vm_data_for_process, m_pose_and_structure);
      }
    }

    return error;
  }

  void Optimizer::SetImagePair(const std::pair<cv::Mat,cv::Mat> _pm_images) {
    mpm_images = _pm_images;
    return;
  }

  void Optimizer::SetTargetData(const std::vector<cv::Mat>& _vm_noised_data) {
    switch (me_mode) {
      case BA2Viewes::POSE: {
        /*
         * _vm_noised_data = [1st-camera, 2nd-camera]
         */
        mm_noised_structure = cv::Mat::zeros(3,3,CV_64F);
        mpm_noised_poses = std::make_pair(_vm_noised_data[0],_vm_noised_data[1]);
      }break;
      case BA2Viewes::STRUCTURE: {
        /*
         * _vm_noised_data = [3d-point]
         */
        mm_noised_structure = _vm_noised_data[0].clone();
        mpm_noised_poses = std::make_pair(cv::Mat::eye(3,4,CV_64F),cv::Mat::eye(3,4,CV_64F));
      }break;
      case BA2Viewes::FULL: {
        /*
         * _vm_noised_data = [3d-point, 1st-camera(not noised), 2nd-camera]
         */
        mm_noised_structure = _vm_noised_data[0].clone();
        mpm_noised_poses = std::make_pair(_vm_noised_data[1],_vm_noised_data[2]);
      }
    }
  }

  bool Optimizer::IsConverged(const double current_error, const double previous_error) {
    bool b_stop_optimization = false;
    if(previous_error < 0.0) {
      // This case is 1st optimization
      // do nothing
      return false;
    }

    if(previous_error <= current_error + 0.0000001) {
      b_stop_optimization = true;
    }

    return b_stop_optimization;
  }

  void Optimizer::ShowProcess(const std::vector<cv::Mat> vm_data_for_process, const PoseAndStructure& _pose_and_structure) {
    const cv::Mat K = _pose_and_structure.m_Kd;

    std::vector<cv::Mat> vm_point2d_noise(2);

    switch(me_mode)
    {
      case BA2Viewes::POSE : {
        assert( vm_data_for_process.size() == 2 );
        const int N_points = _pose_and_structure.m_point3d.cols;
        cv::Mat point3d_homo = cv::Mat::ones(4, N_points, CV_64F);
        _pose_and_structure.m_point3d.copyTo(point3d_homo.rowRange(0,3));

        vm_point2d_noise[0] = K * vm_data_for_process[0] * point3d_homo;
        vm_point2d_noise[1] = K * vm_data_for_process[1] * point3d_homo;

        // 正規化
        for(int i = 0; i < (int)vm_point2d_noise[0].cols; i++) {
          vm_point2d_noise[0].at<double>(0,i) = vm_point2d_noise[0].at<double>(0,i)/vm_point2d_noise[0].at<double>(2,i);
          vm_point2d_noise[0].at<double>(1,i) = vm_point2d_noise[0].at<double>(1,i)/vm_point2d_noise[0].at<double>(2,i);
          vm_point2d_noise[0].at<double>(2,i) = vm_point2d_noise[0].at<double>(2,i)/vm_point2d_noise[0].at<double>(2,i);

          vm_point2d_noise[1].at<double>(0,i) = vm_point2d_noise[1].at<double>(0,i)/vm_point2d_noise[1].at<double>(2,i);
          vm_point2d_noise[1].at<double>(1,i) = vm_point2d_noise[1].at<double>(1,i)/vm_point2d_noise[1].at<double>(2,i);
          vm_point2d_noise[1].at<double>(2,i) = vm_point2d_noise[1].at<double>(2,i)/vm_point2d_noise[1].at<double>(2,i);
        }
      } break;
      case BA2Viewes::STRUCTURE: {
        assert( vm_data_for_process.size() == 1 );
        cv::Mat point3d_homo = cv::Mat::ones(4,vm_data_for_process[0].cols,CV_64F);
        const int N_points = _pose_and_structure.m_point3d.cols;
        vm_data_for_process[0].copyTo(point3d_homo.rowRange(0,3));
        vm_point2d_noise[0] = K * _pose_and_structure.vp_pose_and_structure[0].first * point3d_homo;
        vm_point2d_noise[1] = K * _pose_and_structure.vp_pose_and_structure[1].first * point3d_homo;

        // 正規化
        for(int i = 0; i < (int)vm_point2d_noise[0].cols; i++) {
          vm_point2d_noise[0].at<double>(0,i) = vm_point2d_noise[0].at<double>(0,i)/vm_point2d_noise[0].at<double>(2,i);
          vm_point2d_noise[0].at<double>(1,i) = vm_point2d_noise[0].at<double>(1,i)/vm_point2d_noise[0].at<double>(2,i);
          vm_point2d_noise[0].at<double>(2,i) = vm_point2d_noise[0].at<double>(2,i)/vm_point2d_noise[0].at<double>(2,i);

          vm_point2d_noise[1].at<double>(0,i) = vm_point2d_noise[1].at<double>(0,i)/vm_point2d_noise[1].at<double>(2,i);
          vm_point2d_noise[1].at<double>(1,i) = vm_point2d_noise[1].at<double>(1,i)/vm_point2d_noise[1].at<double>(2,i);
          vm_point2d_noise[1].at<double>(2,i) = vm_point2d_noise[1].at<double>(2,i)/vm_point2d_noise[1].at<double>(2,i);
        }

      } break;
      case BA2Viewes::FULL: {
        assert( vm_data_for_process.size() == 3 );
        const int N_points = _pose_and_structure.m_point3d.cols;
        cv::Mat point3d_homo = cv::Mat::ones(4, N_points, CV_64F);
        vm_data_for_process[0].copyTo(point3d_homo.rowRange(0,3));

        vm_point2d_noise[0] = K * vm_data_for_process[1] * point3d_homo;
        vm_point2d_noise[1] = K * vm_data_for_process[2] * point3d_homo;

        // 正規化
        for(int i = 0; i < (int)vm_point2d_noise[0].cols; i++) {
          vm_point2d_noise[0].at<double>(0,i) = vm_point2d_noise[0].at<double>(0,i)/vm_point2d_noise[0].at<double>(2,i);
          vm_point2d_noise[0].at<double>(1,i) = vm_point2d_noise[0].at<double>(1,i)/vm_point2d_noise[0].at<double>(2,i);
          vm_point2d_noise[0].at<double>(2,i) = vm_point2d_noise[0].at<double>(2,i)/vm_point2d_noise[0].at<double>(2,i);

          vm_point2d_noise[1].at<double>(0,i) = vm_point2d_noise[1].at<double>(0,i)/vm_point2d_noise[1].at<double>(2,i);
          vm_point2d_noise[1].at<double>(1,i) = vm_point2d_noise[1].at<double>(1,i)/vm_point2d_noise[1].at<double>(2,i);
          vm_point2d_noise[1].at<double>(2,i) = vm_point2d_noise[1].at<double>(2,i)/vm_point2d_noise[1].at<double>(2,i);
        }
      } break;
    }
    std::pair< std::vector<cv::Point2d>,std::vector<cv::Point2d> > pv_point2d;
    pv_point2d.first.reserve(vm_point2d_noise[0].cols);
    pv_point2d.second.reserve(vm_point2d_noise[1].cols); 
    for(int i = 0; i < vm_point2d_noise[0].cols; i++) {
      pv_point2d.first.push_back(cv::Point2d{vm_point2d_noise[0].at<double>(0,i),vm_point2d_noise[0].at<double>(1,i)});
      pv_point2d.second.push_back(cv::Point2d{vm_point2d_noise[1].at<double>(0,i),vm_point2d_noise[1].at<double>(1,i)});
    }

    cv::Mat m_image0 = mpm_images.first.clone();
    cv::Mat m_image1 = mpm_images.second.clone();
    for(int i = 0; i < vm_point2d_noise[0].cols; i++) {
      cv::drawMarker(m_image0, _pose_and_structure.vp_pose_and_structure[0].second[i], 
                     cv::Scalar(255, 0, 0), cv::MARKER_CROSS, 10, 2);
      cv::drawMarker(m_image0, pv_point2d.first[i], 
                     cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 10, 2);
      cv::drawMarker(m_image1, _pose_and_structure.vp_pose_and_structure[1].second[i], 
                     cv::Scalar(255, 0, 0), cv::MARKER_CROSS, 10, 2);
      cv::drawMarker(m_image1, pv_point2d.second[i], 
                     cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 10, 2);
      
    }

    cv::Mat mat_for_viewer;
    cv::hconcat(m_image0, m_image1, mat_for_viewer);
    cv::putText( mat_for_viewer, "BLUE : Feature Points", cv::Point{10,20}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,0,0),2);
    cv::putText( mat_for_viewer, "GREEN : Reprojected Points", cv::Point{10,45}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0),2);
    cv::resize(mat_for_viewer, mat_for_viewer,cv::Size(m_image0.cols,m_image0.rows/2));
    cv::imshow(ms_window_name, mat_for_viewer);
    cv::waitKey(1);
#if 0
    static int count = 1;
    cv::imwrite(std::to_string(count)+".png", mat_for_viewer);
    count = count + 1;

#endif
    return;
  }

  void Optimizer::SetVerbose(const bool b_verbose) {
    if(b_verbose) {
      if( (!mpm_images.first.empty()) and (!mpm_images.second.empty()) ) {
        mb_verbose = b_verbose;
      }
      else {
        std::cout << "[WARN] : The pair of images is empty, it is need in debug.\n";
        std::cout << "         Please use SetImagePair()." << std::endl;
        mb_verbose = false;
        return;
      }
    }
    mb_verbose = b_verbose;
    return;
  }

  void Optimizer::Spin() {
    if(mb_verbose) {
      while(1) {
        if(cv::waitKey(1)=='q') {
          break;
        }
      }
    }
  }

} // namespace
