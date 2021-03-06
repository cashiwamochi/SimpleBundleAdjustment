#include <vector>
#include <opencv2/opencv.hpp>


namespace BA2Viewes {

  enum BAMode {
    POSE = 0,
    STRUCTURE = 1,
    FULL = 2
  };

  struct PoseAndStructure {
    cv::Mat m_Kd;
    std::vector< std::pair<cv::Mat, std::vector<cv::Point2d> > > vp_pose_and_structure; // 2
    cv::Mat m_point3d;
  };

  class Optimizer {
  public:
    Optimizer(const PoseAndStructure _pose_and_structure, const BAMode _mode);
    ~Optimizer(){ if(mb_verbose) {cv::destroyWindow(ms_window_name);} };
    double Run();
    void SetTargetData(const std::vector<cv::Mat>& _vm_noised_data);
    void SetImagePair(const std::pair<cv::Mat,cv::Mat> _pm_images);
    void SetVerbose(const bool b_verbose);
    void Spin();

  private:
    double OptimizeOnlyStructure(); // Structure-only BA
    double OptimizeOnlyPose(); // Pose-only BA
    double OptimizeAll(); // Full BA

    cv::Mat ComputeJ(const std::vector<cv::Mat>& vm_data_for_process, const PoseAndStructure& _pose_and_structure);
    double ComputeReprojectionError(cv::Mat& mat_reprojection_error, const std::vector<cv::Mat> vm_data_for_process, const PoseAndStructure& _pose_and_structure);

    cv::Mat ComputeUpdateParams(const cv::Mat& J, const cv::Mat& mat_reprojection_error);
    std::vector<cv::Mat> UpdateParams(const std::vector<cv::Mat>& vm_data_for_process, const cv::Mat& _delta_x);

    bool IsConverged(const double current_error, const double previous_error);

    void ShowProcess(const std::vector<cv::Mat> vm_data_for_process, const PoseAndStructure& _pose_and_structure);

    const PoseAndStructure m_pose_and_structure;
    const BAMode me_mode;
    const int m_MAXITER = 30;


    cv::Mat mm_noised_structure;
    std::pair<cv::Mat, cv::Mat> mpm_noised_poses;

    bool mb_verbose;
    std::pair<cv::Mat,cv::Mat> mpm_images;
    const std::string ms_window_name = "debug-window ( Press q to exit )";
  };
}
