#include "camera_imu_time_sync/utils.hpp"
#include <ros/ros.h>

// 计算点云之间的角度变换 ICP算法
double calcAngleBetweenPointclouds(
    const pcl::PointCloud<pcl::PointXYZ>& prev_pointcloud,
    const pcl::PointCloud<pcl::PointXYZ>& pointcloud) 
{
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_pointcloud_sampled(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_sampled(new pcl::PointCloud<pcl::PointXYZ>);

    // shared_pointers needed by icp, no-op destructor to prevent them being cleaned up after use
    // 输入的点云转换成点云指针（记录） 避免作为局部变量之后被析构
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr prev_pointcloud_ptr(&prev_pointcloud, [](const pcl::PointCloud<pcl::PointXYZ>*) {});
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr pointcloud_ptr(&pointcloud, [](const pcl::PointCloud<pcl::PointXYZ>*) {});
    // 重采样滤波减小计算量
    constexpr int kMaxSamples = 2000;
    pcl::RandomSample<pcl::PointXYZ> sample(false);
    sample.setSample(kMaxSamples);

    sample.setInputCloud(prev_pointcloud_ptr);
    sample.filter(*prev_pointcloud_sampled);

    sample.setInputCloud(pointcloud_ptr);
    sample.filter(*pointcloud_sampled);
    // 输入src 和 tgt
    icp.setInputSource(prev_pointcloud_sampled);
    icp.setInputTarget(pointcloud_sampled);
    // 对齐
    pcl::PointCloud<pcl::PointXYZ> final;
    icp.align(final);
    // 获取变换矩阵
    Eigen::Matrix4f tform = icp.getFinalTransformation();
    // 得到轴角表示法的角度
    double angle = Eigen::AngleAxisd(tform.topLeftCorner<3, 3>().cast<double>()).angle();

    return angle;
}

double calcAngleBetweenImages(const cv::Mat& prev_image,
                              const cv::Mat& image, float focal_length) {
  constexpr int kMaxCorners = 100;
  constexpr double kQualityLevel = 0.01;
  constexpr double kMinDistance = 10;

  std::vector<cv::Point2f> prev_points; // 接收提取后的角点
  // 图像角点检测
  cv::goodFeaturesToTrack(prev_image, prev_points, kMaxCorners, kQualityLevel, kMinDistance);
  // 未能找到角点会报错
  if (prev_points.size() == 0) {
    ROS_ERROR("Tracking has failed cannot calculate angle");
    return 0.0;
  }

  std::vector<cv::Point2f> points;
  std::vector<uint8_t> valid;
  std::vector<float> err;
  // prev_image 和 image是输入量分别代表不同尺度的金字塔（前后两帧图像？？？？）
  // point（输出二维点的矢量） valid（输出状态向量） 和err都是输出量
  cv::calcOpticalFlowPyrLK(prev_image, image, prev_points, points, valid, err);

  std::vector<cv::Point2f> tracked_prev_points, tracked_points;
  for (size_t i = 0; i < prev_points.size(); ++i) {
    // 把找到的光流点对应的像素点提取出来
    if (valid[i]) {
      tracked_prev_points.push_back(prev_points[i]);
      tracked_points.push_back(points[i]);
    }
  }

  /*cv::Mat viz_image;
  cv::cvtColor(prev_image, viz_image, cv::COLOR_GRAY2BGR);

  for (size_t i = 0; i < tracked_points.size(); ++i) {
    cv::arrowedLine(viz_image, tracked_prev_points[i], tracked_points[i],
                    cv::Scalar(0, 255, 0));
  }

  cv::namedWindow("Tracked Points", cv::WINDOW_AUTOSIZE);
  cv::imshow("Tracked Points", viz_image);
  cv::waitKey(1);*/

  // close enough for most cameras given the low level of accuracy needed
  // 图像中心点
  const cv::Point2f offset(image.cols / 2.0, image.rows / 2.0);

  for (size_t i = 0; i < tracked_points.size(); ++i) {
    // 从像素坐标转换到相机坐标
    tracked_prev_points[i] = (tracked_prev_points[i] - offset) / focal_length;
    tracked_points[i] = (tracked_points[i] - offset) / focal_length;
    //std::cout << tracked_prev_points[i].x << " " << tracked_prev_points[i].y << std::endl;
  }

  constexpr double kMaxEpipoleDistance = 1e-3; // 指定置信度
  constexpr double kInlierProbability = 0.99; // 输出掩码

  std::vector<uint8_t> inliers; // 输出
  // 求解基本矩阵 即两幅图上特征点对应的变换关系 输入的是提取到光流的点
  cv::Mat cv_F = cv::findFundamentalMat(tracked_prev_points, tracked_points, cv::FM_LMEDS,
                                     kMaxEpipoleDistance, kInlierProbability, inliers);
  Eigen::Matrix3d E, W;
  // 将基本矩阵转换成eigen类
  cv::cv2eigen(cv_F, E);
  // 对基本矩阵进行svd分解
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(E, Eigen::ComputeThinU | Eigen::ComputeThinV);
  W << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;

  Eigen::Matrix3d Ra = svd.matrixU() * W * svd.matrixV().transpose();
  Eigen::Matrix3d Rb = svd.matrixU() * W.transpose() * svd.matrixV().transpose();
  // 角度变换取最小的那个
  double angle = std::min(Eigen::AngleAxisd(Ra).angle(), Eigen::AngleAxisd(Rb).angle());

  return angle;
}