#include <iostream>
#include <memory>
#include <opencv4/opencv2/core.hpp>
#include <vector>

#include <mutex>
#include <shared_mutex>
#include <map>

using namespace std;

namespace MonocularVisualOdometry {


// Inputs
struct Observation {
  int id_view;
  cv::Vec6d pose;
  cv::Mat mat_pose_4x4;
  bool is_keyframe;

  int id_point3d;
  cv::Point3d point3d;

  cv::Point2d point2d;

  bool is_optimized;

  Observation(
      // Vertex: Camera Pose
      const int &id_view, const cv::Vec6d &pose, const cv::Mat &mat_pose_4x4, const bool &is_keyframe,
      // Vertex: Point 3D
      const int &id_point3d, const cv::Point3d &point3d,
      // Edge Point 2D
      const cv::Point2d &point2d,
      const bool& is_optimized)
  {
    this->id_view = id_view;
    this->pose = pose;
    this->mat_pose_4x4 = mat_pose_4x4;
    this->is_keyframe = is_keyframe;
    this->id_point3d = id_point3d;
    this->point3d = point3d;
    this->is_optimized = is_optimized;
  }
};


class Map
{
  int count_observations = 0;
  int count_keyframe_observation = 0;

  int curr_view_id = 0;

  std::vector<MonocularVisualOdometry::Observation> observations;
  std::map<int, std::vector<int>> map_p3d_to_cam_pose;
  std::map<int, std::vector<int>> map_cam_pose_to_p3d;
  std::map<std::pair<int, int>, int> map_camp_pose_p3d_to_observations;

  mutable std::shared_mutex mutex;

  void print_constraints_info(const
      std::map<int, Observation>& observations)
  {
    std::cout << "Constraints:" << std::endl;
    std::shared_lock lock(mutex);
    for (auto it=observations.begin(); it!=observations.end();it++)
    {
      auto id_obs = it->first;
      auto observation = it->second;

      std::cout << "Point3D id: " << observation.id_point3d << " | View id: " << observation.id_view
        << " | Point3D: " << observation.point3d <<
        " | Point2D: " << observation.point2d << " | is keyframe: " << observation.is_keyframe <<
        " | observation id: " << id_obs << std::endl;
    }
  }

public:

  explicit Map() {}

  void push_observation(
      const MonocularVisualOdometry::Observation &observation)
  {
    std::unique_lock lock(mutex);

    int id_observation = count_observations;
    int id_point3d = observation.id_point3d;
    int id_view = observation.id_view;

    curr_view_id = id_view;

    if (observation.is_keyframe)
      count_keyframe_observation++;

    observations.push_back(observation);

    map_p3d_to_cam_pose[id_point3d].push_back(id_view);
    map_cam_pose_to_p3d[id_view].push_back(id_point3d);
    map_camp_pose_p3d_to_observations[{id_view, id_point3d}] = id_observation;

    count_observations++;
  }

  std::map<int, MonocularVisualOdometry::Observation>
  get_edge_of_point3Ds(const int& min_seen,
                       const int& index_start_view_id,
                       const int& index_end_view_id,
                       const bool& only_keyframe,
                       const bool& print_info)
  {
    std::shared_lock lock(mutex);

    std::cout << "Search constraints over Point3D between view " << index_start_view_id <<
      " and " << index_end_view_id << " | Min seen: "
      << min_seen << " | Search only keyframe: " << only_keyframe << std::endl;

    std::map<int, MonocularVisualOdometry::Observation> obs;

    for (auto it=map_p3d_to_cam_pose.begin(); it!=map_p3d_to_cam_pose.end();it++)
    {

      auto id_p3d = it->first;
      auto count_seen = it->second.size();
/*      if (print_info)
      {
        std::cout << "Query Point3D id: " << id_p3d << " seen count " << count_seen << std::endl;
        for (const auto frame_id : it->second)
          std::cout << "Seen in frame_id:  " << frame_id << std::endl;
      }*/
      if (count_seen >= min_seen)
      {
        for (const int& id_view : it->second)
        {
          int id_obs = map_camp_pose_p3d_to_observations[{id_view, id_p3d}];
          MonocularVisualOdometry::Observation observation = observations.at(id_obs);

          if (observation.id_view < index_start_view_id or
              observation.id_view > index_end_view_id)
            continue;

          if (only_keyframe)
          {
            if (observation.is_keyframe)
            {
              obs.insert(std::make_pair(id_obs, observation));
            }
          } else
          {
            obs.insert(std::make_pair(id_obs, observation));
          }
        }
      }
      }
      std::cout << "Total generated constraint count: " << obs.size() << std::endl;
      std::cout << "Total observation count in map: " << count_observations << std::endl;
      if (print_info)
        print_constraints_info(obs);
      std::cout << "##########################################################"
                   "###################################################################" << std::endl;
    return obs;
  }


  std::vector<cv::Point2d>
  get_points2D_of_frame(const int& id_view,
                        const bool& print_info)
  {
    std::shared_lock lock(mutex);
    std::vector<cv::Point2d> pxs;
    std::cout << "Search for 2D points of view: " << id_view << std::endl;
    std::vector<int> ids_pts = map_cam_pose_to_p3d[id_view];
    for (const int& id_point3d : ids_pts)
    {
      int id_obs = map_camp_pose_p3d_to_observations[{id_view, id_point3d}];
      MonocularVisualOdometry::Observation observation = observations.at(id_obs);
      pxs.push_back(observation.point2d);
      if (print_info)
        std::cout <<  "Point2D: " << observation.point2d << std::endl;
    }
    std::cout << "##########################################################"
                 "###################################################################" << std::endl;
    return pxs;
  }


  std::vector<cv::Point3d>
  get_points3D_of_frame(const int& id_view,
                        const bool& print_info)
  {
    std::shared_lock lock(mutex);

    std::vector<cv::Point3d> pts_3d;
    std::cout << "Search for 3D points of view: " << id_view << std::endl;

    std::vector<int> ids_pts = map_cam_pose_to_p3d[id_view];
    for (const int& id_point3d : ids_pts)
    {
      int id_obs = map_camp_pose_p3d_to_observations[{id_view, id_point3d}];
      MonocularVisualOdometry::Observation observation = observations.at(id_obs);
      pts_3d.push_back(observation.point3d);
      if (print_info)
        std::cout << "Point3D id: " << observation.id_point3d <<
            " | Point3D: " << observation.point3d << std::endl;
    }
    std::cout << "##########################################################"
                 "###################################################################" << std::endl;
    return pts_3d;
  }

  int get_observation_count() { std::shared_lock lock(mutex); return count_observations;}
  int get_keyframe_observation_count() { std::shared_lock lock(mutex); return count_keyframe_observation;}

};

}// namespace MonocularVisualOdometry

typedef std::shared_ptr<MonocularVisualOdometry::Map> MapSharePtr;



int main() {
  MapSharePtr map(new MonocularVisualOdometry::Map);

  // Define vertex poses:
  cv::Vec6d p0 = {0, 0, 0, 0, 0, 0}; // pt0 pt1 pt2p pt3
  cv::Vec6d p1 = {1, 1, 1, 1, 1, 1}; // pt0 pt2p
  cv::Vec6d p2 = {2, 2, 2, 2, 2, 2}; // pt3

  // Define vertex Point3Ds:
  cv::Point3d pt0 = {0,0,0}; // p0 p1
  cv::Point3d pt1 = {1,1,1}; // p0
  cv::Point3d pt2 = {2,2,2}; // p0 p1
  cv::Point3d pt3 = {3,3,3}; // p0 p2

  // Define observations as edges:
  cv::Point2d px0 = {0,0};
  cv::Point2d px1 = {1,1};
  cv::Point2d px2 = {2,2};
  cv::Point2d px3 = {3,3};
  cv::Point2d px4 = {4,4};
  cv::Point2d px5 = {5,5};
  cv::Point2d px6 = {6,6};

  cv::Mat mat_4x4;

  MonocularVisualOdometry::Observation observation0(
      0, p0, mat_4x4,true,0,
    pt0,px0, false);

  MonocularVisualOdometry::Observation observation1(
          0, p0, mat_4x4,true,1,
          pt1,px1, false);

  MonocularVisualOdometry::Observation observation2(
          0, p0, mat_4x4,true,2,
          pt2,px2, false);

  MonocularVisualOdometry::Observation observation3(
          0, p0, mat_4x4,true,3,
          pt3,px3, false);

  MonocularVisualOdometry::Observation observation4(
          1, p1, mat_4x4,true,0,
          pt0,px4, false);

  MonocularVisualOdometry::Observation observation5(
          1, p1, mat_4x4,true,2,
          pt2,px5, false);

  MonocularVisualOdometry::Observation observation6(
          2, p2, mat_4x4,false,3,
          pt3,px6, false);

  map->push_observation(observation0);
  map->push_observation(observation1);
  map->push_observation(observation2);
  map->push_observation(observation3);
  map->push_observation(observation4);
  map->push_observation(observation5);
  map->push_observation(observation6);

  // Example functions:
  std::map<int, MonocularVisualOdometry::Observation> constraints
      = map->get_edge_of_point3Ds(2, 1, 2,
                                  false, true );
  std::vector<cv::Point2d> pxs = map->get_points2D_of_frame(2, true);
  std::vector<cv::Point3d> pts = map->get_points3D_of_frame(1, true);

  return 0;
}
