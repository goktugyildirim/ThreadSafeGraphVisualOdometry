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

  int id_point2d;
  cv::Point2d point2d;

  bool is_optimized;

  Observation(
      // Vertex: Camera Pose
      const int &id_view, const cv::Vec6d &pose, const cv::Mat &mat_pose_4x4, const bool &is_keyframe,
      // Vertex: Point 3D
      const int &id_point3d, const cv::Point3d &point3d,
      // Edge Point 2D
      const int &id_point2d, const cv::Point2d &point2d,
      const bool& is_optimized)
  {
    this->id_view = id_view;
    this->pose = pose;
    this->mat_pose_4x4 = mat_pose_4x4;
    this->is_keyframe = is_keyframe;
    this->id_point3d = id_point3d;
    this->point3d = point3d;
    this->id_point2d = id_point2d;
    this->point2d = point2d;
    this->is_optimized = is_optimized;
  }
};

typedef std::shared_ptr<Observation> ObservationSharePtr;

class Map
{
  int count_observations = 0;
  int count_keyframe_observation = 0;

  int curr_view_id = 0;

  std::vector<ObservationSharePtr> observations;
  std::map<int, std::vector<int>> map_cam_pose_to_observations;
  std::map<int, std::vector<int>> map_p3d_to_observations;
  std::map<int, std::vector<int>> map_p3d_to_cam_pose;
  std::map<int, std::vector<int>> map_cam_pose_to_p3d;

  mutable std::shared_mutex mutex;


  void print_constraints_info(const
     std::vector<ObservationSharePtr>& observations)
  {
    std::shared_lock lock(mutex);

    std::cout << "Constraints:" << std::endl;
    for (const MonocularVisualOdometry::ObservationSharePtr& observation : observations)
    {
      std::cout << "Point3D id: " << observation->id_point3d << " | View id: " << observation->id_view  <<
        " | Point2D id: " << observation->id_point2d <<
        " | Point2D: " << observation->id_point2d << " | Point3D: " << observation->point3d
        << " | Point2D: " << observation->point2d << " | is keyframe: " << observation->is_keyframe <<  std::endl;
    }
  }

 public:

  explicit Map() {}

  void push_observation(
      const MonocularVisualOdometry::ObservationSharePtr &observation)
  {
    std::unique_lock lock(mutex);

    int id_observation = count_observations;
    int id_point3d = observation->id_point3d;
    int id_view = observation->id_view;

    curr_view_id = id_view;

    if (observation->is_keyframe)
      count_keyframe_observation++;

    observations.push_back(observation);

    map_cam_pose_to_observations[id_view].push_back(id_observation);
    map_p3d_to_observations[id_point3d].push_back(id_observation);
    map_p3d_to_cam_pose[id_point3d].push_back(id_view);
    map_cam_pose_to_p3d[id_view].push_back(id_point3d);

    count_observations++;
  }


  std::vector<ObservationSharePtr>
  get_edge_of_point3Ds(const int& min_seen,
                       const int& index_start_view_id,
                       const bool& only_keyframe,
                       const bool& print_info)
  {
    std::shared_lock lock(mutex);

    std::cout << "Search constraints over Point3D | Min seen: "
              << min_seen << " | Only keyframe:" << only_keyframe << std::endl;
    std::vector<ObservationSharePtr> obs;

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
        if (only_keyframe)
        {
          for (const int &id_obs: map_p3d_to_observations[id_p3d])
          {
            ObservationSharePtr observation = observations.at(id_obs);
            if (observation->is_keyframe)
              if (observation->id_view >= index_start_view_id)
              obs.push_back(observation);
          }
        } // eof only_keyframe

        if (!only_keyframe)
        {
          for (const int &id_obs: map_p3d_to_observations[id_p3d])
          {
            ObservationSharePtr observation = observations.at(id_obs);
            if (observation->id_view >= index_start_view_id)
              obs.push_back(observation);
          }
        }  // eof !only_keyframe
      } // eof min_seen
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
    std::vector<int> ids_observation = map_cam_pose_to_observations[id_view];
        for (const int& id_observation : ids_observation)
    {
        ObservationSharePtr observation = observations.at(id_observation);
        pxs.push_back(observation->point2d);
        if (print_info)
          std::cout << "Point2D id: " << observation->id_point2d <<
              " | Point2D:" << observation->point2d << std::endl;
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
      for (const int& id_obs :map_p3d_to_observations[id_point3d])
      {
        MonocularVisualOdometry::ObservationSharePtr observation = observations[id_obs];
        pts_3d.push_back(observation->point3d);
        if (print_info)
          std::cout << "Point3D id: " << observation->id_point3d <<
              " | Point3D:" << observation->point3d << std::endl;
        break;
      }
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

  // Initialization example:
  cv::Vec6d p0 = {0, 0, 0, 0, 0, 0};
  cv::Vec6d p1 = {1, 1, 1, 1, 1, 1};
  std::vector<cv::Point2d> c0_px;
  std::vector<cv::Point2d> c1_px;
  std::vector<cv::Point3d> map_points;
  cv::Mat mat_4x4;
  for (int i=0; i<10; i++)
  {
    cv::Point2d px0 = {0,0};
    c0_px.push_back(px0);
    cv::Point2d px1 = {1,1};
    c1_px.push_back(px1);
    cv::Point3d pt = {0,0,0};
    map_points.push_back(pt);
  }

  int id_point2d = 0;

  // Build observations:
  // Build observations for each frame order by order:
  int view_id0 = 0;
  int counter_point3D = 0;
  for (int i=0; i<c0_px.size(); i++)
  {
    cv::Point2d px = c0_px[i];
    cv::Point3d pt = map_points[i];

    MonocularVisualOdometry::ObservationSharePtr observation =
      std::make_shared<MonocularVisualOdometry::Observation>(0,p0,
                                                               mat_4x4,
                                                               true,
                                                               counter_point3D, pt,
                                                               id_point2d, px,
                                                               false);
    map->push_observation(observation);
    id_point2d++;
    counter_point3D++;
  }

  int view_id1 = 1;
  counter_point3D = 0;
  for (int i=0; i<c1_px.size(); i++)
  {
    cv::Point2d px = c1_px[i];
    cv::Point3d pt = map_points[i];

    MonocularVisualOdometry::ObservationSharePtr observation =
        std::make_shared<MonocularVisualOdometry::Observation>(1,p1,
                                                               mat_4x4,
                                                               true,
                                                               counter_point3D, pt,
                                                               id_point2d, px,
                                                               false);
    map->push_observation(observation);
    id_point2d++;
    counter_point3D++;
  }


  // Example functions:
  std::vector<MonocularVisualOdometry::ObservationSharePtr> constraints
      = map->get_edge_of_point3Ds(0, 0,true, true );
  std::vector<cv::Point2d> pxs = map->get_points2D_of_frame(1, true);
  std::vector<cv::Point3d> pts = map->get_points3D_of_frame(0, true);




  return 0;
}
