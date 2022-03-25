#include <iostream>
#include <memory>
#include <opencv4/opencv2/core.hpp>
#include <vector>

#include <mutex>
#include <shared_mutex>
#include <numeric>
#include <map>

using namespace std;

namespace MonocularVisualOdometry {


// Inputs
struct Observation
{
  bool is_ref_frame{};
  bool is_keyframe;
  int id_frame;
  cv::Vec6d pose;
  cv::Mat mat_pose_4x4;

  int id_point3d;
  cv::Point3d point3d;

  int id_point2d{};
  cv::Point2d point2d;

  bool is_optimized;
  bool is_initialized;

  int id_obs;

  Observation(
      const int &id_frame, const int &id_point3d,
      const cv::Vec6d &pose,  const cv::Mat &mat_pose_4x4, const cv::Point2d &point2d,
      const cv::Point3d &point3d, const bool &is_keyframe, const bool& is_ref_frame,
      const bool& is_optimized, const bool& is_initialized)
  {
    this->id_frame = id_frame;
    this->pose = pose;
    this->mat_pose_4x4 = mat_pose_4x4;
    this->is_keyframe = is_keyframe;
    this->is_ref_frame;
    this->id_point3d = id_point3d;
    this->point3d = point3d;
    this->point2d = point2d;
    this->is_optimized = is_optimized;
    this->is_initialized = is_initialized;
  }
};

class Map
{
  int count_observations = 0;
  int count_keyframe_observation = 0;
  int id_ref_frame = 0;
  int id_last_keyframe = 0;
  int id_curr_frame = 0;


  std::vector<MonocularVisualOdometry::Observation> observations;
  std::map<int, std::vector<int>> map_p3d_to_cam_pose;
  std::map<int, std::vector<int>> map_cam_pose_to_p3d;
  std::map<std::pair<int, int>, int> map_cam_pose_p3d_to_observations;
  std::map<int, std::vector<int>> map_p3d_to_observations;
  std::map<int, std::vector<int>> map_cam_pose_to_observations;

  mutable std::shared_mutex mutex;

  void
  print_observation_info(
    const std::vector<Observation>& observations)
  {
    std::cout << "\nObservations:" << std::endl;
    std::shared_lock lock(mutex);
    for (const Observation& observation : observations)
    {
      std::cout << "Point3D id: " << observation.id_point3d << " | View id: " << observation.id_frame
                << " | Point3D: " << observation.point3d <<
          " | Point2D: " << observation.point2d << " | Pose: " << observation.pose << " | is keyframe: " << observation.is_keyframe <<
          " | observation id: " << observation.id_obs  << " | is optimized: " << observation.is_optimized << std::endl;
    }
  }

public:

  explicit Map() {}

  void
  push_observation(
      const MonocularVisualOdometry::Observation &observation)
  {
    std::unique_lock lock(mutex);

    Observation new_obs = observation;

    if (count_observations == 0)
      id_ref_frame = observation.id_frame;

    new_obs.id_obs = count_observations;
    id_curr_frame = new_obs.id_frame;

    if (new_obs.is_keyframe)
    {
      id_last_keyframe = new_obs.id_frame;
      count_keyframe_observation++;
    }

    observations.push_back(new_obs);

    map_p3d_to_cam_pose[new_obs.id_point3d].push_back(new_obs.id_frame);
    map_cam_pose_to_p3d[new_obs.id_frame].push_back(new_obs.id_point3d);
    map_p3d_to_observations[new_obs.id_point3d].push_back(new_obs.id_obs);
    map_cam_pose_to_observations[new_obs.id_frame].push_back(new_obs.id_obs);
    map_cam_pose_p3d_to_observations[{new_obs.id_frame, new_obs.id_point3d}] = new_obs.id_obs;

    count_observations++;
  }

  std::vector<MonocularVisualOdometry::Observation>
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

    std::vector<MonocularVisualOdometry::Observation> obs;

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
          int id_obs = map_cam_pose_p3d_to_observations[{id_view, id_p3d}];
          MonocularVisualOdometry::Observation observation = observations.at(id_obs);

          if (observation.id_frame < index_start_view_id or
              observation.id_frame > index_end_view_id)
            continue;

          if (only_keyframe)
          {
            if (observation.is_keyframe)
            {
              obs.push_back(observation);
            }
          } else
          {
            obs.push_back(observation);
          }
        }
      }
    }
    std::cout << "Total generated constraint count: " << obs.size() << std::endl;
    std::cout << "Total observation count in map: " << count_observations << std::endl;
    if (print_info)
      print_observation_info(obs);
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
      int id_obs = map_cam_pose_p3d_to_observations[{id_view, id_point3d}];
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
      int id_obs = map_cam_pose_p3d_to_observations[{id_view, id_point3d}];
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

  void
  update_map_old_observations(
    const std::vector<MonocularVisualOdometry::Observation>& old_observations,
    const bool& print_info)
  {
    std::unique_lock lock(mutex);
    if (print_info)
      std::cout << "Update old observations in map:" << std::endl;
    // Iterate over old observations:

    for (const Observation& old_observation : old_observations)
    {
      int id_view = old_observation.id_frame;
      int id_point3d = old_observation.id_point3d;

      for (const int& id_obs : map_p3d_to_observations[id_point3d])
      {
        if (print_info)
          std::cout << "Observation: " << id_obs << " is updated." <<std::endl;
        observations[id_obs].point3d = old_observation.point3d;
        observations[id_obs].is_optimized = true;
      }

      for (const int& id_obs : map_cam_pose_to_observations[id_view])
      {
        if (print_info)
          std::cout << "Observation: " << id_obs << " is updated." <<std::endl;
        observations[id_obs].pose = old_observation.pose;
        observations[id_obs].is_optimized = true;
      }
    }
    if (print_info)
      std::cout << "##########################################################"
                 "###################################################################" << std::endl;
  }


  /*
   * This function generates common 3D point observations seen all
   * camera poses between the last reference frame and current frame.
   */
  std::vector<Observation>
  get_common_observations_ref_frame_to_and_curr_frame(const bool& print_info)
  {
    std::shared_lock lock(mutex);
    std::vector<Observation> obs;
    std::vector<int> view_ids((get_id_curr_keyframe()-get_id_last_ref_frame())+1) ;
    std::iota (std::begin(view_ids), std::end(view_ids), get_id_last_ref_frame());

    // Detect common seen 3D points between reference and current frame:
    std::vector<int> ids_p3d_ref = map_cam_pose_to_p3d[get_id_last_ref_frame()];
    std::vector<int> ids_p3d_curr = map_cam_pose_to_p3d[get_id_curr_keyframe()];
    std::vector<int> ids_common_p3d;
    for (const int& id_p3d_curr : ids_p3d_curr)
    {
      for (const int& id_p3d_ref : ids_p3d_ref)
      {
        if (id_p3d_curr == id_p3d_ref)
        {
          ids_common_p3d.push_back(id_p3d_ref);
          break;
        }
      }
    }

    std::cout << "Count common observed 3D point: " << ids_common_p3d.size() << std::endl;
    if (print_info)
    {
      std::cout << "3D point ids: " ;
      for (const auto id:ids_common_p3d)
        std::cout << id <<  " " ;
    }

    for (const int& id_view : view_ids)
    {
      for (const int& id_p3d : ids_common_p3d)
      {
        int id_obs = map_cam_pose_p3d_to_observations[{id_view, id_p3d}];
        MonocularVisualOdometry::Observation observation = observations.at(id_obs);
        obs.push_back(observation);
      }
    }
    print_observation_info(obs);
    return obs;
  }

  int get_observation_count() { std::shared_lock lock(mutex); return count_observations;}
  int get_keyframe_observation_count() { std::shared_lock lock(mutex); return count_keyframe_observation;}
  int get_id_last_ref_frame() { std::shared_lock lock(mutex); return id_ref_frame;}
  int get_id_last_keyframe() { std::shared_lock lock(mutex); return id_last_keyframe;}
  int get_id_curr_keyframe() { std::shared_lock lock(mutex); return id_curr_frame;}

};

}// namespace MonocularVisualOdometry

typedef std::shared_ptr<MonocularVisualOdometry::Map> MapSharePtr;



int main() {
  MapSharePtr map(new MonocularVisualOdometry::Map);

  // Define vertex poses:
  cv::Vec6d p0 = {0, 0, 0, 0, 0, 0}; // pt0 pt1 pt2 pt3
  cv::Vec6d p1 = {1, 1, 1, 1, 1, 1}; // pt0 pt1 pt2
  cv::Vec6d p2 = {2, 2, 2, 2, 2, 2}; // pt0 pt1
  cv::Vec6d p3 = {3, 3, 3, 3, 3, 3}; // pt1
  // Common seen point3D is pt1

  // Define vertex Point3Ds:
  cv::Point3d pt0 = {0,0,0}; // p0 p1 p2
  cv::Point3d pt1 = {1,1,1}; // p0 p1 p2 p3
  cv::Point3d pt2 = {2,2,2}; // p0 p1
  cv::Point3d pt3 = {3,3,3}; // p0

  // Define observations as edges:
  cv::Point2d px0 = {0,0};
  cv::Point2d px1 = {1,1};
  cv::Point2d px2 = {2,2};
  cv::Point2d px3 = {3,3};
  cv::Point2d px4 = {4,4};
  cv::Point2d px5 = {5,5};
  cv::Point2d px6 = {6,6};
  cv::Point2d px7 = {7,7};
  cv::Point2d px8 = {8,8};
  cv::Point2d px9 = {9,9};

  cv::Mat mat_4x4;



  //Example functions:

  // F1
  //std::vector<MonocularVisualOdometry::Observation> obs = map->get_common_observations_ref_frame_to_and_curr_frame(true);
  // F2

/*
  std::vector<MonocularVisualOdometry::Observation> observations = map->get_edge_of_point3Ds(0, 0, 3, false, true);
  std::vector<MonocularVisualOdometry::Observation> vector_old_obs;
  cv::Point3d new_p = {999,999,999};
  cv::Vec6d new_pose = {999,999,999,999,999,999};

  MonocularVisualOdometry::Observation observation_new(
      0, new_pose, mat_4x4, false, false, 0,
      new_p,px0, true);
  vector_old_obs.push_back(observation_new);
  map->update_map_old_observations(vector_old_obs, true);
  std::vector<MonocularVisualOdometry::Observation> observations2 = map->get_edge_of_point3Ds(0, 0, 3, false, true);
*/

  // F3
  //std::vector<cv::Point2d> pxs = map->get_points2D_of_frame(2, true);
  // F4
  //std::vector<cv::Point3d> pts = map->get_points3D_of_frame(2, true);

/*
  std::map<int, std::map<int, int>> map_out;

  std::map<int, int> map_inner1;
  map_inner1[0] = 0;
  map_inner1[1] = 1;
  map_inner1[2] = 4;
  map_inner1[3] = 9;
  map_inner1[4] = 16;

  std::map<int, int> map_inner2;
  map_inner2[0] = 0;
  map_inner2[1] = -1;
  map_inner2[2] = -4;
  map_inner2[3] = -9;
  map_inner2[4] = -16;


  map_out.emplace(0, map_inner1);
  map_out.emplace(1, map_inner2);


  for (auto it = map_out.begin(); it!= map_out.end(); it++)
  {
    for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++)
    {
      std::cout << it2->second << std::endl;
    }
    std::cout << "*****************************************OUT" << std::endl;
  }

  auto it = map_out[0].find(3);
  map_out[0].erase(it);

  std::cout << "###########################################³" << std::endl;


  for (auto it = map_out.begin(); it!= map_out.end(); it++)
  {
    for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++)
    {
      std::cout << it2->second << std::endl;
    }
    std::cout << "*****************************************OUT" << std::endl;
  }
*/


  std::map<int, int> m;
  m[0] = 0;
  m[1] = 1;
  m[2] = 4;

  for (auto item = m.begin(); item!=m.end(); item++)
    std::cout << item->first << " " << item->second << std::endl;

  auto it_delete = m.find(1);
  m.erase(it_delete);

  for (auto item = m.begin(); item!=m.end(); item++)
    std::cout << item->first << " " << item->second << std::endl;






  return 0;
}