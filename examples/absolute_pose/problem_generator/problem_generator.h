
#pragma once

#include <Eigen/Core>
#include <problem_generator/random.h>
#include <vector>

namespace problem_generator {

struct ProblemGeneratorConfig {
    double point_noise;
    double affine_noise;
    double angle_noise;
    double scale_noise;
    double imu_noise;
    double normal_noise;
    bool known_focal;
    ProblemGeneratorConfig()
        : point_noise(0), affine_noise(0), angle_noise(0), scale_noise(0), imu_noise(0), normal_noise(0), known_focal(false) {}
};

class ProblemGenerator {
  public:
    RandomGenerator generator;
    ProblemGeneratorConfig config;
    ProblemGenerator(ProblemGeneratorConfig _config = ProblemGeneratorConfig()) : config(_config) {}
    int make_random_problems(std::vector<Eigen::Vector3d> &Xs, std::vector<Eigen::Vector3d> &ns,
                              std::vector<Eigen::Vector2d> &ys, std::vector<Eigen::Matrix2d> &As, Eigen::Matrix3d &R,
                              Eigen::Matrix3d &Rxz, Eigen::Vector3d &t, double &f_ref, double &f_query, int N);
    int make_random_problems(std::vector<Eigen::Vector3d> &Xs, std::vector<Eigen::Vector3d> &ns,
                              std::vector<Eigen::Vector2d> &ys, std::vector<Eigen::Matrix2d> &As,
                              std::vector<double> &angle_refs, std::vector<double> &angle_querys,
                              std::vector<double> &qs, Eigen::Matrix3d &R, Eigen::Matrix3d &Rxz, Eigen::Vector3d &t,
                              double &f_ref, double &f_query, int N);
    int make_random_problem(Eigen::Vector3d &X, Eigen::Vector3d &n, Eigen::Vector2d &y, Eigen::Matrix2d &A,
                             double &angle_ref, double &angle_query, double &q, Eigen::Matrix3d &R,
                             Eigen::Matrix3d &Rxz, Eigen::Vector3d &t, double &f_ref, double &f_query);
};
} // namespace problem_generator
