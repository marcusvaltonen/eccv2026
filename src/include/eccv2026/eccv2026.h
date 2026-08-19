#ifndef ECCV2026_H_
#define ECCV2026_H_

#include <Eigen/Dense>
#include <vector>

namespace ECCV2026 {
// Polynomial solvers
Eigen::Vector4d solver_guan_cvpr_2020_ls(const Eigen::VectorXd &data);
std::pair<Eigen::Matrix3d, Eigen::Matrix3d> solver_imu_planar(const Eigen::Matrix<double, 8, 1> &data_);
std::pair<Eigen::Matrix3d, Eigen::Matrix3d> solver_imu_planar_sift(const Eigen::Matrix<double, 8, 1> &data_);
std::pair<Eigen::VectorXd, Eigen::VectorXd>
solver_guan_cvpr_2020_cs(const Eigen::Vector2d &Pi, const Eigen::Vector2d &Pj, const Eigen::Matrix2d &Ac);
Eigen::Vector3d solver_known_rotation(const Eigen::Matrix<double, 8, 1> &data_);
Eigen::Vector3d solver_known_rotation_sift(const Eigen::Matrix<double, 8, 1> &data_);
std::vector<Eigen::Matrix3d> solver_choi_kim_2018(const Eigen::Matrix<double, 8, 1> &data_);
std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double> solver_p35pf(const Eigen::Matrix<double, 2, 4> &y,
                                                                  const Eigen::Matrix<double, 3, 4> &X);
std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double> solver_up1pf_ac(
    const Eigen::Matrix3d &R_ref,
    const Eigen::Vector3d &t_ref,
    const double focal_ref,
    const Eigen::Matrix2d &A, 
    const Eigen::Vector2d &p_ref,
    const double d,
    const Eigen::Vector3d &n, 
    const Eigen::Vector2d &p_query,
    const Eigen::Matrix3d &Rxz);
std::pair<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>>
solver_up1p_sift(
    const Eigen::Matrix3d &R_ref,
    const Eigen::Vector3d &t_ref,
    const Eigen::Vector2d &y, const Eigen::Vector3d &X, const Eigen::Matrix3d &Rxz, double angle_ref,
                 double angle_query, double q, const Eigen::Vector3d &n);
std::pair<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>>
solver_p2p_ori(
        const std::vector<Eigen::Matrix3d> &R_ref,
        const std::vector<Eigen::Vector3d> &t_ref,
        const std::vector<double> &angle_ref,
        const std::vector<double> &angle_query,
        const std::vector<Eigen::Vector3d> &X,
        const std::vector<Eigen::Vector3d> &n,
        const std::vector<Eigen::Vector2d> &p_query
    );
std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double>
solver_up2pf_ori(
    const std::vector<Eigen::Matrix3d> &R_ref,
    const std::vector<Eigen::Vector3d> &t_ref,
    const std::vector<double> &focal_ref,
    const std::vector<double> &angle_ref,
    const std::vector<double> &angle_query,
    const std::vector<Eigen::Vector2d> &p_ref_orig,
    const std::vector<double> &d,
    const std::vector<Eigen::Vector3d> &n, 
    const std::vector<Eigen::Vector2d> &p_query_orig,
    const Eigen::Matrix3d &Rxz
    );

int p4pf_wrapper(const std::vector<Eigen::Vector2d> &x, 
                 const std::vector<Eigen::Vector3d> &X, 
                 std::vector<Eigen::Matrix3d> *output_R, 
                 std::vector<Eigen::Vector3d> *output_T, 
                 std::vector<double> *output_f, 
                 bool filter_solutions = true);
int up2p_wrapper(const std::vector<Eigen::Vector2d> &x, 
                 const std::vector<Eigen::Vector3d> &X,
                 const Eigen::Matrix3d &Rxz,
                 std::vector<Eigen::Matrix3d> *output_R, 
                 std::vector<Eigen::Vector3d> *output_T
                 );
                 
int solver_up3pf(const std::vector<Eigen::Vector2d> &x, 
                 const std::vector<Eigen::Vector3d> &X, 
                 const Eigen::Matrix3d &Rxz,
                 std::vector<Eigen::Matrix3d> *output_R, 
                 std::vector<Eigen::Vector3d> *output_T, 
                 std::vector<double> *output_f);                 

std::tuple<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>, std::vector<double>, std::vector<double>>
solver_up3pfr(
    const Eigen::Matrix<double, 2, 3> &y,
    const Eigen::Matrix<double, 3, 3> &X,
    const Eigen::Matrix3d &Rxz
    );
// Histogram voting
std::pair<Eigen::Matrix3d, std::vector<size_t>>
histogram_voting(const Eigen::Matrix<double, 8, Eigen::Dynamic> &features, double threshold, int range, int solver,
                 int non_minimal_solver);
Eigen::Matrix3d solver_hajder_barath_icra_2020(const Eigen::MatrixXd &data, int solver_type, bool use_fast_solver);
} // namespace ECCV2026
std::vector<Eigen::Matrix<double, 3, 4>> ACP1PCayleySolver(const Eigen::Vector2d &y, const Eigen::Vector3d &X,
                                                           const Eigen::Matrix2d &A, const Eigen::Vector3d &n);
#endif // ECCV2026_H_
