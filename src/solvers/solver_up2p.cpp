#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <complex>
#include <utility>
#include <iostream>
#include <PoseLib/solvers/up2p.h>

namespace ECCV2026 {
int up2p_wrapper(const std::vector<Eigen::Vector2d> &x, 
                 const std::vector<Eigen::Vector3d> &X,
                 const Eigen::Matrix3d &Rxz,
                 std::vector<Eigen::Matrix3d> *output_R, 
                 std::vector<Eigen::Vector3d> *output_T
                 ) {
    output_R->clear();
    output_T->clear();
    std::vector<poselib::CameraPose> pose;
    
    std::vector<Eigen::Vector3d> x_upright;
    std::vector<Eigen::Vector3d> X_upright;

    for (int i = 0; i < 2; ++i) {
        x_upright.push_back(x[i].homogeneous());
        X_upright.push_back(Rxz * X[i]);
    }

    int n_sols = poselib::up2p(x_upright, X_upright, &pose);

    // De-rotate coordinate systems
    for (int i = 0; i < n_sols; ++i) {
        Eigen::Matrix3d R = pose[i].R();
        Eigen::Vector3d t = pose[i].t;
        R = R * Rxz;
        output_R->push_back(R);
        output_T->push_back(t);
    }
    return n_sols;
}
}  // namespace ECCV2026
