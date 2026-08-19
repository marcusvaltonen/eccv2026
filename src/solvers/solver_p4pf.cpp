#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <complex>
#include <utility>
#include <PoseLib/solvers/p4pf.h>

namespace ECCV2026 {
int p4pf_wrapper(const std::vector<Eigen::Vector2d> &x, 
                 const std::vector<Eigen::Vector3d> &X, 
                 std::vector<Eigen::Matrix3d> *output_R, 
                 std::vector<Eigen::Vector3d> *output_T, 
                 std::vector<double> *output_f, 
                 bool filter_solutions = true) {
    output_R->clear();
    output_T->clear();
    output_f->clear();
    std::vector<poselib::CameraPose> pose;
    int nbr_sols_p4pf = poselib::p4pf(x, X,  &pose,  output_f, filter_solutions);        
    for (size_t i = 0; i < nbr_sols_p4pf; i++) {
            output_R->push_back(pose[i].R());
            output_T->push_back(pose[i].t);
    }

    return nbr_sols_p4pf;
}
}  // namespace ECCV2026
