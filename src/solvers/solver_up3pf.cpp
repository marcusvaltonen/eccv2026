#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <complex>
#include <utility>
#include <PoseLib/misc/univariate.h>
#include "gj.h"

// Based on Kukelova et al. (2010), "Closed-Form Solutions to Minimal Absolute Pose Problems with Known Vertical Direction", ACCV
// Only final step is modified (assuming no radial distortion)
namespace ECCV2026 {

static Eigen::Matrix<double, 1, 5> extract_coeffs1(
    const Eigen::Vector2d& x,
    const Eigen::Vector3d& X,
    const Eigen::Matrix3d& Rxz)
{
    Eigen::Matrix<double, 1, 5> c;

    c(0) = -x(1);

    c(1) = x(0);

    c(2) = Rxz(0,0)*x(1)*X(0) + Rxz(1,0)*x(0)*X(0)
         + Rxz(0,1)*x(1)*X(1) + Rxz(1,1)*x(0)*X(1)
         + Rxz(0,2)*x(1)*X(2) + Rxz(1,2)*x(0)*X(2);

    c(3) = 2.0*Rxz(2,0)*x(1)*X(0)
         + 2.0*Rxz(2,1)*x(1)*X(1)
         + 2.0*Rxz(2,2)*x(1)*X(2);

    c(4) = -Rxz(0,0)*x(1)*X(0) + Rxz(1,0)*x(0)*X(0)
           -Rxz(0,1)*x(1)*X(1) + Rxz(1,1)*x(0)*X(1)
           -Rxz(0,2)*x(1)*X(2) + Rxz(1,2)*x(0)*X(2);

    return c;
}

static Eigen::Matrix<double, 1, 3> extract_coeffs2(
    const Eigen::Vector2d& x3,
    const Eigen::Vector3d& X3,
    const Eigen::Matrix3d& Rxz,
    const Eigen::Matrix<double, 2, 3>& c)
{
    Eigen::Matrix<double, 1, 3> k;

    k(0) = Rxz(0,0)*x3(1)*X3(0) + Rxz(1,0)*x3(0)*X3(0)
         + Rxz(0,1)*x3(1)*X3(1) + Rxz(1,1)*x3(0)*X3(1)
         + Rxz(0,2)*x3(1)*X3(2) + Rxz(1,2)*x3(0)*X3(2)
         - x3(0)*c(1,0) + x3(1)*c(0,0);

    k(1) = 2.0*Rxz(2,0)*x3(1)*X3(0)
         + 2.0*Rxz(2,1)*x3(1)*X3(1)
         + 2.0*Rxz(2,2)*x3(1)*X3(2)
         - x3(0)*c(1,1) + x3(1)*c(0,1);

    k(2) = -Rxz(0,0)*x3(1)*X3(0) + Rxz(1,0)*x3(0)*X3(0)
           -Rxz(0,1)*x3(1)*X3(1) + Rxz(1,1)*x3(0)*X3(1)
           -Rxz(0,2)*x3(1)*X3(2) + Rxz(1,2)*x3(0)*X3(2)
           - x3(0)*c(1,2) + x3(1)*c(0,2);

    return k;
}

static Eigen::Matrix<double, 1, 3> extract_coeffs3(
    const Eigen::Vector2d& x,
    const Eigen::Vector3d& X,
    const Eigen::Matrix3d& Rxz,
    double r,
    double ty)
{
    Eigen::Matrix<double, 1, 3> k;
    const double r2 = r * r;

    k(0) = x(1);

    k(1) = 2.0*Rxz(0,0)*x(1)*X(0)*r - Rxz(2,0)*x(1)*X(0)*r2 + Rxz(2,0)*x(1)*X(0)
         + 2.0*Rxz(0,1)*x(1)*X(1)*r - Rxz(2,1)*x(1)*X(1)*r2 + Rxz(2,1)*x(1)*X(1)
         + 2.0*Rxz(0,2)*x(1)*X(2)*r - Rxz(2,2)*x(1)*X(2)*r2 + Rxz(2,2)*x(1)*X(2);

    k(2) = -Rxz(1,0)*X(0)*r2 - Rxz(1,0)*X(0)
           -Rxz(1,1)*X(1)*r2 - Rxz(1,1)*X(1)
           -Rxz(1,2)*X(2)*r2 - Rxz(1,2)*X(2)
           - ty;

    return k;
}

int solver_up3pf(
     const std::vector<Eigen::Vector2d> &x, 
     const std::vector<Eigen::Vector3d> &X,
     const Eigen::Matrix3d &Rxz,
     std::vector<Eigen::Matrix3d> *output_R, 
     std::vector<Eigen::Vector3d> *output_T, 
     std::vector<double> *output_f
) {
    output_R->clear();
    output_T->clear();
    output_f->clear();
    
    // 
    Eigen::MatrixXd M1(2,5);
    M1.row(0) = extract_coeffs1(x[0], X[0], Rxz);
    M1.row(1) = extract_coeffs1(x[1], X[1], Rxz);
    gj(&M1);

    Eigen::Matrix<double, 2, 3> c = M1.block<2, 3>(0, 2);
    Eigen::Matrix<double, 1, 3> coeffs = extract_coeffs2(x[2], X[2], Rxz, c);
    double roots[2];
    int nbr_real_roots = poselib::univariate::solve_quadratic_real(coeffs[0], coeffs[1], coeffs[2], roots);
    
    
    
    for (int i = 0; i < nbr_real_roots; i++) {
        double r = roots[i];
        Eigen::Vector3d poly = {r * r, r, 1.0};

        double tx = -c.row(0).dot(poly);
        double ty = -c.row(1).dot(poly);
        
        Eigen::MatrixXd M2(2,3);
        M2.row(0) = extract_coeffs3(x[0], X[0], Rxz, r, ty);
        M2.row(1) = extract_coeffs3(x[1], X[1], Rxz, r, ty);
        gj(&M2);
        
        double f = -1.0 / M2(1,2);
        double tz = -M2(0,2) * f;
        
        double s = 1 + r * r;
        Eigen::Vector3d t = {tx, ty, tz};
        t /= s;
        
        double cost = (1 - r*r) / s;
        double sint = 2 * r / s;
        Eigen::Matrix3d R;
        R << cost, 0, -sint, 0, 1, 0, sint, 0, cost;
        R = R * Rxz;
        
        output_R->push_back(R);
        output_T->push_back(t);
        output_f->push_back(f);
    }
    
	return nbr_real_roots;
}

}  // namespace ECCV2026
