#include <Eigen/Dense>
#include <tuple>
#include <PoseLib/misc/univariate.h>
#include <cmath>


namespace ECCV2026 {
static const double compute_algebraic_error_unused_constraint_ref(
    const Eigen::Matrix<double, 6, 1> &c,
    const double r,
    const double f);
    
static const Eigen::Matrix<double, 13, 1> generate_coeffs_point_based(
    const Eigen::Matrix3d &R_ref1,
    const Eigen::Vector3d &t_ref1,
    const Eigen::Vector2d &p_ref1,
    const double d1,
    const Eigen::Vector3d &n1, 
    const Eigen::Vector2d &p_query1,
    const Eigen::Matrix3d &Rxz);
    
static const Eigen::Matrix<double, 6, 1> generate_coeffs_ori(
    const Eigen::Matrix3d &R_ref1,
    const Eigen::Vector3d &t_ref1,
    const double c_ref1,
    const double s_ref1,
    const double c_query1,
    const double s_query1,
    const Eigen::Vector2d &p_ref1,
    const double d1,
    const Eigen::Vector3d &n_ref1, 
    const Eigen::Vector2d &p_query1,
    const Eigen::Matrix3d &Rxz);

static const Eigen::Matrix<double, 4, 4> get_M_ref(const Eigen::Matrix<double, 32, 1> &coeffs, const double r, const double focal_query);

static const Eigen::Matrix<double, 2, 2> get_Mbar_ref(const Eigen::Matrix<double, 32, 1> &coeffs, const double r);

static const Eigen::Matrix<double, 5, 1> get_coeffs_up2pfORI_ref(const Eigen::Matrix<double, 32, 1> &coeffs);

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double>
solver_up2pf_ori(
    const std::vector<Eigen::Matrix3d> &R_ref,
    const std::vector<Eigen::Vector3d> &t_ref,
    const std::vector<double> &focal_ref,  // Actually has no effect (only affects scale)
    const std::vector<double> &angle_ref,
    const std::vector<double> &angle_query,
    const std::vector<Eigen::Vector2d> &p_ref_orig,
    const std::vector<double> &d,
    const std::vector<Eigen::Vector3d> &n, 
    const std::vector<Eigen::Vector2d> &p_query_orig,
    const Eigen::Matrix3d &Rxz
    )
{
    // Normalize
    double scale = (p_query_orig[0].norm() + p_query_orig[1].norm()) * 0.5;
    Eigen::Vector2d p_ref[2];
    p_ref[0] = p_ref_orig[0];
    p_ref[1] = p_ref_orig[1];
    Eigen::Vector2d p_query[2];
    p_query[0] = p_query_orig[0] / scale;
    p_query[1] = p_query_orig[1] / scale;
    double c_ref[2], s_ref[2], c_query[2], s_query[2];
    for (int i=0; i < 2; i++) {
        c_ref[i] = cos(angle_ref[i]);
        s_ref[i] = sin(angle_ref[i]);
        c_query[i] = cos(angle_query[i]);
        s_query[i] = sin(angle_query[i]);
    }

    // Generate coefficients
    Eigen::Vector3d n_ref1 = R_ref[0] * n[0];
    Eigen::Vector3d n_ref2 = R_ref[1] * n[1];
    Eigen::Matrix<double, 13, 1> M1 = generate_coeffs_point_based(R_ref[0], t_ref[0], p_ref[0], d[0], n_ref1, p_query[0], Rxz);
    Eigen::Matrix<double, 13, 1> M2 = generate_coeffs_point_based(R_ref[1], t_ref[1], p_ref[1], d[1], n_ref2, p_query[1], Rxz);
    Eigen::Matrix<double, 6, 1> M3 = generate_coeffs_ori(R_ref[0], t_ref[0], c_ref[0], s_ref[0], c_query[0], s_query[0], p_ref[0], d[0], n_ref1, p_query[0], Rxz);
    Eigen::Matrix<double, 6, 1> M4 = generate_coeffs_ori(R_ref[1], t_ref[1], c_ref[1], s_ref[1], c_query[1], s_query[1], p_ref[1], d[1], n_ref2, p_query[1], Rxz);

    Eigen::Matrix<double, 32, 1> coeffs;
    coeffs << M1, M2, M3;

    // Find r
    Eigen::Matrix<double, 5, 1> coeffs_quartic = get_coeffs_up2pfORI_ref(coeffs);
    double* p = coeffs_quartic.data();
    double roots[4];
    int nbr_real_roots = poselib::univariate::solve_quartic_real(p[1] / p[0], p[2] / p[0], p[3] / p[0], p[4] / p[0], roots);

    double r_best;
    double focal_query_best;
    Eigen::Vector3d t_query_best;
    double r_tmp;
    double focal_query_tmp;
    Eigen::Vector3d t_query_tmp;
    double min_err = 1000000000000;
    for (int i=0; i < nbr_real_roots; i++) {
        r_tmp = roots[i];

        // Extract focal_query
        Eigen::Matrix<double, 2, 2> Mbar = get_Mbar_ref(coeffs, r_tmp);
        Eigen::FullPivHouseholderQR<Eigen::Matrix2d> qr(Mbar.transpose() * Mbar);
	    Eigen::Matrix2d Q = qr.matrixQ();
	    Eigen::Vector2d Nbar = Q.rightCols<1>();
        //Eigen::JacobiSVD<Eigen::MatrixXd> svd(Mbar, Eigen::ComputeFullV);
        //Eigen::MatrixXd Nbar = svd.matrixV().rightCols(1);
        focal_query_tmp= Nbar(0) / Nbar(1);
        
        // Extract t_query
        Eigen::Matrix<double, 4, 4> M = get_M_ref(coeffs, r_tmp, focal_query_tmp);
        Eigen::FullPivHouseholderQR<Eigen::Matrix4d> qr2(M.transpose() * M);
	    Eigen::Matrix4d Q2 = qr2.matrixQ();
	    Eigen::Vector4d N = Q2.rightCols<1>();
        //Eigen::JacobiSVD<Eigen::Matrix<double, 5, 4>> svd2(M, Eigen::ComputeFullV);
        //Eigen::Matrix<double, 4, 1> N = svd2.matrixV().rightCols(1);
        t_query_tmp = N.hnormalized();
        
        // Use previously unused constraint to discard false solutions
        double err = compute_algebraic_error_unused_constraint_ref(M4, r_tmp, focal_query_tmp);
        if (err < min_err) {
            r_best = r_tmp;
            focal_query_best = focal_query_tmp;
            t_query_best = t_query_tmp;
            min_err = err;
        }
    }

    // Transform to expected format
    focal_query_best *= scale;
    double w = 1 + r_best * r_best;
    double c = (1 - r_best * r_best) / w;
    double s = 2 * r_best / w;
    Eigen::Matrix3d R;
    R << c, 0, -s, 0, 1, 0, s, 0, c;
    R = R * Rxz;
    t_query_best /= w;
    
    return std::make_tuple(R, t_query_best, focal_query_best);
}

static const Eigen::Matrix<double, 4, 4> get_M_ref(
    const Eigen::Matrix<double, 32, 1> &c,
    const double r,
    const double f)
{
    Eigen::Matrix<double, 4, 4> M;
    // TODO: pre-compute monomials
    M << -f,0,c[0],std::pow(r,2)*f*c[1] + std::pow(r,2)*c[2] + r*f*c[3] + r*c[4] + f*c[5] + c[6],
	0,-f,c[7],std::pow(r,2)*f*c[8] + std::pow(r,2)*c[9] + r*c[10] + f*c[11] + c[12],
	-f,0,c[13],std::pow(r,2)*f*c[14] + std::pow(r,2)*c[15] + r*f*c[16] + r*c[17] + f*c[18] + c[19],
	0,-f,c[20],std::pow(r,2)*f*c[21] + std::pow(r,2)*c[22] + r*c[23] + f*c[24] + c[25];

    return M;
}

static const Eigen::Matrix<double, 2, 2> get_Mbar_ref(
    const Eigen::Matrix<double, 32, 1> &c,
    const double r)
{
    Eigen::Matrix<double, 2, 2> M;

    // TODO: double r2 = r * r;
    // and possibly some more like r2*c[0]
    
    M << std::pow(r,2)*c[0]*c[8] - std::pow(r,2)*c[0]*c[21] - std::pow(r,2)*c[1]*c[7] + std::pow(r,2)*c[1]*c[20] + std::pow(r,2)*c[7]*c[14] - std::pow(r,2)*c[8]*c[13] + std::pow(r,2)*c[13]*c[21] - std::pow(r,2)*c[14]*c[20] - r*c[3]*c[7] + r*c[3]*c[20] + r*c[7]*c[16] - r*c[16]*c[20] + c[0]*c[11] - c[0]*c[24] - c[5]*c[7] + c[5]*c[20] + c[7]*c[18] - c[11]*c[13] + c[13]*c[24] - c[18]*c[20],
std::pow(r,2)*c[0]*c[9] - std::pow(r,2)*c[0]*c[22] - std::pow(r,2)*c[2]*c[7] + std::pow(r,2)*c[2]*c[20] + std::pow(r,2)*c[7]*c[15] - std::pow(r,2)*c[9]*c[13] + std::pow(r,2)*c[13]*c[22] - std::pow(r,2)*c[15]*c[20] + r*c[0]*c[10] - r*c[0]*c[23] - r*c[4]*c[7] + r*c[4]*c[20] + r*c[7]*c[17] - r*c[10]*c[13] + r*c[13]*c[23] - r*c[17]*c[20] + c[0]*c[12] - c[0]*c[25] - c[6]*c[7] + c[6]*c[20] + c[7]*c[19] - c[12]*c[13] + c[13]*c[25] - c[19]*c[20],
std::pow(r,2)*c[26] + r*c[28] + c[30],
std::pow(r,2)*c[27] + r*c[29] + c[31];
    return M;
}

static const Eigen::Matrix<double, 5, 1> get_coeffs_up2pfORI_ref(const Eigen::Matrix<double, 32, 1> &c)
{
    Eigen::Matrix<double, 5, 1> coeffs;
    coeffs << c[0]*c[8]*c[27] - c[0]*c[9]*c[26] - c[0]*c[21]*c[27] + c[0]*c[22]*c[26] - c[1]*c[7]*c[27] + c[1]*c[20]*c[27] + c[2]*c[7]*c[26] - c[2]*c[20]*c[26] + c[7]*c[14]*c[27] - c[7]*c[15]*c[26] - c[8]*c[13]*c[27] + c[9]*c[13]*c[26] + c[13]*c[21]*c[27] - c[13]*c[22]*c[26] - c[14]*c[20]*c[27] + c[15]*c[20]*c[26],
c[0]*c[8]*c[29] - c[0]*c[9]*c[28] - c[0]*c[10]*c[26] - c[0]*c[21]*c[29] + c[0]*c[22]*c[28] + c[0]*c[23]*c[26] - c[1]*c[7]*c[29] + c[1]*c[20]*c[29] + c[2]*c[7]*c[28] - c[2]*c[20]*c[28] - c[3]*c[7]*c[27] + c[3]*c[20]*c[27] + c[4]*c[7]*c[26] - c[4]*c[20]*c[26] + c[7]*c[14]*c[29] - c[7]*c[15]*c[28] + c[7]*c[16]*c[27] - c[7]*c[17]*c[26] - c[8]*c[13]*c[29] + c[9]*c[13]*c[28] + c[10]*c[13]*c[26] + c[13]*c[21]*c[29] - c[13]*c[22]*c[28] - c[13]*c[23]*c[26] - c[14]*c[20]*c[29] + c[15]*c[20]*c[28] - c[16]*c[20]*c[27] + c[17]*c[20]*c[26],
c[0]*c[8]*c[31] - c[0]*c[9]*c[30] - c[0]*c[10]*c[28] + c[0]*c[11]*c[27] - c[0]*c[12]*c[26] - c[0]*c[21]*c[31] + c[0]*c[22]*c[30] + c[0]*c[23]*c[28] - c[0]*c[24]*c[27] + c[0]*c[25]*c[26] - c[1]*c[7]*c[31] + c[1]*c[20]*c[31] + c[2]*c[7]*c[30] - c[2]*c[20]*c[30] - c[3]*c[7]*c[29] + c[3]*c[20]*c[29] + c[4]*c[7]*c[28] - c[4]*c[20]*c[28] - c[5]*c[7]*c[27] + c[5]*c[20]*c[27] + c[6]*c[7]*c[26] - c[6]*c[20]*c[26] + c[7]*c[14]*c[31] - c[7]*c[15]*c[30] + c[7]*c[16]*c[29] - c[7]*c[17]*c[28] + c[7]*c[18]*c[27] - c[7]*c[19]*c[26] - c[8]*c[13]*c[31] + c[9]*c[13]*c[30] + c[10]*c[13]*c[28] - c[11]*c[13]*c[27] + c[12]*c[13]*c[26] + c[13]*c[21]*c[31] - c[13]*c[22]*c[30] - c[13]*c[23]*c[28] + c[13]*c[24]*c[27] - c[13]*c[25]*c[26] - c[14]*c[20]*c[31] + c[15]*c[20]*c[30] - c[16]*c[20]*c[29] + c[17]*c[20]*c[28] - c[18]*c[20]*c[27] + c[19]*c[20]*c[26],
-c[0]*c[10]*c[30] + c[0]*c[11]*c[29] - c[0]*c[12]*c[28] + c[0]*c[23]*c[30] - c[0]*c[24]*c[29] + c[0]*c[25]*c[28] - c[3]*c[7]*c[31] + c[3]*c[20]*c[31] + c[4]*c[7]*c[30] - c[4]*c[20]*c[30] - c[5]*c[7]*c[29] + c[5]*c[20]*c[29] + c[6]*c[7]*c[28] - c[6]*c[20]*c[28] + c[7]*c[16]*c[31] - c[7]*c[17]*c[30] + c[7]*c[18]*c[29] - c[7]*c[19]*c[28] + c[10]*c[13]*c[30] - c[11]*c[13]*c[29] + c[12]*c[13]*c[28] - c[13]*c[23]*c[30] + c[13]*c[24]*c[29] - c[13]*c[25]*c[28] - c[16]*c[20]*c[31] + c[17]*c[20]*c[30] - c[18]*c[20]*c[29] + c[19]*c[20]*c[28],
c[0]*c[11]*c[31] - c[0]*c[12]*c[30] - c[0]*c[24]*c[31] + c[0]*c[25]*c[30] - c[5]*c[7]*c[31] + c[5]*c[20]*c[31] + c[6]*c[7]*c[30] - c[6]*c[20]*c[30] + c[7]*c[18]*c[31] - c[7]*c[19]*c[30] - c[11]*c[13]*c[31] + c[12]*c[13]*c[30] + c[13]*c[24]*c[31] - c[13]*c[25]*c[30] - c[18]*c[20]*c[31] + c[19]*c[20]*c[30];

    return coeffs;
}


static const Eigen::Matrix<double, 13, 1> generate_coeffs_point_based(
    const Eigen::Matrix3d &R_ref1,
    const Eigen::Vector3d &t_ref1,
    const Eigen::Vector2d &p_ref1,
    const double d1,
    const Eigen::Vector3d &n1, 
    const Eigen::Vector2d &p_query1,
    const Eigen::Matrix3d &Rxz)
{

    Eigen::Matrix<double, 13, 1> M1;
    M1 <<
    p_query1(0),
    -R_ref1(0,0)*t_ref1(0)*Rxz(0,0) + R_ref1(0,0)*p_ref1(0)*d1*Rxz(0,0) - R_ref1(1,0)*t_ref1(1)*Rxz(0,0) + R_ref1(1,0)*p_ref1(1)*d1*Rxz(0,0) - R_ref1(2,0)*t_ref1(2)*Rxz(0,0) + R_ref1(2,0)*d1*Rxz(0,0) - R_ref1(0,1)*t_ref1(0)*Rxz(0,1) + R_ref1(0,1)*p_ref1(0)*d1*Rxz(0,1) - R_ref1(1,1)*t_ref1(1)*Rxz(0,1) + R_ref1(1,1)*p_ref1(1)*d1*Rxz(0,1) - R_ref1(2,1)*t_ref1(2)*Rxz(0,1) + R_ref1(2,1)*d1*Rxz(0,1) - R_ref1(0,2)*t_ref1(0)*Rxz(0,2) + R_ref1(0,2)*p_ref1(0)*d1*Rxz(0,2) - R_ref1(1,2)*t_ref1(1)*Rxz(0,2) + R_ref1(1,2)*p_ref1(1)*d1*Rxz(0,2) - R_ref1(2,2)*t_ref1(2)*Rxz(0,2) + R_ref1(2,2)*d1*Rxz(0,2),
    R_ref1(0,0)*t_ref1(0)*p_query1(0)*Rxz(2,0) - R_ref1(0,0)*p_ref1(0)*d1*p_query1(0)*Rxz(2,0) + R_ref1(1,0)*t_ref1(1)*p_query1(0)*Rxz(2,0) - R_ref1(1,0)*p_ref1(1)*d1*p_query1(0)*Rxz(2,0) + R_ref1(2,0)*t_ref1(2)*p_query1(0)*Rxz(2,0) - R_ref1(2,0)*d1*p_query1(0)*Rxz(2,0) + R_ref1(0,1)*t_ref1(0)*p_query1(0)*Rxz(2,1) - R_ref1(0,1)*p_ref1(0)*d1*p_query1(0)*Rxz(2,1) + R_ref1(1,1)*t_ref1(1)*p_query1(0)*Rxz(2,1) - R_ref1(1,1)*p_ref1(1)*d1*p_query1(0)*Rxz(2,1) + R_ref1(2,1)*t_ref1(2)*p_query1(0)*Rxz(2,1) - R_ref1(2,1)*d1*p_query1(0)*Rxz(2,1) + R_ref1(0,2)*t_ref1(0)*p_query1(0)*Rxz(2,2) - R_ref1(0,2)*p_ref1(0)*d1*p_query1(0)*Rxz(2,2) + R_ref1(1,2)*t_ref1(1)*p_query1(0)*Rxz(2,2) - R_ref1(1,2)*p_ref1(1)*d1*p_query1(0)*Rxz(2,2) + R_ref1(2,2)*t_ref1(2)*p_query1(0)*Rxz(2,2) - R_ref1(2,2)*d1*p_query1(0)*Rxz(2,2),
    -2*R_ref1(0,0)*t_ref1(0)*Rxz(2,0) + 2*R_ref1(0,0)*p_ref1(0)*d1*Rxz(2,0) - 2*R_ref1(1,0)*t_ref1(1)*Rxz(2,0) + 2*R_ref1(1,0)*p_ref1(1)*d1*Rxz(2,0) - 2*R_ref1(2,0)*t_ref1(2)*Rxz(2,0) + 2*R_ref1(2,0)*d1*Rxz(2,0) - 2*R_ref1(0,1)*t_ref1(0)*Rxz(2,1) + 2*R_ref1(0,1)*p_ref1(0)*d1*Rxz(2,1) - 2*R_ref1(1,1)*t_ref1(1)*Rxz(2,1) + 2*R_ref1(1,1)*p_ref1(1)*d1*Rxz(2,1) - 2*R_ref1(2,1)*t_ref1(2)*Rxz(2,1) + 2*R_ref1(2,1)*d1*Rxz(2,1) - 2*R_ref1(0,2)*t_ref1(0)*Rxz(2,2) + 2*R_ref1(0,2)*p_ref1(0)*d1*Rxz(2,2) - 2*R_ref1(1,2)*t_ref1(1)*Rxz(2,2) + 2*R_ref1(1,2)*p_ref1(1)*d1*Rxz(2,2) - 2*R_ref1(2,2)*t_ref1(2)*Rxz(2,2) + 2*R_ref1(2,2)*d1*Rxz(2,2),
    -2*R_ref1(0,0)*t_ref1(0)*p_query1(0)*Rxz(0,0) + 2*R_ref1(0,0)*p_ref1(0)*d1*p_query1(0)*Rxz(0,0) - 2*R_ref1(1,0)*t_ref1(1)*p_query1(0)*Rxz(0,0) + 2*R_ref1(1,0)*p_ref1(1)*d1*p_query1(0)*Rxz(0,0) - 2*R_ref1(2,0)*t_ref1(2)*p_query1(0)*Rxz(0,0) + 2*R_ref1(2,0)*d1*p_query1(0)*Rxz(0,0) - 2*R_ref1(0,1)*t_ref1(0)*p_query1(0)*Rxz(0,1) + 2*R_ref1(0,1)*p_ref1(0)*d1*p_query1(0)*Rxz(0,1) - 2*R_ref1(1,1)*t_ref1(1)*p_query1(0)*Rxz(0,1) + 2*R_ref1(1,1)*p_ref1(1)*d1*p_query1(0)*Rxz(0,1) - 2*R_ref1(2,1)*t_ref1(2)*p_query1(0)*Rxz(0,1) + 2*R_ref1(2,1)*d1*p_query1(0)*Rxz(0,1) - 2*R_ref1(0,2)*t_ref1(0)*p_query1(0)*Rxz(0,2) + 2*R_ref1(0,2)*p_ref1(0)*d1*p_query1(0)*Rxz(0,2) - 2*R_ref1(1,2)*t_ref1(1)*p_query1(0)*Rxz(0,2) + 2*R_ref1(1,2)*p_ref1(1)*d1*p_query1(0)*Rxz(0,2) - 2*R_ref1(2,2)*t_ref1(2)*p_query1(0)*Rxz(0,2) + 2*R_ref1(2,2)*d1*p_query1(0)*Rxz(0,2),
    R_ref1(0,0)*t_ref1(0)*Rxz(0,0) - R_ref1(0,0)*p_ref1(0)*d1*Rxz(0,0) + R_ref1(1,0)*t_ref1(1)*Rxz(0,0) - R_ref1(1,0)*p_ref1(1)*d1*Rxz(0,0) + R_ref1(2,0)*t_ref1(2)*Rxz(0,0) - R_ref1(2,0)*d1*Rxz(0,0) + R_ref1(0,1)*t_ref1(0)*Rxz(0,1) - R_ref1(0,1)*p_ref1(0)*d1*Rxz(0,1) + R_ref1(1,1)*t_ref1(1)*Rxz(0,1) - R_ref1(1,1)*p_ref1(1)*d1*Rxz(0,1) + R_ref1(2,1)*t_ref1(2)*Rxz(0,1) - R_ref1(2,1)*d1*Rxz(0,1) + R_ref1(0,2)*t_ref1(0)*Rxz(0,2) - R_ref1(0,2)*p_ref1(0)*d1*Rxz(0,2) + R_ref1(1,2)*t_ref1(1)*Rxz(0,2) - R_ref1(1,2)*p_ref1(1)*d1*Rxz(0,2) + R_ref1(2,2)*t_ref1(2)*Rxz(0,2) - R_ref1(2,2)*d1*Rxz(0,2),
    -R_ref1(0,0)*t_ref1(0)*p_query1(0)*Rxz(2,0) + R_ref1(0,0)*p_ref1(0)*d1*p_query1(0)*Rxz(2,0) - R_ref1(1,0)*t_ref1(1)*p_query1(0)*Rxz(2,0) + R_ref1(1,0)*p_ref1(1)*d1*p_query1(0)*Rxz(2,0) - R_ref1(2,0)*t_ref1(2)*p_query1(0)*Rxz(2,0) + R_ref1(2,0)*d1*p_query1(0)*Rxz(2,0) - R_ref1(0,1)*t_ref1(0)*p_query1(0)*Rxz(2,1) + R_ref1(0,1)*p_ref1(0)*d1*p_query1(0)*Rxz(2,1) - R_ref1(1,1)*t_ref1(1)*p_query1(0)*Rxz(2,1) + R_ref1(1,1)*p_ref1(1)*d1*p_query1(0)*Rxz(2,1) - R_ref1(2,1)*t_ref1(2)*p_query1(0)*Rxz(2,1) + R_ref1(2,1)*d1*p_query1(0)*Rxz(2,1) - R_ref1(0,2)*t_ref1(0)*p_query1(0)*Rxz(2,2) + R_ref1(0,2)*p_ref1(0)*d1*p_query1(0)*Rxz(2,2) - R_ref1(1,2)*t_ref1(1)*p_query1(0)*Rxz(2,2) + R_ref1(1,2)*p_ref1(1)*d1*p_query1(0)*Rxz(2,2) - R_ref1(2,2)*t_ref1(2)*p_query1(0)*Rxz(2,2) + R_ref1(2,2)*d1*p_query1(0)*Rxz(2,2),
    p_query1(1),
    R_ref1(0,0)*t_ref1(0)*Rxz(1,0) - R_ref1(0,0)*p_ref1(0)*d1*Rxz(1,0) + R_ref1(1,0)*t_ref1(1)*Rxz(1,0) - R_ref1(1,0)*p_ref1(1)*d1*Rxz(1,0) + R_ref1(2,0)*t_ref1(2)*Rxz(1,0) - R_ref1(2,0)*d1*Rxz(1,0) + R_ref1(0,1)*t_ref1(0)*Rxz(1,1) - R_ref1(0,1)*p_ref1(0)*d1*Rxz(1,1) + R_ref1(1,1)*t_ref1(1)*Rxz(1,1) - R_ref1(1,1)*p_ref1(1)*d1*Rxz(1,1) + R_ref1(2,1)*t_ref1(2)*Rxz(1,1) - R_ref1(2,1)*d1*Rxz(1,1) + R_ref1(0,2)*t_ref1(0)*Rxz(1,2) - R_ref1(0,2)*p_ref1(0)*d1*Rxz(1,2) + R_ref1(1,2)*t_ref1(1)*Rxz(1,2) - R_ref1(1,2)*p_ref1(1)*d1*Rxz(1,2) + R_ref1(2,2)*t_ref1(2)*Rxz(1,2) - R_ref1(2,2)*d1*Rxz(1,2),
    R_ref1(0,0)*t_ref1(0)*p_query1(1)*Rxz(2,0) - R_ref1(0,0)*p_ref1(0)*d1*p_query1(1)*Rxz(2,0) + R_ref1(1,0)*t_ref1(1)*p_query1(1)*Rxz(2,0) - R_ref1(1,0)*p_ref1(1)*d1*p_query1(1)*Rxz(2,0) + R_ref1(2,0)*t_ref1(2)*p_query1(1)*Rxz(2,0) - R_ref1(2,0)*d1*p_query1(1)*Rxz(2,0) + R_ref1(0,1)*t_ref1(0)*p_query1(1)*Rxz(2,1) - R_ref1(0,1)*p_ref1(0)*d1*p_query1(1)*Rxz(2,1) + R_ref1(1,1)*t_ref1(1)*p_query1(1)*Rxz(2,1) - R_ref1(1,1)*p_ref1(1)*d1*p_query1(1)*Rxz(2,1) + R_ref1(2,1)*t_ref1(2)*p_query1(1)*Rxz(2,1) - R_ref1(2,1)*d1*p_query1(1)*Rxz(2,1) + R_ref1(0,2)*t_ref1(0)*p_query1(1)*Rxz(2,2) - R_ref1(0,2)*p_ref1(0)*d1*p_query1(1)*Rxz(2,2) + R_ref1(1,2)*t_ref1(1)*p_query1(1)*Rxz(2,2) - R_ref1(1,2)*p_ref1(1)*d1*p_query1(1)*Rxz(2,2) + R_ref1(2,2)*t_ref1(2)*p_query1(1)*Rxz(2,2) - R_ref1(2,2)*d1*p_query1(1)*Rxz(2,2),
    -2*R_ref1(0,0)*t_ref1(0)*p_query1(1)*Rxz(0,0) + 2*R_ref1(0,0)*p_ref1(0)*d1*p_query1(1)*Rxz(0,0) - 2*R_ref1(1,0)*t_ref1(1)*p_query1(1)*Rxz(0,0) + 2*R_ref1(1,0)*p_ref1(1)*d1*p_query1(1)*Rxz(0,0) - 2*R_ref1(2,0)*t_ref1(2)*p_query1(1)*Rxz(0,0) + 2*R_ref1(2,0)*d1*p_query1(1)*Rxz(0,0) - 2*R_ref1(0,1)*t_ref1(0)*p_query1(1)*Rxz(0,1) + 2*R_ref1(0,1)*p_ref1(0)*d1*p_query1(1)*Rxz(0,1) - 2*R_ref1(1,1)*t_ref1(1)*p_query1(1)*Rxz(0,1) + 2*R_ref1(1,1)*p_ref1(1)*d1*p_query1(1)*Rxz(0,1) - 2*R_ref1(2,1)*t_ref1(2)*p_query1(1)*Rxz(0,1) + 2*R_ref1(2,1)*d1*p_query1(1)*Rxz(0,1) - 2*R_ref1(0,2)*t_ref1(0)*p_query1(1)*Rxz(0,2) + 2*R_ref1(0,2)*p_ref1(0)*d1*p_query1(1)*Rxz(0,2) - 2*R_ref1(1,2)*t_ref1(1)*p_query1(1)*Rxz(0,2) + 2*R_ref1(1,2)*p_ref1(1)*d1*p_query1(1)*Rxz(0,2) - 2*R_ref1(2,2)*t_ref1(2)*p_query1(1)*Rxz(0,2) + 2*R_ref1(2,2)*d1*p_query1(1)*Rxz(0,2),
    R_ref1(0,0)*t_ref1(0)*Rxz(1,0) - R_ref1(0,0)*p_ref1(0)*d1*Rxz(1,0) + R_ref1(1,0)*t_ref1(1)*Rxz(1,0) - R_ref1(1,0)*p_ref1(1)*d1*Rxz(1,0) + R_ref1(2,0)*t_ref1(2)*Rxz(1,0) - R_ref1(2,0)*d1*Rxz(1,0) + R_ref1(0,1)*t_ref1(0)*Rxz(1,1) - R_ref1(0,1)*p_ref1(0)*d1*Rxz(1,1) + R_ref1(1,1)*t_ref1(1)*Rxz(1,1) - R_ref1(1,1)*p_ref1(1)*d1*Rxz(1,1) + R_ref1(2,1)*t_ref1(2)*Rxz(1,1) - R_ref1(2,1)*d1*Rxz(1,1) + R_ref1(0,2)*t_ref1(0)*Rxz(1,2) - R_ref1(0,2)*p_ref1(0)*d1*Rxz(1,2) + R_ref1(1,2)*t_ref1(1)*Rxz(1,2) - R_ref1(1,2)*p_ref1(1)*d1*Rxz(1,2) + R_ref1(2,2)*t_ref1(2)*Rxz(1,2) - R_ref1(2,2)*d1*Rxz(1,2),
    -R_ref1(0,0)*t_ref1(0)*p_query1(1)*Rxz(2,0) + R_ref1(0,0)*p_ref1(0)*d1*p_query1(1)*Rxz(2,0) - R_ref1(1,0)*t_ref1(1)*p_query1(1)*Rxz(2,0) + R_ref1(1,0)*p_ref1(1)*d1*p_query1(1)*Rxz(2,0) - R_ref1(2,0)*t_ref1(2)*p_query1(1)*Rxz(2,0) + R_ref1(2,0)*d1*p_query1(1)*Rxz(2,0) - R_ref1(0,1)*t_ref1(0)*p_query1(1)*Rxz(2,1) + R_ref1(0,1)*p_ref1(0)*d1*p_query1(1)*Rxz(2,1) - R_ref1(1,1)*t_ref1(1)*p_query1(1)*Rxz(2,1) + R_ref1(1,1)*p_ref1(1)*d1*p_query1(1)*Rxz(2,1) - R_ref1(2,1)*t_ref1(2)*p_query1(1)*Rxz(2,1) + R_ref1(2,1)*d1*p_query1(1)*Rxz(2,1) - R_ref1(0,2)*t_ref1(0)*p_query1(1)*Rxz(2,2) + R_ref1(0,2)*p_ref1(0)*d1*p_query1(1)*Rxz(2,2) - R_ref1(1,2)*t_ref1(1)*p_query1(1)*Rxz(2,2) + R_ref1(1,2)*p_ref1(1)*d1*p_query1(1)*Rxz(2,2) - R_ref1(2,2)*t_ref1(2)*p_query1(1)*Rxz(2,2) + R_ref1(2,2)*d1*p_query1(1)*Rxz(2,2);

    return M1;
}


static const Eigen::Matrix<double, 6, 1> generate_coeffs_ori(
    const Eigen::Matrix3d &R_ref1,
    const Eigen::Vector3d &t_ref1,
    const double c_ref1,
    const double s_ref1,
    const double c_query1,
    const double s_query1,
    const Eigen::Vector2d &p_ref1,
    const double d1,
    const Eigen::Vector3d &n_ref1, 
    const Eigen::Vector2d &p_query1,
    const Eigen::Matrix3d &Rxz)
{
    Eigen::Matrix<double, 6, 1> M;
M << -R_ref1(0,0)*c_ref1*c_query1*p_ref1(1)*n_ref1(1)*Rxz(1,0) - R_ref1(0,0)*c_ref1*c_query1*n_ref1(2)*Rxz(1,0) - R_ref1(0,0)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*Rxz(0,0) - R_ref1(0,0)*c_ref1*s_query1*n_ref1(2)*Rxz(0,0) + R_ref1(0,0)*s_ref1*c_query1*p_ref1(0)*n_ref1(1)*Rxz(1,0) + R_ref1(0,0)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*Rxz(0,0) + R_ref1(1,0)*c_ref1*c_query1*p_ref1(1)*n_ref1(0)*Rxz(1,0) + R_ref1(1,0)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*Rxz(0,0) - R_ref1(1,0)*s_ref1*c_query1*p_ref1(0)*n_ref1(0)*Rxz(1,0) - R_ref1(1,0)*s_ref1*c_query1*n_ref1(2)*Rxz(1,0) - R_ref1(1,0)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*Rxz(0,0) - R_ref1(1,0)*s_ref1*s_query1*n_ref1(2)*Rxz(0,0) + R_ref1(2,0)*c_ref1*c_query1*n_ref1(0)*Rxz(1,0) + R_ref1(2,0)*c_ref1*s_query1*n_ref1(0)*Rxz(0,0) + R_ref1(2,0)*s_ref1*c_query1*n_ref1(1)*Rxz(1,0) + R_ref1(2,0)*s_ref1*s_query1*n_ref1(1)*Rxz(0,0) - R_ref1(0,1)*c_ref1*c_query1*p_ref1(1)*n_ref1(1)*Rxz(1,1) - R_ref1(0,1)*c_ref1*c_query1*n_ref1(2)*Rxz(1,1) - R_ref1(0,1)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*Rxz(0,1) - R_ref1(0,1)*c_ref1*s_query1*n_ref1(2)*Rxz(0,1) + R_ref1(0,1)*s_ref1*c_query1*p_ref1(0)*n_ref1(1)*Rxz(1,1) + R_ref1(0,1)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*Rxz(0,1) + R_ref1(1,1)*c_ref1*c_query1*p_ref1(1)*n_ref1(0)*Rxz(1,1) + R_ref1(1,1)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*Rxz(0,1) - R_ref1(1,1)*s_ref1*c_query1*p_ref1(0)*n_ref1(0)*Rxz(1,1) - R_ref1(1,1)*s_ref1*c_query1*n_ref1(2)*Rxz(1,1) - R_ref1(1,1)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*Rxz(0,1) - R_ref1(1,1)*s_ref1*s_query1*n_ref1(2)*Rxz(0,1) + R_ref1(2,1)*c_ref1*c_query1*n_ref1(0)*Rxz(1,1) + R_ref1(2,1)*c_ref1*s_query1*n_ref1(0)*Rxz(0,1) + R_ref1(2,1)*s_ref1*c_query1*n_ref1(1)*Rxz(1,1) + R_ref1(2,1)*s_ref1*s_query1*n_ref1(1)*Rxz(0,1) - R_ref1(0,2)*c_ref1*c_query1*p_ref1(1)*n_ref1(1)*Rxz(1,2) - R_ref1(0,2)*c_ref1*c_query1*n_ref1(2)*Rxz(1,2) - R_ref1(0,2)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*Rxz(0,2) - R_ref1(0,2)*c_ref1*s_query1*n_ref1(2)*Rxz(0,2) + R_ref1(0,2)*s_ref1*c_query1*p_ref1(0)*n_ref1(1)*Rxz(1,2) + R_ref1(0,2)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*Rxz(0,2) + R_ref1(1,2)*c_ref1*c_query1*p_ref1(1)*n_ref1(0)*Rxz(1,2) + R_ref1(1,2)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*Rxz(0,2) - R_ref1(1,2)*s_ref1*c_query1*p_ref1(0)*n_ref1(0)*Rxz(1,2) - R_ref1(1,2)*s_ref1*c_query1*n_ref1(2)*Rxz(1,2) - R_ref1(1,2)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*Rxz(0,2) - R_ref1(1,2)*s_ref1*s_query1*n_ref1(2)*Rxz(0,2) + R_ref1(2,2)*c_ref1*c_query1*n_ref1(0)*Rxz(1,2) + R_ref1(2,2)*c_ref1*s_query1*n_ref1(0)*Rxz(0,2) + R_ref1(2,2)*s_ref1*c_query1*n_ref1(1)*Rxz(1,2) + R_ref1(2,2)*s_ref1*s_query1*n_ref1(1)*Rxz(0,2),
-R_ref1(0,0)*c_ref1*c_query1*p_ref1(1)*n_ref1(1)*p_query1(1)*Rxz(2,0) - R_ref1(0,0)*c_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(2,0) + R_ref1(0,0)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*p_query1(0)*Rxz(2,0) + R_ref1(0,0)*c_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(2,0) + R_ref1(0,0)*s_ref1*c_query1*p_ref1(0)*n_ref1(1)*p_query1(1)*Rxz(2,0) - R_ref1(0,0)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*p_query1(0)*Rxz(2,0) + R_ref1(1,0)*c_ref1*c_query1*p_ref1(1)*n_ref1(0)*p_query1(1)*Rxz(2,0) - R_ref1(1,0)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*p_query1(0)*Rxz(2,0) - R_ref1(1,0)*s_ref1*c_query1*p_ref1(0)*n_ref1(0)*p_query1(1)*Rxz(2,0) - R_ref1(1,0)*s_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(2,0) + R_ref1(1,0)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*p_query1(0)*Rxz(2,0) + R_ref1(1,0)*s_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(2,0) + R_ref1(2,0)*c_ref1*c_query1*n_ref1(0)*p_query1(1)*Rxz(2,0) - R_ref1(2,0)*c_ref1*s_query1*n_ref1(0)*p_query1(0)*Rxz(2,0) + R_ref1(2,0)*s_ref1*c_query1*n_ref1(1)*p_query1(1)*Rxz(2,0) - R_ref1(2,0)*s_ref1*s_query1*n_ref1(1)*p_query1(0)*Rxz(2,0) - R_ref1(0,1)*c_ref1*c_query1*p_ref1(1)*n_ref1(1)*p_query1(1)*Rxz(2,1) - R_ref1(0,1)*c_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(2,1) + R_ref1(0,1)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*p_query1(0)*Rxz(2,1) + R_ref1(0,1)*c_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(2,1) + R_ref1(0,1)*s_ref1*c_query1*p_ref1(0)*n_ref1(1)*p_query1(1)*Rxz(2,1) - R_ref1(0,1)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*p_query1(0)*Rxz(2,1) + R_ref1(1,1)*c_ref1*c_query1*p_ref1(1)*n_ref1(0)*p_query1(1)*Rxz(2,1) - R_ref1(1,1)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*p_query1(0)*Rxz(2,1) - R_ref1(1,1)*s_ref1*c_query1*p_ref1(0)*n_ref1(0)*p_query1(1)*Rxz(2,1) - R_ref1(1,1)*s_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(2,1) + R_ref1(1,1)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*p_query1(0)*Rxz(2,1) + R_ref1(1,1)*s_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(2,1) + R_ref1(2,1)*c_ref1*c_query1*n_ref1(0)*p_query1(1)*Rxz(2,1) - R_ref1(2,1)*c_ref1*s_query1*n_ref1(0)*p_query1(0)*Rxz(2,1) + R_ref1(2,1)*s_ref1*c_query1*n_ref1(1)*p_query1(1)*Rxz(2,1) - R_ref1(2,1)*s_ref1*s_query1*n_ref1(1)*p_query1(0)*Rxz(2,1) - R_ref1(0,2)*c_ref1*c_query1*p_ref1(1)*n_ref1(1)*p_query1(1)*Rxz(2,2) - R_ref1(0,2)*c_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(2,2) + R_ref1(0,2)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*p_query1(0)*Rxz(2,2) + R_ref1(0,2)*c_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(2,2) + R_ref1(0,2)*s_ref1*c_query1*p_ref1(0)*n_ref1(1)*p_query1(1)*Rxz(2,2) - R_ref1(0,2)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*p_query1(0)*Rxz(2,2) + R_ref1(1,2)*c_ref1*c_query1*p_ref1(1)*n_ref1(0)*p_query1(1)*Rxz(2,2) - R_ref1(1,2)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*p_query1(0)*Rxz(2,2) - R_ref1(1,2)*s_ref1*c_query1*p_ref1(0)*n_ref1(0)*p_query1(1)*Rxz(2,2) - R_ref1(1,2)*s_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(2,2) + R_ref1(1,2)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*p_query1(0)*Rxz(2,2) + R_ref1(1,2)*s_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(2,2) + R_ref1(2,2)*c_ref1*c_query1*n_ref1(0)*p_query1(1)*Rxz(2,2) - R_ref1(2,2)*c_ref1*s_query1*n_ref1(0)*p_query1(0)*Rxz(2,2) + R_ref1(2,2)*s_ref1*c_query1*n_ref1(1)*p_query1(1)*Rxz(2,2) - R_ref1(2,2)*s_ref1*s_query1*n_ref1(1)*p_query1(0)*Rxz(2,2),
-2*R_ref1(0,0)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*Rxz(2,0) - 2*R_ref1(0,0)*c_ref1*s_query1*n_ref1(2)*Rxz(2,0) + 2*R_ref1(0,0)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*Rxz(2,0) + 2*R_ref1(1,0)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*Rxz(2,0) - 2*R_ref1(1,0)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*Rxz(2,0) - 2*R_ref1(1,0)*s_ref1*s_query1*n_ref1(2)*Rxz(2,0) + 2*R_ref1(2,0)*c_ref1*s_query1*n_ref1(0)*Rxz(2,0) + 2*R_ref1(2,0)*s_ref1*s_query1*n_ref1(1)*Rxz(2,0) - 2*R_ref1(0,1)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*Rxz(2,1) - 2*R_ref1(0,1)*c_ref1*s_query1*n_ref1(2)*Rxz(2,1) + 2*R_ref1(0,1)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*Rxz(2,1) + 2*R_ref1(1,1)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*Rxz(2,1) - 2*R_ref1(1,1)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*Rxz(2,1) - 2*R_ref1(1,1)*s_ref1*s_query1*n_ref1(2)*Rxz(2,1) + 2*R_ref1(2,1)*c_ref1*s_query1*n_ref1(0)*Rxz(2,1) + 2*R_ref1(2,1)*s_ref1*s_query1*n_ref1(1)*Rxz(2,1) - 2*R_ref1(0,2)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*Rxz(2,2) - 2*R_ref1(0,2)*c_ref1*s_query1*n_ref1(2)*Rxz(2,2) + 2*R_ref1(0,2)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*Rxz(2,2) + 2*R_ref1(1,2)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*Rxz(2,2) - 2*R_ref1(1,2)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*Rxz(2,2) - 2*R_ref1(1,2)*s_ref1*s_query1*n_ref1(2)*Rxz(2,2) + 2*R_ref1(2,2)*c_ref1*s_query1*n_ref1(0)*Rxz(2,2) + 2*R_ref1(2,2)*s_ref1*s_query1*n_ref1(1)*Rxz(2,2),
2*R_ref1(0,0)*c_ref1*c_query1*p_ref1(1)*n_ref1(1)*p_query1(1)*Rxz(0,0) + 2*R_ref1(0,0)*c_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(0,0) - 2*R_ref1(0,0)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*p_query1(0)*Rxz(0,0) - 2*R_ref1(0,0)*c_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(0,0) - 2*R_ref1(0,0)*s_ref1*c_query1*p_ref1(0)*n_ref1(1)*p_query1(1)*Rxz(0,0) + 2*R_ref1(0,0)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*p_query1(0)*Rxz(0,0) - 2*R_ref1(1,0)*c_ref1*c_query1*p_ref1(1)*n_ref1(0)*p_query1(1)*Rxz(0,0) + 2*R_ref1(1,0)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*p_query1(0)*Rxz(0,0) + 2*R_ref1(1,0)*s_ref1*c_query1*p_ref1(0)*n_ref1(0)*p_query1(1)*Rxz(0,0) + 2*R_ref1(1,0)*s_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(0,0) - 2*R_ref1(1,0)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*p_query1(0)*Rxz(0,0) - 2*R_ref1(1,0)*s_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(0,0) - 2*R_ref1(2,0)*c_ref1*c_query1*n_ref1(0)*p_query1(1)*Rxz(0,0) + 2*R_ref1(2,0)*c_ref1*s_query1*n_ref1(0)*p_query1(0)*Rxz(0,0) - 2*R_ref1(2,0)*s_ref1*c_query1*n_ref1(1)*p_query1(1)*Rxz(0,0) + 2*R_ref1(2,0)*s_ref1*s_query1*n_ref1(1)*p_query1(0)*Rxz(0,0) + 2*R_ref1(0,1)*c_ref1*c_query1*p_ref1(1)*n_ref1(1)*p_query1(1)*Rxz(0,1) + 2*R_ref1(0,1)*c_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(0,1) - 2*R_ref1(0,1)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*p_query1(0)*Rxz(0,1) - 2*R_ref1(0,1)*c_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(0,1) - 2*R_ref1(0,1)*s_ref1*c_query1*p_ref1(0)*n_ref1(1)*p_query1(1)*Rxz(0,1) + 2*R_ref1(0,1)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*p_query1(0)*Rxz(0,1) - 2*R_ref1(1,1)*c_ref1*c_query1*p_ref1(1)*n_ref1(0)*p_query1(1)*Rxz(0,1) + 2*R_ref1(1,1)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*p_query1(0)*Rxz(0,1) + 2*R_ref1(1,1)*s_ref1*c_query1*p_ref1(0)*n_ref1(0)*p_query1(1)*Rxz(0,1) + 2*R_ref1(1,1)*s_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(0,1) - 2*R_ref1(1,1)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*p_query1(0)*Rxz(0,1) - 2*R_ref1(1,1)*s_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(0,1) - 2*R_ref1(2,1)*c_ref1*c_query1*n_ref1(0)*p_query1(1)*Rxz(0,1) + 2*R_ref1(2,1)*c_ref1*s_query1*n_ref1(0)*p_query1(0)*Rxz(0,1) - 2*R_ref1(2,1)*s_ref1*c_query1*n_ref1(1)*p_query1(1)*Rxz(0,1) + 2*R_ref1(2,1)*s_ref1*s_query1*n_ref1(1)*p_query1(0)*Rxz(0,1) + 2*R_ref1(0,2)*c_ref1*c_query1*p_ref1(1)*n_ref1(1)*p_query1(1)*Rxz(0,2) + 2*R_ref1(0,2)*c_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(0,2) - 2*R_ref1(0,2)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*p_query1(0)*Rxz(0,2) - 2*R_ref1(0,2)*c_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(0,2) - 2*R_ref1(0,2)*s_ref1*c_query1*p_ref1(0)*n_ref1(1)*p_query1(1)*Rxz(0,2) + 2*R_ref1(0,2)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*p_query1(0)*Rxz(0,2) - 2*R_ref1(1,2)*c_ref1*c_query1*p_ref1(1)*n_ref1(0)*p_query1(1)*Rxz(0,2) + 2*R_ref1(1,2)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*p_query1(0)*Rxz(0,2) + 2*R_ref1(1,2)*s_ref1*c_query1*p_ref1(0)*n_ref1(0)*p_query1(1)*Rxz(0,2) + 2*R_ref1(1,2)*s_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(0,2) - 2*R_ref1(1,2)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*p_query1(0)*Rxz(0,2) - 2*R_ref1(1,2)*s_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(0,2) - 2*R_ref1(2,2)*c_ref1*c_query1*n_ref1(0)*p_query1(1)*Rxz(0,2) + 2*R_ref1(2,2)*c_ref1*s_query1*n_ref1(0)*p_query1(0)*Rxz(0,2) - 2*R_ref1(2,2)*s_ref1*c_query1*n_ref1(1)*p_query1(1)*Rxz(0,2) + 2*R_ref1(2,2)*s_ref1*s_query1*n_ref1(1)*p_query1(0)*Rxz(0,2),
-R_ref1(0,0)*c_ref1*c_query1*p_ref1(1)*n_ref1(1)*Rxz(1,0) - R_ref1(0,0)*c_ref1*c_query1*n_ref1(2)*Rxz(1,0) + R_ref1(0,0)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*Rxz(0,0) + R_ref1(0,0)*c_ref1*s_query1*n_ref1(2)*Rxz(0,0) + R_ref1(0,0)*s_ref1*c_query1*p_ref1(0)*n_ref1(1)*Rxz(1,0) - R_ref1(0,0)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*Rxz(0,0) + R_ref1(1,0)*c_ref1*c_query1*p_ref1(1)*n_ref1(0)*Rxz(1,0) - R_ref1(1,0)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*Rxz(0,0) - R_ref1(1,0)*s_ref1*c_query1*p_ref1(0)*n_ref1(0)*Rxz(1,0) - R_ref1(1,0)*s_ref1*c_query1*n_ref1(2)*Rxz(1,0) + R_ref1(1,0)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*Rxz(0,0) + R_ref1(1,0)*s_ref1*s_query1*n_ref1(2)*Rxz(0,0) + R_ref1(2,0)*c_ref1*c_query1*n_ref1(0)*Rxz(1,0) - R_ref1(2,0)*c_ref1*s_query1*n_ref1(0)*Rxz(0,0) + R_ref1(2,0)*s_ref1*c_query1*n_ref1(1)*Rxz(1,0) - R_ref1(2,0)*s_ref1*s_query1*n_ref1(1)*Rxz(0,0) - R_ref1(0,1)*c_ref1*c_query1*p_ref1(1)*n_ref1(1)*Rxz(1,1) - R_ref1(0,1)*c_ref1*c_query1*n_ref1(2)*Rxz(1,1) + R_ref1(0,1)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*Rxz(0,1) + R_ref1(0,1)*c_ref1*s_query1*n_ref1(2)*Rxz(0,1) + R_ref1(0,1)*s_ref1*c_query1*p_ref1(0)*n_ref1(1)*Rxz(1,1) - R_ref1(0,1)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*Rxz(0,1) + R_ref1(1,1)*c_ref1*c_query1*p_ref1(1)*n_ref1(0)*Rxz(1,1) - R_ref1(1,1)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*Rxz(0,1) - R_ref1(1,1)*s_ref1*c_query1*p_ref1(0)*n_ref1(0)*Rxz(1,1) - R_ref1(1,1)*s_ref1*c_query1*n_ref1(2)*Rxz(1,1) + R_ref1(1,1)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*Rxz(0,1) + R_ref1(1,1)*s_ref1*s_query1*n_ref1(2)*Rxz(0,1) + R_ref1(2,1)*c_ref1*c_query1*n_ref1(0)*Rxz(1,1) - R_ref1(2,1)*c_ref1*s_query1*n_ref1(0)*Rxz(0,1) + R_ref1(2,1)*s_ref1*c_query1*n_ref1(1)*Rxz(1,1) - R_ref1(2,1)*s_ref1*s_query1*n_ref1(1)*Rxz(0,1) - R_ref1(0,2)*c_ref1*c_query1*p_ref1(1)*n_ref1(1)*Rxz(1,2) - R_ref1(0,2)*c_ref1*c_query1*n_ref1(2)*Rxz(1,2) + R_ref1(0,2)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*Rxz(0,2) + R_ref1(0,2)*c_ref1*s_query1*n_ref1(2)*Rxz(0,2) + R_ref1(0,2)*s_ref1*c_query1*p_ref1(0)*n_ref1(1)*Rxz(1,2) - R_ref1(0,2)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*Rxz(0,2) + R_ref1(1,2)*c_ref1*c_query1*p_ref1(1)*n_ref1(0)*Rxz(1,2) - R_ref1(1,2)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*Rxz(0,2) - R_ref1(1,2)*s_ref1*c_query1*p_ref1(0)*n_ref1(0)*Rxz(1,2) - R_ref1(1,2)*s_ref1*c_query1*n_ref1(2)*Rxz(1,2) + R_ref1(1,2)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*Rxz(0,2) + R_ref1(1,2)*s_ref1*s_query1*n_ref1(2)*Rxz(0,2) + R_ref1(2,2)*c_ref1*c_query1*n_ref1(0)*Rxz(1,2) - R_ref1(2,2)*c_ref1*s_query1*n_ref1(0)*Rxz(0,2) + R_ref1(2,2)*s_ref1*c_query1*n_ref1(1)*Rxz(1,2) - R_ref1(2,2)*s_ref1*s_query1*n_ref1(1)*Rxz(0,2),
R_ref1(0,0)*c_ref1*c_query1*p_ref1(1)*n_ref1(1)*p_query1(1)*Rxz(2,0) + R_ref1(0,0)*c_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(2,0) - R_ref1(0,0)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*p_query1(0)*Rxz(2,0) - R_ref1(0,0)*c_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(2,0) - R_ref1(0,0)*s_ref1*c_query1*p_ref1(0)*n_ref1(1)*p_query1(1)*Rxz(2,0) + R_ref1(0,0)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*p_query1(0)*Rxz(2,0) - R_ref1(1,0)*c_ref1*c_query1*p_ref1(1)*n_ref1(0)*p_query1(1)*Rxz(2,0) + R_ref1(1,0)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*p_query1(0)*Rxz(2,0) + R_ref1(1,0)*s_ref1*c_query1*p_ref1(0)*n_ref1(0)*p_query1(1)*Rxz(2,0) + R_ref1(1,0)*s_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(2,0) - R_ref1(1,0)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*p_query1(0)*Rxz(2,0) - R_ref1(1,0)*s_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(2,0) - R_ref1(2,0)*c_ref1*c_query1*n_ref1(0)*p_query1(1)*Rxz(2,0) + R_ref1(2,0)*c_ref1*s_query1*n_ref1(0)*p_query1(0)*Rxz(2,0) - R_ref1(2,0)*s_ref1*c_query1*n_ref1(1)*p_query1(1)*Rxz(2,0) + R_ref1(2,0)*s_ref1*s_query1*n_ref1(1)*p_query1(0)*Rxz(2,0) + R_ref1(0,1)*c_ref1*c_query1*p_ref1(1)*n_ref1(1)*p_query1(1)*Rxz(2,1) + R_ref1(0,1)*c_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(2,1) - R_ref1(0,1)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*p_query1(0)*Rxz(2,1) - R_ref1(0,1)*c_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(2,1) - R_ref1(0,1)*s_ref1*c_query1*p_ref1(0)*n_ref1(1)*p_query1(1)*Rxz(2,1) + R_ref1(0,1)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*p_query1(0)*Rxz(2,1) - R_ref1(1,1)*c_ref1*c_query1*p_ref1(1)*n_ref1(0)*p_query1(1)*Rxz(2,1) + R_ref1(1,1)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*p_query1(0)*Rxz(2,1) + R_ref1(1,1)*s_ref1*c_query1*p_ref1(0)*n_ref1(0)*p_query1(1)*Rxz(2,1) + R_ref1(1,1)*s_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(2,1) - R_ref1(1,1)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*p_query1(0)*Rxz(2,1) - R_ref1(1,1)*s_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(2,1) - R_ref1(2,1)*c_ref1*c_query1*n_ref1(0)*p_query1(1)*Rxz(2,1) + R_ref1(2,1)*c_ref1*s_query1*n_ref1(0)*p_query1(0)*Rxz(2,1) - R_ref1(2,1)*s_ref1*c_query1*n_ref1(1)*p_query1(1)*Rxz(2,1) + R_ref1(2,1)*s_ref1*s_query1*n_ref1(1)*p_query1(0)*Rxz(2,1) + R_ref1(0,2)*c_ref1*c_query1*p_ref1(1)*n_ref1(1)*p_query1(1)*Rxz(2,2) + R_ref1(0,2)*c_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(2,2) - R_ref1(0,2)*c_ref1*s_query1*p_ref1(1)*n_ref1(1)*p_query1(0)*Rxz(2,2) - R_ref1(0,2)*c_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(2,2) - R_ref1(0,2)*s_ref1*c_query1*p_ref1(0)*n_ref1(1)*p_query1(1)*Rxz(2,2) + R_ref1(0,2)*s_ref1*s_query1*p_ref1(0)*n_ref1(1)*p_query1(0)*Rxz(2,2) - R_ref1(1,2)*c_ref1*c_query1*p_ref1(1)*n_ref1(0)*p_query1(1)*Rxz(2,2) + R_ref1(1,2)*c_ref1*s_query1*p_ref1(1)*n_ref1(0)*p_query1(0)*Rxz(2,2) + R_ref1(1,2)*s_ref1*c_query1*p_ref1(0)*n_ref1(0)*p_query1(1)*Rxz(2,2) + R_ref1(1,2)*s_ref1*c_query1*n_ref1(2)*p_query1(1)*Rxz(2,2) - R_ref1(1,2)*s_ref1*s_query1*p_ref1(0)*n_ref1(0)*p_query1(0)*Rxz(2,2) - R_ref1(1,2)*s_ref1*s_query1*n_ref1(2)*p_query1(0)*Rxz(2,2) - R_ref1(2,2)*c_ref1*c_query1*n_ref1(0)*p_query1(1)*Rxz(2,2) + R_ref1(2,2)*c_ref1*s_query1*n_ref1(0)*p_query1(0)*Rxz(2,2) - R_ref1(2,2)*s_ref1*c_query1*n_ref1(1)*p_query1(1)*Rxz(2,2) + R_ref1(2,2)*s_ref1*s_query1*n_ref1(1)*p_query1(0)*Rxz(2,2);

return M;
}


static const double compute_algebraic_error_unused_constraint_ref(
    const Eigen::Matrix<double, 6, 1> &c,
    const double r,
    const double f)
{
    Eigen::Matrix<double, 6, 1> monomials = { r * r * f, r * r, r * f, r, f, 1 };
    double eqs6 = c.transpose() * monomials;
    double err = std::abs(eqs6);
    return err;
}
} // namespace ECCV2026
