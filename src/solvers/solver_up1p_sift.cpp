#include <Eigen/Dense>
#include <unsupported/Eigen/Polynomials>

namespace ECCV2026 {
static Eigen::Matrix<double, 4, 6> build_linear_constraints(const Eigen::Matrix3d &R_ref, const Eigen::Vector3d &t_ref,
                                                            const double s_ref, const double c_ref,
                                                            const double s_query, const double c_query, const double q,
                                                            const Eigen::Vector2d &p_ref, const double d,
                                                            const Eigen::Vector3d &n, const Eigen::Vector2d &p_query,
                                                            const Eigen::Matrix3d &Rxz) {
    const Eigen::Vector3d n_ref = R_ref * n;
    Eigen::Matrix<double, 4, 6> M;
    M << -1, 0, p_query(0),
        R_ref(2, 0) * Rxz(0, 0) * d + R_ref(2, 1) * Rxz(0, 1) * d + R_ref(2, 2) * Rxz(0, 2) * d -
            R_ref(0, 0) * Rxz(0, 0) * t_ref(0) - R_ref(0, 1) * Rxz(0, 1) * t_ref(0) -
            R_ref(0, 2) * Rxz(0, 2) * t_ref(0) - R_ref(1, 0) * Rxz(0, 0) * t_ref(1) -
            R_ref(1, 1) * Rxz(0, 1) * t_ref(1) - R_ref(1, 2) * Rxz(0, 2) * t_ref(1) -
            R_ref(2, 0) * Rxz(0, 0) * t_ref(2) - R_ref(2, 1) * Rxz(0, 1) * t_ref(2) -
            R_ref(2, 2) * Rxz(0, 2) * t_ref(2) + R_ref(0, 0) * Rxz(0, 0) * d * p_ref(0) +
            R_ref(0, 1) * Rxz(0, 1) * d * p_ref(0) + R_ref(0, 2) * Rxz(0, 2) * d * p_ref(0) +
            R_ref(1, 0) * Rxz(0, 0) * d * p_ref(1) + R_ref(1, 1) * Rxz(0, 1) * d * p_ref(1) +
            R_ref(1, 2) * Rxz(0, 2) * d * p_ref(1) - R_ref(2, 0) * Rxz(2, 0) * d * p_query(0) -
            R_ref(2, 1) * Rxz(2, 1) * d * p_query(0) - R_ref(2, 2) * Rxz(2, 2) * d * p_query(0) +
            R_ref(0, 0) * Rxz(2, 0) * t_ref(0) * p_query(0) + R_ref(0, 1) * Rxz(2, 1) * t_ref(0) * p_query(0) +
            R_ref(0, 2) * Rxz(2, 2) * t_ref(0) * p_query(0) + R_ref(1, 0) * Rxz(2, 0) * t_ref(1) * p_query(0) +
            R_ref(1, 1) * Rxz(2, 1) * t_ref(1) * p_query(0) + R_ref(1, 2) * Rxz(2, 2) * t_ref(1) * p_query(0) +
            R_ref(2, 0) * Rxz(2, 0) * t_ref(2) * p_query(0) + R_ref(2, 1) * Rxz(2, 1) * t_ref(2) * p_query(0) +
            R_ref(2, 2) * Rxz(2, 2) * t_ref(2) * p_query(0) - R_ref(0, 0) * Rxz(2, 0) * d * p_ref(0) * p_query(0) -
            R_ref(0, 1) * Rxz(2, 1) * d * p_ref(0) * p_query(0) - R_ref(0, 2) * Rxz(2, 2) * d * p_ref(0) * p_query(0) -
            R_ref(1, 0) * Rxz(2, 0) * d * p_ref(1) * p_query(0) - R_ref(1, 1) * Rxz(2, 1) * d * p_ref(1) * p_query(0) -
            R_ref(1, 2) * Rxz(2, 2) * d * p_ref(1) * p_query(0),
        2 * R_ref(2, 0) * Rxz(2, 0) * d + 2 * R_ref(2, 1) * Rxz(2, 1) * d + 2 * R_ref(2, 2) * Rxz(2, 2) * d -
            2 * R_ref(0, 0) * Rxz(2, 0) * t_ref(0) - 2 * R_ref(0, 1) * Rxz(2, 1) * t_ref(0) -
            2 * R_ref(0, 2) * Rxz(2, 2) * t_ref(0) - 2 * R_ref(1, 0) * Rxz(2, 0) * t_ref(1) -
            2 * R_ref(1, 1) * Rxz(2, 1) * t_ref(1) - 2 * R_ref(1, 2) * Rxz(2, 2) * t_ref(1) -
            2 * R_ref(2, 0) * Rxz(2, 0) * t_ref(2) - 2 * R_ref(2, 1) * Rxz(2, 1) * t_ref(2) -
            2 * R_ref(2, 2) * Rxz(2, 2) * t_ref(2) + 2 * R_ref(0, 0) * Rxz(2, 0) * d * p_ref(0) +
            2 * R_ref(0, 1) * Rxz(2, 1) * d * p_ref(0) + 2 * R_ref(0, 2) * Rxz(2, 2) * d * p_ref(0) +
            2 * R_ref(1, 0) * Rxz(2, 0) * d * p_ref(1) + 2 * R_ref(1, 1) * Rxz(2, 1) * d * p_ref(1) +
            2 * R_ref(1, 2) * Rxz(2, 2) * d * p_ref(1) + 2 * R_ref(2, 0) * Rxz(0, 0) * d * p_query(0) +
            2 * R_ref(2, 1) * Rxz(0, 1) * d * p_query(0) + 2 * R_ref(2, 2) * Rxz(0, 2) * d * p_query(0) -
            2 * R_ref(0, 0) * Rxz(0, 0) * t_ref(0) * p_query(0) - 2 * R_ref(0, 1) * Rxz(0, 1) * t_ref(0) * p_query(0) -
            2 * R_ref(0, 2) * Rxz(0, 2) * t_ref(0) * p_query(0) - 2 * R_ref(1, 0) * Rxz(0, 0) * t_ref(1) * p_query(0) -
            2 * R_ref(1, 1) * Rxz(0, 1) * t_ref(1) * p_query(0) - 2 * R_ref(1, 2) * Rxz(0, 2) * t_ref(1) * p_query(0) -
            2 * R_ref(2, 0) * Rxz(0, 0) * t_ref(2) * p_query(0) - 2 * R_ref(2, 1) * Rxz(0, 1) * t_ref(2) * p_query(0) -
            2 * R_ref(2, 2) * Rxz(0, 2) * t_ref(2) * p_query(0) +
            2 * R_ref(0, 0) * Rxz(0, 0) * d * p_ref(0) * p_query(0) +
            2 * R_ref(0, 1) * Rxz(0, 1) * d * p_ref(0) * p_query(0) +
            2 * R_ref(0, 2) * Rxz(0, 2) * d * p_ref(0) * p_query(0) +
            2 * R_ref(1, 0) * Rxz(0, 0) * d * p_ref(1) * p_query(0) +
            2 * R_ref(1, 1) * Rxz(0, 1) * d * p_ref(1) * p_query(0) +
            2 * R_ref(1, 2) * Rxz(0, 2) * d * p_ref(1) * p_query(0),
        R_ref(0, 0) * Rxz(0, 0) * t_ref(0) - R_ref(2, 1) * Rxz(0, 1) * d - R_ref(2, 2) * Rxz(0, 2) * d -
            R_ref(2, 0) * Rxz(0, 0) * d + R_ref(0, 1) * Rxz(0, 1) * t_ref(0) + R_ref(0, 2) * Rxz(0, 2) * t_ref(0) +
            R_ref(1, 0) * Rxz(0, 0) * t_ref(1) + R_ref(1, 1) * Rxz(0, 1) * t_ref(1) +
            R_ref(1, 2) * Rxz(0, 2) * t_ref(1) + R_ref(2, 0) * Rxz(0, 0) * t_ref(2) +
            R_ref(2, 1) * Rxz(0, 1) * t_ref(2) + R_ref(2, 2) * Rxz(0, 2) * t_ref(2) -
            R_ref(0, 0) * Rxz(0, 0) * d * p_ref(0) - R_ref(0, 1) * Rxz(0, 1) * d * p_ref(0) -
            R_ref(0, 2) * Rxz(0, 2) * d * p_ref(0) - R_ref(1, 0) * Rxz(0, 0) * d * p_ref(1) -
            R_ref(1, 1) * Rxz(0, 1) * d * p_ref(1) - R_ref(1, 2) * Rxz(0, 2) * d * p_ref(1) +
            R_ref(2, 0) * Rxz(2, 0) * d * p_query(0) + R_ref(2, 1) * Rxz(2, 1) * d * p_query(0) +
            R_ref(2, 2) * Rxz(2, 2) * d * p_query(0) - R_ref(0, 0) * Rxz(2, 0) * t_ref(0) * p_query(0) -
            R_ref(0, 1) * Rxz(2, 1) * t_ref(0) * p_query(0) - R_ref(0, 2) * Rxz(2, 2) * t_ref(0) * p_query(0) -
            R_ref(1, 0) * Rxz(2, 0) * t_ref(1) * p_query(0) - R_ref(1, 1) * Rxz(2, 1) * t_ref(1) * p_query(0) -
            R_ref(1, 2) * Rxz(2, 2) * t_ref(1) * p_query(0) - R_ref(2, 0) * Rxz(2, 0) * t_ref(2) * p_query(0) -
            R_ref(2, 1) * Rxz(2, 1) * t_ref(2) * p_query(0) - R_ref(2, 2) * Rxz(2, 2) * t_ref(2) * p_query(0) +
            R_ref(0, 0) * Rxz(2, 0) * d * p_ref(0) * p_query(0) + R_ref(0, 1) * Rxz(2, 1) * d * p_ref(0) * p_query(0) +
            R_ref(0, 2) * Rxz(2, 2) * d * p_ref(0) * p_query(0) + R_ref(1, 0) * Rxz(2, 0) * d * p_ref(1) * p_query(0) +
            R_ref(1, 1) * Rxz(2, 1) * d * p_ref(1) * p_query(0) + R_ref(1, 2) * Rxz(2, 2) * d * p_ref(1) * p_query(0),
        0, -1, p_query(1),
        R_ref(0, 0) * Rxz(1, 0) * t_ref(0) - R_ref(2, 1) * Rxz(1, 1) * d - R_ref(2, 2) * Rxz(1, 2) * d -
            R_ref(2, 0) * Rxz(1, 0) * d + R_ref(0, 1) * Rxz(1, 1) * t_ref(0) + R_ref(0, 2) * Rxz(1, 2) * t_ref(0) +
            R_ref(1, 0) * Rxz(1, 0) * t_ref(1) + R_ref(1, 1) * Rxz(1, 1) * t_ref(1) +
            R_ref(1, 2) * Rxz(1, 2) * t_ref(1) + R_ref(2, 0) * Rxz(1, 0) * t_ref(2) +
            R_ref(2, 1) * Rxz(1, 1) * t_ref(2) + R_ref(2, 2) * Rxz(1, 2) * t_ref(2) -
            R_ref(0, 0) * Rxz(1, 0) * d * p_ref(0) - R_ref(0, 1) * Rxz(1, 1) * d * p_ref(0) -
            R_ref(0, 2) * Rxz(1, 2) * d * p_ref(0) - R_ref(1, 0) * Rxz(1, 0) * d * p_ref(1) -
            R_ref(1, 1) * Rxz(1, 1) * d * p_ref(1) - R_ref(1, 2) * Rxz(1, 2) * d * p_ref(1) -
            R_ref(2, 0) * Rxz(2, 0) * d * p_query(1) - R_ref(2, 1) * Rxz(2, 1) * d * p_query(1) -
            R_ref(2, 2) * Rxz(2, 2) * d * p_query(1) + R_ref(0, 0) * Rxz(2, 0) * t_ref(0) * p_query(1) +
            R_ref(0, 1) * Rxz(2, 1) * t_ref(0) * p_query(1) + R_ref(0, 2) * Rxz(2, 2) * t_ref(0) * p_query(1) +
            R_ref(1, 0) * Rxz(2, 0) * t_ref(1) * p_query(1) + R_ref(1, 1) * Rxz(2, 1) * t_ref(1) * p_query(1) +
            R_ref(1, 2) * Rxz(2, 2) * t_ref(1) * p_query(1) + R_ref(2, 0) * Rxz(2, 0) * t_ref(2) * p_query(1) +
            R_ref(2, 1) * Rxz(2, 1) * t_ref(2) * p_query(1) + R_ref(2, 2) * Rxz(2, 2) * t_ref(2) * p_query(1) -
            R_ref(0, 0) * Rxz(2, 0) * d * p_ref(0) * p_query(1) - R_ref(0, 1) * Rxz(2, 1) * d * p_ref(0) * p_query(1) -
            R_ref(0, 2) * Rxz(2, 2) * d * p_ref(0) * p_query(1) - R_ref(1, 0) * Rxz(2, 0) * d * p_ref(1) * p_query(1) -
            R_ref(1, 1) * Rxz(2, 1) * d * p_ref(1) * p_query(1) - R_ref(1, 2) * Rxz(2, 2) * d * p_ref(1) * p_query(1),
        2 * R_ref(2, 0) * Rxz(0, 0) * d * p_query(1) + 2 * R_ref(2, 1) * Rxz(0, 1) * d * p_query(1) +
            2 * R_ref(2, 2) * Rxz(0, 2) * d * p_query(1) - 2 * R_ref(0, 0) * Rxz(0, 0) * t_ref(0) * p_query(1) -
            2 * R_ref(0, 1) * Rxz(0, 1) * t_ref(0) * p_query(1) - 2 * R_ref(0, 2) * Rxz(0, 2) * t_ref(0) * p_query(1) -
            2 * R_ref(1, 0) * Rxz(0, 0) * t_ref(1) * p_query(1) - 2 * R_ref(1, 1) * Rxz(0, 1) * t_ref(1) * p_query(1) -
            2 * R_ref(1, 2) * Rxz(0, 2) * t_ref(1) * p_query(1) - 2 * R_ref(2, 0) * Rxz(0, 0) * t_ref(2) * p_query(1) -
            2 * R_ref(2, 1) * Rxz(0, 1) * t_ref(2) * p_query(1) - 2 * R_ref(2, 2) * Rxz(0, 2) * t_ref(2) * p_query(1) +
            2 * R_ref(0, 0) * Rxz(0, 0) * d * p_ref(0) * p_query(1) +
            2 * R_ref(0, 1) * Rxz(0, 1) * d * p_ref(0) * p_query(1) +
            2 * R_ref(0, 2) * Rxz(0, 2) * d * p_ref(0) * p_query(1) +
            2 * R_ref(1, 0) * Rxz(0, 0) * d * p_ref(1) * p_query(1) +
            2 * R_ref(1, 1) * Rxz(0, 1) * d * p_ref(1) * p_query(1) +
            2 * R_ref(1, 2) * Rxz(0, 2) * d * p_ref(1) * p_query(1),
        R_ref(0, 0) * Rxz(1, 0) * t_ref(0) - R_ref(2, 1) * Rxz(1, 1) * d - R_ref(2, 2) * Rxz(1, 2) * d -
            R_ref(2, 0) * Rxz(1, 0) * d + R_ref(0, 1) * Rxz(1, 1) * t_ref(0) + R_ref(0, 2) * Rxz(1, 2) * t_ref(0) +
            R_ref(1, 0) * Rxz(1, 0) * t_ref(1) + R_ref(1, 1) * Rxz(1, 1) * t_ref(1) +
            R_ref(1, 2) * Rxz(1, 2) * t_ref(1) + R_ref(2, 0) * Rxz(1, 0) * t_ref(2) +
            R_ref(2, 1) * Rxz(1, 1) * t_ref(2) + R_ref(2, 2) * Rxz(1, 2) * t_ref(2) -
            R_ref(0, 0) * Rxz(1, 0) * d * p_ref(0) - R_ref(0, 1) * Rxz(1, 1) * d * p_ref(0) -
            R_ref(0, 2) * Rxz(1, 2) * d * p_ref(0) - R_ref(1, 0) * Rxz(1, 0) * d * p_ref(1) -
            R_ref(1, 1) * Rxz(1, 1) * d * p_ref(1) - R_ref(1, 2) * Rxz(1, 2) * d * p_ref(1) +
            R_ref(2, 0) * Rxz(2, 0) * d * p_query(1) + R_ref(2, 1) * Rxz(2, 1) * d * p_query(1) +
            R_ref(2, 2) * Rxz(2, 2) * d * p_query(1) - R_ref(0, 0) * Rxz(2, 0) * t_ref(0) * p_query(1) -
            R_ref(0, 1) * Rxz(2, 1) * t_ref(0) * p_query(1) - R_ref(0, 2) * Rxz(2, 2) * t_ref(0) * p_query(1) -
            R_ref(1, 0) * Rxz(2, 0) * t_ref(1) * p_query(1) - R_ref(1, 1) * Rxz(2, 1) * t_ref(1) * p_query(1) -
            R_ref(1, 2) * Rxz(2, 2) * t_ref(1) * p_query(1) - R_ref(2, 0) * Rxz(2, 0) * t_ref(2) * p_query(1) -
            R_ref(2, 1) * Rxz(2, 1) * t_ref(2) * p_query(1) - R_ref(2, 2) * Rxz(2, 2) * t_ref(2) * p_query(1) +
            R_ref(0, 0) * Rxz(2, 0) * d * p_ref(0) * p_query(1) + R_ref(0, 1) * Rxz(2, 1) * d * p_ref(0) * p_query(1) +
            R_ref(0, 2) * Rxz(2, 2) * d * p_ref(0) * p_query(1) + R_ref(1, 0) * Rxz(2, 0) * d * p_ref(1) * p_query(1) +
            R_ref(1, 1) * Rxz(2, 1) * d * p_ref(1) * p_query(1) + R_ref(1, 2) * Rxz(2, 2) * d * p_ref(1) * p_query(1),
        0, 0, -c_query * n_ref(2) * q - c_query * n_ref(0) * q * p_ref(0) - c_query * n_ref(1) * q * p_ref(1),
        R_ref(2, 0) * Rxz(0, 0) * c_ref * d * n_ref(0) - R_ref(0, 1) * Rxz(0, 1) * c_ref * d * n_ref(2) -
            R_ref(0, 2) * Rxz(0, 2) * c_ref * d * n_ref(2) - R_ref(0, 0) * Rxz(0, 0) * c_ref * d * n_ref(2) +
            R_ref(2, 1) * Rxz(0, 1) * c_ref * d * n_ref(0) + R_ref(2, 2) * Rxz(0, 2) * c_ref * d * n_ref(0) -
            R_ref(1, 0) * Rxz(0, 0) * d * n_ref(2) * s_ref - R_ref(1, 1) * Rxz(0, 1) * d * n_ref(2) * s_ref -
            R_ref(1, 2) * Rxz(0, 2) * d * n_ref(2) * s_ref + R_ref(2, 0) * Rxz(0, 0) * d * n_ref(1) * s_ref +
            R_ref(2, 1) * Rxz(0, 1) * d * n_ref(1) * s_ref + R_ref(2, 2) * Rxz(0, 2) * d * n_ref(1) * s_ref +
            R_ref(2, 0) * Rxz(2, 0) * c_query * d * n_ref(2) * q +
            R_ref(2, 1) * Rxz(2, 1) * c_query * d * n_ref(2) * q +
            R_ref(2, 2) * Rxz(2, 2) * c_query * d * n_ref(2) * q -
            R_ref(0, 0) * Rxz(0, 0) * c_ref * d * n_ref(1) * p_ref(1) -
            R_ref(0, 1) * Rxz(0, 1) * c_ref * d * n_ref(1) * p_ref(1) -
            R_ref(0, 2) * Rxz(0, 2) * c_ref * d * n_ref(1) * p_ref(1) +
            R_ref(1, 0) * Rxz(0, 0) * c_ref * d * n_ref(0) * p_ref(1) +
            R_ref(1, 1) * Rxz(0, 1) * c_ref * d * n_ref(0) * p_ref(1) +
            R_ref(1, 2) * Rxz(0, 2) * c_ref * d * n_ref(0) * p_ref(1) +
            R_ref(0, 0) * Rxz(2, 0) * c_ref * d * n_ref(2) * p_query(0) +
            R_ref(0, 1) * Rxz(2, 1) * c_ref * d * n_ref(2) * p_query(0) +
            R_ref(0, 2) * Rxz(2, 2) * c_ref * d * n_ref(2) * p_query(0) -
            R_ref(2, 0) * Rxz(2, 0) * c_ref * d * n_ref(0) * p_query(0) -
            R_ref(2, 1) * Rxz(2, 1) * c_ref * d * n_ref(0) * p_query(0) -
            R_ref(2, 2) * Rxz(2, 2) * c_ref * d * n_ref(0) * p_query(0) -
            R_ref(0, 0) * Rxz(2, 0) * c_query * n_ref(2) * q * t_ref(0) -
            R_ref(0, 1) * Rxz(2, 1) * c_query * n_ref(2) * q * t_ref(0) -
            R_ref(0, 2) * Rxz(2, 2) * c_query * n_ref(2) * q * t_ref(0) -
            R_ref(1, 0) * Rxz(2, 0) * c_query * n_ref(2) * q * t_ref(1) -
            R_ref(1, 1) * Rxz(2, 1) * c_query * n_ref(2) * q * t_ref(1) -
            R_ref(1, 2) * Rxz(2, 2) * c_query * n_ref(2) * q * t_ref(1) -
            R_ref(2, 0) * Rxz(2, 0) * c_query * n_ref(2) * q * t_ref(2) -
            R_ref(2, 1) * Rxz(2, 1) * c_query * n_ref(2) * q * t_ref(2) -
            R_ref(2, 2) * Rxz(2, 2) * c_query * n_ref(2) * q * t_ref(2) +
            R_ref(0, 0) * Rxz(0, 0) * d * n_ref(1) * s_ref * p_ref(0) +
            R_ref(0, 1) * Rxz(0, 1) * d * n_ref(1) * s_ref * p_ref(0) +
            R_ref(0, 2) * Rxz(0, 2) * d * n_ref(1) * s_ref * p_ref(0) -
            R_ref(1, 0) * Rxz(0, 0) * d * n_ref(0) * s_ref * p_ref(0) -
            R_ref(1, 1) * Rxz(0, 1) * d * n_ref(0) * s_ref * p_ref(0) -
            R_ref(1, 2) * Rxz(0, 2) * d * n_ref(0) * s_ref * p_ref(0) +
            R_ref(1, 0) * Rxz(2, 0) * d * n_ref(2) * s_ref * p_query(0) +
            R_ref(1, 1) * Rxz(2, 1) * d * n_ref(2) * s_ref * p_query(0) +
            R_ref(1, 2) * Rxz(2, 2) * d * n_ref(2) * s_ref * p_query(0) -
            R_ref(2, 0) * Rxz(2, 0) * d * n_ref(1) * s_ref * p_query(0) -
            R_ref(2, 1) * Rxz(2, 1) * d * n_ref(1) * s_ref * p_query(0) -
            R_ref(2, 2) * Rxz(2, 2) * d * n_ref(1) * s_ref * p_query(0) +
            R_ref(0, 0) * Rxz(2, 0) * c_query * d * n_ref(2) * q * p_ref(0) +
            R_ref(0, 1) * Rxz(2, 1) * c_query * d * n_ref(2) * q * p_ref(0) +
            R_ref(0, 2) * Rxz(2, 2) * c_query * d * n_ref(2) * q * p_ref(0) +
            R_ref(1, 0) * Rxz(2, 0) * c_query * d * n_ref(2) * q * p_ref(1) +
            R_ref(1, 1) * Rxz(2, 1) * c_query * d * n_ref(2) * q * p_ref(1) +
            R_ref(1, 2) * Rxz(2, 2) * c_query * d * n_ref(2) * q * p_ref(1) +
            R_ref(2, 0) * Rxz(2, 0) * c_query * d * n_ref(0) * q * p_ref(0) +
            R_ref(2, 0) * Rxz(2, 0) * c_query * d * n_ref(1) * q * p_ref(1) +
            R_ref(2, 1) * Rxz(2, 1) * c_query * d * n_ref(0) * q * p_ref(0) +
            R_ref(2, 1) * Rxz(2, 1) * c_query * d * n_ref(1) * q * p_ref(1) +
            R_ref(2, 2) * Rxz(2, 2) * c_query * d * n_ref(0) * q * p_ref(0) +
            R_ref(2, 2) * Rxz(2, 2) * c_query * d * n_ref(1) * q * p_ref(1) +
            R_ref(0, 0) * Rxz(2, 0) * c_ref * d * n_ref(1) * p_ref(1) * p_query(0) +
            R_ref(0, 1) * Rxz(2, 1) * c_ref * d * n_ref(1) * p_ref(1) * p_query(0) +
            R_ref(0, 2) * Rxz(2, 2) * c_ref * d * n_ref(1) * p_ref(1) * p_query(0) -
            R_ref(1, 0) * Rxz(2, 0) * c_ref * d * n_ref(0) * p_ref(1) * p_query(0) -
            R_ref(1, 1) * Rxz(2, 1) * c_ref * d * n_ref(0) * p_ref(1) * p_query(0) -
            R_ref(1, 2) * Rxz(2, 2) * c_ref * d * n_ref(0) * p_ref(1) * p_query(0) -
            R_ref(0, 0) * Rxz(2, 0) * c_query * n_ref(0) * q * t_ref(0) * p_ref(0) -
            R_ref(0, 0) * Rxz(2, 0) * c_query * n_ref(1) * q * t_ref(0) * p_ref(1) -
            R_ref(0, 1) * Rxz(2, 1) * c_query * n_ref(0) * q * t_ref(0) * p_ref(0) -
            R_ref(0, 1) * Rxz(2, 1) * c_query * n_ref(1) * q * t_ref(0) * p_ref(1) -
            R_ref(0, 2) * Rxz(2, 2) * c_query * n_ref(0) * q * t_ref(0) * p_ref(0) -
            R_ref(0, 2) * Rxz(2, 2) * c_query * n_ref(1) * q * t_ref(0) * p_ref(1) -
            R_ref(1, 0) * Rxz(2, 0) * c_query * n_ref(0) * q * t_ref(1) * p_ref(0) -
            R_ref(1, 0) * Rxz(2, 0) * c_query * n_ref(1) * q * t_ref(1) * p_ref(1) -
            R_ref(1, 1) * Rxz(2, 1) * c_query * n_ref(0) * q * t_ref(1) * p_ref(0) -
            R_ref(1, 1) * Rxz(2, 1) * c_query * n_ref(1) * q * t_ref(1) * p_ref(1) -
            R_ref(1, 2) * Rxz(2, 2) * c_query * n_ref(0) * q * t_ref(1) * p_ref(0) -
            R_ref(1, 2) * Rxz(2, 2) * c_query * n_ref(1) * q * t_ref(1) * p_ref(1) -
            R_ref(2, 0) * Rxz(2, 0) * c_query * n_ref(0) * q * t_ref(2) * p_ref(0) -
            R_ref(2, 0) * Rxz(2, 0) * c_query * n_ref(1) * q * t_ref(2) * p_ref(1) -
            R_ref(2, 1) * Rxz(2, 1) * c_query * n_ref(0) * q * t_ref(2) * p_ref(0) -
            R_ref(2, 1) * Rxz(2, 1) * c_query * n_ref(1) * q * t_ref(2) * p_ref(1) -
            R_ref(2, 2) * Rxz(2, 2) * c_query * n_ref(0) * q * t_ref(2) * p_ref(0) -
            R_ref(2, 2) * Rxz(2, 2) * c_query * n_ref(1) * q * t_ref(2) * p_ref(1) -
            R_ref(0, 0) * Rxz(2, 0) * d * n_ref(1) * s_ref * p_ref(0) * p_query(0) -
            R_ref(0, 1) * Rxz(2, 1) * d * n_ref(1) * s_ref * p_ref(0) * p_query(0) -
            R_ref(0, 2) * Rxz(2, 2) * d * n_ref(1) * s_ref * p_ref(0) * p_query(0) +
            R_ref(1, 0) * Rxz(2, 0) * d * n_ref(0) * s_ref * p_ref(0) * p_query(0) +
            R_ref(1, 1) * Rxz(2, 1) * d * n_ref(0) * s_ref * p_ref(0) * p_query(0) +
            R_ref(1, 2) * Rxz(2, 2) * d * n_ref(0) * s_ref * p_ref(0) * p_query(0) +
            R_ref(0, 0) * Rxz(2, 0) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(0) +
            R_ref(0, 1) * Rxz(2, 1) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(0) +
            R_ref(0, 2) * Rxz(2, 2) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(0) +
            R_ref(1, 0) * Rxz(2, 0) * c_query * d * n_ref(1) * q * p_ref(1) * p_ref(1) +
            R_ref(1, 1) * Rxz(2, 1) * c_query * d * n_ref(1) * q * p_ref(1) * p_ref(1) +
            R_ref(1, 2) * Rxz(2, 2) * c_query * d * n_ref(1) * q * p_ref(1) * p_ref(1) +
            R_ref(0, 0) * Rxz(2, 0) * c_query * d * n_ref(1) * q * p_ref(0) * p_ref(1) +
            R_ref(0, 1) * Rxz(2, 1) * c_query * d * n_ref(1) * q * p_ref(0) * p_ref(1) +
            R_ref(0, 2) * Rxz(2, 2) * c_query * d * n_ref(1) * q * p_ref(0) * p_ref(1) +
            R_ref(1, 0) * Rxz(2, 0) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(1) +
            R_ref(1, 1) * Rxz(2, 1) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(1) +
            R_ref(1, 2) * Rxz(2, 2) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(1),
        2 * R_ref(2, 0) * Rxz(2, 0) * c_ref * d * n_ref(0) - 2 * R_ref(0, 1) * Rxz(2, 1) * c_ref * d * n_ref(2) -
            2 * R_ref(0, 2) * Rxz(2, 2) * c_ref * d * n_ref(2) - 2 * R_ref(0, 0) * Rxz(2, 0) * c_ref * d * n_ref(2) +
            2 * R_ref(2, 1) * Rxz(2, 1) * c_ref * d * n_ref(0) + 2 * R_ref(2, 2) * Rxz(2, 2) * c_ref * d * n_ref(0) -
            2 * R_ref(1, 0) * Rxz(2, 0) * d * n_ref(2) * s_ref - 2 * R_ref(1, 1) * Rxz(2, 1) * d * n_ref(2) * s_ref -
            2 * R_ref(1, 2) * Rxz(2, 2) * d * n_ref(2) * s_ref + 2 * R_ref(2, 0) * Rxz(2, 0) * d * n_ref(1) * s_ref +
            2 * R_ref(2, 1) * Rxz(2, 1) * d * n_ref(1) * s_ref + 2 * R_ref(2, 2) * Rxz(2, 2) * d * n_ref(1) * s_ref -
            2 * R_ref(2, 0) * Rxz(0, 0) * c_query * d * n_ref(2) * q -
            2 * R_ref(2, 1) * Rxz(0, 1) * c_query * d * n_ref(2) * q -
            2 * R_ref(2, 2) * Rxz(0, 2) * c_query * d * n_ref(2) * q -
            2 * R_ref(0, 0) * Rxz(2, 0) * c_ref * d * n_ref(1) * p_ref(1) -
            2 * R_ref(0, 1) * Rxz(2, 1) * c_ref * d * n_ref(1) * p_ref(1) -
            2 * R_ref(0, 2) * Rxz(2, 2) * c_ref * d * n_ref(1) * p_ref(1) +
            2 * R_ref(1, 0) * Rxz(2, 0) * c_ref * d * n_ref(0) * p_ref(1) +
            2 * R_ref(1, 1) * Rxz(2, 1) * c_ref * d * n_ref(0) * p_ref(1) +
            2 * R_ref(1, 2) * Rxz(2, 2) * c_ref * d * n_ref(0) * p_ref(1) -
            2 * R_ref(0, 0) * Rxz(0, 0) * c_ref * d * n_ref(2) * p_query(0) -
            2 * R_ref(0, 1) * Rxz(0, 1) * c_ref * d * n_ref(2) * p_query(0) -
            2 * R_ref(0, 2) * Rxz(0, 2) * c_ref * d * n_ref(2) * p_query(0) +
            2 * R_ref(2, 0) * Rxz(0, 0) * c_ref * d * n_ref(0) * p_query(0) +
            2 * R_ref(2, 1) * Rxz(0, 1) * c_ref * d * n_ref(0) * p_query(0) +
            2 * R_ref(2, 2) * Rxz(0, 2) * c_ref * d * n_ref(0) * p_query(0) +
            2 * R_ref(0, 0) * Rxz(0, 0) * c_query * n_ref(2) * q * t_ref(0) +
            2 * R_ref(0, 1) * Rxz(0, 1) * c_query * n_ref(2) * q * t_ref(0) +
            2 * R_ref(0, 2) * Rxz(0, 2) * c_query * n_ref(2) * q * t_ref(0) +
            2 * R_ref(1, 0) * Rxz(0, 0) * c_query * n_ref(2) * q * t_ref(1) +
            2 * R_ref(1, 1) * Rxz(0, 1) * c_query * n_ref(2) * q * t_ref(1) +
            2 * R_ref(1, 2) * Rxz(0, 2) * c_query * n_ref(2) * q * t_ref(1) +
            2 * R_ref(2, 0) * Rxz(0, 0) * c_query * n_ref(2) * q * t_ref(2) +
            2 * R_ref(2, 1) * Rxz(0, 1) * c_query * n_ref(2) * q * t_ref(2) +
            2 * R_ref(2, 2) * Rxz(0, 2) * c_query * n_ref(2) * q * t_ref(2) +
            2 * R_ref(0, 0) * Rxz(2, 0) * d * n_ref(1) * s_ref * p_ref(0) +
            2 * R_ref(0, 1) * Rxz(2, 1) * d * n_ref(1) * s_ref * p_ref(0) +
            2 * R_ref(0, 2) * Rxz(2, 2) * d * n_ref(1) * s_ref * p_ref(0) -
            2 * R_ref(1, 0) * Rxz(2, 0) * d * n_ref(0) * s_ref * p_ref(0) -
            2 * R_ref(1, 1) * Rxz(2, 1) * d * n_ref(0) * s_ref * p_ref(0) -
            2 * R_ref(1, 2) * Rxz(2, 2) * d * n_ref(0) * s_ref * p_ref(0) -
            2 * R_ref(1, 0) * Rxz(0, 0) * d * n_ref(2) * s_ref * p_query(0) -
            2 * R_ref(1, 1) * Rxz(0, 1) * d * n_ref(2) * s_ref * p_query(0) -
            2 * R_ref(1, 2) * Rxz(0, 2) * d * n_ref(2) * s_ref * p_query(0) +
            2 * R_ref(2, 0) * Rxz(0, 0) * d * n_ref(1) * s_ref * p_query(0) +
            2 * R_ref(2, 1) * Rxz(0, 1) * d * n_ref(1) * s_ref * p_query(0) +
            2 * R_ref(2, 2) * Rxz(0, 2) * d * n_ref(1) * s_ref * p_query(0) -
            2 * R_ref(0, 0) * Rxz(0, 0) * c_query * d * n_ref(2) * q * p_ref(0) -
            2 * R_ref(0, 1) * Rxz(0, 1) * c_query * d * n_ref(2) * q * p_ref(0) -
            2 * R_ref(0, 2) * Rxz(0, 2) * c_query * d * n_ref(2) * q * p_ref(0) -
            2 * R_ref(1, 0) * Rxz(0, 0) * c_query * d * n_ref(2) * q * p_ref(1) -
            2 * R_ref(1, 1) * Rxz(0, 1) * c_query * d * n_ref(2) * q * p_ref(1) -
            2 * R_ref(1, 2) * Rxz(0, 2) * c_query * d * n_ref(2) * q * p_ref(1) -
            2 * R_ref(2, 0) * Rxz(0, 0) * c_query * d * n_ref(0) * q * p_ref(0) -
            2 * R_ref(2, 0) * Rxz(0, 0) * c_query * d * n_ref(1) * q * p_ref(1) -
            2 * R_ref(2, 1) * Rxz(0, 1) * c_query * d * n_ref(0) * q * p_ref(0) -
            2 * R_ref(2, 1) * Rxz(0, 1) * c_query * d * n_ref(1) * q * p_ref(1) -
            2 * R_ref(2, 2) * Rxz(0, 2) * c_query * d * n_ref(0) * q * p_ref(0) -
            2 * R_ref(2, 2) * Rxz(0, 2) * c_query * d * n_ref(1) * q * p_ref(1) -
            2 * R_ref(0, 0) * Rxz(0, 0) * c_ref * d * n_ref(1) * p_ref(1) * p_query(0) -
            2 * R_ref(0, 1) * Rxz(0, 1) * c_ref * d * n_ref(1) * p_ref(1) * p_query(0) -
            2 * R_ref(0, 2) * Rxz(0, 2) * c_ref * d * n_ref(1) * p_ref(1) * p_query(0) +
            2 * R_ref(1, 0) * Rxz(0, 0) * c_ref * d * n_ref(0) * p_ref(1) * p_query(0) +
            2 * R_ref(1, 1) * Rxz(0, 1) * c_ref * d * n_ref(0) * p_ref(1) * p_query(0) +
            2 * R_ref(1, 2) * Rxz(0, 2) * c_ref * d * n_ref(0) * p_ref(1) * p_query(0) +
            2 * R_ref(0, 0) * Rxz(0, 0) * c_query * n_ref(0) * q * t_ref(0) * p_ref(0) +
            2 * R_ref(0, 0) * Rxz(0, 0) * c_query * n_ref(1) * q * t_ref(0) * p_ref(1) +
            2 * R_ref(0, 1) * Rxz(0, 1) * c_query * n_ref(0) * q * t_ref(0) * p_ref(0) +
            2 * R_ref(0, 1) * Rxz(0, 1) * c_query * n_ref(1) * q * t_ref(0) * p_ref(1) +
            2 * R_ref(0, 2) * Rxz(0, 2) * c_query * n_ref(0) * q * t_ref(0) * p_ref(0) +
            2 * R_ref(0, 2) * Rxz(0, 2) * c_query * n_ref(1) * q * t_ref(0) * p_ref(1) +
            2 * R_ref(1, 0) * Rxz(0, 0) * c_query * n_ref(0) * q * t_ref(1) * p_ref(0) +
            2 * R_ref(1, 0) * Rxz(0, 0) * c_query * n_ref(1) * q * t_ref(1) * p_ref(1) +
            2 * R_ref(1, 1) * Rxz(0, 1) * c_query * n_ref(0) * q * t_ref(1) * p_ref(0) +
            2 * R_ref(1, 1) * Rxz(0, 1) * c_query * n_ref(1) * q * t_ref(1) * p_ref(1) +
            2 * R_ref(1, 2) * Rxz(0, 2) * c_query * n_ref(0) * q * t_ref(1) * p_ref(0) +
            2 * R_ref(1, 2) * Rxz(0, 2) * c_query * n_ref(1) * q * t_ref(1) * p_ref(1) +
            2 * R_ref(2, 0) * Rxz(0, 0) * c_query * n_ref(0) * q * t_ref(2) * p_ref(0) +
            2 * R_ref(2, 0) * Rxz(0, 0) * c_query * n_ref(1) * q * t_ref(2) * p_ref(1) +
            2 * R_ref(2, 1) * Rxz(0, 1) * c_query * n_ref(0) * q * t_ref(2) * p_ref(0) +
            2 * R_ref(2, 1) * Rxz(0, 1) * c_query * n_ref(1) * q * t_ref(2) * p_ref(1) +
            2 * R_ref(2, 2) * Rxz(0, 2) * c_query * n_ref(0) * q * t_ref(2) * p_ref(0) +
            2 * R_ref(2, 2) * Rxz(0, 2) * c_query * n_ref(1) * q * t_ref(2) * p_ref(1) +
            2 * R_ref(0, 0) * Rxz(0, 0) * d * n_ref(1) * s_ref * p_ref(0) * p_query(0) +
            2 * R_ref(0, 1) * Rxz(0, 1) * d * n_ref(1) * s_ref * p_ref(0) * p_query(0) +
            2 * R_ref(0, 2) * Rxz(0, 2) * d * n_ref(1) * s_ref * p_ref(0) * p_query(0) -
            2 * R_ref(1, 0) * Rxz(0, 0) * d * n_ref(0) * s_ref * p_ref(0) * p_query(0) -
            2 * R_ref(1, 1) * Rxz(0, 1) * d * n_ref(0) * s_ref * p_ref(0) * p_query(0) -
            2 * R_ref(1, 2) * Rxz(0, 2) * d * n_ref(0) * s_ref * p_ref(0) * p_query(0) -
            2 * R_ref(0, 0) * Rxz(0, 0) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(0) -
            2 * R_ref(0, 1) * Rxz(0, 1) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(0) -
            2 * R_ref(0, 2) * Rxz(0, 2) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(0) -
            2 * R_ref(1, 0) * Rxz(0, 0) * c_query * d * n_ref(1) * q * p_ref(1) * p_ref(1) -
            2 * R_ref(1, 1) * Rxz(0, 1) * c_query * d * n_ref(1) * q * p_ref(1) * p_ref(1) -
            2 * R_ref(1, 2) * Rxz(0, 2) * c_query * d * n_ref(1) * q * p_ref(1) * p_ref(1) -
            2 * R_ref(0, 0) * Rxz(0, 0) * c_query * d * n_ref(1) * q * p_ref(0) * p_ref(1) -
            2 * R_ref(0, 1) * Rxz(0, 1) * c_query * d * n_ref(1) * q * p_ref(0) * p_ref(1) -
            2 * R_ref(0, 2) * Rxz(0, 2) * c_query * d * n_ref(1) * q * p_ref(0) * p_ref(1) -
            2 * R_ref(1, 0) * Rxz(0, 0) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(1) -
            2 * R_ref(1, 1) * Rxz(0, 1) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(1) -
            2 * R_ref(1, 2) * Rxz(0, 2) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(1),
        R_ref(0, 0) * Rxz(0, 0) * c_ref * d * n_ref(2) + R_ref(0, 1) * Rxz(0, 1) * c_ref * d * n_ref(2) +
            R_ref(0, 2) * Rxz(0, 2) * c_ref * d * n_ref(2) - R_ref(2, 0) * Rxz(0, 0) * c_ref * d * n_ref(0) -
            R_ref(2, 1) * Rxz(0, 1) * c_ref * d * n_ref(0) - R_ref(2, 2) * Rxz(0, 2) * c_ref * d * n_ref(0) +
            R_ref(1, 0) * Rxz(0, 0) * d * n_ref(2) * s_ref + R_ref(1, 1) * Rxz(0, 1) * d * n_ref(2) * s_ref +
            R_ref(1, 2) * Rxz(0, 2) * d * n_ref(2) * s_ref - R_ref(2, 0) * Rxz(0, 0) * d * n_ref(1) * s_ref -
            R_ref(2, 1) * Rxz(0, 1) * d * n_ref(1) * s_ref - R_ref(2, 2) * Rxz(0, 2) * d * n_ref(1) * s_ref -
            R_ref(2, 0) * Rxz(2, 0) * c_query * d * n_ref(2) * q -
            R_ref(2, 1) * Rxz(2, 1) * c_query * d * n_ref(2) * q -
            R_ref(2, 2) * Rxz(2, 2) * c_query * d * n_ref(2) * q +
            R_ref(0, 0) * Rxz(0, 0) * c_ref * d * n_ref(1) * p_ref(1) +
            R_ref(0, 1) * Rxz(0, 1) * c_ref * d * n_ref(1) * p_ref(1) +
            R_ref(0, 2) * Rxz(0, 2) * c_ref * d * n_ref(1) * p_ref(1) -
            R_ref(1, 0) * Rxz(0, 0) * c_ref * d * n_ref(0) * p_ref(1) -
            R_ref(1, 1) * Rxz(0, 1) * c_ref * d * n_ref(0) * p_ref(1) -
            R_ref(1, 2) * Rxz(0, 2) * c_ref * d * n_ref(0) * p_ref(1) -
            R_ref(0, 0) * Rxz(2, 0) * c_ref * d * n_ref(2) * p_query(0) -
            R_ref(0, 1) * Rxz(2, 1) * c_ref * d * n_ref(2) * p_query(0) -
            R_ref(0, 2) * Rxz(2, 2) * c_ref * d * n_ref(2) * p_query(0) +
            R_ref(2, 0) * Rxz(2, 0) * c_ref * d * n_ref(0) * p_query(0) +
            R_ref(2, 1) * Rxz(2, 1) * c_ref * d * n_ref(0) * p_query(0) +
            R_ref(2, 2) * Rxz(2, 2) * c_ref * d * n_ref(0) * p_query(0) +
            R_ref(0, 0) * Rxz(2, 0) * c_query * n_ref(2) * q * t_ref(0) +
            R_ref(0, 1) * Rxz(2, 1) * c_query * n_ref(2) * q * t_ref(0) +
            R_ref(0, 2) * Rxz(2, 2) * c_query * n_ref(2) * q * t_ref(0) +
            R_ref(1, 0) * Rxz(2, 0) * c_query * n_ref(2) * q * t_ref(1) +
            R_ref(1, 1) * Rxz(2, 1) * c_query * n_ref(2) * q * t_ref(1) +
            R_ref(1, 2) * Rxz(2, 2) * c_query * n_ref(2) * q * t_ref(1) +
            R_ref(2, 0) * Rxz(2, 0) * c_query * n_ref(2) * q * t_ref(2) +
            R_ref(2, 1) * Rxz(2, 1) * c_query * n_ref(2) * q * t_ref(2) +
            R_ref(2, 2) * Rxz(2, 2) * c_query * n_ref(2) * q * t_ref(2) -
            R_ref(0, 0) * Rxz(0, 0) * d * n_ref(1) * s_ref * p_ref(0) -
            R_ref(0, 1) * Rxz(0, 1) * d * n_ref(1) * s_ref * p_ref(0) -
            R_ref(0, 2) * Rxz(0, 2) * d * n_ref(1) * s_ref * p_ref(0) +
            R_ref(1, 0) * Rxz(0, 0) * d * n_ref(0) * s_ref * p_ref(0) +
            R_ref(1, 1) * Rxz(0, 1) * d * n_ref(0) * s_ref * p_ref(0) +
            R_ref(1, 2) * Rxz(0, 2) * d * n_ref(0) * s_ref * p_ref(0) -
            R_ref(1, 0) * Rxz(2, 0) * d * n_ref(2) * s_ref * p_query(0) -
            R_ref(1, 1) * Rxz(2, 1) * d * n_ref(2) * s_ref * p_query(0) -
            R_ref(1, 2) * Rxz(2, 2) * d * n_ref(2) * s_ref * p_query(0) +
            R_ref(2, 0) * Rxz(2, 0) * d * n_ref(1) * s_ref * p_query(0) +
            R_ref(2, 1) * Rxz(2, 1) * d * n_ref(1) * s_ref * p_query(0) +
            R_ref(2, 2) * Rxz(2, 2) * d * n_ref(1) * s_ref * p_query(0) -
            R_ref(0, 0) * Rxz(2, 0) * c_query * d * n_ref(2) * q * p_ref(0) -
            R_ref(0, 1) * Rxz(2, 1) * c_query * d * n_ref(2) * q * p_ref(0) -
            R_ref(0, 2) * Rxz(2, 2) * c_query * d * n_ref(2) * q * p_ref(0) -
            R_ref(1, 0) * Rxz(2, 0) * c_query * d * n_ref(2) * q * p_ref(1) -
            R_ref(1, 1) * Rxz(2, 1) * c_query * d * n_ref(2) * q * p_ref(1) -
            R_ref(1, 2) * Rxz(2, 2) * c_query * d * n_ref(2) * q * p_ref(1) -
            R_ref(2, 0) * Rxz(2, 0) * c_query * d * n_ref(0) * q * p_ref(0) -
            R_ref(2, 0) * Rxz(2, 0) * c_query * d * n_ref(1) * q * p_ref(1) -
            R_ref(2, 1) * Rxz(2, 1) * c_query * d * n_ref(0) * q * p_ref(0) -
            R_ref(2, 1) * Rxz(2, 1) * c_query * d * n_ref(1) * q * p_ref(1) -
            R_ref(2, 2) * Rxz(2, 2) * c_query * d * n_ref(0) * q * p_ref(0) -
            R_ref(2, 2) * Rxz(2, 2) * c_query * d * n_ref(1) * q * p_ref(1) -
            R_ref(0, 0) * Rxz(2, 0) * c_ref * d * n_ref(1) * p_ref(1) * p_query(0) -
            R_ref(0, 1) * Rxz(2, 1) * c_ref * d * n_ref(1) * p_ref(1) * p_query(0) -
            R_ref(0, 2) * Rxz(2, 2) * c_ref * d * n_ref(1) * p_ref(1) * p_query(0) +
            R_ref(1, 0) * Rxz(2, 0) * c_ref * d * n_ref(0) * p_ref(1) * p_query(0) +
            R_ref(1, 1) * Rxz(2, 1) * c_ref * d * n_ref(0) * p_ref(1) * p_query(0) +
            R_ref(1, 2) * Rxz(2, 2) * c_ref * d * n_ref(0) * p_ref(1) * p_query(0) +
            R_ref(0, 0) * Rxz(2, 0) * c_query * n_ref(0) * q * t_ref(0) * p_ref(0) +
            R_ref(0, 0) * Rxz(2, 0) * c_query * n_ref(1) * q * t_ref(0) * p_ref(1) +
            R_ref(0, 1) * Rxz(2, 1) * c_query * n_ref(0) * q * t_ref(0) * p_ref(0) +
            R_ref(0, 1) * Rxz(2, 1) * c_query * n_ref(1) * q * t_ref(0) * p_ref(1) +
            R_ref(0, 2) * Rxz(2, 2) * c_query * n_ref(0) * q * t_ref(0) * p_ref(0) +
            R_ref(0, 2) * Rxz(2, 2) * c_query * n_ref(1) * q * t_ref(0) * p_ref(1) +
            R_ref(1, 0) * Rxz(2, 0) * c_query * n_ref(0) * q * t_ref(1) * p_ref(0) +
            R_ref(1, 0) * Rxz(2, 0) * c_query * n_ref(1) * q * t_ref(1) * p_ref(1) +
            R_ref(1, 1) * Rxz(2, 1) * c_query * n_ref(0) * q * t_ref(1) * p_ref(0) +
            R_ref(1, 1) * Rxz(2, 1) * c_query * n_ref(1) * q * t_ref(1) * p_ref(1) +
            R_ref(1, 2) * Rxz(2, 2) * c_query * n_ref(0) * q * t_ref(1) * p_ref(0) +
            R_ref(1, 2) * Rxz(2, 2) * c_query * n_ref(1) * q * t_ref(1) * p_ref(1) +
            R_ref(2, 0) * Rxz(2, 0) * c_query * n_ref(0) * q * t_ref(2) * p_ref(0) +
            R_ref(2, 0) * Rxz(2, 0) * c_query * n_ref(1) * q * t_ref(2) * p_ref(1) +
            R_ref(2, 1) * Rxz(2, 1) * c_query * n_ref(0) * q * t_ref(2) * p_ref(0) +
            R_ref(2, 1) * Rxz(2, 1) * c_query * n_ref(1) * q * t_ref(2) * p_ref(1) +
            R_ref(2, 2) * Rxz(2, 2) * c_query * n_ref(0) * q * t_ref(2) * p_ref(0) +
            R_ref(2, 2) * Rxz(2, 2) * c_query * n_ref(1) * q * t_ref(2) * p_ref(1) +
            R_ref(0, 0) * Rxz(2, 0) * d * n_ref(1) * s_ref * p_ref(0) * p_query(0) +
            R_ref(0, 1) * Rxz(2, 1) * d * n_ref(1) * s_ref * p_ref(0) * p_query(0) +
            R_ref(0, 2) * Rxz(2, 2) * d * n_ref(1) * s_ref * p_ref(0) * p_query(0) -
            R_ref(1, 0) * Rxz(2, 0) * d * n_ref(0) * s_ref * p_ref(0) * p_query(0) -
            R_ref(1, 1) * Rxz(2, 1) * d * n_ref(0) * s_ref * p_ref(0) * p_query(0) -
            R_ref(1, 2) * Rxz(2, 2) * d * n_ref(0) * s_ref * p_ref(0) * p_query(0) -
            R_ref(0, 0) * Rxz(2, 0) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(0) -
            R_ref(0, 1) * Rxz(2, 1) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(0) -
            R_ref(0, 2) * Rxz(2, 2) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(0) -
            R_ref(1, 0) * Rxz(2, 0) * c_query * d * n_ref(1) * q * p_ref(1) * p_ref(1) -
            R_ref(1, 1) * Rxz(2, 1) * c_query * d * n_ref(1) * q * p_ref(1) * p_ref(1) -
            R_ref(1, 2) * Rxz(2, 2) * c_query * d * n_ref(1) * q * p_ref(1) * p_ref(1) -
            R_ref(0, 0) * Rxz(2, 0) * c_query * d * n_ref(1) * q * p_ref(0) * p_ref(1) -
            R_ref(0, 1) * Rxz(2, 1) * c_query * d * n_ref(1) * q * p_ref(0) * p_ref(1) -
            R_ref(0, 2) * Rxz(2, 2) * c_query * d * n_ref(1) * q * p_ref(0) * p_ref(1) -
            R_ref(1, 0) * Rxz(2, 0) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(1) -
            R_ref(1, 1) * Rxz(2, 1) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(1) -
            R_ref(1, 2) * Rxz(2, 2) * c_query * d * n_ref(0) * q * p_ref(0) * p_ref(1),
        0, 0, -n_ref(2) * q * s_query - n_ref(0) * q * s_query * p_ref(0) - n_ref(1) * q * s_query * p_ref(1),
        R_ref(0, 0) * Rxz(1, 0) * c_ref * d * n_ref(2) + R_ref(0, 1) * Rxz(1, 1) * c_ref * d * n_ref(2) +
            R_ref(0, 2) * Rxz(1, 2) * c_ref * d * n_ref(2) - R_ref(2, 0) * Rxz(1, 0) * c_ref * d * n_ref(0) -
            R_ref(2, 1) * Rxz(1, 1) * c_ref * d * n_ref(0) - R_ref(2, 2) * Rxz(1, 2) * c_ref * d * n_ref(0) +
            R_ref(1, 0) * Rxz(1, 0) * d * n_ref(2) * s_ref + R_ref(1, 1) * Rxz(1, 1) * d * n_ref(2) * s_ref +
            R_ref(1, 2) * Rxz(1, 2) * d * n_ref(2) * s_ref - R_ref(2, 0) * Rxz(1, 0) * d * n_ref(1) * s_ref -
            R_ref(2, 1) * Rxz(1, 1) * d * n_ref(1) * s_ref - R_ref(2, 2) * Rxz(1, 2) * d * n_ref(1) * s_ref +
            R_ref(0, 0) * Rxz(1, 0) * c_ref * d * n_ref(1) * p_ref(1) +
            R_ref(0, 1) * Rxz(1, 1) * c_ref * d * n_ref(1) * p_ref(1) +
            R_ref(0, 2) * Rxz(1, 2) * c_ref * d * n_ref(1) * p_ref(1) -
            R_ref(1, 0) * Rxz(1, 0) * c_ref * d * n_ref(0) * p_ref(1) -
            R_ref(1, 1) * Rxz(1, 1) * c_ref * d * n_ref(0) * p_ref(1) -
            R_ref(1, 2) * Rxz(1, 2) * c_ref * d * n_ref(0) * p_ref(1) +
            R_ref(0, 0) * Rxz(2, 0) * c_ref * d * n_ref(2) * p_query(1) +
            R_ref(0, 1) * Rxz(2, 1) * c_ref * d * n_ref(2) * p_query(1) +
            R_ref(0, 2) * Rxz(2, 2) * c_ref * d * n_ref(2) * p_query(1) -
            R_ref(2, 0) * Rxz(2, 0) * c_ref * d * n_ref(0) * p_query(1) -
            R_ref(2, 1) * Rxz(2, 1) * c_ref * d * n_ref(0) * p_query(1) -
            R_ref(2, 2) * Rxz(2, 2) * c_ref * d * n_ref(0) * p_query(1) +
            R_ref(2, 0) * Rxz(2, 0) * d * n_ref(2) * q * s_query +
            R_ref(2, 1) * Rxz(2, 1) * d * n_ref(2) * q * s_query +
            R_ref(2, 2) * Rxz(2, 2) * d * n_ref(2) * q * s_query -
            R_ref(0, 0) * Rxz(1, 0) * d * n_ref(1) * s_ref * p_ref(0) -
            R_ref(0, 1) * Rxz(1, 1) * d * n_ref(1) * s_ref * p_ref(0) -
            R_ref(0, 2) * Rxz(1, 2) * d * n_ref(1) * s_ref * p_ref(0) +
            R_ref(1, 0) * Rxz(1, 0) * d * n_ref(0) * s_ref * p_ref(0) +
            R_ref(1, 1) * Rxz(1, 1) * d * n_ref(0) * s_ref * p_ref(0) +
            R_ref(1, 2) * Rxz(1, 2) * d * n_ref(0) * s_ref * p_ref(0) +
            R_ref(1, 0) * Rxz(2, 0) * d * n_ref(2) * s_ref * p_query(1) +
            R_ref(1, 1) * Rxz(2, 1) * d * n_ref(2) * s_ref * p_query(1) +
            R_ref(1, 2) * Rxz(2, 2) * d * n_ref(2) * s_ref * p_query(1) -
            R_ref(2, 0) * Rxz(2, 0) * d * n_ref(1) * s_ref * p_query(1) -
            R_ref(2, 1) * Rxz(2, 1) * d * n_ref(1) * s_ref * p_query(1) -
            R_ref(2, 2) * Rxz(2, 2) * d * n_ref(1) * s_ref * p_query(1) -
            R_ref(0, 0) * Rxz(2, 0) * n_ref(2) * q * s_query * t_ref(0) -
            R_ref(0, 1) * Rxz(2, 1) * n_ref(2) * q * s_query * t_ref(0) -
            R_ref(0, 2) * Rxz(2, 2) * n_ref(2) * q * s_query * t_ref(0) -
            R_ref(1, 0) * Rxz(2, 0) * n_ref(2) * q * s_query * t_ref(1) -
            R_ref(1, 1) * Rxz(2, 1) * n_ref(2) * q * s_query * t_ref(1) -
            R_ref(1, 2) * Rxz(2, 2) * n_ref(2) * q * s_query * t_ref(1) -
            R_ref(2, 0) * Rxz(2, 0) * n_ref(2) * q * s_query * t_ref(2) -
            R_ref(2, 1) * Rxz(2, 1) * n_ref(2) * q * s_query * t_ref(2) -
            R_ref(2, 2) * Rxz(2, 2) * n_ref(2) * q * s_query * t_ref(2) +
            R_ref(0, 0) * Rxz(2, 0) * c_ref * d * n_ref(1) * p_ref(1) * p_query(1) +
            R_ref(0, 1) * Rxz(2, 1) * c_ref * d * n_ref(1) * p_ref(1) * p_query(1) +
            R_ref(0, 2) * Rxz(2, 2) * c_ref * d * n_ref(1) * p_ref(1) * p_query(1) -
            R_ref(1, 0) * Rxz(2, 0) * c_ref * d * n_ref(0) * p_ref(1) * p_query(1) -
            R_ref(1, 1) * Rxz(2, 1) * c_ref * d * n_ref(0) * p_ref(1) * p_query(1) -
            R_ref(1, 2) * Rxz(2, 2) * c_ref * d * n_ref(0) * p_ref(1) * p_query(1) +
            R_ref(0, 0) * Rxz(2, 0) * d * n_ref(2) * q * s_query * p_ref(0) +
            R_ref(0, 1) * Rxz(2, 1) * d * n_ref(2) * q * s_query * p_ref(0) +
            R_ref(0, 2) * Rxz(2, 2) * d * n_ref(2) * q * s_query * p_ref(0) +
            R_ref(1, 0) * Rxz(2, 0) * d * n_ref(2) * q * s_query * p_ref(1) +
            R_ref(1, 1) * Rxz(2, 1) * d * n_ref(2) * q * s_query * p_ref(1) +
            R_ref(1, 2) * Rxz(2, 2) * d * n_ref(2) * q * s_query * p_ref(1) +
            R_ref(2, 0) * Rxz(2, 0) * d * n_ref(0) * q * s_query * p_ref(0) +
            R_ref(2, 0) * Rxz(2, 0) * d * n_ref(1) * q * s_query * p_ref(1) +
            R_ref(2, 1) * Rxz(2, 1) * d * n_ref(0) * q * s_query * p_ref(0) +
            R_ref(2, 1) * Rxz(2, 1) * d * n_ref(1) * q * s_query * p_ref(1) +
            R_ref(2, 2) * Rxz(2, 2) * d * n_ref(0) * q * s_query * p_ref(0) +
            R_ref(2, 2) * Rxz(2, 2) * d * n_ref(1) * q * s_query * p_ref(1) -
            R_ref(0, 0) * Rxz(2, 0) * d * n_ref(1) * s_ref * p_ref(0) * p_query(1) -
            R_ref(0, 1) * Rxz(2, 1) * d * n_ref(1) * s_ref * p_ref(0) * p_query(1) -
            R_ref(0, 2) * Rxz(2, 2) * d * n_ref(1) * s_ref * p_ref(0) * p_query(1) +
            R_ref(1, 0) * Rxz(2, 0) * d * n_ref(0) * s_ref * p_ref(0) * p_query(1) +
            R_ref(1, 1) * Rxz(2, 1) * d * n_ref(0) * s_ref * p_ref(0) * p_query(1) +
            R_ref(1, 2) * Rxz(2, 2) * d * n_ref(0) * s_ref * p_ref(0) * p_query(1) -
            R_ref(0, 0) * Rxz(2, 0) * n_ref(0) * q * s_query * t_ref(0) * p_ref(0) -
            R_ref(0, 0) * Rxz(2, 0) * n_ref(1) * q * s_query * t_ref(0) * p_ref(1) -
            R_ref(0, 1) * Rxz(2, 1) * n_ref(0) * q * s_query * t_ref(0) * p_ref(0) -
            R_ref(0, 1) * Rxz(2, 1) * n_ref(1) * q * s_query * t_ref(0) * p_ref(1) -
            R_ref(0, 2) * Rxz(2, 2) * n_ref(0) * q * s_query * t_ref(0) * p_ref(0) -
            R_ref(0, 2) * Rxz(2, 2) * n_ref(1) * q * s_query * t_ref(0) * p_ref(1) -
            R_ref(1, 0) * Rxz(2, 0) * n_ref(0) * q * s_query * t_ref(1) * p_ref(0) -
            R_ref(1, 0) * Rxz(2, 0) * n_ref(1) * q * s_query * t_ref(1) * p_ref(1) -
            R_ref(1, 1) * Rxz(2, 1) * n_ref(0) * q * s_query * t_ref(1) * p_ref(0) -
            R_ref(1, 1) * Rxz(2, 1) * n_ref(1) * q * s_query * t_ref(1) * p_ref(1) -
            R_ref(1, 2) * Rxz(2, 2) * n_ref(0) * q * s_query * t_ref(1) * p_ref(0) -
            R_ref(1, 2) * Rxz(2, 2) * n_ref(1) * q * s_query * t_ref(1) * p_ref(1) -
            R_ref(2, 0) * Rxz(2, 0) * n_ref(0) * q * s_query * t_ref(2) * p_ref(0) -
            R_ref(2, 0) * Rxz(2, 0) * n_ref(1) * q * s_query * t_ref(2) * p_ref(1) -
            R_ref(2, 1) * Rxz(2, 1) * n_ref(0) * q * s_query * t_ref(2) * p_ref(0) -
            R_ref(2, 1) * Rxz(2, 1) * n_ref(1) * q * s_query * t_ref(2) * p_ref(1) -
            R_ref(2, 2) * Rxz(2, 2) * n_ref(0) * q * s_query * t_ref(2) * p_ref(0) -
            R_ref(2, 2) * Rxz(2, 2) * n_ref(1) * q * s_query * t_ref(2) * p_ref(1) +
            R_ref(0, 0) * Rxz(2, 0) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(0) +
            R_ref(0, 1) * Rxz(2, 1) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(0) +
            R_ref(0, 2) * Rxz(2, 2) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(0) +
            R_ref(1, 0) * Rxz(2, 0) * d * n_ref(1) * q * s_query * p_ref(1) * p_ref(1) +
            R_ref(1, 1) * Rxz(2, 1) * d * n_ref(1) * q * s_query * p_ref(1) * p_ref(1) +
            R_ref(1, 2) * Rxz(2, 2) * d * n_ref(1) * q * s_query * p_ref(1) * p_ref(1) +
            R_ref(0, 0) * Rxz(2, 0) * d * n_ref(1) * q * s_query * p_ref(0) * p_ref(1) +
            R_ref(0, 1) * Rxz(2, 1) * d * n_ref(1) * q * s_query * p_ref(0) * p_ref(1) +
            R_ref(0, 2) * Rxz(2, 2) * d * n_ref(1) * q * s_query * p_ref(0) * p_ref(1) +
            R_ref(1, 0) * Rxz(2, 0) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(1) +
            R_ref(1, 1) * Rxz(2, 1) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(1) +
            R_ref(1, 2) * Rxz(2, 2) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(1),
        2 * R_ref(2, 0) * Rxz(0, 0) * c_ref * d * n_ref(0) * p_query(1) -
            2 * R_ref(0, 1) * Rxz(0, 1) * c_ref * d * n_ref(2) * p_query(1) -
            2 * R_ref(0, 2) * Rxz(0, 2) * c_ref * d * n_ref(2) * p_query(1) -
            2 * R_ref(0, 0) * Rxz(0, 0) * c_ref * d * n_ref(2) * p_query(1) +
            2 * R_ref(2, 1) * Rxz(0, 1) * c_ref * d * n_ref(0) * p_query(1) +
            2 * R_ref(2, 2) * Rxz(0, 2) * c_ref * d * n_ref(0) * p_query(1) -
            2 * R_ref(2, 0) * Rxz(0, 0) * d * n_ref(2) * q * s_query -
            2 * R_ref(2, 1) * Rxz(0, 1) * d * n_ref(2) * q * s_query -
            2 * R_ref(2, 2) * Rxz(0, 2) * d * n_ref(2) * q * s_query -
            2 * R_ref(1, 0) * Rxz(0, 0) * d * n_ref(2) * s_ref * p_query(1) -
            2 * R_ref(1, 1) * Rxz(0, 1) * d * n_ref(2) * s_ref * p_query(1) -
            2 * R_ref(1, 2) * Rxz(0, 2) * d * n_ref(2) * s_ref * p_query(1) +
            2 * R_ref(2, 0) * Rxz(0, 0) * d * n_ref(1) * s_ref * p_query(1) +
            2 * R_ref(2, 1) * Rxz(0, 1) * d * n_ref(1) * s_ref * p_query(1) +
            2 * R_ref(2, 2) * Rxz(0, 2) * d * n_ref(1) * s_ref * p_query(1) +
            2 * R_ref(0, 0) * Rxz(0, 0) * n_ref(2) * q * s_query * t_ref(0) +
            2 * R_ref(0, 1) * Rxz(0, 1) * n_ref(2) * q * s_query * t_ref(0) +
            2 * R_ref(0, 2) * Rxz(0, 2) * n_ref(2) * q * s_query * t_ref(0) +
            2 * R_ref(1, 0) * Rxz(0, 0) * n_ref(2) * q * s_query * t_ref(1) +
            2 * R_ref(1, 1) * Rxz(0, 1) * n_ref(2) * q * s_query * t_ref(1) +
            2 * R_ref(1, 2) * Rxz(0, 2) * n_ref(2) * q * s_query * t_ref(1) +
            2 * R_ref(2, 0) * Rxz(0, 0) * n_ref(2) * q * s_query * t_ref(2) +
            2 * R_ref(2, 1) * Rxz(0, 1) * n_ref(2) * q * s_query * t_ref(2) +
            2 * R_ref(2, 2) * Rxz(0, 2) * n_ref(2) * q * s_query * t_ref(2) -
            2 * R_ref(0, 0) * Rxz(0, 0) * c_ref * d * n_ref(1) * p_ref(1) * p_query(1) -
            2 * R_ref(0, 1) * Rxz(0, 1) * c_ref * d * n_ref(1) * p_ref(1) * p_query(1) -
            2 * R_ref(0, 2) * Rxz(0, 2) * c_ref * d * n_ref(1) * p_ref(1) * p_query(1) +
            2 * R_ref(1, 0) * Rxz(0, 0) * c_ref * d * n_ref(0) * p_ref(1) * p_query(1) +
            2 * R_ref(1, 1) * Rxz(0, 1) * c_ref * d * n_ref(0) * p_ref(1) * p_query(1) +
            2 * R_ref(1, 2) * Rxz(0, 2) * c_ref * d * n_ref(0) * p_ref(1) * p_query(1) -
            2 * R_ref(0, 0) * Rxz(0, 0) * d * n_ref(2) * q * s_query * p_ref(0) -
            2 * R_ref(0, 1) * Rxz(0, 1) * d * n_ref(2) * q * s_query * p_ref(0) -
            2 * R_ref(0, 2) * Rxz(0, 2) * d * n_ref(2) * q * s_query * p_ref(0) -
            2 * R_ref(1, 0) * Rxz(0, 0) * d * n_ref(2) * q * s_query * p_ref(1) -
            2 * R_ref(1, 1) * Rxz(0, 1) * d * n_ref(2) * q * s_query * p_ref(1) -
            2 * R_ref(1, 2) * Rxz(0, 2) * d * n_ref(2) * q * s_query * p_ref(1) -
            2 * R_ref(2, 0) * Rxz(0, 0) * d * n_ref(0) * q * s_query * p_ref(0) -
            2 * R_ref(2, 0) * Rxz(0, 0) * d * n_ref(1) * q * s_query * p_ref(1) -
            2 * R_ref(2, 1) * Rxz(0, 1) * d * n_ref(0) * q * s_query * p_ref(0) -
            2 * R_ref(2, 1) * Rxz(0, 1) * d * n_ref(1) * q * s_query * p_ref(1) -
            2 * R_ref(2, 2) * Rxz(0, 2) * d * n_ref(0) * q * s_query * p_ref(0) -
            2 * R_ref(2, 2) * Rxz(0, 2) * d * n_ref(1) * q * s_query * p_ref(1) +
            2 * R_ref(0, 0) * Rxz(0, 0) * d * n_ref(1) * s_ref * p_ref(0) * p_query(1) +
            2 * R_ref(0, 1) * Rxz(0, 1) * d * n_ref(1) * s_ref * p_ref(0) * p_query(1) +
            2 * R_ref(0, 2) * Rxz(0, 2) * d * n_ref(1) * s_ref * p_ref(0) * p_query(1) -
            2 * R_ref(1, 0) * Rxz(0, 0) * d * n_ref(0) * s_ref * p_ref(0) * p_query(1) -
            2 * R_ref(1, 1) * Rxz(0, 1) * d * n_ref(0) * s_ref * p_ref(0) * p_query(1) -
            2 * R_ref(1, 2) * Rxz(0, 2) * d * n_ref(0) * s_ref * p_ref(0) * p_query(1) +
            2 * R_ref(0, 0) * Rxz(0, 0) * n_ref(0) * q * s_query * t_ref(0) * p_ref(0) +
            2 * R_ref(0, 0) * Rxz(0, 0) * n_ref(1) * q * s_query * t_ref(0) * p_ref(1) +
            2 * R_ref(0, 1) * Rxz(0, 1) * n_ref(0) * q * s_query * t_ref(0) * p_ref(0) +
            2 * R_ref(0, 1) * Rxz(0, 1) * n_ref(1) * q * s_query * t_ref(0) * p_ref(1) +
            2 * R_ref(0, 2) * Rxz(0, 2) * n_ref(0) * q * s_query * t_ref(0) * p_ref(0) +
            2 * R_ref(0, 2) * Rxz(0, 2) * n_ref(1) * q * s_query * t_ref(0) * p_ref(1) +
            2 * R_ref(1, 0) * Rxz(0, 0) * n_ref(0) * q * s_query * t_ref(1) * p_ref(0) +
            2 * R_ref(1, 0) * Rxz(0, 0) * n_ref(1) * q * s_query * t_ref(1) * p_ref(1) +
            2 * R_ref(1, 1) * Rxz(0, 1) * n_ref(0) * q * s_query * t_ref(1) * p_ref(0) +
            2 * R_ref(1, 1) * Rxz(0, 1) * n_ref(1) * q * s_query * t_ref(1) * p_ref(1) +
            2 * R_ref(1, 2) * Rxz(0, 2) * n_ref(0) * q * s_query * t_ref(1) * p_ref(0) +
            2 * R_ref(1, 2) * Rxz(0, 2) * n_ref(1) * q * s_query * t_ref(1) * p_ref(1) +
            2 * R_ref(2, 0) * Rxz(0, 0) * n_ref(0) * q * s_query * t_ref(2) * p_ref(0) +
            2 * R_ref(2, 0) * Rxz(0, 0) * n_ref(1) * q * s_query * t_ref(2) * p_ref(1) +
            2 * R_ref(2, 1) * Rxz(0, 1) * n_ref(0) * q * s_query * t_ref(2) * p_ref(0) +
            2 * R_ref(2, 1) * Rxz(0, 1) * n_ref(1) * q * s_query * t_ref(2) * p_ref(1) +
            2 * R_ref(2, 2) * Rxz(0, 2) * n_ref(0) * q * s_query * t_ref(2) * p_ref(0) +
            2 * R_ref(2, 2) * Rxz(0, 2) * n_ref(1) * q * s_query * t_ref(2) * p_ref(1) -
            2 * R_ref(0, 0) * Rxz(0, 0) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(0) -
            2 * R_ref(0, 1) * Rxz(0, 1) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(0) -
            2 * R_ref(0, 2) * Rxz(0, 2) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(0) -
            2 * R_ref(1, 0) * Rxz(0, 0) * d * n_ref(1) * q * s_query * p_ref(1) * p_ref(1) -
            2 * R_ref(1, 1) * Rxz(0, 1) * d * n_ref(1) * q * s_query * p_ref(1) * p_ref(1) -
            2 * R_ref(1, 2) * Rxz(0, 2) * d * n_ref(1) * q * s_query * p_ref(1) * p_ref(1) -
            2 * R_ref(0, 0) * Rxz(0, 0) * d * n_ref(1) * q * s_query * p_ref(0) * p_ref(1) -
            2 * R_ref(0, 1) * Rxz(0, 1) * d * n_ref(1) * q * s_query * p_ref(0) * p_ref(1) -
            2 * R_ref(0, 2) * Rxz(0, 2) * d * n_ref(1) * q * s_query * p_ref(0) * p_ref(1) -
            2 * R_ref(1, 0) * Rxz(0, 0) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(1) -
            2 * R_ref(1, 1) * Rxz(0, 1) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(1) -
            2 * R_ref(1, 2) * Rxz(0, 2) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(1),
        R_ref(0, 0) * Rxz(1, 0) * c_ref * d * n_ref(2) + R_ref(0, 1) * Rxz(1, 1) * c_ref * d * n_ref(2) +
            R_ref(0, 2) * Rxz(1, 2) * c_ref * d * n_ref(2) - R_ref(2, 0) * Rxz(1, 0) * c_ref * d * n_ref(0) -
            R_ref(2, 1) * Rxz(1, 1) * c_ref * d * n_ref(0) - R_ref(2, 2) * Rxz(1, 2) * c_ref * d * n_ref(0) +
            R_ref(1, 0) * Rxz(1, 0) * d * n_ref(2) * s_ref + R_ref(1, 1) * Rxz(1, 1) * d * n_ref(2) * s_ref +
            R_ref(1, 2) * Rxz(1, 2) * d * n_ref(2) * s_ref - R_ref(2, 0) * Rxz(1, 0) * d * n_ref(1) * s_ref -
            R_ref(2, 1) * Rxz(1, 1) * d * n_ref(1) * s_ref - R_ref(2, 2) * Rxz(1, 2) * d * n_ref(1) * s_ref +
            R_ref(0, 0) * Rxz(1, 0) * c_ref * d * n_ref(1) * p_ref(1) +
            R_ref(0, 1) * Rxz(1, 1) * c_ref * d * n_ref(1) * p_ref(1) +
            R_ref(0, 2) * Rxz(1, 2) * c_ref * d * n_ref(1) * p_ref(1) -
            R_ref(1, 0) * Rxz(1, 0) * c_ref * d * n_ref(0) * p_ref(1) -
            R_ref(1, 1) * Rxz(1, 1) * c_ref * d * n_ref(0) * p_ref(1) -
            R_ref(1, 2) * Rxz(1, 2) * c_ref * d * n_ref(0) * p_ref(1) -
            R_ref(0, 0) * Rxz(2, 0) * c_ref * d * n_ref(2) * p_query(1) -
            R_ref(0, 1) * Rxz(2, 1) * c_ref * d * n_ref(2) * p_query(1) -
            R_ref(0, 2) * Rxz(2, 2) * c_ref * d * n_ref(2) * p_query(1) +
            R_ref(2, 0) * Rxz(2, 0) * c_ref * d * n_ref(0) * p_query(1) +
            R_ref(2, 1) * Rxz(2, 1) * c_ref * d * n_ref(0) * p_query(1) +
            R_ref(2, 2) * Rxz(2, 2) * c_ref * d * n_ref(0) * p_query(1) -
            R_ref(2, 0) * Rxz(2, 0) * d * n_ref(2) * q * s_query -
            R_ref(2, 1) * Rxz(2, 1) * d * n_ref(2) * q * s_query -
            R_ref(2, 2) * Rxz(2, 2) * d * n_ref(2) * q * s_query -
            R_ref(0, 0) * Rxz(1, 0) * d * n_ref(1) * s_ref * p_ref(0) -
            R_ref(0, 1) * Rxz(1, 1) * d * n_ref(1) * s_ref * p_ref(0) -
            R_ref(0, 2) * Rxz(1, 2) * d * n_ref(1) * s_ref * p_ref(0) +
            R_ref(1, 0) * Rxz(1, 0) * d * n_ref(0) * s_ref * p_ref(0) +
            R_ref(1, 1) * Rxz(1, 1) * d * n_ref(0) * s_ref * p_ref(0) +
            R_ref(1, 2) * Rxz(1, 2) * d * n_ref(0) * s_ref * p_ref(0) -
            R_ref(1, 0) * Rxz(2, 0) * d * n_ref(2) * s_ref * p_query(1) -
            R_ref(1, 1) * Rxz(2, 1) * d * n_ref(2) * s_ref * p_query(1) -
            R_ref(1, 2) * Rxz(2, 2) * d * n_ref(2) * s_ref * p_query(1) +
            R_ref(2, 0) * Rxz(2, 0) * d * n_ref(1) * s_ref * p_query(1) +
            R_ref(2, 1) * Rxz(2, 1) * d * n_ref(1) * s_ref * p_query(1) +
            R_ref(2, 2) * Rxz(2, 2) * d * n_ref(1) * s_ref * p_query(1) +
            R_ref(0, 0) * Rxz(2, 0) * n_ref(2) * q * s_query * t_ref(0) +
            R_ref(0, 1) * Rxz(2, 1) * n_ref(2) * q * s_query * t_ref(0) +
            R_ref(0, 2) * Rxz(2, 2) * n_ref(2) * q * s_query * t_ref(0) +
            R_ref(1, 0) * Rxz(2, 0) * n_ref(2) * q * s_query * t_ref(1) +
            R_ref(1, 1) * Rxz(2, 1) * n_ref(2) * q * s_query * t_ref(1) +
            R_ref(1, 2) * Rxz(2, 2) * n_ref(2) * q * s_query * t_ref(1) +
            R_ref(2, 0) * Rxz(2, 0) * n_ref(2) * q * s_query * t_ref(2) +
            R_ref(2, 1) * Rxz(2, 1) * n_ref(2) * q * s_query * t_ref(2) +
            R_ref(2, 2) * Rxz(2, 2) * n_ref(2) * q * s_query * t_ref(2) -
            R_ref(0, 0) * Rxz(2, 0) * c_ref * d * n_ref(1) * p_ref(1) * p_query(1) -
            R_ref(0, 1) * Rxz(2, 1) * c_ref * d * n_ref(1) * p_ref(1) * p_query(1) -
            R_ref(0, 2) * Rxz(2, 2) * c_ref * d * n_ref(1) * p_ref(1) * p_query(1) +
            R_ref(1, 0) * Rxz(2, 0) * c_ref * d * n_ref(0) * p_ref(1) * p_query(1) +
            R_ref(1, 1) * Rxz(2, 1) * c_ref * d * n_ref(0) * p_ref(1) * p_query(1) +
            R_ref(1, 2) * Rxz(2, 2) * c_ref * d * n_ref(0) * p_ref(1) * p_query(1) -
            R_ref(0, 0) * Rxz(2, 0) * d * n_ref(2) * q * s_query * p_ref(0) -
            R_ref(0, 1) * Rxz(2, 1) * d * n_ref(2) * q * s_query * p_ref(0) -
            R_ref(0, 2) * Rxz(2, 2) * d * n_ref(2) * q * s_query * p_ref(0) -
            R_ref(1, 0) * Rxz(2, 0) * d * n_ref(2) * q * s_query * p_ref(1) -
            R_ref(1, 1) * Rxz(2, 1) * d * n_ref(2) * q * s_query * p_ref(1) -
            R_ref(1, 2) * Rxz(2, 2) * d * n_ref(2) * q * s_query * p_ref(1) -
            R_ref(2, 0) * Rxz(2, 0) * d * n_ref(0) * q * s_query * p_ref(0) -
            R_ref(2, 0) * Rxz(2, 0) * d * n_ref(1) * q * s_query * p_ref(1) -
            R_ref(2, 1) * Rxz(2, 1) * d * n_ref(0) * q * s_query * p_ref(0) -
            R_ref(2, 1) * Rxz(2, 1) * d * n_ref(1) * q * s_query * p_ref(1) -
            R_ref(2, 2) * Rxz(2, 2) * d * n_ref(0) * q * s_query * p_ref(0) -
            R_ref(2, 2) * Rxz(2, 2) * d * n_ref(1) * q * s_query * p_ref(1) +
            R_ref(0, 0) * Rxz(2, 0) * d * n_ref(1) * s_ref * p_ref(0) * p_query(1) +
            R_ref(0, 1) * Rxz(2, 1) * d * n_ref(1) * s_ref * p_ref(0) * p_query(1) +
            R_ref(0, 2) * Rxz(2, 2) * d * n_ref(1) * s_ref * p_ref(0) * p_query(1) -
            R_ref(1, 0) * Rxz(2, 0) * d * n_ref(0) * s_ref * p_ref(0) * p_query(1) -
            R_ref(1, 1) * Rxz(2, 1) * d * n_ref(0) * s_ref * p_ref(0) * p_query(1) -
            R_ref(1, 2) * Rxz(2, 2) * d * n_ref(0) * s_ref * p_ref(0) * p_query(1) +
            R_ref(0, 0) * Rxz(2, 0) * n_ref(0) * q * s_query * t_ref(0) * p_ref(0) +
            R_ref(0, 0) * Rxz(2, 0) * n_ref(1) * q * s_query * t_ref(0) * p_ref(1) +
            R_ref(0, 1) * Rxz(2, 1) * n_ref(0) * q * s_query * t_ref(0) * p_ref(0) +
            R_ref(0, 1) * Rxz(2, 1) * n_ref(1) * q * s_query * t_ref(0) * p_ref(1) +
            R_ref(0, 2) * Rxz(2, 2) * n_ref(0) * q * s_query * t_ref(0) * p_ref(0) +
            R_ref(0, 2) * Rxz(2, 2) * n_ref(1) * q * s_query * t_ref(0) * p_ref(1) +
            R_ref(1, 0) * Rxz(2, 0) * n_ref(0) * q * s_query * t_ref(1) * p_ref(0) +
            R_ref(1, 0) * Rxz(2, 0) * n_ref(1) * q * s_query * t_ref(1) * p_ref(1) +
            R_ref(1, 1) * Rxz(2, 1) * n_ref(0) * q * s_query * t_ref(1) * p_ref(0) +
            R_ref(1, 1) * Rxz(2, 1) * n_ref(1) * q * s_query * t_ref(1) * p_ref(1) +
            R_ref(1, 2) * Rxz(2, 2) * n_ref(0) * q * s_query * t_ref(1) * p_ref(0) +
            R_ref(1, 2) * Rxz(2, 2) * n_ref(1) * q * s_query * t_ref(1) * p_ref(1) +
            R_ref(2, 0) * Rxz(2, 0) * n_ref(0) * q * s_query * t_ref(2) * p_ref(0) +
            R_ref(2, 0) * Rxz(2, 0) * n_ref(1) * q * s_query * t_ref(2) * p_ref(1) +
            R_ref(2, 1) * Rxz(2, 1) * n_ref(0) * q * s_query * t_ref(2) * p_ref(0) +
            R_ref(2, 1) * Rxz(2, 1) * n_ref(1) * q * s_query * t_ref(2) * p_ref(1) +
            R_ref(2, 2) * Rxz(2, 2) * n_ref(0) * q * s_query * t_ref(2) * p_ref(0) +
            R_ref(2, 2) * Rxz(2, 2) * n_ref(1) * q * s_query * t_ref(2) * p_ref(1) -
            R_ref(0, 0) * Rxz(2, 0) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(0) -
            R_ref(0, 1) * Rxz(2, 1) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(0) -
            R_ref(0, 2) * Rxz(2, 2) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(0) -
            R_ref(1, 0) * Rxz(2, 0) * d * n_ref(1) * q * s_query * p_ref(1) * p_ref(1) -
            R_ref(1, 1) * Rxz(2, 1) * d * n_ref(1) * q * s_query * p_ref(1) * p_ref(1) -
            R_ref(1, 2) * Rxz(2, 2) * d * n_ref(1) * q * s_query * p_ref(1) * p_ref(1) -
            R_ref(0, 0) * Rxz(2, 0) * d * n_ref(1) * q * s_query * p_ref(0) * p_ref(1) -
            R_ref(0, 1) * Rxz(2, 1) * d * n_ref(1) * q * s_query * p_ref(0) * p_ref(1) -
            R_ref(0, 2) * Rxz(2, 2) * d * n_ref(1) * q * s_query * p_ref(0) * p_ref(1) -
            R_ref(1, 0) * Rxz(2, 0) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(1) -
            R_ref(1, 1) * Rxz(2, 1) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(1) -
            R_ref(1, 2) * Rxz(2, 2) * d * n_ref(0) * q * s_query * p_ref(0) * p_ref(1);
    return M;
}

std::pair<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>>
solver_up1p_sift(const Eigen::Matrix3d &R_ref, const Eigen::Vector3d &t_ref, const Eigen::Vector2d &y, const Eigen::Vector3d &X, const Eigen::Matrix3d &Rxz, double angle_ref,
                 double angle_query, double q, const Eigen::Vector3d &n) {
    double s_ref = sin(angle_ref);
    double c_ref = cos(angle_ref);
    double s_query = sin(angle_query);
    double c_query = cos(angle_query);

    // OUTPUTS
    std::vector<Eigen::Matrix3d> Rsolns;
    std::vector<Eigen::Vector3d> tsolns;

    double d = X(2);
    Eigen::Vector2d p_ref; // "x"
    p_ref << X(0) / d, X(1) / d;
    Eigen::Matrix<double, 4, 6> M =
        build_linear_constraints(R_ref, t_ref, s_ref, c_ref, s_query, c_query, q, p_ref, d, n, y, Rxz); // p_query = y

    // G-J elimination
    const Eigen::Matrix<double, 4, 3> C = M.block<4, 4>(0, 0).partialPivLu().solve(M.block<4, 3>(0, 3));

    Eigen::Vector3d poly(C(3, 2), C(3, 1), C(3, 0));
    Eigen::PolynomialSolver<double, 2> poly_solver(poly);
    std::vector<double> r_solns;
    poly_solver.realRoots(r_solns);

    int nsolns = r_solns.size();
    for (int i = 0; i < nsolns; i++) {
        double r = r_solns[i];

        Eigen::Matrix3d Ry;
        double s = (1 + r * r);
        Ry << 1 - r * r, 0, -2 * r, 0, 1 + r * r, 0, 2 * r, 0, 1 - r * r;
        Ry /= s;

        Eigen::Vector3d mon(r * r, r, 1);
        Eigen::Vector3d t = -C.block<3, 3>(0, 0) * mon;
        t /= s;

        Eigen::Matrix3d Rsoln = Ry * Rxz;
        Eigen::Vector3d tsoln = t;

        Rsolns.push_back(Rsoln);
        tsolns.push_back(tsoln);
    }
    return std::make_pair(Rsolns, tsolns);
}
} // namespace ECCV2026
