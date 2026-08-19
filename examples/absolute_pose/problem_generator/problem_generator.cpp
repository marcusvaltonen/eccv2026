
#include <Eigen/Geometry>
#include <affine2sift/affine2sift.h>
#include <iostream>
#include <problem_generator/problem_generator.h>
#include <problem_generator/random.h>

namespace problem_generator {
static Eigen::Matrix3d calculateHomography(const Eigen::Vector3d &X, const Eigen::Vector3d &n, const Eigen::Matrix3d &R,
                                           const Eigen::Vector3d &t, const Eigen::Matrix3d &K, const Eigen::Matrix3d &K_ref_inv) {
    double d = -n.dot(X);

    Eigen::Vector3d v = -n / d;
    Eigen::Matrix3d H = K * (R + t * v.transpose()) * K_ref_inv;

    return H;
}

static Eigen::Matrix2d affineFromHomography(const Eigen::Matrix3d &H, const Eigen::Vector2d &x,
                                            const Eigen::Vector2d &y) {
    double s = H(2, 0) * x(0) + H(2, 1) * x(1) + H(2, 2);

    Eigen::Matrix2d A;
    A << (H(0, 0) - H(2, 0) * y(0)) / s, (H(0, 1) - H(2, 1) * y(0)) / s, (H(1, 0) - H(2, 0) * y(1)) / s,
        (H(1, 1) - H(2, 1) * y(1)) / s;

    return A;
}

int ProblemGenerator::make_random_problem(Eigen::Vector3d &X, Eigen::Vector3d &n, Eigen::Vector2d &y,
                                           Eigen::Matrix2d &A, double &angle_ref, double &angle_query, double &q,
                                           Eigen::Matrix3d &R, Eigen::Matrix3d &Rxz, Eigen::Vector3d &t, double &f_ref, double &f_query) {
    std::vector<Eigen::Vector3d> Xs;
    std::vector<Eigen::Vector3d> ns;
    std::vector<Eigen::Vector2d> ys;
    std::vector<Eigen::Matrix2d> As;
    std::vector<double> angle_refs;
    std::vector<double> angle_querys;
    std::vector<double> qs;

    int out = make_random_problems(Xs, ns, ys, As, angle_refs, angle_querys, qs, R, Rxz, t, f_ref, f_query, 1);
    X = Xs[0];
    n = ns[0];
    y = ys[0];
    A = As[0];
    angle_ref = angle_refs[0];
    angle_query = angle_querys[0];
    q = qs[0];
    return out;
}

int ProblemGenerator::make_random_problems(std::vector<Eigen::Vector3d> &Xs, std::vector<Eigen::Vector3d> &ns,
                                            std::vector<Eigen::Vector2d> &ys, std::vector<Eigen::Matrix2d> &As,
                                            Eigen::Matrix3d &R, Eigen::Matrix3d &Rxz, Eigen::Vector3d &t, double &f_ref, double &f_query,
                                            int N) {
    std::vector<double> angle_refs;
    std::vector<double> angle_querys;
    std::vector<double> qs;
    int out = make_random_problems(Xs, ns, ys, As, angle_refs, angle_querys, qs, R, Rxz, t, f_ref, f_query, N);
    return out;
}

int ProblemGenerator::make_random_problems(std::vector<Eigen::Vector3d> &Xs, std::vector<Eigen::Vector3d> &ns,
                                            std::vector<Eigen::Vector2d> &ys, std::vector<Eigen::Matrix2d> &As,
                                            std::vector<double> &angle_refs, std::vector<double> &angle_querys,
                                            std::vector<double> &qs, Eigen::Matrix3d &R, Eigen::Matrix3d &Rxz,
                                            Eigen::Vector3d &t, double &f_ref, double &f_query, int N) {
    // generate random rotation and IMU data
    double x = generator.randn();
    double z = generator.randn();
    Eigen::AngleAxisd xAngle(x, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd yAngle(generator.randn(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd zAngle(z, Eigen::Vector3d::UnitZ());

    if (config.imu_noise > 0) {
        Eigen::AngleAxisd xAngleNoise(x + config.imu_noise * generator.randn() * M_PI / 180. / 2.0, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd zAngleNoise(z + config.imu_noise * generator.randn() * M_PI / 180. / 2.0, Eigen::Vector3d::UnitZ());
        Eigen::Quaterniond q_xz = xAngleNoise * zAngleNoise;
        Rxz = q_xz.toRotationMatrix();
    } else {
        Eigen::Quaterniond q_xz = xAngle * zAngle;
        Rxz = q_xz.toRotationMatrix();
    }

    Eigen::Quaterniond q_mat = yAngle * xAngle * zAngle;
    R = q_mat.toRotationMatrix();

    // generate random translation
    t = generator.rand_unit_vector() * 2;

    // generate random focal length in interval [1, 10]
    if (config.known_focal) {
        f_query = 1;
        f_ref = 1;
    } else {
        f_query = 700 + 500 * generator.rand();
        f_ref = 1; // 700 + 500 * generator.rand();
    }
    Eigen::Matrix3d K;
    K << f_query, 0, 0, 0, f_query, 0, 0, 0, 1;
    Eigen::Matrix3d K_ref_inv;
    K_ref_inv << 1 / f_ref, 0, 0, 0, 1 / f_ref, 0, 0, 0, 1;

    // Reset vectors
    Xs.clear();
    ns.clear();
    ys.clear();
    As.clear();
    angle_refs.clear();
    angle_querys.clear();
    qs.clear();
    for (int i = 0; i < N; i++) {
        Eigen::Vector3d X;
        Eigen::Vector3d n;
        Eigen::Vector2d y;
        Eigen::Matrix2d A;
        double s_ref, c_ref, s_query, c_query, q;
        double angle_ref, angle_query;
        // iterate until valid configuration found
        int attempt = 0;
        while (attempt <= 100) {
            attempt += 1;
            // make 3D point
            X = generator.randn3();

            // compute depth in reference camera
            double d = X(2);

            // check if behind reference camera
            if (d < 0)
                continue;

            // project to reference camera
            Eigen::Vector2d x = X.head(2) / X(2);
            x *= f_ref;

            // sample random normal vector
            n = generator.rand_unit_vector();

            // project to query camera
            Eigen::Vector3d PX = K * (R * X + t);
            y = PX.head(2) / PX(2);

            // check if behind query camera
            if (PX(2) < 0)
                continue;

            // calculate homography
            Eigen::Matrix3d H = calculateHomography(X, n, R, t, K, K_ref_inv);

            // calculate affine from homography
            A = affineFromHomography(H, x, y);

            // check if determinant is positive
            if (A.determinant() < 0)
                continue;

            // convert affine to SIFT
            affine2sift(A, s_ref, c_ref, s_query, c_query, q);
            
            // solve for w
            // qu * qv = q
            double qu = q;
            double qv = q;
            double w;
            if ( abs(c_query*s_ref) > abs(c_ref*c_query) )
                w = (c_ref*c_query*qu - A(0,0) + qv*s_ref*s_query) / (c_query*s_ref);
            else
                w = (A(0,1) - c_query*qu*s_ref + c_ref*qv*s_query)/(c_ref*c_query);

            angle_ref = std::atan2(s_ref, c_ref);
            angle_query = std::atan2(s_query, c_query);

            if (config.point_noise > 0) {
                y += generator.randn2() * config.point_noise;
            }

            // add noise to affine
            if (config.affine_noise > 0) {
                A(0, 0) += (A(0, 0) * config.affine_noise) * generator.randn();
                A(0, 1) += (A(0, 1) * config.affine_noise) * generator.randn();
                A(1, 0) += (A(1, 0) * config.affine_noise) * generator.randn();
                A(1, 1) += (A(1, 1) * config.affine_noise) * generator.randn();
            }
            // add noise to angles
            if (config.angle_noise > 0) {
                double rand_angle = generator.randn() * config.angle_noise * M_PI / 180.;
                angle_ref += rand_angle;
                rand_angle = generator.randn() * config.angle_noise * M_PI / 180.;
                angle_query += rand_angle;
            }
            
            // add noise to normal vector
            if ( config.normal_noise > 0 )
            {
                Eigen::Vector3d noiser = generator.rand_unit_vector();
                double rand_angle = generator.randn() * config.normal_noise * M_PI / 180.;
                Eigen::Matrix3d randR = Eigen::AngleAxisd(rand_angle,noiser).toRotationMatrix();
                n = randR * n;
            }

            // add noise to scale
            if (config.scale_noise > 0) {
                double log_q = log(q);
                double log_noise = generator.randn() * config.scale_noise;
                log_q += log_noise;
                q = exp(log_q);
                qv = q;
            }

            
            // re-make affine matrix if necessary
            if ( config.angle_noise > 0 || config.scale_noise > 0 )
            {
                Eigen::Matrix2d R2, U, R1t;
                R2 << c_query, -s_query, s_query, c_query;
                U << qu, w, 0, qv;
                R1t << c_ref, s_ref, -s_ref, c_ref;
                A = R2 * U * R1t;
                if (A.determinant() < 0) {
                    std::cout << "FAILED" << std::endl;
                    continue;
                }
            }
            break;
        }
        if (attempt == 100) {
            return 1;
        }
        Xs.push_back(X);
        ns.push_back(n);
        ys.push_back(y);
        As.push_back(A);
        angle_refs.push_back(angle_ref);
        angle_querys.push_back(angle_query);
        qs.push_back(q);
    }
    return 0;
}

} // namespace problem_generator
