#include <Eigen/Dense>
#include <PoseLib/misc/re3q3.h>
#include <utility>

namespace ECCV2026
{
    static inline
    Eigen::Matrix<double,3,12> build_constraints(
        const Eigen::Matrix3d &R_ref,
        const Eigen::Vector3d &t_ref,
        const double s_ref,
        const double c_ref,
        const double s_query,
        const double c_query,
        const Eigen::Vector2d &p_ref,
        const double d,
        const Eigen::Vector3d &n,
        const Eigen::Vector2d &p_query )
{
        const Eigen::Vector3d n_ref = R_ref * n;
        Eigen::Matrix<double,3,12> M;
        M << -1, 0, p_query(0), R_ref(0,0)*t_ref(0) - R_ref(2,0)*d + R_ref(1,0)*t_ref(1) + R_ref(2,0)*t_ref(2) - R_ref(0,0)*d*p_ref(0) - R_ref(1,0)*d*p_ref(1), 0, R_ref(2,0)*d*p_query(0) - R_ref(0,0)*t_ref(0)*p_query(0) - R_ref(1,0)*t_ref(1)*p_query(0) - R_ref(2,0)*t_ref(2)*p_query(0) + R_ref(0,0)*d*p_ref(0)*p_query(0) + R_ref(1,0)*d*p_ref(1)*p_query(0), R_ref(0,1)*t_ref(0) - R_ref(2,1)*d + R_ref(1,1)*t_ref(1) + R_ref(2,1)*t_ref(2) - R_ref(0,1)*d*p_ref(0) - R_ref(1,1)*d*p_ref(1), 0, R_ref(2,1)*d*p_query(0) - R_ref(0,1)*t_ref(0)*p_query(0) - R_ref(1,1)*t_ref(1)*p_query(0) - R_ref(2,1)*t_ref(2)*p_query(0) + R_ref(0,1)*d*p_ref(0)*p_query(0) + R_ref(1,1)*d*p_ref(1)*p_query(0), R_ref(0,2)*t_ref(0) - R_ref(2,2)*d + R_ref(1,2)*t_ref(1) + R_ref(2,2)*t_ref(2) - R_ref(0,2)*d*p_ref(0) - R_ref(1,2)*d*p_ref(1), 0, R_ref(2,2)*d*p_query(0) - R_ref(0,2)*t_ref(0)*p_query(0) - R_ref(1,2)*t_ref(1)*p_query(0) - R_ref(2,2)*t_ref(2)*p_query(0) + R_ref(0,2)*d*p_ref(0)*p_query(0) + R_ref(1,2)*d*p_ref(1)*p_query(0),
        0, -1, p_query(1), 0, R_ref(0,0)*t_ref(0) - R_ref(2,0)*d + R_ref(1,0)*t_ref(1) + R_ref(2,0)*t_ref(2) - R_ref(0,0)*d*p_ref(0) - R_ref(1,0)*d*p_ref(1), R_ref(2,0)*d*p_query(1) - R_ref(0,0)*t_ref(0)*p_query(1) - R_ref(1,0)*t_ref(1)*p_query(1) - R_ref(2,0)*t_ref(2)*p_query(1) + R_ref(0,0)*d*p_ref(0)*p_query(1) + R_ref(1,0)*d*p_ref(1)*p_query(1), 0, R_ref(0,1)*t_ref(0) - R_ref(2,1)*d + R_ref(1,1)*t_ref(1) + R_ref(2,1)*t_ref(2) - R_ref(0,1)*d*p_ref(0) - R_ref(1,1)*d*p_ref(1), R_ref(2,1)*d*p_query(1) - R_ref(0,1)*t_ref(0)*p_query(1) - R_ref(1,1)*t_ref(1)*p_query(1) - R_ref(2,1)*t_ref(2)*p_query(1) + R_ref(0,1)*d*p_ref(0)*p_query(1) + R_ref(1,1)*d*p_ref(1)*p_query(1), 0, R_ref(0,2)*t_ref(0) - R_ref(2,2)*d + R_ref(1,2)*t_ref(1) + R_ref(2,2)*t_ref(2) - R_ref(0,2)*d*p_ref(0) - R_ref(1,2)*d*p_ref(1), R_ref(2,2)*d*p_query(1) - R_ref(0,2)*t_ref(0)*p_query(1) - R_ref(1,2)*t_ref(1)*p_query(1) - R_ref(2,2)*t_ref(2)*p_query(1) + R_ref(0,2)*d*p_ref(0)*p_query(1) + R_ref(1,2)*d*p_ref(1)*p_query(1),
        0, 0, 0, R_ref(0,0)*c_ref*d*n_ref(2)*s_query - R_ref(2,0)*c_ref*d*n_ref(0)*s_query + R_ref(1,0)*d*n_ref(2)*s_ref*s_query - R_ref(2,0)*d*n_ref(1)*s_ref*s_query + R_ref(0,0)*c_ref*d*n_ref(1)*s_query*p_ref(1) - R_ref(1,0)*c_ref*d*n_ref(0)*s_query*p_ref(1) - R_ref(0,0)*d*n_ref(1)*s_ref*s_query*p_ref(0) + R_ref(1,0)*d*n_ref(0)*s_ref*s_query*p_ref(0), R_ref(2,0)*c_ref*c_query*d*n_ref(0) - R_ref(0,0)*c_ref*c_query*d*n_ref(2) - R_ref(1,0)*c_query*d*n_ref(2)*s_ref + R_ref(2,0)*c_query*d*n_ref(1)*s_ref - R_ref(0,0)*c_ref*c_query*d*n_ref(1)*p_ref(1) + R_ref(1,0)*c_ref*c_query*d*n_ref(0)*p_ref(1) + R_ref(0,0)*c_query*d*n_ref(1)*s_ref*p_ref(0) - R_ref(1,0)*c_query*d*n_ref(0)*s_ref*p_ref(0), R_ref(0,0)*c_ref*c_query*d*n_ref(2)*p_query(1) - R_ref(2,0)*c_ref*c_query*d*n_ref(0)*p_query(1) - R_ref(0,0)*c_ref*d*n_ref(2)*s_query*p_query(0) + R_ref(1,0)*c_query*d*n_ref(2)*s_ref*p_query(1) + R_ref(2,0)*c_ref*d*n_ref(0)*s_query*p_query(0) - R_ref(2,0)*c_query*d*n_ref(1)*s_ref*p_query(1) - R_ref(1,0)*d*n_ref(2)*s_ref*s_query*p_query(0) + R_ref(2,0)*d*n_ref(1)*s_ref*s_query*p_query(0) + R_ref(0,0)*c_ref*c_query*d*n_ref(1)*p_ref(1)*p_query(1) - R_ref(1,0)*c_ref*c_query*d*n_ref(0)*p_ref(1)*p_query(1) - R_ref(0,0)*c_ref*d*n_ref(1)*s_query*p_ref(1)*p_query(0) - R_ref(0,0)*c_query*d*n_ref(1)*s_ref*p_ref(0)*p_query(1) + R_ref(1,0)*c_ref*d*n_ref(0)*s_query*p_ref(1)*p_query(0) + R_ref(1,0)*c_query*d*n_ref(0)*s_ref*p_ref(0)*p_query(1) + R_ref(0,0)*d*n_ref(1)*s_ref*s_query*p_ref(0)*p_query(0) - R_ref(1,0)*d*n_ref(0)*s_ref*s_query*p_ref(0)*p_query(0), R_ref(0,1)*c_ref*d*n_ref(2)*s_query - R_ref(2,1)*c_ref*d*n_ref(0)*s_query + R_ref(1,1)*d*n_ref(2)*s_ref*s_query - R_ref(2,1)*d*n_ref(1)*s_ref*s_query + R_ref(0,1)*c_ref*d*n_ref(1)*s_query*p_ref(1) - R_ref(1,1)*c_ref*d*n_ref(0)*s_query*p_ref(1) - R_ref(0,1)*d*n_ref(1)*s_ref*s_query*p_ref(0) + R_ref(1,1)*d*n_ref(0)*s_ref*s_query*p_ref(0), R_ref(2,1)*c_ref*c_query*d*n_ref(0) - R_ref(0,1)*c_ref*c_query*d*n_ref(2) - R_ref(1,1)*c_query*d*n_ref(2)*s_ref + R_ref(2,1)*c_query*d*n_ref(1)*s_ref - R_ref(0,1)*c_ref*c_query*d*n_ref(1)*p_ref(1) + R_ref(1,1)*c_ref*c_query*d*n_ref(0)*p_ref(1) + R_ref(0,1)*c_query*d*n_ref(1)*s_ref*p_ref(0) - R_ref(1,1)*c_query*d*n_ref(0)*s_ref*p_ref(0), R_ref(0,1)*c_ref*c_query*d*n_ref(2)*p_query(1) - R_ref(2,1)*c_ref*c_query*d*n_ref(0)*p_query(1) - R_ref(0,1)*c_ref*d*n_ref(2)*s_query*p_query(0) + R_ref(1,1)*c_query*d*n_ref(2)*s_ref*p_query(1) + R_ref(2,1)*c_ref*d*n_ref(0)*s_query*p_query(0) - R_ref(2,1)*c_query*d*n_ref(1)*s_ref*p_query(1) - R_ref(1,1)*d*n_ref(2)*s_ref*s_query*p_query(0) + R_ref(2,1)*d*n_ref(1)*s_ref*s_query*p_query(0) + R_ref(0,1)*c_ref*c_query*d*n_ref(1)*p_ref(1)*p_query(1) - R_ref(1,1)*c_ref*c_query*d*n_ref(0)*p_ref(1)*p_query(1) - R_ref(0,1)*c_ref*d*n_ref(1)*s_query*p_ref(1)*p_query(0) - R_ref(0,1)*c_query*d*n_ref(1)*s_ref*p_ref(0)*p_query(1) + R_ref(1,1)*c_ref*d*n_ref(0)*s_query*p_ref(1)*p_query(0) + R_ref(1,1)*c_query*d*n_ref(0)*s_ref*p_ref(0)*p_query(1) + R_ref(0,1)*d*n_ref(1)*s_ref*s_query*p_ref(0)*p_query(0) - R_ref(1,1)*d*n_ref(0)*s_ref*s_query*p_ref(0)*p_query(0), R_ref(0,2)*c_ref*d*n_ref(2)*s_query - R_ref(2,2)*c_ref*d*n_ref(0)*s_query + R_ref(1,2)*d*n_ref(2)*s_ref*s_query - R_ref(2,2)*d*n_ref(1)*s_ref*s_query + R_ref(0,2)*c_ref*d*n_ref(1)*s_query*p_ref(1) - R_ref(1,2)*c_ref*d*n_ref(0)*s_query*p_ref(1) - R_ref(0,2)*d*n_ref(1)*s_ref*s_query*p_ref(0) + R_ref(1,2)*d*n_ref(0)*s_ref*s_query*p_ref(0), R_ref(2,2)*c_ref*c_query*d*n_ref(0) - R_ref(0,2)*c_ref*c_query*d*n_ref(2) - R_ref(1,2)*c_query*d*n_ref(2)*s_ref + R_ref(2,2)*c_query*d*n_ref(1)*s_ref - R_ref(0,2)*c_ref*c_query*d*n_ref(1)*p_ref(1) + R_ref(1,2)*c_ref*c_query*d*n_ref(0)*p_ref(1) + R_ref(0,2)*c_query*d*n_ref(1)*s_ref*p_ref(0) - R_ref(1,2)*c_query*d*n_ref(0)*s_ref*p_ref(0), R_ref(0,2)*c_ref*c_query*d*n_ref(2)*p_query(1) - R_ref(2,2)*c_ref*c_query*d*n_ref(0)*p_query(1) - R_ref(0,2)*c_ref*d*n_ref(2)*s_query*p_query(0) + R_ref(1,2)*c_query*d*n_ref(2)*s_ref*p_query(1) + R_ref(2,2)*c_ref*d*n_ref(0)*s_query*p_query(0) - R_ref(2,2)*c_query*d*n_ref(1)*s_ref*p_query(1) - R_ref(1,2)*d*n_ref(2)*s_ref*s_query*p_query(0) + R_ref(2,2)*d*n_ref(1)*s_ref*s_query*p_query(0) + R_ref(0,2)*c_ref*c_query*d*n_ref(1)*p_ref(1)*p_query(1) - R_ref(1,2)*c_ref*c_query*d*n_ref(0)*p_ref(1)*p_query(1) - R_ref(0,2)*c_ref*d*n_ref(1)*s_query*p_ref(1)*p_query(0) - R_ref(0,2)*c_query*d*n_ref(1)*s_ref*p_ref(0)*p_query(1) + R_ref(1,2)*c_ref*d*n_ref(0)*s_query*p_ref(1)*p_query(0) + R_ref(1,2)*c_query*d*n_ref(0)*s_ref*p_ref(0)*p_query(1) + R_ref(0,2)*d*n_ref(1)*s_ref*s_query*p_ref(0)*p_query(0) - R_ref(1,2)*d*n_ref(0)*s_ref*s_query*p_ref(0)*p_query(0);
        return M;
    }
std::pair<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>>
solver_p2p_ori(
        const std::vector<Eigen::Matrix3d> &R_ref,
        const std::vector<Eigen::Vector3d> &t_ref,
        const std::vector<double> &angle_ref,
        const std::vector<double> &angle_query,
        const std::vector<Eigen::Vector3d> &X,
        const std::vector<Eigen::Vector3d> &n,
        const std::vector<Eigen::Vector2d> &p_query
    )
    {

    std::vector<Eigen::Matrix<double,3,12>> Ms;
    for (int i=0; i < 2; i++) {
        double s_ref = sin(angle_ref[i]);
        double c_ref = cos(angle_ref[i]);
        double s_query = sin(angle_query[i]);
        double c_query = cos(angle_query[i]);


        double d = X[i](2);
        Eigen::Vector2d p_ref = X[i].hnormalized();
        Eigen::Matrix<double,3,12> M = build_constraints(R_ref[i],t_ref[i],s_ref,c_ref,s_query,c_query,p_ref,d,n[i],p_query[i]);
        Ms.push_back(M);
    }

    Eigen::Matrix<double,6,12> M;
    M << Ms[0], Ms[1];

    // OUTPUTS
    std::vector<Eigen::Matrix3d> Rsolns;
    std::vector<Eigen::Vector3d> tsolns;
 
    // G-J elimination
    const Eigen::Matrix<double,6,9> G = M.block<6,6>(0,0).partialPivLu().solve(M.block<6,9>(0,3));
    const Eigen::Matrix<double,3,9> T = G.block<3,9>(0,0);
    const Eigen::Matrix<double,3,9> C = G.block<3,9>(3,0);

    Eigen::Matrix<double,3,8> cayley_solutions;
    Eigen::Matrix<double,3,10> C3Q3;
    poselib::re3q3::rotation_to_3q3(C,&C3Q3);
    int nsolns = poselib::re3q3::re3q3(C3Q3, &cayley_solutions);

    for ( int i = 0; i < nsolns; i++ )
    {
        Eigen::Matrix3d R;
        poselib::re3q3::cayley_param(cayley_solutions.col(i),&R);
        
        Eigen::Matrix<double,9,1> X;
        X << R(0,0),R(1,0),R(2,0), R(0,1),R(1,1),R(2,1), R(0,2),R(1,2),R(2,2);
        Eigen::Vector3d t = -T*X;
        
        Rsolns.push_back(R);
        tsolns.push_back(t);
    }
    return std::make_pair(Rsolns, tsolns);
}
} // namespace ECCV2026
