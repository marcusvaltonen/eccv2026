#include "eccv2026/eccv2026.h"

#include <gtest/gtest.h>


TEST(ECCV2026Tests, TestSolverP35Pf) {
    const double tol = 1e-8;

    Eigen::Matrix<double, 2, 4> y;
    y << -3.511091220107755, -7.527070199692494, 8.916313274472833, 3.825943935873279, 6.898907830435459,
        9.015711613055140, 31.142244863628367, 2.000730182984090;

    Eigen::Matrix<double, 3, 4> X;
    X << 0.777003526719788, 2.508772473185106, 0.317317472324612, -0.162501941548495, 0.622393924172013,
        1.063459639041023, 0.078020080978078, 0.690051897857806, 0.647380884516047, 1.156921653327394,
        1.324385449160224, 0.555756771412105;

    auto output = ECCV2026::solver_p35pf(y, X);

    Eigen::Matrix3d R;
    R << -0.865329056542071, 0.013314553063081, 0.501027291253421, 0.488305174805843, -0.202934531924287,
        0.848749451846954, 0.112976458449863, 0.979101781445452, 0.169103581885392;
    ASSERT_TRUE(std::get<0>(output).isApprox(R, tol));

    Eigen::Vector3d t;
    t << -0.042887577710841, -0.050789377713144, -0.074706725189861;
    ASSERT_TRUE(std::get<1>(output).isApprox(t, tol));

    ASSERT_NEAR(std::get<2>(output), 6.716781251796911, tol);
}

TEST(ECCV2026Tests, TestSolverUP1PfAC) {
    const double tol = 1e-8;

    Eigen::Matrix3d R_ref;
    R_ref << 1, 0, 0,
             0, 1, 0,
             0, 0, 1;

    Eigen::Vector3d t_ref;
    t_ref << 0, 0, 0;

    double focal_ref = 1;

    Eigen::Matrix2d A_orig;
    A_orig << 1224.90305929698, 367.971886245288,
             -2869.57458472315, -109.91170017409;

    Eigen::Vector2d p_ref_orig;
    p_ref_orig << 0.965058147365262, -2.54733828081291;

    double d = 0.922551;

    Eigen::Vector3d n;
    n << 0.513876091678939, -0.797380365904498, 0.316410989807226;

    Eigen::Vector2d p_query_orig;
    p_query_orig << 452.00309391118, -1348.69389572756;

    Eigen::Matrix3d Rxz;
    Rxz << 0.465304228419325, 0.885150820490552, -1.38777878078145e-17,
          -0.878575379712747, 0.461847664490549, 0.121663622201657,
           0.10769065501565, -0.0566105978552424, 0.992571389388578;
           
    auto output = ECCV2026::solver_up1pf_ac(
        R_ref, t_ref, focal_ref, A_orig, p_ref_orig,
        d, n, p_query_orig, Rxz);

    // Expected outputs
    Eigen::Matrix3d R_expected;
    R_expected << 0.428623133323543,  0.650090262103427,  0.627427175613399,
                 -0.878575379712747,  0.461847664490549,  0.121663622201657,
                 -0.210683439649422, -0.603389912016205,  0.769111891947177;

    Eigen::Vector3d t_expected;
    t_expected << 1.13854201692199, 0.0508551107137336, -1.64351438976271;
    
    ASSERT_TRUE(std::get<0>(output).isApprox(R_expected, tol));
    ASSERT_TRUE(std::get<1>(output).isApprox(t_expected, tol));
    ASSERT_NEAR(std::get<2>(output), 234.57211046484613, tol);
}


TEST(ECCV2026Tests, TestSolverUP2PfORI) {
    const double tol = 1e-8;
    
    std::vector<Eigen::Matrix3d> R_ref(2);
    std::vector<Eigen::Vector3d> t_ref(2);
    std::vector<double> focal_ref(2);
    std::vector<double> angle_ref(2);
    std::vector<double> angle_query(2);
    std::vector<Eigen::Vector2d> p_ref_orig(2);
    std::vector<double> d(2);
    std::vector<Eigen::Vector3d> n(2);
    std::vector<Eigen::Vector2d> p_query_orig(2);

    R_ref[0] = Eigen::Matrix3d::Identity();
    t_ref[0] = Eigen::Vector3d(0.0, 0.0, 0.0);
    focal_ref[0]   = 1.0;
    angle_ref[0]   = 1.94761;
    angle_query[0] = 1.68414;
    p_ref_orig[0]  = Eigen::Vector2d(0.965058147365262, -2.54733828081291);
    d[0]           = 0.922551;
    n[0]           = Eigen::Vector3d(0.513876091678939,
                                     -0.797380365904498,
                                      0.316410989807226);
    p_query_orig[0] = Eigen::Vector2d(452.00309391118, -1348.69389572756);

    R_ref[1] = Eigen::Matrix3d::Identity();
    t_ref[1] = Eigen::Vector3d(0.0, 0.0, 0.0);
    focal_ref[1]   = 1.0;
    angle_ref[1]   = 2.14154;
    angle_query[1] = 2.17492;
    p_ref_orig[1]  = Eigen::Vector2d(0.955333658566738, -1.31622728146566);
    d[1]           = 1.27214;
    n[1]           = Eigen::Vector3d(-0.975193435988501,
                                     -0.156816499002404,
                                     -0.156225311795399);
    p_query_orig[1] = Eigen::Vector2d(3600.77926138146, -4301.28372421395);

    Eigen::Matrix3d Rxz;
    Rxz << 0.465304228419325, 0.885150820490552, -1.38777878078145e-17,
          -0.878575379712747, 0.461847664490549, 0.121663622201657,
           0.10769065501565, -0.0566105978552424, 0.992571389388578;

    auto [R_out, t_out, f_out] = ECCV2026::solver_up2pf_ori(
        R_ref, t_ref, focal_ref, angle_ref, angle_query,
        p_ref_orig, d, n, p_query_orig, Rxz);
           
    // Expected outputs
    Eigen::Matrix3d R_expected;
    R_expected << 0.428623541687554,  0.650091431644584,  0.627425684853205,
                 -0.878575379712747,  0.461847664490549,  0.121663622201657,
                 -0.210682608854695, -0.603388651954046,  0.769113108079208;


    Eigen::Vector3d t_expected;
    t_expected << 1.138547142314, 0.0508510372597529, -1.64351222448313;
    
    ASSERT_TRUE(R_out.isApprox(R_expected, tol));
    ASSERT_TRUE(t_out.isApprox(t_expected, tol));
    ASSERT_NEAR(f_out, 234.57239310976527, tol);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
