#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eccv2026/eccv2026.h>
#include <fstream>
#include <iostream>
#include <problem_generator/problem_generator.h>
#include <stdlib.h>
#include <chrono>

using namespace problem_generator;


int main(int argc, char **argv) {
    if (argc != 7) {
        std::cout << "Six arguments expected - [point_noise] [angle_noise] [affine_noise] [normal_noise] [imu_noise] [nprob]" << std::endl;
        return 1;
    }

    srand(1234);

    ProblemGenerator generator;
    generator.config.known_focal = false;

    double point_noise = atof(argv[1]);
    double angle_noise = atof(argv[2]);
    double affine_noise = atof(argv[3]);
    double normal_noise = atof(argv[4]);
    double imu_noise = atof(argv[5]);
    
    generator.config.point_noise = point_noise;
    generator.config.angle_noise = angle_noise;
    generator.config.affine_noise = affine_noise;
    generator.config.normal_noise = normal_noise;
    generator.config.imu_noise = imu_noise;

    int nprob = atoi(argv[6]);

    double mean_rot_err = 0;
    double mean_pos_err = 0;
    double mean_focal_err = 0;

    double mean_rot_err2 = 0;
    double mean_pos_err2 = 0;
    double mean_focal_err2 = 0;

    double mean_rot_err3 = 0;
    double mean_pos_err3 = 0;

    double mean_rot_err4 = 0;
    double mean_pos_err4 = 0;
    
    double mean_rot_err5 = 0;
    double mean_pos_err5 = 0;
    double mean_focal_err5 = 0;
    
    double mean_rot_err6 = 0;
    double mean_pos_err6 = 0;
    
    double mean_rot_err7 = 0;
    double mean_pos_err7 = 0;
    double mean_focal_err7 = 0;

    double mean_rot_err8 = 0;
    double mean_pos_err8 = 0;
    
    double mean_rot_err9 = 0;
    double mean_pos_err9 = 0;
    double mean_focal_err9 = 0;

    int N = 4;

    Eigen::MatrixXd stats = Eigen::MatrixXd(3, nprob);
    Eigen::MatrixXd stats2 = Eigen::MatrixXd(3, nprob);
    Eigen::MatrixXd stats3 = Eigen::MatrixXd(2, nprob);
    Eigen::MatrixXd stats4 = Eigen::MatrixXd(2, nprob);
    Eigen::MatrixXd stats5 = Eigen::MatrixXd(3, nprob);
    Eigen::MatrixXd stats6 = Eigen::MatrixXd(2, nprob);
    Eigen::MatrixXd stats7 = Eigen::MatrixXd(3, nprob);
    Eigen::MatrixXd stats8 = Eigen::MatrixXd(2, nprob);
    Eigen::MatrixXd stats9 = Eigen::MatrixXd(3, nprob);
    
    std::vector<long> runtimes1;
    std::vector<long> runtimes2;
    std::vector<long> runtimes3;
    std::vector<long> runtimes4;
    std::vector<long> runtimes5;
    std::vector<long> runtimes6;
    std::vector<long> runtimes7;
    std::vector<long> runtimes8;
    std::vector<long> runtimes9;

    for (int i = 0; i < nprob; i++) {
        // std::cout << "iter: " << i << std::endl;
        // Generate problem
        std::vector<Eigen::Vector3d> Xs;
        std::vector<Eigen::Vector3d> ns;
        std::vector<Eigen::Vector2d> ys;
        std::vector<Eigen::Matrix2d> As;
        std::vector<double> angle_refs;
        std::vector<double> angle_querys;
        std::vector<double> qs;
        Eigen::Matrix3d R;
        Eigen::Matrix3d Rxz;
        Eigen::Vector3d t;
        double f, f_ref;
        int flag = 1;
        while (flag > 0) {
            flag = generator.make_random_problems(Xs, ns, ys, As, angle_refs, angle_querys, qs, R, Rxz, t, f_ref, f, N);
        }

        Eigen::Quaterniond q(R);
        Eigen::Vector3d c(-R.transpose() * t);

        // Solve UP1PfAC - Our
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        Eigen::Vector3d O = Eigen::Vector3d::Zero();
        
        std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double> out;
        auto start_time = std::chrono::high_resolution_clock::now();
        out = ECCV2026::solver_up1pf_ac(I, O, f_ref, As[0], Xs[0].hnormalized(), Xs[0](2), ns[0], ys[0], Rxz);
        auto end_time = std::chrono::high_resolution_clock::now();
        runtimes1.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        Eigen::Matrix3d R_est = std::get<0>(out);
        Eigen::Vector3d t_est = std::get<1>(out);
        double f_est = std::get<2>(out);
                const Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ";\n", "[", "]", "[", "]");
        std::cout << "=== INPUTS ===" << std::endl;
std::cout << "R_ref:\n" << I.format(HeavyFmt) << std::endl;
std::cout << "t_ref:\n" << O.format(HeavyFmt) << std::endl;
std::cout << "focal_ref: " << f_ref << std::endl;
std::cout << "A_orig:\n" << As[0].format(HeavyFmt) << std::endl;
std::cout << "p_ref_orig:\n" << Xs[0].hnormalized().format(HeavyFmt) << std::endl;
std::cout << "d: " << Xs[0](2) << std::endl;
std::cout << "n:\n" << ns[0].format(HeavyFmt) << std::endl;
std::cout << "p_query_orig:\n" << ys[0].format(HeavyFmt) << std::endl;
std::cout << "Rxz:\n" << Rxz.format(HeavyFmt) << std::endl;


std::cout << "\n=== OUTPUTS ===" << std::endl;
std::cout << "R (rotation matrix):\n" << std::get<0>(out).format(HeavyFmt) << std::endl;
std::cout << "t (translation vector):\n" << std::get<1>(out).format(HeavyFmt) << std::endl;
std::cout << "focal: " << std::get<2>(out) << std::endl;

        Eigen::Quaterniond qsoln(R_est);
        Eigen::Vector3d csoln(-R_est.transpose() * t_est);
        double rot_err = q.angularDistance(qsoln);
        double pos_err = (c - csoln).norm();
        double focal_err = std::abs(f - f_est) / f;

        mean_rot_err += rot_err;
        mean_pos_err += pos_err;
        mean_focal_err += focal_err;
        stats(0, i) = rot_err;
        stats(1, i) = pos_err;
        stats(2, i) = focal_err;

        // Solve P3.5Pf - Larsson et al. (2017)
        Eigen::Matrix<double, 2, 4> y;
        Eigen::Matrix<double, 3, 4> X;

        y << ys[0][0], ys[1][0], ys[2][0], ys[3][0], ys[0][1], ys[1][1], ys[2][1], ys[3][1];

        X << Xs[0][0], Xs[1][0], Xs[2][0], Xs[3][0], Xs[0][1], Xs[1][1], Xs[2][1], Xs[3][1], Xs[0][2], Xs[1][2],
            Xs[2][2], Xs[3][2];
        start_time = std::chrono::high_resolution_clock::now();
        out = ECCV2026::solver_p35pf(y, X);
        end_time = std::chrono::high_resolution_clock::now();
        runtimes2.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        R_est = std::get<0>(out);
        t_est = std::get<1>(out);
        f_est = std::get<2>(out);

        qsoln = R_est;
        csoln = -R_est.transpose() * t_est;
        rot_err = q.angularDistance(qsoln);
        pos_err = (c - csoln).norm();
        focal_err = std::abs(f - f_est) / f;

        mean_rot_err2 += rot_err;
        mean_pos_err2 += pos_err;
        mean_focal_err2 += focal_err;
        stats2(0, i) = rot_err;
        stats2(1, i) = pos_err;
        stats2(2, i) = focal_err;

        // Solve P1PAC - Ventura et al. (2023)
        Eigen::Vector2d y_tmp = ys[0] / f; // Normalize with focal length

        double min_rot_err = INFINITY;
        double min_pos_err = INFINITY;
        
        start_time = std::chrono::high_resolution_clock::now();
        std::vector<Eigen::Matrix<double, 3, 4>> out2 =
            ACP1PCayleySolver(y_tmp, Xs[0], As[0] * f_ref / f , ns[0]); // Need to normalize the affine features as well
        end_time = std::chrono::high_resolution_clock::now();
        runtimes3.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        
        for (auto c_ : out2) {
            Eigen::Matrix3d R_est2 = c_.block(0, 0, 3, 3);
            Eigen::Vector3d t_est2 = c_.block(0, 3, 3, 1);
            Eigen::Quaterniond qsoln2(R_est2);
            Eigen::Vector3d csoln2(-R_est2.transpose() * t_est2);
            double my_rot_err = q.angularDistance(qsoln2);
            double my_pos_err = (c - csoln2).norm();
            if (my_rot_err < min_rot_err)
                min_rot_err = my_rot_err;
            if (my_pos_err < min_pos_err)
                min_pos_err = my_pos_err;
        }
        mean_rot_err3 += min_rot_err;
        mean_pos_err3 += min_pos_err;
        stats3(0, i) = min_rot_err;
        stats3(1, i) = min_pos_err;

        // Solve UP1PSIFT - Ventura et al. (CVPR, 2024)
        min_rot_err = INFINITY;
        min_pos_err = INFINITY;
        
        start_time = std::chrono::high_resolution_clock::now();
        std::pair<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>> out3 = ECCV2026::solver_up1p_sift(
            I, O, y_tmp, Xs[0], Rxz, angle_refs[0], angle_querys[0], qs[0] * f_ref / f, ns[0]); // Need to normalize the scale as well
        end_time = std::chrono::high_resolution_clock::now();
        runtimes4.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        
        for (size_t i = 0; i < out3.first.size(); i++) {
            Eigen::Matrix3d R_est2 = out3.first[i];
            Eigen::Vector3d t_est2 = out3.second[i];
            Eigen::Quaterniond qsoln2(R_est2);
            Eigen::Vector3d csoln2(-R_est2.transpose() * t_est2);
            double my_rot_err = q.angularDistance(qsoln2);
            double my_pos_err = (c - csoln2).norm();

            if (my_rot_err < min_rot_err)
                min_rot_err = my_rot_err;
            if (my_pos_err < min_pos_err)
                min_pos_err = my_pos_err;
        }
        mean_rot_err4 += min_rot_err;
        mean_pos_err4 += min_pos_err;
        stats4(0, i) = min_rot_err;
        stats4(1, i) = min_pos_err;

        // Solve P4Pf - Kukelova et al. (2013)
        min_rot_err = INFINITY;
        min_pos_err = INFINITY;
        double min_focal_err = INFINITY;
        std::vector<Eigen::Matrix3d> output_R;
        std::vector<Eigen::Vector3d> output_T;
        std::vector<double> output_focal;
        
        start_time = std::chrono::high_resolution_clock::now();
        int nbr_sols_p4pf = ECCV2026::p4pf_wrapper(ys, Xs, &output_R, &output_T, &output_focal, true);
        end_time = std::chrono::high_resolution_clock::now();
        runtimes5.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        
        for (size_t i = 0; i < nbr_sols_p4pf; i++) {
            Eigen::Quaterniond qsoln2(output_R[i]);
            Eigen::Vector3d csoln2(-output_R[i].transpose() * output_T[i]);
            double my_rot_err = q.angularDistance(qsoln2);
            double my_pos_err = (c - csoln2).norm();
        	double my_focal_err = std::abs(f - output_focal[i]) / f;
            if (my_rot_err < min_rot_err)
                min_rot_err = my_rot_err;
            if (my_pos_err < min_pos_err)
                min_pos_err = my_pos_err;
            if (my_focal_err < min_focal_err)
                min_focal_err = my_focal_err;
        }
        mean_rot_err5 += min_rot_err;
        mean_pos_err5 += min_pos_err;
        mean_focal_err5 += min_focal_err;
        stats5(0, i) = min_rot_err;
        stats5(1, i) = min_pos_err;
        stats5(2, i) = min_focal_err;
        
        // Solve P2PORI - Ventura et al. (CVPR, 2024)
        min_rot_err = INFINITY;
        min_pos_err = INFINITY;
        std::vector<Eigen::Matrix3d> Is = { I, I };
        std::vector<Eigen::Vector3d> Os = { O, O };
        std::vector<Eigen::Vector2d> ys_norm;
        ys_norm.push_back(ys[0] / f);
        ys_norm.push_back(ys[1] / f);
      
        start_time = std::chrono::high_resolution_clock::now();
        std::pair<std::vector<Eigen::Matrix3d>, std::vector<Eigen::Vector3d>> out4 = ECCV2026::solver_p2p_ori(
            Is, Os, angle_refs, angle_querys, Xs, ns, ys_norm);
            end_time = std::chrono::high_resolution_clock::now();
        runtimes6.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        
        for (size_t j = 0; j < out4.first.size(); j++) {
            Eigen::Matrix3d R_est2 = out4.first[j];
            Eigen::Vector3d t_est2 = out4.second[j];
            Eigen::Quaterniond qsoln2(R_est2);
            Eigen::Vector3d csoln2(-R_est2.transpose() * t_est2);
            double my_rot_err = q.angularDistance(qsoln2);
            double my_pos_err = (c - csoln2).norm();

            if (my_rot_err < min_rot_err)
                min_rot_err = my_rot_err;
            if (my_pos_err < min_pos_err)
                min_pos_err = my_pos_err;
        }
        mean_rot_err6 += min_rot_err;
        mean_pos_err6 += min_pos_err;
        stats6(0, i) = min_rot_err;
        stats6(1, i) = min_pos_err;
        
        // Solve UP2PfORI - OUR
        std::vector<double> fs = { f_ref, f_ref };
        std::vector<Eigen::Vector2d> p_ref_origs;
        p_ref_origs.push_back(Xs[0].hnormalized());
        p_ref_origs.push_back(Xs[1].hnormalized());
        std::vector<double> ds;
        ds.push_back(Xs[0](2));
        ds.push_back(Xs[1](2));
      
        std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double> out5;
        
        start_time = std::chrono::high_resolution_clock::now();
        out5 = ECCV2026::solver_up2pf_ori(Is, Os, fs, angle_refs, angle_querys, p_ref_origs, ds, ns, ys, Rxz);
        end_time = std::chrono::high_resolution_clock::now();
        runtimes7.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        


	std::cout << "=== INPUTS ===\n" << std::endl;
for (size_t i = 0; i < 2; ++i) {
    std::cout << "--- Sample " << i << " ---" << std::endl;
    std::cout << "R_ref:\n" << Is[i].format(HeavyFmt) << std::endl;
    std::cout << "t_ref:\n" << Os[i].format(HeavyFmt) << std::endl;
    std::cout << "focal_ref: " << fs[i] << std::endl;
    std::cout << "angle_ref: " << angle_refs[i] << std::endl;
    std::cout << "angle_query: " << angle_querys[i] << std::endl;
    std::cout << "p_ref_orig:\n" << p_ref_origs[i].format(HeavyFmt) << std::endl;
    std::cout << "d: " << ds[i] << std::endl;
    std::cout << "n:\n" << ns[i].format(HeavyFmt) << std::endl;
    std::cout << "p_query_orig:\n" << ys[i].format(HeavyFmt) << std::endl;
}
std::cout << "Rxz:\n" << Rxz.format(HeavyFmt) << std::endl;


std::cout << "\n=== OUTPUTS ===\n" << std::endl;
std::cout << "R (rotation matrix):\n" << std::get<0>(out5).format(HeavyFmt) << std::endl;
std::cout << "t (translation vector):\n" << std::get<1>(out5).format(HeavyFmt) << std::endl;
std::cout << "focal: " << std::get<2>(out5) << std::endl;

        
        R_est = std::get<0>(out5);
        t_est = std::get<1>(out5);
        f_est = std::get<2>(out5);

        qsoln = R_est;
        csoln = -R_est.transpose() * t_est;
        rot_err = q.angularDistance(qsoln);
        pos_err = (c - csoln).norm();
        focal_err = std::abs(f - f_est) / f;
        
        mean_rot_err7 += rot_err;
        mean_pos_err7 += pos_err;
        mean_focal_err7 += focal_err;
        stats7(0, i) = rot_err;
        stats7(1, i) = pos_err;
        stats7(2, i) = focal_err;
        
        // Solve UP2P - Kukelova et al., ACCV (2010)
        min_rot_err = INFINITY;
        min_pos_err = INFINITY;
        
        start_time = std::chrono::high_resolution_clock::now();
        int nbr_sols_up2p = ECCV2026::up2p_wrapper(ys_norm, Xs, Rxz, &output_R, &output_T);
        end_time = std::chrono::high_resolution_clock::now();
        runtimes8.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        
        for (size_t j = 0; j < nbr_sols_up2p; j++) {
            Eigen::Quaterniond qsoln2(output_R[j]);
            Eigen::Vector3d csoln2(-output_R[j].transpose() * output_T[i]);
            double my_rot_err = q.angularDistance(qsoln2);
            double my_pos_err = (c - csoln2).norm();
        	double my_focal_err = std::abs(f - output_focal[i]) / f;
            if (my_rot_err < min_rot_err)
                min_rot_err = my_rot_err;
            if (my_pos_err < min_pos_err)
                min_pos_err = my_pos_err;
            if (my_focal_err < min_focal_err)
                min_focal_err = my_focal_err;
        }
        mean_rot_err8 += min_rot_err;
        mean_pos_err8 += min_pos_err;

        stats8(0, i) = min_rot_err;
        stats8(1, i) = min_pos_err;
        
        // Solve UP3Pf - Kukelova et al., ACCV (2010) - modified
        min_rot_err = INFINITY;
        min_pos_err = INFINITY;
        min_focal_err = INFINITY;
        
        start_time = std::chrono::high_resolution_clock::now();

        int nbr_sols_up3pf = ECCV2026::solver_up3pf(ys, Xs, Rxz, &output_R, &output_T, &output_focal);
        end_time = std::chrono::high_resolution_clock::now();
        runtimes9.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        
        for (size_t j = 0; j < nbr_sols_up3pf; j++) {
            Eigen::Quaterniond qsoln2(output_R[j]);
            Eigen::Vector3d csoln2(-output_R[j].transpose() * output_T[j]);
            double my_rot_err = q.angularDistance(qsoln2);
            double my_pos_err = (c - csoln2).norm();
        	double my_focal_err = std::abs(f - output_focal[j]) / f;
            if (my_rot_err < min_rot_err)
                min_rot_err = my_rot_err;
            if (my_pos_err < min_pos_err)
                min_pos_err = my_pos_err;
            if (my_focal_err < min_focal_err)
                min_focal_err = my_focal_err;
        }
        mean_rot_err9 += min_rot_err;
        mean_pos_err9 += min_pos_err;
        mean_focal_err9 += min_focal_err;

        stats9(0, i) = min_rot_err;
        stats9(1, i) = min_pos_err;
        stats9(2, i) = min_focal_err;
    }

    mean_rot_err /= nprob;
    mean_pos_err /= nprob;
    mean_focal_err /= nprob;
    mean_rot_err2 /= nprob;
    mean_pos_err2 /= nprob;
    mean_focal_err2 /= nprob;
    mean_rot_err3 /= nprob;
    mean_pos_err3 /= nprob;
    mean_rot_err4 /= nprob;
    mean_pos_err4 /= nprob;
    mean_rot_err5 /= nprob;
    mean_pos_err5 /= nprob;
    mean_focal_err5 /= nprob;
    mean_rot_err6 /= nprob;
    mean_pos_err6 /= nprob;
    mean_rot_err7 /= nprob;
    mean_pos_err7 /= nprob;
    mean_focal_err7 /= nprob;
    mean_rot_err8 /= nprob;
    mean_pos_err8 /= nprob;
    mean_rot_err9 /= nprob;
    mean_pos_err9 /= nprob;
    mean_focal_err9 /= nprob;
    
    std::sort(runtimes1.begin(), runtimes1.end());
    std::sort(runtimes2.begin(), runtimes2.end());
    std::sort(runtimes3.begin(), runtimes3.end());
    std::sort(runtimes4.begin(), runtimes4.end());
    std::sort(runtimes5.begin(), runtimes5.end());
    std::sort(runtimes6.begin(), runtimes6.end());
    std::sort(runtimes7.begin(), runtimes7.end());
    std::sort(runtimes8.begin(), runtimes8.end());
    std::sort(runtimes9.begin(), runtimes9.end());
    int mid = runtimes1.size() / 2;


    std::cout << "Tested " << nprob << " problems.\n\n";
    
    std::cout << "UP1PfAC (Our)\n";
    std::cout << "median runtime: " << runtimes1[mid] << " ns\n";
    std::cout << "average norm. focal length error: " << mean_focal_err << "\n";
    std::cout << "average position error: " << mean_pos_err << "\n";
    std::cout << "average rotation error: " << mean_rot_err << "\n\n";
    
    std::cout << "P3.5Pf (Larsson et al., 2017)\n";
    std::cout << "median runtime: " << runtimes2[mid] << " ns\n";
    std::cout << "average norm. focal length error: " << mean_focal_err2 << "\n";
    std::cout << "average position error: " << mean_pos_err2 << "\n";
    std::cout << "average rotation error: " << mean_rot_err2 << "\n\n";
    
    std::cout << "P1PAC (Ventura et al., 2023)\n";
    std::cout << "median runtime: " << runtimes3[mid] << " ns\n";
    std::cout << "average position error: " << mean_pos_err3 << "\n";
    std::cout << "average rotation error: " << mean_rot_err3 << "\n\n";
    
    std::cout << "UP1PSIFT (Ventura et al., 2024)\n";
    std::cout << "median runtime: " << runtimes4[mid] << " ns\n";
    std::cout << "average position error: " << mean_pos_err4 << "\n";
    std::cout << "average rotation error: " << mean_rot_err4 << "\n\n";
    
    std::cout << "P4Pf (Kukelova et al., 2013)\n";
    std::cout << "median runtime: " << runtimes5[mid] << " ns\n";
    std::cout << "average norm. focal length error: " << mean_focal_err5 << "\n";
    std::cout << "average position error: " << mean_pos_err5 << "\n";
    std::cout << "average rotation error: " << mean_rot_err5 << "\n\n";
    
    std::cout << "P2PORI (Ventura et al., 2024)\n";
    std::cout << "median runtime: " << runtimes6[mid] << " ns\n";
    std::cout << "average position error: " << mean_pos_err6 << "\n";
    std::cout << "average rotation error: " << mean_rot_err6 << "\n\n";
    
    std::cout << "UP2PfORI (Our)\n";
    std::cout << "median runtime: " << runtimes7[mid] << " ns\n";
    std::cout << "average norm. focal length error: " << mean_focal_err7 << "\n";
    std::cout << "average position error: " << mean_pos_err7 << "\n";
    std::cout << "average rotation error: " << mean_rot_err7 << "\n\n";
    
    std::cout << "UP2P (Kukelova et al., 2010)\n";
    std::cout << "median runtime: " << runtimes8[mid] << " ns\n";
    std::cout << "average position error: " << mean_pos_err8 << "\n";
    std::cout << "average rotation error: " << mean_rot_err8 << "\n\n";
    
    std::cout << "UP3Pf (Kukelova et al., 2010 - modified)\n";
    std::cout << "median runtime: " << runtimes9[mid] << " ns\n";
    std::cout << "average norm. focal length error: " << mean_focal_err9 << "\n";
    std::cout << "average position error: " << mean_pos_err9 << "\n";
    std::cout << "average rotation error: " << mean_rot_err9 << "\n";

    return 0;
}
