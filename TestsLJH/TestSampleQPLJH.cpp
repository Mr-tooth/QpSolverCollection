/* Author: Lai Junhang */

#include <limits>
#include <string>
#include <qp_solver_collection/QpSolverCollection.h>
#include <vector>
using QpSolverCollection::QpCoeff;
using QpSolverCollection::QpSolverType;

std::vector<double> durationList;
void solveOneQP(const QpCoeff & qp_coeff, const Eigen::VectorXd & x_gt, const QpSolverType & qp_solver_type)
{
    
    if (QpSolverCollection::isQpSolverEnabled(qp_solver_type))
    {
        auto qp_solver = QpSolverCollection::allocateQpSolver(qp_solver_type);
        QpCoeff qp_coeff_copied = qp_coeff;
        qp_solver->printInfo();

        // Start the timer
        auto start = std::chrono::high_resolution_clock::now();

        Eigen::VectorXd x_opt = qp_solver->solve(qp_coeff_copied);

        // End the timer
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate the elapsed time
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        durationList.push_back(duration.count());
        double thre = 1e-6;
        if(qp_solver_type == QpSolverType::OSQP)
        {
          thre = 1e-3;
        }
        else if(qp_solver_type == QpSolverType::PROXQP)
        {
          // Set proxqp_->settings.eps_abs to 1e-8 to satisfy thre of 1e-6
          thre = 1e-4;
        }

        auto err = (x_opt-x_gt).norm();
        if(err < thre)
            std::cout<<"QP Solution of " << std::to_string(qp_solver_type) << " is correct:\n"
            << "  solution: " << x_opt.transpose() << "\n  ground truth: " << x_gt.transpose()
            << "\n  error: " << err  
            << "\n elapsed time: "<< duration.count() <<" microseconds\n"<<std::endl;
    }
    else
    {
        std::cout << "[solveOneQP] Skip QP solver " << std::to_string(qp_solver_type) << " because it is not enabled."
                << std::endl;
    }
    

    
}

int main()
{

    // QP coefficients are copied from https://github.com/jrl-umi3218/eigen-qld/blob/master/tests/QPTest.cpp
    int dim_var = 6;
    int dim_eq = 3;
    int dim_ineq = 2;
    QpCoeff qp_coeff;
    qp_coeff.setup(dim_var, dim_eq, dim_ineq);
    qp_coeff.obj_mat_.setIdentity();
    qp_coeff.obj_vec_ << 1., 2., 3., 4., 5., 6.;
    qp_coeff.eq_mat_ << 1., -1., 1., 0., 3., 1., -1., 0., -3., -4., 5., 6., 2., 5., 3., 0., 1., 0.;
    qp_coeff.eq_vec_ << 1., 2., 3.;
    qp_coeff.ineq_mat_ << 0., 1., 0., 1., 2., -1., -1., 0., 2., 1., 1., 0.;
    qp_coeff.ineq_vec_ << -1., 2.5;
    qp_coeff.x_min_ << -1000., -10000., 0., -1000., -1000., -1000.;
    qp_coeff.x_max_ << 10000., 100., 1.5, 100., 100., 1000.;
    Eigen::VectorXd x_gt(dim_var);
    x_gt << 1.7975426, -0.3381487, 0.1633880, -4.9884023, 0.6054943, -3.1155623;


    solveOneQP(qp_coeff, x_gt, QpSolverCollection::QpSolverType::OSQP);
    solveOneQP(qp_coeff, x_gt, QpSolverCollection::QpSolverType::QPMAD);
    solveOneQP(qp_coeff, x_gt, QpSolverCollection::QpSolverType::qpOASES);
    solveOneQP(qp_coeff, x_gt, QpSolverCollection::QpSolverType::JRLQP);
    solveOneQP(qp_coeff, x_gt, QpSolverCollection::QpSolverType::QuadProg);
    solveOneQP(qp_coeff, x_gt, QpSolverCollection::QpSolverType::QLD);


    std::cout<<"DurationList: \n"
    <<"OSQP    : "<<durationList.at(0)<<" us\n"
    <<"QPMAD   : "<<durationList.at(1)<<" us\n"
    <<"qpOASES : "<<durationList.at(2)<<" us\n"
    <<"JRLQP   : "<<durationList.at(3)<<" us\n"
    <<"QuadProg: "<<durationList.at(4)<<" us\n"
    <<"QLD     : "<<durationList.at(5)<<" us\n"
    <<std::endl;

    return 0;
}