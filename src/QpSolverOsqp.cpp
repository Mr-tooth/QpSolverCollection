/* Author: Masaki Murooka */

#include <qp_solver_collection/QpSolverOptions.h>

#if ENABLE_OSQP
#  include <limits>

#  include <qp_solver_collection/QpSolverCollection.h>

#  include <OsqpEigen/OsqpEigen.h>
#  define OSQP_EIGEN_DEBUG_OUTPUT

using namespace QpSolverCollection;

QpSolverOsqp::QpSolverOsqp()
{
  type_ = QpSolverType::OSQP;
  osqp_ = std::make_unique<OsqpEigen::Solver>();
}

Eigen::VectorXd QpSolverOsqp::solve(int dim_var,
                                    int dim_eq,
                                    int dim_ineq,
                                    Eigen::Ref<Eigen::MatrixXd> Q,
                                    const Eigen::Ref<const Eigen::VectorXd> & c,
                                    const Eigen::Ref<const Eigen::MatrixXd> & A,
                                    const Eigen::Ref<const Eigen::VectorXd> & b,
                                    const Eigen::Ref<const Eigen::MatrixXd> & C,
                                    const Eigen::Ref<const Eigen::VectorXd> & d,
                                    const Eigen::Ref<const Eigen::VectorXd> & x_min,
                                    const Eigen::Ref<const Eigen::VectorXd> & x_max)
{
  int dim_eq_ineq_with_bound = dim_eq + dim_ineq + dim_var;
  Eigen::MatrixXd AC_with_bound(dim_eq_ineq_with_bound, dim_var);
  Eigen::VectorXd bd_with_bound_min(dim_eq_ineq_with_bound);
  Eigen::VectorXd bd_with_bound_max(dim_eq_ineq_with_bound);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim_var, dim_var);
  AC_with_bound << A, C, I;
  bd_with_bound_min << b, Eigen::VectorXd::Constant(dim_ineq, -1 * std::numeric_limits<double>::infinity()), x_min;
  bd_with_bound_max << b, d, x_max;

  auto sparse_start_time = clock::now();
  // Matrices and vectors must be hold during solver's lifetime
  Q_sparse_ = Q.sparseView();
  AC_with_bound_sparse_ = AC_with_bound.sparseView();
  // You must pass unconst vectors to OSQP
  c_ = c;
  bd_with_bound_min_ = bd_with_bound_min;
  bd_with_bound_max_ = bd_with_bound_max;
  auto sparse_end_time = clock::now();
  sparse_duration_ =
      1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(sparse_end_time - sparse_start_time).count();

  // osqp_->settings()->setAbsoluteTolerance(1e-2);
  // osqp_->settings()->setRelativeTolerance(1e-2);
  // osqp_->settings()->setScaledTerimination(1);
  // QSC_INFO_STREAM("max_iter: " << osqp_->settings()->getSettings()->max_iter << ", " <<
  //                 "eps_abs: " << osqp_->settings()->getSettings()->eps_abs << ", " <<
  //                 "eps_rel: " << osqp_->settings()->getSettings()->eps_rel << ", " <<
  //                 "scaled_termination: " << osqp_->settings()->getSettings()->scaled_termination);

  osqp_->settings()->setVerbosity(false);
  osqp_->settings()->setWarmStart(true);
  if(!solve_failed_ && !force_initialize_ && osqp_->isInitialized() && dim_var == osqp_->data()->getData()->n
     && dim_eq_ineq_with_bound == osqp_->data()->getData()->m)
  {
    // Update only matrices and vectors
    osqp_->updateHessianMatrix(Q_sparse_);
    osqp_->updateGradient(c_);
    osqp_->updateLinearConstraintsMatrix(AC_with_bound_sparse_);
    osqp_->updateBounds(bd_with_bound_min_, bd_with_bound_max_);
  }
  else
  {
    // Initialize fully
    if(osqp_->isInitialized())
    {
      osqp_->clearSolver();
      osqp_->data()->clearHessianMatrix();
      osqp_->data()->clearLinearConstraintsMatrix();
    }

    osqp_->data()->setNumberOfVariables(dim_var);
    osqp_->data()->setNumberOfConstraints(dim_eq_ineq_with_bound);
    osqp_->data()->setHessianMatrix(Q_sparse_);
    osqp_->data()->setGradient(c_);
    osqp_->data()->setLinearConstraintsMatrix(AC_with_bound_sparse_);
    osqp_->data()->setLowerBound(bd_with_bound_min_);
    osqp_->data()->setUpperBound(bd_with_bound_max_);
    osqp_->initSolver();
  }

  osqp_->solve();

  if(osqp_->workspace()->info->status_val == OSQP_SOLVED)
  {
    solve_failed_ = false;
  }
  else
  {
    solve_failed_ = true;
    QSC_WARN_STREAM("[QpSolverOsqp::solve] Failed to solve: " << osqp_->workspace()->info->status);
  }

  return osqp_->getSolution();
}

namespace QpSolverCollection
{
std::shared_ptr<QpSolver> allocateQpSolverOsqp()
{
  return std::make_shared<QpSolverOsqp>();
}
} // namespace QpSolverCollection
#endif
