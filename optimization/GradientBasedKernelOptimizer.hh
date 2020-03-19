#ifndef _GRADIENT_BASED_KERNEL_OPTIMIZER_HH
#define _GRADIENT_BASED_KERNEL_OPTIMIZER_HH

#include "KernelOptimizer.hh"

#include <nlopt.hpp>

#include <Eigen/Core>

#include <vector>
#include <iostream>
#include <type_traits>
#include <cstddef>
#include <limits>


namespace GaussianProcess
{

  namespace Optimization 
  {
    
    struct GradientBasedKernelOptimizer 
    {

      template<typename OptParameters_T>
      explicit GradientBasedKernelOptimizer(const OptParameters_T&)
      {
        init(OptParameters_T::gradient_aglorithm, OptParameters_T::maxiterations, OptParameters_T::ftolerance, OptParameters_T::xreltolerance);
      }

      explicit GradientBasedKernelOptimizer()
      {
        init();
      }

      void init(nlopt::algorithm algorithm = nlopt::LD_MMA, double maxiterations = 100.0, double ftolerance = -1.0, double xreltolerance = -1.0)
      {
        m_algorithm     = algorithm;
        m_iterations    = maxiterations;
        m_ftolerance    = ftolerance;
        m_xreltolerance = xreltolerance;
      }

      template<typename GaussianProcess_T>
      void operator() (GaussianProcess_T& model)
      {

        nlopt::opt optimizer(m_algorithm, model.numKernelParameters());
        optimizer.set_max_objective(optimizationf<GaussianProcess_T>, &model);

        std::vector<double> params(model.numKernelParameters());
        Eigen::VectorXd     optparams = model.getKernelParameters();
        Eigen::VectorXd::Map(&params[0], model.numKernelParameters()) = optparams;

        optimizer.set_maxeval(m_iterations);
        optimizer.set_ftol_rel(m_ftolerance);
        optimizer.set_xtol_rel(m_xreltolerance);

        double likelihood;
        try 
        {
          optimizer.optimize(params, likelihood);
        }
        catch (std::runtime_error& err) 
        {
          std::cout << err.what() << std::endl;
        }
      }

    private:

      template<typename GaussianProcess_T>
      static double optimizationf(const std::vector<double>& parameters, std::vector<double>& gradient, void* ptrModel)
      {
        Eigen::VectorXd  opt_param = Eigen::VectorXd::Map(parameters.data(), parameters.size());
        GaussianProcess_T* ptrModelOpt = reinterpret_cast<GaussianProcess_T*>(ptrModel);
        ptrModelOpt->recompute(opt_param);
                
        double value{ std::numeric_limits<double>::infinity() };

        if (!gradient.empty()) 
        {
          ptrModelOpt->eval(true);
          Eigen::VectorXd gradient = -1.0*ptrModelOpt->negLogLikelihoodGradient();
          value =  ptrModelOpt->negLogLikelihood();
          Eigen::VectorXd::Map(&gradient[0], gradient.size()) =gradient;
        }
        else
        {
          ptrModelOpt->eval(false);
          value =ptrModelOpt->negLogLikelihood();

        }
        return value;
      }

      nlopt::algorithm        m_algorithm;
      double                  m_ftolerance;
      double                  m_xreltolerance;
      double                  m_iterations;

    };//class GradientBasedKernelOptimizer

  }//namespace Optimization

}//namespace GaussianProcess 

#endif