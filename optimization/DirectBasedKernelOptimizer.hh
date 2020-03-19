#ifndef _DIRECT_BASED_KERNEL_OPTIMIZER_HH
#define _DIRECT_BASED_KERNEL_OPTIMIZER_HH

#include "KernelOptimizer.hh"

#include <nlopt.hpp>

#include <Eigen/Core>

#include <iostream>
#include <type_traits>
#include <cstddef>
#include <limits>

namespace GaussianProcess 
{

  namespace Optimization 
  {

    struct DirectBasedKernelOptimizer
    {
      
      template<typename OptParameters_T>
      explicit DirectBasedKernelOptimizer(const OptParameters_T&)
      {
        init(OptParameters_T::direct_algorithm, OptParameters_T::maxiterations, OptParameters_T::ftolerance, OptParameters_T::xreltolerance);
      }

      explicit DirectBasedKernelOptimizer()
      {
        init();
      }

      void init( nlopt::algorithm algorithm_t = nlopt::LN_COBYLA, double maxiterations = 100.0, double ftolerance = -1.0, double xreltolerance = -1.0)
      {
        m_algorithm = algorithm_t;
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
      static double optimizationf(const std::vector<double>& params, std::vector<double>& gradient, void* ptrModel) 
      {
        GaussianProcess_T* ptrModelOpt = reinterpret_cast<GaussianProcess_T*>(ptrModel);
        Eigen::VectorXd  optparam = Eigen::VectorXd::Map(params.data(), params.size());
        ptrModelOpt->recompute(optparam);
        ptrModelOpt->eval(false);
        return  ptrModelOpt->negLogLikelihood();
      }

      nlopt::algorithm   m_algorithm;
      double             m_ftolerance;
      double             m_xreltolerance;
      double             m_iterations;

    };//class DirectBasedKernelOptimizer

  }//namespace Optimization

}//namespace GaussianProcess  

#endif