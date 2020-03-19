#ifndef _KERNEL_OPTIMIZER_HH
#define _KERNEL_OPTIMIZER_HH

#include <Eigen/Core>

#include <limits>
namespace GaussianProcess 
{
  namespace Optimization 
  {
    struct KernelOptimizer
    {

      template<typename GaussianProcess_T, typename WrapGaussianProcess_T>
      void operator()(GaussianProcess_T& gaussianProcessModel, WrapGaussianProcess_T& wrapper)
      {
        KernelOptImpl<GaussianProcess_T>  kernelOptimizer(gaussianProcessModel);
        wrapper(kernelOptimizer);
        gaussianProcessModel = kernelOptimizer.getModel();
        gaussianProcessModel.fullCovariance();
        setOptimizationData(kernelOptimizer);
      }

      template<typename GaussianProcess_T, typename WrapGaussianProcess_T>
      void operator()(GaussianProcess_T* ptrGaussianProcessModel, WrapGaussianProcess_T& wrapper) 
      {
        KernelOptImpl<GaussianProcess_T*>  kernelOptimizer(ptrGaussianProcessModel);
        wrapper(kernelOptimizer);
        ptrGaussianProcessModel->fullCovariance();
        setOptimizationData(kernelOptimizer);
      }

      Eigen::VectorXd getParameters() const 
      {
        return m_params;
      }

      Eigen::VectorXd getGradient() const
      {
        return m_gradient;
      }

      double getLikelihood() const
      {
        return m_likelihood;
      }

    private:

      template<typename KernelOptImpl_T>
      void setOptimizationData(KernelOptImpl_T&  opt_impl_t) 
      {
        m_gradient   =  opt_impl_t.negLogLikelihoodGradient();
        m_params     =  opt_impl_t.getKernelParameters();
        m_likelihood =  opt_impl_t.negLogLikelihood();
      }

      Eigen::VectorXd    m_gradient;
      Eigen::VectorXd    m_params;
      double             m_likelihood;

      template<typename WrapGaussianProcess_T>
      struct KernelOptImpl 
      {
        explicit KernelOptImpl(const WrapGaussianProcess_T& gpr) : m_gpr{ gpr }, m_gradient{ Eigen::VectorXd::Ones(gpr.m_kernel.numOptParameters())*std::numeric_limits<double>::infinity() },
          m_likelihood{ -1.0*std::numeric_limits<double>::infinity() }
          {}

        void eval(bool eval_gradient = true)
        {
          if (eval_gradient) 
          {
            m_gradient =  m_gpr.negLogLikelihoodGrad();
          }
          m_gpr.negLogLikelihood(m_likelihood);
        }

        void recompute(const Eigen::VectorXd& params) 
        {
          m_gpr.m_kernel.setParameters(params);
          m_gpr.fullCovariance();
        }
     
        Eigen::VectorXd negLogLikelihoodGradient() const
        {
          return m_gradient;
        }
      
        double negLogLikelihood() const 
        {
          return m_likelihood;
        }

        Eigen::VectorXd getKernelParameters() const 
        {
          return m_gpr.m_kernel.getParameters();
        }

        std::size_t numKernelParameters() const 
        {
          return m_gpr.m_kernel.numOptParameters();
        }

        WrapGaussianProcess_T getModel() const
        {
          return m_gpr;
        }
                
      private:

        WrapGaussianProcess_T   m_gpr;
        Eigen::VectorXd         m_gradient;              
        double                  m_likelihood;
      
      };//class KernelOptImpl<GaussianProcess_T>

      template<typename WrapGaussianProcess_T>
      struct KernelOptImpl<WrapGaussianProcess_T*>
      {
        explicit KernelOptImpl(WrapGaussianProcess_T* ptr_gp) : m_ptrGPR{ ptr_gp }, m_gradient{ Eigen::VectorXd::Ones(ptr_gp->numKernelParameters())*std::numeric_limits<double>::infinity() },
          m_likelihood{ -1.0*std::numeric_limits<double>::infinity() }
          {}

        void eval(bool eval_gradient = true) 
        {
          if (eval_gradient) 
          {         
            m_gradient = m_ptrGPR->negLogLikelihoodGrad();
          }
          m_ptrGPR->negLogLikelihood(m_likelihood);
        }

        void recompute(const Eigen::VectorXd& params) 
        {
          m_ptrGPR->setKernelParameters(params);
          m_ptrGPR->fullCovariance();
        }

        Eigen::VectorXd negLogLikelihoodGradient()  const
        {
          return m_gradient;
        }

        double negLogLikelihood() const 
        {
           return m_likelihood;
        }

        Eigen::VectorXd  getKernelParameters() const
        {
          return m_ptrGPR->getKernelParameters();
        }

        std::size_t numKernelParameters() const
        {
          return m_ptrGPR->numKernelParameters();
        }

      private:
        WrapGaussianProcess_T*     m_ptrGPR;
        Eigen::VectorXd            m_gradient;
        double                     m_likelihood;

      };//class KernelOptImpl<GaussianProcess_T*>
    
    };//class KernelOptimizator

  }//namespace optimization 

}//namespace GaussianProcess 
#endif
