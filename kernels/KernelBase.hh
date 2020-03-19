#ifndef _KERNEL_BASE_HH
#define _KERNEL_BASE_HH

#include <Eigen/Core>

namespace GaussianProcess {

  namespace Kernels {

    template<typename Kernel_T>
    struct KernelBase {
            
       explicit KernelBase(double noise,  bool noiseOpt, double jitter = 1e-08) : m_noise{noise},  m_jitter{ jitter }, m_noiseOpt{ noiseOpt }
       { }
            
       template <typename KernelParams_T>
       explicit  KernelBase(const KernelParams_T& params) :m_noise{  params.noise }, m_jitter{ params.jitter },  m_noiseOpt{params.noise_opt}
       { }
                
       KernelBase(const KernelBase& other) = default;
            
       KernelBase(KernelBase&&) = default;
            
       KernelBase& operator=(const KernelBase&) = default;
            
       KernelBase& operator=(KernelBase&&)      = default;

       double operator() (double x1, double x2, std::size_t idx1, std::size_t idx2) const 
       {
          double variance { static_cast<const Kernel_T*>(this)->cov(x1,x2)};
          variance += ((idx1 == idx2) ? (m_noise + m_jitter) : 0.0);
          return variance;
       }
            
       double operator()(double x1, double x2) const
       {
          return static_cast<const Kernel_T*>(this)->cov(x1, x2);
       }

       Eigen::VectorXd  gradient(double x1, double x2, std::size_t idx1, std::size_t idx2) const
       {
          Eigen::VectorXd grad = static_cast<const Kernel_T*>(this)->grad(x1, x2);
          if (m_noiseOpt)
          {
            grad.conservativeResize(static_cast<const Kernel_T*>(this)->numOptParams() + 1);
            grad[static_cast<const Kernel_T*>(this)->numOptParams()] = 2.0*((idx1 == idx2) ? 2.0*m_noise : 0.0);
          }
          return grad;
       }
            
       void setParameters(const Eigen::VectorXd& param)
       {
          if (m_noiseOpt) 
          {
            static_cast<Kernel_T*>(this)->setParams(param.head(static_cast<const Kernel_T*>(this)->numOptParams()));
            m_noise = std::exp(2.0*param[static_cast<const Kernel_T*>(this)->numOptParams()]);
            return;
          }
            static_cast<Kernel_T*>(this)->setParams(param);
       }

       Eigen::VectorXd getParameters() const
       {
          Eigen::VectorXd  kernelParams = static_cast<const Kernel_T*>(this)->getParams();
          if (m_noiseOpt) 
          {
            kernelParams.conservativeResize(static_cast<const Kernel_T*>(this)->numOptParams() + 1);
            kernelParams[static_cast<const Kernel_T*>(this)->numOptParams()] = std::log(std::sqrt(m_noise));
          }
          return kernelParams;
       }

       std::size_t numOptParameters() const
       {
          return static_cast<const Kernel_T*>(this)->numOptParams() + (m_noiseOpt? 1 : 0);
       }

       double  noise() const
       {
          return m_noise;
       }
  
       double       m_noise;
       double       m_jitter;
       bool         m_noiseOpt;
    
    };//class KernelBase
   
  }//namespace kernels

}//namespace GaussianProcess 


#endif