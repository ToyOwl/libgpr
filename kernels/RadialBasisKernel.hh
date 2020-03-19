#ifndef _RADIAL_BASIS_HH
#define _RADIAL_BASIS_HH

#include <kernels/KernelBase.hh>

#include <cmath>

namespace GaussianProcess 
{
    
  namespace Kernels 
  {
            
    struct RadialBasisKernel : public KernelBase<RadialBasisKernel>
    {
      friend class KernelBase<RadialBasisKernel>;

      explicit RadialBasisKernel(double sf2, double l, double noise, bool noiseOpt = true) : KernelBase<RadialBasisKernel>(noise, noiseOpt), m_sf2{std::sqrt(sf2)}, m_l{l }, m_numOptParams{2}
      {  }

      template<typename KernelParams_T>
      explicit RadialBasisKernel(const KernelParams_T& params) : KernelBase<RadialBasisKernel>(params), m_sf2{ std::sqrt(params.sf2) }, m_l{ params.l }, m_numOptParams{ 2 }
      { }

      RadialBasisKernel(const RadialBasisKernel&)               =default;

      RadialBasisKernel(RadialBasisKernel&&)                    =default;

      RadialBasisKernel&  operator=(const RadialBasisKernel&)   =default;
            
      RadialBasisKernel&  operator=(RadialBasisKernel&&)        =default;

    private:

      double cov(double x1, double x2) const
      {
        double z {(x1-x2)*(x1-x2)/(m_l*m_l)};
        return m_sf2 * exp(-0.5*z);
      }
            
      Eigen::VectorXd grad(double x1, double x2) const
      {
        Eigen::VectorXd gradient(m_numOptParams);
        double term = (x1 - x2)*(x1 - x2) /(m_l*m_l);
        double k    = m_sf2 * exp(-.5*term);
        gradient[0] = k * term;
        gradient[1] = 2.0*k;
        return gradient;
      }

      void setParams(const Eigen::VectorXd& param)
      {
         m_l   = exp(param[0]);
         m_sf2 = exp(2.0*param[1]);
      }

      Eigen::VectorXd getParams() const
      {
        Eigen::VectorXd  parameters(m_numOptParams);
        parameters[0] = std::log(m_l);
        parameters[1] =std::log(std::sqrt(m_sf2));
        return parameters;
      }

      std::size_t numOptParams()  const
      {
         return m_numOptParams;
      }

      std::size_t   m_numOptParams;
      double        m_sf2;
      double        m_l;

    };//class RadialBasisKernel
  
  }//namespace kernels

}//namespace gaussian_process
#endif
