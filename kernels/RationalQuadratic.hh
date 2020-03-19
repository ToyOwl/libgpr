#ifndef _RATIONAL_QUADRATIC_HH
#define _RATIONAL_QUADRATIC_HH

#include <kernels/KernelBase.hh>

#include <cmath>

namespace GaussianProcess 
{

  namespace Kernels 
  {

    struct RationalQuadratic : public KernelBase<RationalQuadratic>
    {
      friend class KernelBase<RationalQuadratic>;

      explicit RationalQuadratic(double sf2, double l, double alpha, double noise, bool noiseOpt=true) : KernelBase<RationalQuadratic>(noise, noiseOpt), m_sf2{ std::sqrt(sf2) }, m_l{ l }, m_alpha{alpha}, m_numOptParams{ 3 }
      {  }

      template<typename KernelParams_T>
      explicit RationalQuadratic(const KernelParams_T& params) : KernelBase<RationalQuadratic>(params), m_sf2{ std::sqrt(params.sf2) }, m_l{ params.l }, m_alpha{params.alpha}, m_numOptParams{ 3 }
      {  }

      RationalQuadratic(const RationalQuadratic&)             =default;
           
      RationalQuadratic(RationalQuadratic&&)                  =default;

      RationalQuadratic& operator=(const RationalQuadratic&)  =default;
            
      RationalQuadratic& operator=(RationalQuadratic&&)       =default;

    private:

      double cov(double x1, double x2) const
      {
        double z{ .5*(x1 - x2)*(x1 - x2) /(m_l*m_l)};
        return m_sf2 * std::pow((1.0 + z/m_alpha), -m_alpha);
      }

      Eigen::VectorXd grad(double x1, double x2) const
      {
        Eigen::VectorXd gradient(m_numOptParams);
        double z     = .5*(x1 - x2)*(x1 - x2) / (m_l*m_l);
        double k     = 1 + 0.5*z / m_alpha;
        double sf2_k = m_sf2 *std::pow(k, -m_alpha);
        gradient[0]  = m_sf2 * z*std::pow(k, -m_alpha - 1);
        gradient[1]  = 2.0* sf2_k;
        gradient[2]  = sf2_k*(0.5*z / k - m_alpha * std::log(k));
        return gradient;
      }

      void setParams(const Eigen::VectorXd& param)
      {
         m_l     = std::exp(param[0]);
         m_sf2   = std::exp(2.0*param[1]);
         m_alpha = std::exp(param[2]);
      }

      Eigen::VectorXd getParams() const
      {
        Eigen::VectorXd  parameters(m_numOptParams);
        parameters[0] = std::log(m_l);
        parameters[1] = std::log(std::sqrt(m_sf2));
        parameters[2] = std::log(m_alpha);
        return parameters;
      }

      std::size_t numOptParams() const
      {
        return m_numOptParams;
      }
       
      std::size_t  m_numOptParams;
      double       m_alpha;
      double       m_sf2;
      double       m_l;
        
    };//class RationalQuadratic
   
  }//namespace kernels

}//namespace GaussianProcess 
#endif

