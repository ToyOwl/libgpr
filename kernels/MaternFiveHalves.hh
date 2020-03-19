#ifndef _MATERN_FIVE_HALVES_HH
#define _MATERN_FIVE_HALVES_HH

#include <kernels/KernelBase.hh>

#include <cmath>

namespace GaussianProcess 
{

  namespace Kernels
  {

    struct MaternFiveHalves : public KernelBase<MaternFiveHalves>
    {
      friend class KernelBase<MaternFiveHalves>;

      explicit MaternFiveHalves(double sf2, double l, double noise, bool noiseOpt= true) : KernelBase<MaternFiveHalves>(noise, noiseOpt), m_sf2{ std::sqrt(sf2) }, m_l{ l }, m_sqrt5{ sqrt(5.0) }, m_numOptParams{ 2 }
      { }

      template<typename KernelParams_T>
      explicit MaternFiveHalves(const KernelParams_T& params) : KernelBase<MaternFiveHalves>(params), m_sf2{ std::sqrt(params.sf2) }, m_l{ params.l }, m_sqrt5{ sqrt(5.0) }, m_numOptParams{ 2 }
      { }

      MaternFiveHalves(const MaternFiveHalves&)             =default;

      MaternFiveHalves(MaternFiveHalves&&)                  =default;

      MaternFiveHalves&  operator=(const MaternFiveHalves&) =default;

      MaternFiveHalves&  operator=(MaternFiveHalves&&)      =default;

    private:

      double cov(double x1,  double x2) const
      {
        double z = std::abs((x1 - x2)*m_sqrt5 / m_l);
        return m_sf2 * exp(-z)*(1 + z+ std::pow(z, 2.0)/3.0);
      }

      Eigen::VectorXd  grad(double x1, double x2) const
      {
        Eigen::VectorXd gradient(m_numOptParams);
        double z = std::abs((x1 - x2)*m_sqrt5 / m_l);
        double k = m_sf2 * exp(-z);
        gradient[0] = k * (std::pow(z, 2.0) + std::pow(z, 3.0))/3.0;
        gradient[1] = 2.0*k*(1 + z + std::pow(z, 2.0)/3.0);
        return gradient;
      }

      void  setParams(const Eigen::VectorXd& param)
      {
        m_l = exp(param[0]);
        m_sf2 = exp(2.0*param[1]);
      }

      Eigen::VectorXd  getParams() const
      {
        Eigen::VectorXd  parameters(m_numOptParams);
        parameters[0] = std::log(m_l);
        parameters[1] = std::log(std::sqrt(m_sf2));
        return parameters;
      }

      std::size_t numOptParams()  const
      {
        return m_numOptParams;
      }

      std::size_t  m_numOptParams;
      double       m_sf2;
      double       m_l;
      double       m_sqrt5;
    
    };//class MaternFiveHalves
  
  }//namespace Kernels

}//namespace GaussianProcess 
#endif