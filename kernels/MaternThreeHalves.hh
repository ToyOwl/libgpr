#ifndef _MATERN_THREE_HALVES_HH
#define _MATERN_THREE_HALVES_HH

#include <kernels/KernelBase.hh>

#include <cmath>

namespace GaussianProcess 
{

  namespace Kernels {

    struct MaternThreeHalves : public KernelBase<MaternThreeHalves>
    {
      
      friend class KernelBase<MaternThreeHalves>;

      explicit MaternThreeHalves(double sf2, double l, double noise, bool noiseOpt=true) : KernelBase<MaternThreeHalves>(noise, noiseOpt), m_sf2{ std::sqrt(sf2) }, m_l{ l }, m_sqrt3{ sqrt(3.0) }, m_numOptParams{ 2 }
      { }

      template<typename KernelParams_T>
      explicit MaternThreeHalves(const KernelParams_T& params) : KernelBase<MaternThreeHalves>(params), m_sf2{ std::sqrt(params.sf2) }, m_l{ params.l }, m_sqrt3{ sqrt(3.0) }, m_numOptParams{ 2 }
      { }

      MaternThreeHalves(const MaternThreeHalves&)             =default;
      
      MaternThreeHalves(MaternThreeHalves&&)                  =default;

      MaternThreeHalves&  operator=(const MaternThreeHalves&) =default;
      
      MaternThreeHalves&  operator=(MaternThreeHalves&&)      =default;

    private:

      double cov(double x1, double x2) const
      {
        double z = std::abs((x1 - x2)*m_sqrt3 / m_l);
        return m_sf2 * exp(-z)*(1 + z);
      }

      Eigen::VectorXd grad(double x1, double x2) const
      {
        Eigen::VectorXd gradient(m_numOptParams);
        double z = std::abs((x1 - x2)*m_sqrt3 / m_l);
        double _k = m_sf2 * exp(-z);
        gradient[0] = _k * z * z;
        gradient[1] = 2.0*_k*(1 + z);
        return gradient;
      }

      void setParams(const Eigen::VectorXd& param)
      {
        m_l = exp(param[0]);
        m_sf2 = exp(2.0*param[1]);
      }

      Eigen::VectorXd getParams() const
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
      double       m_sqrt3;

    };//class MaternThreeHalves
  
  }//namespace kernels

}//GaussianProcess 
#endif
