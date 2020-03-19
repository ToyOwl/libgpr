#ifndef _MATERN_THREE_HALVES_PERIODIC_HH
#define _MATERN_THREE_HALVES_PERIODIC_HH

#include <kernels/KernelBase.hh>

#define _USE_MATH_DEFINES
#include <math.h>

#include <Eigen/Core>

namespace GaussianProcess 
{

  namespace Kernels 
  {
    struct MaternThreeHalvesPeriodic : public KernelBase<MaternThreeHalvesPeriodic>
    {
      friend class KernelBase<MaternThreeHalvesPeriodic>;

      explicit MaternThreeHalvesPeriodic(double sf2, double l, double t, double noise,  bool noiseOpt=true) : KernelBase<MaternThreeHalvesPeriodic>(noise, noiseOpt), m_sf2{ std::sqrt(sf2) }, m_l{ l }, m_t{ t },
        m_sqrt3{ sqrt(3.0) },  m_numOptParams{ 3 }
      { }

      template<typename KernelParams_T>
      explicit MaternThreeHalvesPeriodic(const KernelParams_T& params) : KernelBase<MaternThreeHalvesPeriodic>(params), m_sf2{ std::sqrt(params.sf2) }, m_l{ params.l }, m_t{params.t },
        m_sqrt3{ sqrt(3.0) }, m_numOptParams{ 3 }
      { }

      MaternThreeHalvesPeriodic(const MaternThreeHalvesPeriodic&) = default;
      
      MaternThreeHalvesPeriodic(MaternThreeHalvesPeriodic&&) = default;

      MaternThreeHalvesPeriodic&  operator=(const MaternThreeHalvesPeriodic&) = default;
      
      MaternThreeHalvesPeriodic&  operator=(MaternThreeHalvesPeriodic&&) = default;

    private:
      double cov(double x1, double x2) const
      {
        double _z = m_sqrt3*std::abs(std::sin(M_PI*std::abs((x1 -x2))/m_t)/m_l);
        return m_sf2 * (1 + _z)* std::exp(-_z);
      }

      Eigen::VectorXd grad(double x1, double x2) const
      {
        Eigen::VectorXd gradient(m_numOptParams);
        double _z = M_PI*std::abs((x1 -x2))/m_t;
        double _k = m_sqrt3*std::abs(std::sin(_z)/m_l);
        gradient[0] = m_sf2*std::pow(_k, 2.0)*std::exp(-_k);
        gradient[1] = 2.0*m_sf2*(1+_k)*std::exp(-_k);
        gradient[2] = m_sqrt3 *m_sf2*_k*std::exp(-_k)*std::cos(_z) / (m_l*m_t);
        return gradient;
      }


      void setParams(const Eigen::VectorXd& param)
      {
        m_l   = std::exp(param[0]);
        m_sf2 = std::exp(2.0*param[1]);
        m_t =   std::exp(param[2]);
      }

      Eigen::VectorXd getParams() const
      {
        Eigen::VectorXd  parameters(m_numOptParams);
        parameters[0] = std::log(m_l);
        parameters[1] = std::log(std::sqrt(m_sf2));
        parameters[2] = std::log(m_t);
        return parameters;
      }

     std::size_t numOptParams()  const
     {
        return m_numOptParams;
     }
      
     std::size_t  m_numOptParams;
     double m_sf2;
     double m_l;
     double m_t;
     double m_sqrt3;
        
    };//class  MaternThreeHalvesPeriodic
  
  }//namespace Kernels

}//namespace GaussianProcess
#endif
