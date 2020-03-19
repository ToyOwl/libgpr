#ifndef _GAUSSIAN_PROCESSES_REGRESSION_HH
#define _GAUSSIAN_PROCESSES_REGRESSION_HH

#include <kernels/KernelBase.hh>
#include <optimization/KernelOptimizer.hh>

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <limits>
#include <utility>
#include <vector>

#include <memory>
#define _USE_MATH_DEFINES
#include <math.h>


namespace GaussianProcess
{

  namespace Models 
  {

    struct GaussianProcessRegression 
    {

      friend class Optimization::KernelOptimizer;
            
      virtual void compute(const std::vector<double>& samples, const std::vector<double>& observations)  =0;
         
      virtual void addSample(double sample,  double observation)                                         =0;

      virtual void predict(double sample, double& mean, double& variance)                                =0;
    
      virtual std::vector<double> samples()                 const                                        =0;
          
      virtual std::vector<double> observations()            const                                        =0;

      virtual std::size_t numKernelParameters()             const                                        =0;

      virtual Eigen::VectorXd getKernelParameters()         const                                        =0;

    private:

      virtual void  setKernelParameters(const Eigen::VectorXd&)                                          =0;

      virtual void  fullCovariance()                                                                     =0;

      virtual void negLogLikelihood(double& refLikelihood)  const                                        =0;
            
      virtual Eigen::VectorXd negLogLikelihoodGrad()        const                                        =0;     
        
    };//class GaussianProcessRegression
        
    
    template <typename Kernel_T>
    struct FullGaussianProcessRegression: public GaussianProcessRegression 
    {
           
      friend class Optimization::KernelOptimizer;
           
      explicit FullGaussianProcessRegression(const Kernel_T& refKernel) : m_kernel{ refKernel }
      { }
           
      explicit FullGaussianProcessRegression(const Kernel_T& refKernel, const std::vector<Eigen::VectorXd>& samples, const std::vector<Eigen::VectorXd>& observations) :m_kernel{ refKernel }
      { }

      FullGaussianProcessRegression(const FullGaussianProcessRegression&)                    = default;

      FullGaussianProcessRegression(FullGaussianProcessRegression&&)                         = default;

      FullGaussianProcessRegression& operator=(const FullGaussianProcessRegression&)         = default;
                     
      FullGaussianProcessRegression& operator=(FullGaussianProcessRegression&&)              = default;

      void compute(const std::vector<double>& samples, const std::vector<double>& observations) 
      {
        m_samples       = Eigen::VectorXd::Map(samples.data(), samples.size());
        m_observations  = Eigen::VectorXd::Map(observations.data(), observations.size());
        demeanObservations();
        fullCovariance();
      }
           
      std::vector<double> samples()  const
      {
        std::vector<double> outsamples(m_samples.size());
        Eigen::VectorXd::Map(&outsamples[0], m_samples.size())  = m_samples;
        return outsamples;
      }

      std::vector<double> observations()  const
      {
        std::vector <double> outobservations(m_observations.size());
        Eigen::VectorXd::Map(&outobservations[0], m_observations.size()) = m_observations;
        return outobservations;
      }

      std::size_t numKernelParameters()             const
      {
        return  m_kernel.numOptParameters();
      }

      Eigen::VectorXd getKernelParameters() const
      {
        return m_kernel.getParameters();
      }

      void addSample(double sample, double observation) override
      {
        m_samples.conservativeResize(m_samples.rows() +1);
        m_samples(m_samples.rows() - 1) = sample;
        m_observations.conservativeResize(m_observations.rows() + 1);
        m_observations(m_observations.rows()-1) = observation;
        demeanObservations();
        incrementalCovariance();               
      }
           
      void predict(double sample, double& mean, double& variance) override
      {
        if (m_observations.size() == 0) 
        {
          mean = sample;
          variance = m_kernel(sample, sample);       
        }
        else 
        {
          Eigen::VectorXd  k, v;                 
          kStar(sample, k);
          v = m_L.triangularView<Eigen::Lower>().solve(k);
          mean = (k.transpose()*m_alpha)(0);
          mean += m_meanobs;
          double d_variance = m_kernel(sample, sample) - v.transpose()*v; 
          variance = d_variance < std::numeric_limits<double>::epsilon() ? 0.0 : d_variance;
          variance = d_variance +m_kernel.noise();
        }
      }

    private:
               
      void fullCovariance()
      {
        std::size_t nsamples  = m_samples.size();
        m_covariance.resize(nsamples, nsamples);

        for (std::size_t idx{ 0 }; idx < nsamples; idx++) 
        {
          for (std::size_t jdx{ 0 }; jdx <= idx; ++jdx) 
          {
            m_covariance(idx, jdx) =  m_kernel(m_samples[idx], m_samples[jdx], idx, jdx);
          }
        }
        for (std::size_t idx{ 0 }; idx < nsamples; idx++) 
        {
          for (std::size_t jdx{ 0 }; jdx < idx;  ++jdx)
          {
            m_covariance(jdx, idx) =  m_covariance(idx, jdx);
          }
        }
        m_L  = Eigen::LLT<Eigen::MatrixXd>(m_covariance).matrixL();
        evalAlpha();
      }

      void incrementalCovariance()
      {
        std::size_t    nsamples { static_cast<std::size_t>(m_samples.size()) };
        double l_idx;

        m_covariance.conservativeResize(nsamples, nsamples);
        m_L.conservativeResizeLike(Eigen::MatrixXd::Zero(nsamples, nsamples));
                    
        for (size_t idx{ 0 }; idx < nsamples; idx++) 
        {
          m_covariance(idx, nsamples - 1) = m_kernel(m_samples[idx], m_samples[nsamples - 1], idx, nsamples - 1);
          m_covariance(nsamples - 1, idx) = m_covariance(idx, nsamples - 1);
        }
                    
        for (std::size_t idx{ 0 }; idx < nsamples -1; idx++) 
        {
          l_idx = m_covariance(nsamples - 1, idx) - (m_L.block(idx, 0, 1, idx)*m_L.block(nsamples - 1, 0, 1, idx).transpose())(0, 0);
          m_L(nsamples - 1, idx) = l_idx / m_L(idx, idx);
        }
        l_idx = m_covariance(nsamples - 1, nsamples -1) - (m_L.block(nsamples - 1, 0, 1, nsamples - 1)*m_L.block(nsamples - 1, 0, 1, nsamples - 1).transpose())(0, 0);
        m_L(nsamples - 1,  nsamples -1) = l_idx / sqrt(l_idx);
        evalAlpha();
      }

      void evalAlpha()
      {               
        Eigen::TriangularView<Eigen::MatrixXd, Eigen::Lower>  triangL  = m_L.triangularView<Eigen::Lower>();
        m_alpha = triangL.solve(m_dobservations);
        triangL.adjoint().solveInPlace(m_alpha);      
      }

      void kStar(double testSample, Eigen::VectorXd&  refCovariance)
      {
        Eigen::VectorXd outCov(m_samples.size());
        for (std::size_t idx{0}; idx < m_samples.size(); idx++)
        {
          outCov[idx] = m_kernel(testSample, m_samples[idx]);
        }
        refCovariance= outCov;
      }

      void demeanObservations()
      {
        m_meanobs = m_observations.mean();
        m_dobservations = m_observations.array() - m_meanobs;
      }

    void negLogLikelihood(double& outLikelihood) const
    {
      outLikelihood = -1.*(m_L.diagonal().array().log().sum()) - .5* (m_dobservations.transpose()*m_alpha).trace() -
                        .5*(m_dobservations.rows() * std::log(2.0*M_PI));
    }
                
    Eigen::VectorXd negLogLikelihoodGrad() const 
    {
      std::size_t  nsamples{(std::size_t) m_dobservations.rows() };
                    
      Eigen::MatrixXd invcovar = Eigen::MatrixXd::Identity(nsamples, nsamples);
      Eigen::VectorXd gradient = Eigen::VectorXd::Zero(m_kernel.numOptParameters());

      m_L.triangularView<Eigen::Lower>().solveInPlace(invcovar);
      m_L.triangularView<Eigen::Lower>().transpose().solveInPlace(invcovar);
                    
      Eigen::MatrixXd w = m_alpha * m_alpha.transpose() - invcovar;

      for (std::size_t idx{ 0 }; idx < nsamples; idx++) 
      {
        for (std::size_t jdx{ 0 }; jdx <= idx; jdx++) 
        {
          Eigen::VectorXd grad = m_kernel.gradient(m_samples[idx], m_samples[jdx], idx, jdx);
          if (idx != jdx)
            gradient += w(idx, jdx)*grad;
          else
            gradient += 0.5*w(idx, jdx)*grad;
        }    
      }
      return -1.0*gradient;
    }

    virtual void setKernelParameters(const Eigen::VectorXd& params) 
    {
      m_kernel.setParameters(params);
    };
              

     
      Eigen::VectorXd        m_samples;
      Eigen::VectorXd        m_observations;
      Eigen::VectorXd        m_dobservations;
      Eigen::VectorXd        m_alpha;
      Eigen::MatrixXd        m_L, m_covariance;

      Kernel_T m_kernel;

      double                 m_meanobs;

    };//class FullGaussianProcessRegression
       
  }//namespace Models

}//namespace GaussianProcess 

#endif
