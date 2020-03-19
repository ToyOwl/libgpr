#include <sstream>
#include <iterator>

#include <optimization/GradientBasedKernelOptimizer.hh>

#include <kernels/RationalQuadratic.hh>
#include <kernels/MaternThreeHalves.hh>
#include <kernels/MaternFiveHalves.hh>
#include <kernels/MaternThreeHalvesPeriodic.hh>
#include <kernels/RationalQuadratic.hh>
#include <kernels/RadialBasisKernel.hh>

#include <optimization/DirectBasedKernelOptimizer.hh>
#include <optimization/KernelOptimizer.hh>

#include <models/GaussianProcessRegressionModels.hh>

#include <cxxopts/cxxopts.hpp>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <memory>

using namespace GaussianProcess::Kernels;
using namespace GaussianProcess::Models;
using namespace GaussianProcess::Optimization;




void readCsvFile(const std::string& filename, const std::string& delim, std::vector<double>& samples, std::vector<double>& obseravations);

template<typename OptimizationData_T>
void optimize(std::unique_ptr<GaussianProcessRegression>&,  bool useGradientOptimizer, const OptimizationData_T& optData);

std::vector<double> upsampling(const std::vector<double>& samples, double factor);

void predictAndWrite(std::unique_ptr<GaussianProcessRegression>&, const std::vector<double>, const std::string&);


struct OptimizationData
{
  static double            maxiterations;
  static double            ftolerance;
  static double            xreltolerance;

  static nlopt::algorithm  gradient_aglorithm;
  static nlopt::algorithm  direct_algorithm;
};

double OptimizationData::maxiterations = 100.;
double OptimizationData::ftolerance    = -1.;
double OptimizationData::xreltolerance = -1.;
nlopt::algorithm OptimizationData::gradient_aglorithm = nlopt::LD_MMA;
nlopt::algorithm OptimizationData::direct_algorithm = nlopt::LN_COBYLA;


struct  KernelData
{
  static double   alpha;
  static double   t;
  static double   sf2;
  static double   l;
  static double   noise;
  static double   jitter;
  static bool     noise_opt;
};

double KernelData::alpha       =1.;
double KernelData::t           =1.;
double KernelData::sf2         =1.;
double KernelData::l           =1.;
double KernelData::noise       =0.1;
double KernelData::jitter      =1e-08;
bool KernelData::noise_opt     =true;

const std::vector<std::string> kernames = { "rbf", "matern32", "matern32p", "matern52", "rq" };

int  main(int argc, char** argv)
{
  cxxopts::Options cmdopt("libgpr","test program for libgpr library");
  cmdopt.add_options()
    ("i,input",      "csv input  file name",                          cxxopts::value<std::string>())
    ("o,output",     "csv output file name",                          cxxopts::value<std::string>()->default_value("out.data"))
    ("u,upsampling", "upsampling factor" ,                            cxxopts::value<double>()->default_value("10.0"))
    ("n,noise",      "{true, false } GPR model for noisy observations and  noise free observations",
                                                                      cxxopts::value<bool>()->default_value("true"))
    ("k,kernel",     "{rbf, matern32, matern32p, matern52, rq } covariance kernel function", 
                                                                      cxxopts::value<std::string>()->default_value("rbf"))
    ("t,optimization", "{gradient, direct } optimization solver type for the kernel parameters",
                                                                      cxxopts::value<std::string>()->default_value("direct"))
    ("m,max",     "optimization solver maximum  iteration numbers",   cxxopts::value<double>()->default_value("100."))
    ("a,alpha",   "Rational Quadratic kernel degree",                 cxxopts::value<double>()->default_value("1.0"))
    ("s,sigma",   "kernel bandwidth parameter",                       cxxopts::value<double>()->default_value("1.0"))
    ("l,length",  "kernel characteristic length scale",               cxxopts::value<double>()->default_value("1.0"))
    ("p,period",  "Matern 3/2 periodic period parameter",             cxxopts::value<double>()->default_value("1.0"))
    ("e,epsilon", "prior error level",                                cxxopts::value<double>()->default_value(".1"))
    ("j,jitter",  "jitter",                                           cxxopts::value<double>()->default_value("1e-08"))
    ("h,help", "Print usage");

  auto  result = cmdopt.parse(argc, argv);

  if (result.count("help"))
  {
    std::cout << cmdopt.help() << std::endl;
    exit(0);
  }

  std::string infile, outfile, kerneltype, optimizertype;
  if (result.count("input"))
    infile = result["input"].as<std::string>();

  
  
  const double upsfactor = result["upsampling"].as<double>();
  
  if (infile.empty())
  {
    std::cout << "input csv file name is empty" << std::endl;
    exit(0);
  }

  outfile          = result["output"].as<std::string>();
  kerneltype       = result["kernel"].as<std::string>();
  optimizertype    = result["optimization"].as<std::string>();
  outfile          = result["output"].as<std::string>();
  
  OptimizationData optimizationData;
  KernelData       kernelData;

  optimizationData.maxiterations
                    = result["max"].as<double>();
  kernelData.noise_opt 
                    = result["noise"].as<bool>();
  kernelData.noise
                    = result["epsilon"].as<double>();
  kernelData.jitter
                    = result["jitter"].as<double>();
  kernelData.l
                    = result["length"].as<double>();
  kernelData.t
                    = result["period"].as<double>();
  kernelData.alpha
                    = result["alpha"].as<double>();

  std::unique_ptr<GaussianProcessRegression>  ptrModel;
  std::vector<double> samples;
  std::vector<double> observations;
  std::vector<double> upsampled;
  try
  {
    readCsvFile(infile, " ", samples, observations);
  }
  catch (...)
  {
    std::cout << "can't read file " + infile << std::endl;
    exit(1);
  }
  upsampled = upsampling(samples, upsfactor);
  
  const bool useGradientOptimizer = (optimizertype == "gradient"? true: false);
  
  auto iter = std::find(kernames.begin(), kernames.end(), kerneltype);
  if (iter == kernames.end())
    kerneltype = "rbf";

  //Radial Basis Function Kernel
  if(kerneltype == "rbf")
  {
    RadialBasisKernel kernel(kernelData);
    ptrModel.reset(new FullGaussianProcessRegression<RadialBasisKernel>(kernel));
    ptrModel->compute(samples, observations);
    
    optimize(ptrModel, useGradientOptimizer, optimizationData);
    predictAndWrite(ptrModel, upsampled, outfile);
  }

  //Rational Quadratic
  else if (kerneltype == "rq") 
  {
    RationalQuadratic kernel(kernelData);
    ptrModel.reset(new FullGaussianProcessRegression<RationalQuadratic>(kernel));
    ptrModel->compute(samples, observations);

    optimize(ptrModel, useGradientOptimizer, optimizationData);
    predictAndWrite(ptrModel, upsampled, outfile);
  }

  //Matern 3/2 Kernel
  else if (kerneltype == "matern32")
  {
    MaternThreeHalves kernel(kernelData);
    ptrModel.reset(new FullGaussianProcessRegression<MaternThreeHalves>(kernel));
    ptrModel->compute(samples, observations);

    optimize(ptrModel, useGradientOptimizer, optimizationData);
    predictAndWrite(ptrModel, upsampled, outfile);
  }

 
  //Matern 5/2 Kernel
  else if (kerneltype == "matern52")
  {
    MaternFiveHalves kernel(kernelData);
    ptrModel.reset(new FullGaussianProcessRegression<MaternFiveHalves>(kernel));
    ptrModel->compute(samples, observations);

    optimize(ptrModel, useGradientOptimizer, optimizationData);
    predictAndWrite(ptrModel, upsampled, outfile);
  }

  //Matern pereodic 3/2 Kernel
  else if (kerneltype == "matern32p") 
  {
    MaternThreeHalvesPeriodic kernel(kernelData);
    ptrModel.reset(new FullGaussianProcessRegression<MaternThreeHalvesPeriodic>(kernel));
    ptrModel->compute(samples, observations);

    optimize(ptrModel, useGradientOptimizer, optimizationData);
    predictAndWrite(ptrModel, upsampled, outfile);
  }

  return 0;
}

void readCsvFile(const std::string& filename, const std::string& delim, std::vector<double>& samples, std::vector<double>& obseravations)
{
  std::ifstream csvfile(filename);
  std::string line = "";

  samples.clear();
  obseravations.clear();

  while (std::getline(csvfile, line))
  {

    std::istringstream iss(line);
    std::vector<std::string> splitted_line ( std::istream_iterator<std::string>{iss},
                      std::istream_iterator<std::string>() );
    double sample{ std::stod(splitted_line[0]) };
    double observation{ std::stod(splitted_line[1]) };
    samples.push_back(sample);
    obseravations.push_back(observation);
  }
  csvfile.close();
}

template<typename OptimizationData_T>
void optimize(std::unique_ptr<GaussianProcessRegression>& ptrGaussianModel,  bool useGradientOptimizer, const OptimizationData_T& optData)
{
  if (ptrGaussianModel != nullptr)
  {
    KernelOptimizer  kernelOptimizer;
    if (useGradientOptimizer)
    {
      GradientBasedKernelOptimizer  gradOptimizer(optData);
      kernelOptimizer(ptrGaussianModel.get(), gradOptimizer);
    }
    else
    {
      DirectBasedKernelOptimizer   directOptimizer(optData);
      kernelOptimizer(ptrGaussianModel.get(), directOptimizer);
    }
  }
}

std::vector<double> upsampling(const std::vector<double>& samples, double factor)
{
  std::vector<double> upsampled;
  if (samples.size() > 1)
  {
    std::size_t numElemets = samples.size();
    double minSample = samples[0];
    double maxSample = samples[numElemets - 1];
    double step = std::abs(maxSample - minSample) / (factor*numElemets);
    for (double sample{ minSample }; sample < maxSample; sample += step)
    {
      upsampled.push_back(sample);
    }
  }
  return upsampled;
}

void predictAndWrite(std::unique_ptr<GaussianProcessRegression>& ptrGaussianModel, const std::vector<double> samples, const std::string& fileName)
{
  if (ptrGaussianModel != nullptr && !fileName.empty() && !samples.empty())
  {
    std::ofstream  outFile(fileName);
    double  mean, variance;
    for (auto sample : samples)
    {
      ptrGaussianModel->predict(sample, mean, variance);
      outFile << sample << " " << mean << " " << variance << std::endl;
    }
    outFile.close();
  }
}