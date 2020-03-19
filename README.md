## Simple header-only library of 1-D Full Gaussian Process Regression model.
![f(\mathbf{x})\sim \mathscr{GP}\left(\mathit{0},\Kappa\left(\mathbf{x},\mathbf{x}^'\right)\right)](https://render.githubusercontent.com/render/math?math=f(%5Cmathbf%7Bx%7D)%5Csim%20%5Cmathscr%7BGP%7D%5Cleft(%5Cmathit%7B0%7D%2C%5CKappa%5Cleft(%5Cmathbf%7Bx%7D%2C%5Cmathbf%7Bx%7D%5E'%5Cright)%5Cright))
## Features:
- predictions with noise-free observations 
- predictions with  noisy observations
- estimation the kernels parameters using likelihood maximization
## Implemented kernels:
#### Radial Basis Function
 ![\Kappa\left(\mathbf{x,}\mathbf{x}^'\right)=\sigma_f^2\exp\left(-\frac{1}{2\ell^2}\left\|\mathbf{x}-\mathbf{x}^'\right\|^2\right)](https://render.githubusercontent.com/render/math?math=%5CKappa%5Cleft(%5Cmathbf%7Bx%2C%7D%5Cmathbf%7Bx%7D%5E'%5Cright)%3D%5Csigma_f%5E2%5Cexp%5Cleft(-%5Cfrac%7B1%7D%7B2%5Cell%5E2%7D%5Cleft%5C%7C%5Cmathbf%7Bx%7D-%5Cmathbf%7Bx%7D%5E'%5Cright%5C%7C%5E2%5Cright))
#### Matern 3/2
 ![\Kappa_{\nu=3/2}\left(\mathbf{x,}\mathbf{x}^'\right)=\sigma_f^2\left(1+\frac{\sqrt{3}\left\|\mathbf{x}-\mathbf{x}^'\right\|}{\ell}\right)\exp\left(-\frac{\sqrt{3}\left\|\mathbf{x}-\mathbf{x}^'\right\|}{\ell}\right)](https://render.githubusercontent.com/render/math?math=%5CKappa_%7B%5Cnu%3D3%2F2%7D%5Cleft(%5Cmathbf%7Bx%2C%7D%5Cmathbf%7Bx%7D%5E'%5Cright)%3D%5Csigma_f%5E2%5Cleft(1%2B%5Cfrac%7B%5Csqrt%7B3%7D%5Cleft%5C%7C%5Cmathbf%7Bx%7D-%5Cmathbf%7Bx%7D%5E'%5Cright%5C%7C%7D%7B%5Cell%7D%5Cright)%5Cexp%5Cleft(-%5Cfrac%7B%5Csqrt%7B3%7D%5Cleft%5C%7C%5Cmathbf%7Bx%7D-%5Cmathbf%7Bx%7D%5E'%5Cright%5C%7C%7D%7B%5Cell%7D%5Cright))
#### Matern 3/2 periodic, based on realisation https://github.com/mblum/libgp
 ![\Kappa_{\nu=3/2p}\left(\mathbf{x,}\mathbf{x}^'\right)=\sigma_f^2\left(1+\frac{\sqrt{3}}{\ell}\Bigg|\sin\left(\frac{\pi\left\|\mathbf{x}-\mathbf{x}^'\right\|}{\mathit{P}}\right)\Bigg|\right)\exp\left(-\frac{\sqrt{3}}{\ell}\Bigg|\sin\left(\frac{\pi\left\|\mathbf{x}-\mathbf{x}^'\right\|}{\mathit{P}}\right)\Bigg|\right)](https://render.githubusercontent.com/render/math?math=%5CKappa_%7B%5Cnu%3D3%2F2p%7D%5Cleft(%5Cmathbf%7Bx%2C%7D%5Cmathbf%7Bx%7D%5E'%5Cright)%3D%5Csigma_f%5E2%5Cleft(1%2B%5Cfrac%7B%5Csqrt%7B3%7D%7D%7B%5Cell%7D%5CBigg%7C%5Csin%5Cleft(%5Cfrac%7B%5Cpi%5Cleft%5C%7C%5Cmathbf%7Bx%7D-%5Cmathbf%7Bx%7D%5E'%5Cright%5C%7C%7D%7B%5Cmathit%7BP%7D%7D%5Cright)%5CBigg%7C%5Cright)%5Cexp%5Cleft(-%5Cfrac%7B%5Csqrt%7B3%7D%7D%7B%5Cell%7D%5CBigg%7C%5Csin%5Cleft(%5Cfrac%7B%5Cpi%5Cleft%5C%7C%5Cmathbf%7Bx%7D-%5Cmathbf%7Bx%7D%5E'%5Cright%5C%7C%7D%7B%5Cmathit%7BP%7D%7D%5Cright)%5CBigg%7C%5Cright))
#### Matern 5/2 
 ![\Kappa_{\nu=5/2}\left(\mathbf{x,}\mathbf{x}^'\right)=\sigma_f^2\left(1+\frac{\sqrt{5}\left\|\mathbf{x}-\mathbf{x}^'\right\|}{\ell}+\frac{5\left\|\mathbf{x}-\mathbf{x}^'\right\|^2}{3\ell^2}\right)\exp\left(-\frac{\sqrt{5}\left\|\mathbf{x}-\mathbf{x}^'\right\|}{\ell}\right)](https://render.githubusercontent.com/render/math?math=%5CKappa_%7B%5Cnu%3D5%2F2%7D%5Cleft(%5Cmathbf%7Bx%2C%7D%5Cmathbf%7Bx%7D%5E'%5Cright)%3D%5Csigma_f%5E2%5Cleft(1%2B%5Cfrac%7B%5Csqrt%7B5%7D%5Cleft%5C%7C%5Cmathbf%7Bx%7D-%5Cmathbf%7Bx%7D%5E'%5Cright%5C%7C%7D%7B%5Cell%7D%2B%5Cfrac%7B5%5Cleft%5C%7C%5Cmathbf%7Bx%7D-%5Cmathbf%7Bx%7D%5E'%5Cright%5C%7C%5E2%7D%7B3%5Cell%5E2%7D%5Cright)%5Cexp%5Cleft(-%5Cfrac%7B%5Csqrt%7B5%7D%5Cleft%5C%7C%5Cmathbf%7Bx%7D-%5Cmathbf%7Bx%7D%5E'%5Cright%5C%7C%7D%7B%5Cell%7D%5Cright))
#### Rational Quadratic 
 ![\Kappa_{\mathit{RQ}}\left(\mathbf{x,}\mathbf{x}^'\right)=\sigma_f^2\left(1+\frac{\left\|\mathbf{x}-\mathbf{x}^'\right\|^{2}}{2 \alpha \ell^{2}}\right)^{-\alpha}](https://render.githubusercontent.com/render/math?math=%5CKappa_%7B%5Cmathit%7BRQ%7D%7D%5Cleft(%5Cmathbf%7Bx%2C%7D%5Cmathbf%7Bx%7D%5E'%5Cright)%3D%5Csigma_f%5E2%5Cleft(1%2B%5Cfrac%7B%5Cleft%5C%7C%5Cmathbf%7Bx%7D-%5Cmathbf%7Bx%7D%5E'%5Cright%5C%7C%5E%7B2%7D%7D%7B2%20%5Calpha%20%5Cell%5E%7B2%7D%7D%5Cright)%5E%7B-%5Calpha%7D)
## Extended kernels for noisy observations
for noisy observations ![y=f(\mathbf{x})+\epsilon](https://render.githubusercontent.com/render/math?math=y%3Df(%5Cmathbf%7Bx%7D)%2B%5Cepsilon), where ![\epsilon \sim \mathcal{N}\left(0, \sigma_{y}^{2}\right)](https://render.githubusercontent.com/render/math?math=%5Cepsilon%20%5Csim%20%5Cmathcal%7BN%7D%5Cleft(0%2C%20%5Csigma_%7By%7D%5E%7B2%7D%5Cright))
 for with case kernels represented as ![Cov\left\[y_{p}, y_{q}\right\]=\Kappa\left(\mathbf{x}_{p}, \mathbf{x}_{q}\right)+\sigma_{y}^{2} \delta_{p q}](https://render.githubusercontent.com/render/math?math=Cov%5Cleft%5By_%7Bp%7D%2C%20y_%7Bq%7D%5Cright%5D%3D%5CKappa%5Cleft(%5Cmathbf%7Bx%7D_%7Bp%7D%2C%20%5Cmathbf%7Bx%7D_%7Bq%7D%5Cright)%2B%5Csigma_%7By%7D%5E%7B2%7D%20%5Cdelta_%7Bp%20q%7D) where ![\delta_{p q}=\mathbf{I}(p=q)](https://render.githubusercontent.com/render/math?math=%5Cdelta_%7Bp%20q%7D%3D%5Cmathbf%7BI%7D(p%3Dq))
## Dependencies:
- Eigen 3.x
- NLopt 2.6.x
## Third party
- FindNLopt.cmake from here https://github.com/dartsim/dart
- cxxopts command line parser 
## Test application
- input CSV file: sample, observations, with space delimiters
- output CSV file: samples, predictions, variances
## Test application keys
* i,input   -input   CSV file with  the {sample 'space, \t' observation} structure
* o,output  -output CSV file with  the  {sample 'space' prediction 'space' variance}  structure
* k,kernel  -{rbf, matern32, matern32p, matern52, rq} covariance kernel function, default = rbf
* s,sigma   -kernel bandwidth parameter, default = 1.0
* l,length  -kernel characteristic length scale, default = 1.0
* p,period  -period for Matern 3/2 kernel, default = 1.0
* a,alpha   -rational quadratic kernel degree, default =1.0
* n,noise   -{true, false} GPR model for noisy observations and  noise-free observations, default = true
* e,epsilon -prior error level, default  =.01
* j,jitter  -jitter,default =1e-08
* f,factor  -upsampling factor,default = 10.0
* m,max     -optimization solver maximum  iteration numbers, default=100.
* t,optimization -{gradient, direct} optimization  solver type for the kernel parameters, default = direct
## Experiments
![result](https://github.com/ToyOwl/libgpr/blob/master/img/result.PNG)
 with keys --input=data/parameter.dat --kernel=matern52
