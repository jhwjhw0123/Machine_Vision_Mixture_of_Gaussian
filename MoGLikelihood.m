function [ HiddenLike ] = MoGLikelihood(mixGaussEst,data)
%MOGLIKELIHOOD Summary of this function goes here
%   Detailed explanation goes here
%Please see fitMixGauss to check out how could this work
[nDim, nData] = size(data);
cData = 1:1:nData;
HiddenLike = [];
for cGaussian = 1:1:mixGaussEst.k
      sigma = mixGaussEst.cov(:,:,cGaussian);
      mu = mixGaussEst.mean(:,cGaussian)*ones(1,nData);
      ThisHiddenLike = (det(2*pi*sigma)^(-0.5))*exp(sum((-0.5*((data(:,cData)-mu)).*mldivide(sigma,(data(:,cData)-mu))),1));
      HiddenLike = [HiddenLike; ThisHiddenLike];
end
end

