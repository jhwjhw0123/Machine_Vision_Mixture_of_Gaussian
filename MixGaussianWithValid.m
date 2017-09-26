function mixGaussEst = MixGaussianWithValid(DataItrain,nGaussian,MeanInitial)
%MIXGAUSSIANWITHVALID Summary of this function goes here
%   Detailed explanation goes here
%FITMIXGAUSS Summary of this function goes here
%   Detailed explanation goes here
  k = nGaussian;
  [nDim, nData] = size(DataItrain);
  postHidden = zeros(k,nData);      %%In one dimension it is called Responsibilities

  %in the E-M algorithm, we calculate a complete posterior distribution over
  %the (nData) hidden variables in the E-Step.  In the M-Step, we
  %update the parameters of the Gaussians (mean, cov, w).  

  %we will initialize the values to random values
  mixGaussEst.d = nDim;
  mixGaussEst.k = k;
  mixGaussEst.weight = (1/k)*ones(1,k);
  mixGaussEst.mean = MeanInitial;
  for cGauss =1:1:k
      mixGaussEst.cov(:,:,cGauss) = (0.5+1.5*rand(1))*eye(nDim,nDim);        %Diagonal Cov Matrix
  end
  
  logLike = getMixGaussLogLike(DataItrain,mixGaussEst);
  %fprintf('Log Likelihood Iter 0 : %4.3f\n',logLike);
  LastLikelihood = logLike;
  
  for nInt = 1:1:50
    %E-step
    %MarginalEstep = zeros(1,nData);
    HiddenLike = HiddenLikeCal(mixGaussEst,DataItrain);
    if (sum(sum(HiddenLike)) == 0)
        fprintf('Numerical Unstable, break out. This mean can not be used\n');
        mixGaussEst.Likelihood=0;
        break;
    end
    HiddenJoint = zeros(mixGaussEst.k,nData);
    %Likelihood*Prior of h, comparing to MarginalEstep, I call it 'HiddenJoint'
    %(Don't take the name seriously, I'm a non-native speaker ^_^)
    for GaussianInd = 1:1:mixGaussEst.k
        HiddenJoint(GaussianInd,:) = HiddenLike(GaussianInd,:)*mixGaussEst.weight(GaussianInd);
    end
    %Our new 'MarginalEstep' is a Cool Matrix!
    MarginalEstep = repmat(sum(HiddenJoint,1),k,1);
    %To divide it as matrix fomula we must clone the matrix into
    %nGaussian*nData.
    postHidden = HiddenJoint./MarginalEstep;
  
    %Maximization Step
    SumpostHidden = zeros(k,1);
    for indx = 1:1:k
         SumpostHidden(indx) = sum(postHidden(indx,:));
    end
    TotalpostHidden = sum(SumpostHidden);
    %for each constituent Gaussian
    for cGauss = 1:1:k 
        %This is equivalent to the equation. Notice that the Calculation of
        %weight here is the same as non-speed-up-one
        mixGaussEst.weight(cGauss) = SumpostHidden(cGauss)/TotalpostHidden; 
        %Here I find out that the Whole Data Matrix could be directly
        %multipled by the chosen Gaussian Distribution(1*nData)*(nData*nDim)
        %The result is a row vector, thus I further transposed it
        %Using Matrix Calculation, Solve mixGaussEst.mean(:,cGauss) quickly
        %Cooooooooooooool
        mixGaussEst.mean(:,cGauss) = (postHidden(cGauss,:)*DataItrain.').'/SumpostHidden(cGauss);
        %Speed-up: Using Vector Calculation to replace for loop! Cooool!
        %cData is a vector now, no need to loop
        cData = 1:1:nData;
        %We need to let R(ik) fit the dimension of data in order to perform
        %Matrix Multiple
        postCalcuHidden = repmat(postHidden(cGauss,:),nDim,1);
        %Basic Idea: sigma(R(ik)*X(i)*trans(X(i))) = X(*)*X
        %Where X(*) is the total data with each colum already multiplied by
        %R(ik)(A littile bit tricky)
        mixGaussEst.cov(:,:,cGauss) = (postCalcuHidden.*(DataItrain(:,cData)-mixGaussEst.mean(:,cGauss)*ones(1,nData)))*((DataItrain(:,cData)-mixGaussEst.mean(:,cGauss)*ones(1,nData)).')/SumpostHidden(cGauss);
    end
   
    %calculate the log likelihood
    logLike = getMixGaussLogLike(DataItrain,mixGaussEst);
    if(abs((logLike-LastLikelihood)/LastLikelihood)<0.001);
        LastLikelihood = logLike;
        fprintf('Log Likelihood Iter %d : %4.3f, Good Enough, Iretation Finished\n',nInt,logLike);
        break;
    end
    %Prevent Near Singular Matrix Situation
    if(logLike == 0)
        fprintf('Numerical Unstable, break out. This mean can not be used\n');
        mixGaussEst.Likelihood=0;
        break;
    end
    LastLikelihood = logLike;
    %fprintf('Log Likelihood Iter %d : %4.3f\n',nInt,logLike);
  end
  if (nInt == 50)
    fprintf('Log Likelihood Iter %d : %4.3f, Up to Maximum Iteration Times, Iretation Finished\n',nInt,LastLikelihood);
  end
  mixGaussEst.Likelihood = LastLikelihood;
end

function logLike = getMixGaussLogLike(DataItrain,mixGaussEst)
%For working rationale, please see the comments of function 'HiddenLikeCal'
%below
lastwarn('');
[nDim,nData] = size(DataItrain);   
cData = 1:1:nData;
cGaussian = (1:1:mixGaussEst.k).';
Weights = diag(diag(repmat(mixGaussEst.weight(cGaussian),3,1)));
sigma1 = mixGaussEst.cov(:,:,1);
sigma2 = mixGaussEst.cov(:,:,2);
sigma3 = mixGaussEst.cov(:,:,3);
mu1 = mixGaussEst.mean(:,1)*ones(1,nData);
mu2 = mixGaussEst.mean(:,2)*ones(1,nData);
mu3 = mixGaussEst.mean(:,3)*ones(1,nData);
HiddenLike1 = (det(2*pi*sigma1)^(-0.5))*exp(sum((-0.5*((DataItrain(:,cData)-mu1)).*mldivide(sigma1,(DataItrain(:,cData)-mu1))),1));
HiddenLike2 = (det(2*pi*sigma2)^(-0.5))*exp(sum(-0.5*((DataItrain(:,cData)-mu2)).*mldivide(sigma2,(DataItrain(:,cData)-mu2)),1));
HiddenLike3 = (det(2*pi*sigma3)^(-0.5))*exp(sum(-0.5*((DataItrain(:,cData)-mu3)).*mldivide(sigma3,(DataItrain(:,cData)-mu3)),1));
TotalLike = [HiddenLike1;HiddenLike2;HiddenLike3];
[warnmsg, msgid] = lastwarn;
if strcmp(msgid,'MATLAB:illConditionedMatrix')||strcmp(msgid,'MATLAB:nearlySingularMatrix')
  logLike = 0;
  lastwarn('');
  return
end
%Weights Mtrix is now a diagnal Matrix, could multiple R(k) to diffetent
%row
LoglikeMatrix = Weights*TotalLike;
%sum log likelihood: Firstly sum each colum, get the sigma(r(k)*x) matrix
%Then log it, it should be a 1*nData Log likelihood matrix
%Then sum over all colums, we get the total likelihood
logLike = sum(log(sum(LoglikeMatrix,1)));
end

function [HiddenLike] = HiddenLikeCal(mixGaussEst,DataItrain)
  %How does this work:
  %We want the HiddenLike Matrix(for p(x|h,seita)), but if we go through
  %every data points, it cause 'slow problem'
  %Alternatively, here I use different matrix to store the information of
  %different Gaussian (So each of them should be 1*nData vector). Then if a
  %concatenate them, it will come to the HiddenLike Matrix
  %This is a 'Historical' Version with only fixed amount of Gaussian
  %See fitMixGauss for new method of calculating Hiddenlike with
  %Matrix Operaion
  lastwarn('');
  [nDim, nData] = size(DataItrain);
  cData = 1:1:nData;
  HiddenLike1 = zeros(1,nData);
  HiddenLike2 = zeros(1,nData);
  HiddenLike3 = zeros(1,nData);
  sigma1 = mixGaussEst.cov(:,:,1);
  sigma2 = mixGaussEst.cov(:,:,2);
  sigma3 = mixGaussEst.cov(:,:,3);
  mu1 = mixGaussEst.mean(:,1)*ones(1,nData);
  mu2 = mixGaussEst.mean(:,2)*ones(1,nData);
  mu3 = mixGaussEst.mean(:,3)*ones(1,nData);
  %Calculation: Notice that if we use whole data matrix X to perform
  %trans(X)*sigma^-1*X,we will get a nData*nData Matrix
  %In this Matrix, the elements on the diagnal is what we want
  %However, nData*nData could be Extremly BIG
  %Alternatively, because we only need corresponding elements to multiple (this is the diagnal)
  %Could use .* here, and then sum each colum up, we will get the wanted
  %vector!
  HiddenLike1 = (det(2*pi*sigma1)^(-0.5))*exp(sum((-0.5*((DataItrain(:,cData)-mu1)).*mldivide(sigma1,(DataItrain(:,cData)-mu1))),1));
  HiddenLike2 = (det(2*pi*sigma2)^(-0.5))*exp(sum(-0.5*((DataItrain(:,cData)-mu2)).*mldivide(sigma2,(DataItrain(:,cData)-mu2)),1));
  HiddenLike3 = (det(2*pi*sigma3)^(-0.5))*exp(sum(-0.5*((DataItrain(:,cData)-mu3)).*mldivide(sigma3,(DataItrain(:,cData)-mu3)),1));
  HiddenLike = [HiddenLike1;HiddenLike2;HiddenLike3];
  [warnmsg, msgid] = lastwarn;
  if strcmp(msgid,'MATLAB:illConditionedMatrix')||strcmp(msgid,'MATLAB:nearlySingularMatrix')
    HiddenLike = zeros(size(HiddenLike));
    lastwarn('');
  end
end
