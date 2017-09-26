clc
clear
%% Load Apple Data(Stored with MAT document)
load('TrainApple','*');
load('TrainNonApple','*');

nDim = size(TrainApple,1);
%% Validation Apple
nGaussianApple = 3;
LikeApple = 0;
MeanApple = 2*zeros(nDim,nGaussianApple);
for i = 1:1:15
  fprintf('Processing the %d interation of apple\n',i);
  mean = 2*randn(nDim,nGaussianApple);
  mixGaussEstApple = MixGaussianWithValid(TrainApple,nGaussianApple,mean);
  if(mixGaussEstApple.Likelihood>LikeApple)
      LikeApple = mixGaussEstApple.Likelihood;
      MeanApple = mean;
  end
end
fprintf('The optimal Mean of Apple Pixels is:\n');
MeanApple
fprintf('\n which could bring a likelihood of %4.3f\n',LikeApple);

%% Validation NoApple
nGaussianNoneApple = 3;
LikeNonApple = 0;
MeanNonApple = 2*zeros(nDim,nGaussianNoneApple);
for i = 1:1:15
  fprintf('Processing the %d interation of Non-Apple\n',i);
  mean = 2*randn(nDim,nGaussianNoneApple);
  mixGaussEstNonApple = MixGaussianWithValid(TrainNonApple,nGaussianNoneApple,mean);
  if(mixGaussEstNonApple.Likelihood>LikeNonApple)
      LikeNonApple = mixGaussEstNonApple.Likelihood;
      MeanNonApple = mean;
  end
end

fprintf('The optimal Mean of Non-Apple Pixels is:\n');
MeanNonApple
fprintf('\n which could bring a likelihood of %4.3f\n',LikeNonApple);