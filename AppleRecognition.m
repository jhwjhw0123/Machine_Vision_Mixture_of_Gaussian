%% Copyrights 
%@Function: Recignize apple from the picture
%@Autor: Chen Wang
%@Version:1.0.0
%All Rights Reserved
clc
clear
close all
%% Load Pictures and process them as Matlab data

if( ~exist('apples', 'dir') || ~exist('testApples', 'dir') )
    display('Please change current directory to the parent folder of both apples/ and testApples/');
end

%Load Apple Pictures
Iapples = cell(3,1);
Iapples{1} = 'apples/Apples_by_kightp_Pat_Knight_flickr.jpg';
Iapples{2} = 'apples/ApplesAndPears_by_srqpix_ClydeRobinson.jpg';
Iapples{3} = 'apples/bobbing-for-apples.jpg';
%Load Apple Masks
IapplesMasks = cell(3,1);
IapplesMasks{1} = 'apples/Apples_by_kightp_Pat_Knight_flickr.png';
IapplesMasks{2} = 'apples/ApplesAndPears_by_srqpix_ClydeRobinson.png';
IapplesMasks{3} = 'apples/bobbing-for-apples.png';

%% Load Apple Data(Stored with MAT document)
load('TrainApple','*');
load('TrainNonApple','*');
load('ValidApple','*');
load('ValidNonApple','*');

%% Fiiting MoG
nGaussianApple = 3;
nGaussianNoneApple = 3;
load('MeanApple');
load('MeanNonApple');
mixGaussEstApple = fitMixGauss(TrainApple,nGaussianApple,MeanApple);
mixGaussEstNonApple = fitMixGauss(TrainNonApple,nGaussianNoneApple,MeanNonApple);
save MoGApple mixGaussEstApple
save MoGNoneApple mixGaussEstNonApple

%% Main Loop
for iImage = 1:1:3
  %% Draw Picture  
  curI = double(imread(Iapples{iImage}))/255;      %width x height x 3
  %Mask Picture Data
  curImask = imread(IapplesMasks{iImage});
  curImask = curImask(:,:,2) > 128;  % Picked green-channel arbitrarily.
  %Recognize Apple Data and Non-Apple Data
  %Open New figure
  figure;
  set(gcf,'Color',[1 1 1]);
  subplot(1,3,1); imagesc(curI); axis off; axis image;
  subplot(1,3,2); imagesc(curImask); colormap(gray); axis off; axis image;
  drawnow;
 %% Judge if it is apple
  curIJudge = reshape(double(curI),size(curI,1)*size(curI,2),3).';
  appleJudge = zeros(1,size(curI,1)*size(curI,2));
  priorApple = 0.5;
  priorNonApple = 0.5;
  nData = size(appleJudge,2);
  cData = 1:1:nData;
  %Please see fixMixGauss about how the codes works to speed up the program
  %Apple
  cGaussianApple = (1:1:mixGaussEstApple.k).';
  WeightsApple = diag(diag(repmat(mixGaussEstApple.weight(cGaussianApple),3,1)));
  AppleTotal = MoGLikelihood(mixGaussEstApple,curIJudge);
  AppleLike = reshape(sum(WeightsApple*AppleTotal,1),size(curI,1),size(curI,2));    %1*nData vector now
  %NonApple
  cGaussianNonapple = (1:1:mixGaussEstNonApple.k).';
  WeightsNonApple = diag(diag(repmat(mixGaussEstNonApple.weight(cGaussianNonapple),3,1)));
  NonAppleTotal = MoGLikelihood(mixGaussEstNonApple,curIJudge);
  NonAppleLike = reshape(sum(WeightsNonApple*NonAppleTotal,1),size(curI,1),size(curI,2));    %1*nData vector now  
  appleJudge = (AppleLike.*priorApple)./(AppleLike.*priorApple + NonAppleLike.*priorNonApple);
  %Output
  clims = [0, 1];
  subplot(1,3,3); imagesc(appleJudge, clims); colormap(hot); axis off; axis image;
end
