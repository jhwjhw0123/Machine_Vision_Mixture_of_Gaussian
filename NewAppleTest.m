%% Function: Test the three self-downloaded picture and quantify them with corresponding Mask Information
clc
clear
close all

if(~exist('Newapples','dir'))
    display('Error: Please Check if the folder of /apples, /testApples and /Newapples exist');
    return;
end

Iapples = cell(3,1);
Iapples{1} = 'Newapples/image_20160910_010741_618.jpg';
Iapples{2} = 'Newapples/apple-tree-429213__340.jpg';
Iapples{3} = 'Newapples/RTR30DB6.jpg';

IapplesMasks = cell(3,1);
IapplesMasks{1} = 'Newapples/image_20160910_010741_618.png';
IapplesMasks{2} = 'Newapples/apple-tree-429213__340.png';
IapplesMasks{3} = 'Newapples/RTR30DB6.png';

load('MoGApple','*');
load('MoGNoneApple','*');
%load('BestPrior','*');
%load('BestTrustLevel','*');     
%Here I choose to use a different validation Prior and Trust threshold
load('NewTestApple','*');
load('NewTestNonApple','*');

Color = rand(11,3);
%Parameters to store the information of Validation and Draw ROC Curve
g_fp = [];
g_tp = [];
g_color_index = [];
g_legend_str = {};
%Parameters to find the best Trust level (Point on the ROC Curve)
pos = size(NewTestApple,2)/(size(NewTestApple,2)+size(NewTestNonApple,2));
optiPont = zeros(1,11);
OptimalTrust = zeros(1,11);

for priorApple = 0:0.1:1
   priorNonApple = 1 - priorApple;
   fp = [];
   tp = [];
   for trustlevel = 0:0.1:1
      % Judge if it is apple
      % True Positive: NewTestApple
      appleJudgeTP = zeros(1,size(NewTestApple,2));
      nData = size(appleJudgeTP,2);
      cData = 1:1:nData;
      %Please see fixMixGauss about how the codes works to speed up the program
      %Apple
      cGaussianApple = (1:1:mixGaussEstApple.k).';
      WeightsApple = diag(diag(repmat(mixGaussEstApple.weight(cGaussianApple),3,1)));
      AppleTotal = MoGLikelihood(mixGaussEstApple,NewTestApple);
      AppleLike = sum(WeightsApple*AppleTotal,1);    %1*nData vector now
      %NonApple
      cGaussianNonapple = (1:1:mixGaussEstNonApple.k).';
      WeightsNonApple = diag(diag(repmat(mixGaussEstNonApple.weight(cGaussianNonapple),3,1)));
      NonAppleTotal = MoGLikelihood(mixGaussEstNonApple,NewTestApple);
      NonAppleLike = sum(WeightsNonApple*NonAppleTotal,1);    %1*nData vector now  
      appleJudgeTP = (AppleLike.*priorApple)./(AppleLike.*priorApple + NonAppleLike.*priorNonApple);
      %Judge Positive
      JudgePositiveTP =  sum(appleJudgeTP>=trustlevel);
      TPrate = JudgePositiveTP/size(appleJudgeTP,2);
      fprintf('PriorApple at %4.3f, Trust level at %4.3f,the True Positive Rate is %4.3f\n',priorApple,trustlevel,TPrate);
      tp = [tp TPrate];
      % False Positive: NewTestNonApple
      appleJudgeFP = zeros(1,size(NewTestNonApple,2));
      nData = size(appleJudgeFP,2);
      cData = 1:1:nData;
      %Please see fixMixGauss about how the codes works to speed up the program
      %Apple
      cGaussianApple = (1:1:mixGaussEstApple.k).';
      WeightsApple = diag(diag(repmat(mixGaussEstApple.weight(cGaussianApple),3,1)));
      AppleTotal = MoGLikelihood(mixGaussEstApple,NewTestNonApple);
      AppleLike = sum(WeightsApple*AppleTotal,1);    %1*nData vector now
      %NonApple
      cGaussianNonapple = (1:1:mixGaussEstNonApple.k).';
      WeightsNonApple = diag(diag(repmat(mixGaussEstNonApple.weight(cGaussianNonapple),3,1)));
      NonAppleTotal = MoGLikelihood(mixGaussEstNonApple,NewTestNonApple);
      NonAppleLike = sum(WeightsNonApple*NonAppleTotal,1);    %1*nData vector now  
      appleJudgeFP = (AppleLike.*priorApple)./(AppleLike.*priorApple + NonAppleLike.*priorNonApple);
      %Judge Positive
      JudgePositiveFP =  sum(appleJudgeFP>=trustlevel);
      FPrate = JudgePositiveFP/size(appleJudgeFP,2);
      fp = [fp FPrate];
      fprintf('PriorApple at %4.3f, Trust level at %4.3f,the Frue Positive Rate is %4.3f\n',priorApple,trustlevel,FPrate);
      fprintf('\n');
      thisAccuracy = pos*TPrate + (1-pos)*(1-FPrate);      %According to Peter Plach's Tutorial
      if (FPrate~=1&&TPrate~=0)
        if (thisAccuracy>optiPont(:,round(10*priorApple+1)))
            optiPont(:,round(10*priorApple+1)) = thisAccuracy;
            OptimalTrust(:,round(10*priorApple+1)) = trustlevel;
        end
      end
   end

    clr = Color(round(10*priorApple+1),:);
    str = strcat('Prior= ',(num2str(priorApple)));

    g_fp = [g_fp; fp];
    g_tp = [g_tp; tp];
    g_color_index = [g_color_index;  clr];
    g_legend_str = [g_legend_str; str];
end

BestPrior = 0;
BestAUC = 0;

for cPrior = 1:1:11
   %Calculating AUC of ROC with trapezoidal approximations 
   %This could help us to select the best prior
   thisAUC = -trapz(g_fp(cPrior,:),g_tp(cPrior,:));
   if (thisAUC>BestAUC)
       BestAUC = thisAUC;
       BestPrior = 0.1*(cPrior-1);
   end
end

BestTrustLevel = OptimalTrust(:,round(10*BestPrior)+1);
WeightedAccuracy = optiPont(:,round(10*BestPrior)+1);
BestTPR = g_tp(round(BestPrior*10)+1,round(BestTrustLevel*10)+1);
BestTNR = 1 - g_fp(round(BestPrior*10)+1,round(BestTrustLevel*10)+1);

fprintf('The Best Apple Prior for new test set should be %4.3f \n',BestPrior);
fprintf('The Corresponding AUC value is %4.3f \n',BestAUC);
fprintf('The Trust Level for the three images should be %4.3f \n',BestTrustLevel);
fprintf('Which could Provide weighted accuracy of %4.3f \n',WeightedAccuracy);
fprintf('The Ture Positive Rate is: %4.3f \n',BestTPR);
fprintf('The Ture Negative Rate is: %4.3f \n',BestTNR);


% Plot the ROC Curves
%  calling plot here with block data 
figure;
for cPrior = 1:1:11
    plot(g_fp(cPrior,:),g_tp(cPrior,:),'color',g_color_index(cPrior,:),'linewidth',2);
    hold on;
end
legend(g_legend_str);
hold off
figure;
plot(g_fp(round(10*BestPrior)+1,:),g_tp(round(10*BestPrior)+1,:),'color',[1.0 0 1.0],'linewidth',2);
hold on
plot(g_fp(round(10*BestPrior)+1,round(10*BestTrustLevel)+1),g_tp(round(10*BestPrior)+1,round(10*BestTrustLevel)+1),'b*');
hold off


%% Plot the Classification Reult
for iImage = 1:1:3
  %% Draw Picture  
  curI = double(imread(Iapples{iImage}))/255;      %width x height x 3
  curIMask = imread(IapplesMasks{iImage});
  %Open New figure
  figure;
  set(gcf,'Color',[1 1 1]);
  subplot(1,2,1); imagesc(curI); axis off; axis image;
  drawnow;
 %% Judge if it is apple
  curIJudge = reshape(double(curI),size(curI,1)*size(curI,2),3).';
  appleJudge = zeros(1,size(curI,1)*size(curI,2));
  priorApple = BestPrior;
  priorNonApple = 1 - priorApple;
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
  subplot(1,2,2); imagesc(appleJudge, clims); colormap(hot); axis off; axis image;
  DeterminedTest = sign(appleJudge-BestTrustLevel);
  DeterminedTest(DeterminedTest<0) = 0;
  figure;
  set(gcf,'Color',[1 1 1]);
  subplot(1,2,1); imagesc(curIMask); axis off; axis image;
  subplot(1,2,2); imagesc(DeterminedTest); colormap(gray); axis off; axis image;
end


