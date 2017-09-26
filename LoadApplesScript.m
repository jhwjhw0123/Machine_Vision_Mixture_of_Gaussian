%% LoadApplesScript.m
clc
clear
close all
%% Load the data and process into training data
if( ~exist('apples', 'dir') || ~exist('testApples', 'dir') )
    display('Please change current directory to the parent folder of both apples/ and testApples/');
    return;    %if folder doesn't exist, program won't start
end

% Note that cells are accessed using curly-brackets {} instead of parentheses ().
Iapples = cell(3,1);
Iapples{1} = 'apples/Apples_by_kightp_Pat_Knight_flickr.jpg';
Iapples{2} = 'apples/ApplesAndPears_by_srqpix_ClydeRobinson.jpg';
Iapples{3} = 'apples/bobbing-for-apples.jpg';

IapplesMasks = cell(3,1);
IapplesMasks{1} = 'apples/Apples_by_kightp_Pat_Knight_flickr.png';
IapplesMasks{2} = 'apples/ApplesAndPears_by_srqpix_ClydeRobinson.png';
IapplesMasks{3} = 'apples/bobbing-for-apples.png';

TrainApple = [];
TrainNonApple = [];
ValidApple = [];
ValidNonApple = [];

for iImage = 1:1:3
  %Picture Data  
  curI = double(imread(Iapples{iImage}))/255;      %width x height x 3
  %Mask Picture Data
  curImask = imread(IapplesMasks{iImage});
  curImask = curImask(:,:,2) > 128;  % Picked green-channel arbitrarily.
  %Recognize Apple Data and Non-Apple Data
  DataAppleCollect = [];
  DataNonAppleCollect = [];
  for nX = 1:1:size(curI,1)
      for nY = 1:1:size(curI,2)
          if(curImask(nX,nY)==1)
              DataAppleCollect = [DataAppleCollect,curI(nX,nY,:)];
          else
              DataNonAppleCollect = [DataNonAppleCollect,curI(nX,nY,:)];
          end
      end
  end
  DataAppleCollect = (squeeze(DataAppleCollect)).';
  DataNonAppleCollect = (squeeze(DataNonAppleCollect)).';
  
  TrainApple = [TrainApple DataAppleCollect(:,1:floor(0.7*size(DataAppleCollect,2)))];
  TrainNonApple = [TrainNonApple DataNonAppleCollect(:,1:floor(0.7*size(DataNonAppleCollect,2)))];
  ValidApple = [ValidApple DataAppleCollect(:,ceil(0.7*size(DataAppleCollect,2)):size(DataAppleCollect,2))];
  ValidNonApple = [ValidNonApple DataNonAppleCollect(:,ceil(0.7*size(DataNonAppleCollect,2)):size(DataNonAppleCollect,2))];
  
end

save TrainApple TrainApple
save TrainNonApple TrainNonApple
save ValidApple ValidApple
save ValidNonApple ValidNonApple

