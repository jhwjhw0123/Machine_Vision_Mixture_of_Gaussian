%% Function: Process the Images to the Equilized ones
close all
clc
clear

Icpmvert = cell(3,1);
Icpmvert{1} = 'Apples_by_kightp_Pat_Knight_flickr.jpg';
Icpmvert{2} = 'ApplesAndPears_by_srqpix_ClydeRobinson.jpg';
Icpmvert{3} = 'bobbing-for-apples.jpg';
ImSave = cell(3,1);

for iImage=1:1:3
    ThisPicture = imread(Icpmvert{iImage});
    %Define Matrix for Matlab function double-check
    EqualizedImage = zeros(size(ThisPicture,1),size(ThisPicture,2),size(ThisPicture,3));
    %Define Matrix for my own Histograph Equilization
    MatEquiIm = zeros(size(ThisPicture));
    %Process R,G,B Respectively
    for Channel = 1:1:3
     ThisChannel = ThisPicture(:,:,Channel);     %Weight*Height Matrix
     ThisStat = zeros(1,256);              %Row vector
      for level = 0:1:255
          LightChannel = double(ThisChannel) - level;
          LightChannel(LightChannel==0) = inf;
          LightChannel(LightChannel~=inf) = 0;
          LightChannel(LightChannel==inf) = 1;
          ThisStat(level+1) = sum(sum(LightChannel));
      end
      %Using Matrix calculation to mimic Culmulative function
      CulmalChannel = tril(ones(256,256))*(ThisStat.');    %Column Vector with culmulative distribution
      HistoChannel = round((CulmalChannel*255)/(size(ThisChannel,1)*size(ThisChannel,2)));   %Equalized Red Channel
      EquThisChannel = double(ThisChannel);
      for level = 0:1:255
          EquThisChannel(EquThisChannel==level) = level+256;
      end
      for level = 256:1:511
          EquThisChannel(EquThisChannel==level) = HistoChannel(level-255);
      end
      %My Channel Histogram Equilized
      EqualizedImage(:,:,Channel) = double(uint8(EquThisChannel))/255;
      %Matlab Channel Histogram Equilized
      MatEquiIm(:,:,Channel) = histeq(ThisChannel);
    end
    MatEquiIm = double(MatEquiIm)/255;
    figure;
    set(gcf,'Color',[1 1 1]);
    subplot(1,3,1); imagesc(ThisPicture); axis off; axis image;
    subplot(1,3,2); imagesc(EqualizedImage); axis off; axis image;
    %Using Matlab Equilization to check if the result is right
    subplot(1,3,3); imagesc(MatEquiIm); axis off; axis image;      
    ImSave{iImage} = EqualizedImage;
end

imwrite(ImSave{1},'Equilized_Apples_by_kightp_Pat_Knight_flickr.jpg');
imwrite(ImSave{2},'Equilized_ApplesAndPears_by_srqpix_ClydeRobinson.jpg');
imwrite(ImSave{3},'Equilized_bobbing-for-apples.jpg');