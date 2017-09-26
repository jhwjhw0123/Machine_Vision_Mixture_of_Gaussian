%% Function: Label the Test apple for further Quatification of the performence
clc;
close all

%% Load the image
if( ~exist('apples', 'dir') || ~exist('testApples', 'dir') || ~exist('Newapples','dir'))
    display('Error: Please Check if the folder of /apples, /testApples and /Newapples exist');
    return;
end

IappleTest = double(imread('testApples/Bbr98ad4z0A-ctgXo3gdwu8-original.jpg'))/255;
IappleTestMask = imread('testApples/Bbr98ad4z0A-ctgXo3gdwu8-original.png');
IappleTestMask = IappleTestMask(:,:,2) > 128;  % Picked green-channel arbitrarily.

DataAppleTest = reshape(IappleTest,size(IappleTest,1)*size(IappleTest,2),3).';
DataAppleMask = reshape(IappleTestMask,1,size(IappleTestMask,1)*size(IappleTestMask,2));
%Get Apple Test Data
ITestApple = (repmat(DataAppleMask,3,1)).*DataAppleTest;
ITestApple(:,all(ITestApple==0,1))=[];
%Get Non-apple Test Data
ITestNonApple = (repmat(~DataAppleMask,3,1)).*DataAppleTest;
ITestNonApple(:,all(ITestNonApple==0,1))=[];

save ITestApple ITestApple
save ITestNonApple ITestNonApple

IappleNewTest = cell(3,1);
IappleNewTest{1} = 'Newapples/image_20160910_010741_618.jpg';
IappleNewTest{2} = 'Newapples/apple-tree-429213__340.jpg';
IappleNewTest{3} = 'Newapples/RTR30DB6.jpg';

IappleNewMask = cell(3,1);
IappleNewMask{1} = 'Newapples/image_20160910_010741_618.png';
IappleNewMask{2} = 'Newapples/apple-tree-429213__340.png';
IappleNewMask{3} = 'Newapples/RTR30DB6.png';
%N.B.: These maks has been pre-processed into Logic Matrices

NewTestApple = [];
NewTestNonApple = [];

for iImage = 1:1:3
    curI = double(imread(IappleNewTest{iImage}))/255;      %width x height x 3
    curImask = imread(IappleNewMask{iImage});
    curIData = reshape(curI,size(curI,1)*size(curI,2),3).';
    curIMaskData = reshape(curImask,size(curImask,1)*size(curImask,2),1).';
    curTestApple = repmat(curIMaskData,3,1).*curIData;
    curTestApple(:,all(curTestApple==0,1))=[];
    NewTestApple = [NewTestApple curTestApple];
    curTestNonApple = repmat(~curIMaskData,3,1).*curIData;
    curTestNonApple(:,all(curTestNonApple==0,1))=[];
    NewTestNonApple = [NewTestNonApple curTestNonApple];
end

save NewTestApple NewTestApple
save NewTestNonApple NewTestNonApple
