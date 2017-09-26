close all
Icpmvert = cell(3,1);
Icpmvert{1} = 'apple-tree-429213__340_Mask.png';
Icpmvert{2} = 'image_20160910_010741_618_Mask.png';
Icpmvert{3} = 'RTR30DB6_Mask .png';
ImSave = cell(3,1);
for ICount = 1:1:3
ThisPicture = imread(Icpmvert{ICount});
LogicPicture = im2bw(ThisPicture,0);
if (ICount == 1)
    LogicPicture = ~LogicPicture;
end
ImSave{ICount} = LogicPicture;
figure;
set(gcf,'Color',[1 1 1]);
subplot(1,2,1); imagesc(ThisPicture); axis off; axis image;
subplot(1,2,2); imagesc(LogicPicture); colormap(gray); axis off; axis image;
end
imwrite(ImSave{1},'apple-tree-429213__340.png');
imwrite(ImSave{2},'image_20160910_010741_618.png');
imwrite(ImSave{3},'RTR30DB6.png');