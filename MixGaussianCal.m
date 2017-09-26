function likelihood = MixGaussianCal(Data,mixGaussEst)
%MIXGAUSSIANCAL Summary of this function goes here
%   Detailed explanation goes here
 nData = size(Data,2);
 likelihood = 1;
 for cData = 1:1:nData
    thisData = Data(:,cData);        
    like = 0;
    for i = 1:1:mixGaussEst.k
        like = like + mixGaussEst.weight(i)*MultiNormal(thisData,mixGaussEst.cov(:,:,i),mixGaussEst.mean(:,i));
    end
    %add to total log like
    likelihood = likelihood * like;        
end
end

