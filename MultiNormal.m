function res = MultiNormal( X, sigma, mu )
% Computes the likelihood that the data X have been generated from the given
% parameters (mu, sigma) of the one-dimensional normal distribution.
DataAmount = size(X,2);
C = zeros(DataAmount,1);
Mul = 1;
for i = 1:1:DataAmount
  C(i) = (det(2*pi*sigma)^(-0.5))*exp(-0.5*((X(:,i)-mu).')*mldivide(sigma,(X(:,i)-mu)));  
end
for j = 1:1:DataAmount
  Mul = Mul*C(j);  
end
% TODO fill out this function
res = Mul;