function res = normal( X, sigma, mu )
% Computes the likelihood that the data X have been generated from the given
% parameters (mu, sigma) of the one-dimensional normal distribution.
C = zeros(length(X),1);
Mul = 1;
for i = 1:1:length(X)
  C(i) = (1/sqrt(2*pi*sigma^2))*exp(-0.5*(X(i)-mu)^2/sigma^2);  
end
for j = 1:1:length(X)
  Mul = Mul*C(j);  
end
% TODO fill out this function
res = Mul;