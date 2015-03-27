function g = activationGradient(z, method, slope)
%ACTIVATIONGRADIENT returns the gradient of the activation function
%evaluated at z
%   g = ACTIVATIONGRADIENT(z) computes the gradient of the activation 
%   function evaluated at z. This should work regardless if z is a matrix 
%   or a vector. In particular, if z is a vector or matrix, you should 
%   return the gradient for each element.

if nargin < 3
    slope = 0;
end
if nargin < 2
    method = 'relu';
end

switch method
    case 'relu'
        g = ones(size(z));
        g(z<0) = slope;
    case 'sigmoid'
        g = activation(z, 'sigmoid') .* ...
            ( ones(size(z)) - activation(z, 'sigmoid') );
end
end
