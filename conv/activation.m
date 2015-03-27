function z = activation(z, method, slope)
%ACTIVATION Compute activation function
%   g = ACTIVATION(z) computes the activation of z.

if nargin < 3
    slope = 0;
end
if nargin < 2
    method = 'relu';
end

switch method
    case 'relu'
        z(z<0) = slope * z(z<0);
    case 'sigmoid'
        z = 1.0 ./ (1.0 + exp(-z));
end
end
