function [ Y ] = bigproduct( A, X )
%BIGPRODUCT Summary of this function goes here
%   Detailed explanation goes here

threshold = 10^6;

if threshold < numel(X)
    unit = floor(threshold / size(X, 1));
    Y = zeros(size(A, 1), size(X, 2));
    for i = 1 : ceil(size(X, 2) / unit)
        from = (i-1) * unit + 1;
        upto = min(i * unit, size(X, 2));
        Xg = gpuArray(X(:, from : upto));
        Y(:, from : upto) = gather(A * Xg);
        clear('Xg');
    end
else
    Xg = gpuArray(X);
    Y  = A * Xg;
end

Y = gather(Y);

end

