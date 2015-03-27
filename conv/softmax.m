function sm = softmax(X, L)
    if 2 > nargin
        L = 1;
    end
    Z = sum(X);
    Z(Z==0) = 1;
    sm = X ./ (ones(size(X,1), 1) * (Z ./ L));
end