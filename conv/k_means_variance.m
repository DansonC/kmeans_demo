function [ lambda ] = k_means_variance( X, k_patches, idx )
%K_MEANS_VARIANCE Summary of this function goes here
%   Detailed explanation goes here

N = size(X, 1);
labels = max(idx);
assert(N == size(idx, 1));

fprintf('getting k_means_variance ... '); tic;

Y = zeros(labels, N);
Y(sub2ind(size(Y), idx, (1:N)')) = 1;

t = Y' * k_patches;
d = sum(abs(X - t), 2);

lambda = zeros(labels, 1);
for i = 1 : labels
    lambda(i) = mean(d(idx == i));
end
toc;
end