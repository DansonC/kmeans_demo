function [ W ] = regression( layer, X, y, X_te, y_te, options )
%REGRESION Summary of this function goes here
%   Detailed explanation goes here

N = size(X, 2);
features = size(X, 1);

labels = max(y);
Y = zeros(labels, N);
Y(sub2ind(size(Y), y, (1:N)')) = 1;

if 5 < nargin && isfield(options, 'lambda')
    options = roptions(options);
    L = bsxfun(@times, Y, options.lambda);
else
    options = roptions;
    L = Y;
end

batch_size = floor(N / options.batch_split);
lr = options.base_lr;
W = cell(1);
W{1,1} = (rand(labels, features + 1) - 0.5) * 2 / N;
X1 = [ones(1, N); X];
V = cell(options.batch_split, 1);

%% try to load a cache
filename = sprintf('W_%dx%dx%d.mat', features, labels, layer);
if exist(filename, 'file')
    fprintf('load %s ...\n', filename);
    load(filename);
    return;
end
    
tic;
for i = 1 : options.iter
    for b = 1 : options.batch_split
        Xg = gpuArray(X1(:, (b-1)*batch_size+1:b*batch_size));
        yg = L(:, (b-1)*batch_size+1:b*batch_size);
        
        Zg = W{1,1} * Xg;
        d = softmax(activation(Zg), sum(yg)) - yg;
        Vt = lr * d * Xg';
        if 1 == mod(i, 100) && b == 1
            lr = lr * options.gamma;
        end
        if 1 == i
            V{b} = zeros(size(Vt)); 
        end
        V{b} = options.momentum * V{b} - Vt;
        W{1,1} = W{1,1} + V{b} - options.decay * W{1,1};
        if (0 == mod(i, 10) || i == options.iter) && b == options.batch_split
            fprintf('%d %.5e', i, lr);
            if 0 < options.verbose
                chk = find(Zg<0);
                fprintf(' o=%.5e d=%.5e', ...
                    size(chk, 1) / numel(Zg), ...
                    sum(sum(Vt.^2))/numel(Vt));
            end
            if b == options.batch_split && N < 10^7
                accuracy = test(W, X, y);
                fprintf(' %.3f', accuracy * 100);
                if exist('X_te', 'var') && 100 < size(X_te, 2)
                    accuracy = test(W, X_te, y_te);
                    fprintf(' / %.2f', accuracy * 100);
                end
            end
            fprintf('\n');
        end
    end
end
toc;
W = gather(W{1,1});
% Save a cache
save(filename, 'W');

if N >= 10^7
    accuracy = test(W, X, y);
    fprintf(' %.3f', accuracy * 100);
    if nargin > 4
        accuracy = test(W, X_te, y_te);
        fprintf(' / %.2f', accuracy * 100);
    end
end

end

