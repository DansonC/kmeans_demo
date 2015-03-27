%% Test
function [accuracy, wrongidx, maxlabel] = test(W, X, y, options)
    N = size(X, 2);
    X1 = X;
    for i = 1 : size(W, 1)
        X1 = [ones(1, size(X1, 2)); X1]; %#ok<AGROW>
        X1 = activation(bigproduct(W{i,1}, X1));
    end
    [~,maxlabel] = max(X1);
    accuracy = size(find(y==maxlabel'), 1) / N;
    wrongidx = find(y~=maxlabel');
end