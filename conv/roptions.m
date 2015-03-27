function [ options ] = roptions( options_ )
%ROPTIONS Summary of this function goes here
%   Detailed explanation goes here

    options.base_lr = .1;
    options.gamma = 1;
    options.momentum = 0;
    options.decay = 0;
    options.iter = 50;
    options.batch_split = 100;
    options.verbose = true;

    if 0 < nargin
        fnames = fieldnames(options_);
        for i = 1 : numel(fnames)
            options.(fnames{i}) = options_.(fnames{i});
        end
    end
end

