function [smoothed] = smooth(scalar,weight)
    smoothed = zeros(length(scalar),1);
    smoothed(1) = scalar(1);
    for i = 2:length(scalar)
        smoothed(i) = smoothed(i-1) * weight + (1 - weight)* scalar(i);
    end
end