function [variance] = compute_variation(data,width_var)
    
variance = zeros(length(data),1);
for idx = 1:length(data)
    if idx < (width_var/2)
        indexes = 1:width_var;
    elseif idx > (length(data) - width_var/2)
        indexes = (length(data) - width_var + 1):length(data);
    else
        indexes = (idx - width_var/2 + 1):(idx + width_var/2 );
    end
    sample = data(indexes);

    variance(idx) = var(sample);

end
end