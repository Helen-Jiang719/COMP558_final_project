function image = linearBlending(warpedImages, warpedWeights) 

    num = zeros(size(warpedImages{1}));
    denom = zeros(size(warpedImages{1}));
    
    for i = 1:length(warpedImages)
        num = num + (double(warpedImages{i}) .* warpedWeights{i});
        denom = denom + warpedWeights{i};
    end
    
    % Blended image
    image = uint8(num./denom);
end
