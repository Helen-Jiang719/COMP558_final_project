function blendedImage = multiBandBlending(input, inputImages, inputWeights)
    % Initialize parameters
    numBands = input.bands;              % Number of pyramid levels
    numImages = length(inputImages);     % Number of images
    gaussFilter = fspecial('gaussian', input.filtSize, input.MBBsigma); % Gaussian filter
    
    % Initialize cell arrays for pyramid levels and blending
    gaussianPyramids = cell(numImages, numBands);  % Gaussian pyramid
    weightPyramids = cell(numImages, numBands);    % Weight pyramids
    laplacianPyramids = cell(numImages, numBands); % Laplacian pyramid
    blendedPyramids = cell(1, numBands);           % Final blend for each band
    maxWeightMask = cell(1, numImages);            % Weights for the warped images
    
    % Step 1: Compute maximum weight masks for each warped image
    weightSum = cellfun(@(x) sum(x, 3), inputWeights, 'UniformOutput', false);
    weightSumMatrix = cat(3, weightSum{:});
    totalWeightSum = sum(weightSumMatrix, 3);
    zeroIndices = totalWeightSum == 0;
    
    [~, maxWeightIndices] = max(weightSumMatrix, [], 3, 'omitnan');
    maxWeightIndices = maxWeightIndices .* imcomplement(zeroIndices);
    
    % Create max weight masks for each image
    for i = 1:numImages
        maxMask = maxWeightIndices == i;
        maxWeightTemp = zeros(size(inputWeights{i}));
        maxMask3D = repmat(maxMask, 1, 1, 3); % Create 3D mask for RGB channels
        maxWeightTemp(maxMask3D) = 1;  % Assign weights
        maxWeightMask{i} = maxWeightTemp;
    end
    
    % Step 2: Build Gaussian pyramids for each image and corresponding weights
    for i = 1:numImages
        gaussianPyramids{i, 1} = double(inputImages{i}) / 255;  % Normalize image
        weightPyramids{i, 1} = maxWeightMask{i};                % Set initial weights
        
        for j = 2:numBands
            % Apply Gaussian filter and resize for next band
            gaussianPyramids{i, j} = imresize(imfilter(gaussianPyramids{i, j-1}, gaussFilter), 0.5);
            weightPyramids{i, j} = imresize(imfilter(weightPyramids{i, j-1}, gaussFilter), 0.5, 'bilinear');
        end
    end
    
    % Step 3: Build Laplacian pyramids from Gaussian pyramids
    for i = 1:numImages
        for j = 1:numBands-1
            h = size(gaussianPyramids{i, j}, 1);
            w = size(gaussianPyramids{i, j}, 2);
            laplacianPyramids{i, j} = gaussianPyramids{i, j} - imresize(gaussianPyramids{i, j+1}, [h, w]);
        end
        laplacianPyramids{i, numBands} = gaussianPyramids{i, numBands};
    end
    
    % Step 4: Perform multi-band blending across the pyramid bands
    for j = 1:numBands
        blendedPyramids{j} = zeros(size(laplacianPyramids{1, j}));  % Initialize blended band
        denominator = zeros(size(laplacianPyramids{1, j}));         % Initialize denominator
        
        for i = 1:numImages
            blendedPyramids{j} = blendedPyramids{j} + laplacianPyramids{i, j} .* weightPyramids{i, j};
            denominator = denominator + weightPyramids{i, j};
        end
        
        % Prevent division by zero
        denominator(denominator == 0) = Inf;
        blendedPyramids{j} = blendedPyramids{j} ./ denominator;
    end
    
    % Step 5: Reconstruct the final image from the blended Laplacian pyramid
    finalImage = blendedPyramids{numBands};
    
    for i = 1:numBands-1
        j = numBands - i;
        h = size(blendedPyramids{j}, 1);
        w = size(blendedPyramids{j}, 2);
        finalImage = blendedPyramids{j} + imresize(finalImage, [h, w]);
    end
    
    % Convert to uint8 for final output
    blendedImage = uint8(255 .* finalImage);
end
