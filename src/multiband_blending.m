%% MULTIBAND BLENDING
clear; clc; close all;
numImages = 3;
img = cell(1, numImages);
imgGray = cell(1, numImages);
for i = 1:numImages
    img{i} = imread(fullfile('NSH/', sprintf('medium0%d.jpg', i)));
end

% Converting images to grayscale 
imgGray = cell(1, numImages);
for i = 1:numImages
    imgGray{i} = rgb2gray(img{i});
end

% RANSAC parameters
numIterations = 2000; 
inlierThreshold = 3; % Threshold for error in pixels 

homographies = cell(1, numImages - 1); % To store homographies 
% Looping over pairs of consecutive images to detect features and match
for i = 1:numImages-1
    ptsCur = detectSURFFeatures(imgGray{i});
    ptsNext = detectSURFFeatures(imgGray{i+1});
    [featuresCur, validPtsCur] = extractFeatures(imgGray{i}, ptsCur);
    [featuresNext, validPtsNext] = extractFeatures(imgGray{i+1}, ptsNext);
    indexPairs = matchFeatures(featuresCur, featuresNext, 'Unique', true, 'MaxRatio', 0.8);
    matchPtsCur = validPtsCur(indexPairs(:, 1));
    matchPtsNext = validPtsNext(indexPairs(:, 2));
    
    bestHomography = []; % Best homography found yet
    bestInlierNum = 0; % To track max num of inliers
    bestInlierIndex = []; % Indexes of Inliers for best homography
    ptsLocationCur = matchPtsCur.Location;
    ptsLocationNext = matchPtsNext.Location;
    
    for j = 1:numIterations
        index = randperm(size(ptsLocationCur, 1), 4);
        randPts1 = ptsLocationCur(index, :);
        randPts2 = ptsLocationNext(index, :);
        H = Homography(randPts1, randPts2);
        [inlierIdx, numInliers] = computeInliers(H, ptsLocationCur, ptsLocationNext, inlierThreshold);
        if numInliers > bestInlierNum
            bestInlierNum = numInliers;
            bestHomography = H;
            bestInlierIndex = inlierIdx;
        end
    end
    inlierPts1 = matchPtsCur(bestInlierIndex, :);
    inlierPts2 = matchPtsNext(bestInlierIndex, :);
    bestHomography = Homography(inlierPts1.Location, inlierPts2.Location);
    homographies{i} = bestHomography; 
end

% Panorama limits
maxX = zeros(numImages, 2);
maxY = zeros(numImages, 2);
mid = ceil(numImages / 2);
HRef = cell(1, numImages); % Cumulative homographies are stored in a cell array
HRef{mid} = eye(3); % Identity matrix for the middle image

% Computing homographies to the left of middle image
for i = mid-1:-1:1
    H_CurToNext = homographies{i}; 
    HRef{i} = H_CurToNext * HRef{i+1}; 
end

% Computing homographies to the right of middle image
for i = mid+1:numImages
    H_PrevToCur = homographies{i-1}; 
    HRef{i} = inv(H_PrevToCur) * HRef{i-1};
end

for i = 1:numImages
    [height, width, ~] = size(img{i});
    corners = [1, 1; width, 1; width, height; 1, height];
    [xTransformed, yTransformed] = transformPointsForward(projective2d(HRef{i}'), corners(:,1), corners(:,2));
    maxX(i,:) = [min(xTransformed), max(xTransformed)];
    maxY(i,:) = [min(yTransformed), max(yTransformed)];
end

% Panorama dimensions
pWidth = ceil(max(maxX(:))) - floor(min(maxX(:))) + 1;
pHeight = ceil(max(maxY(:))) - floor(min(maxY(:))) + 1;
maxPWidth = 5000; 
maxPHeight = 2500;
if pWidth > maxPWidth
    pWidth = maxPWidth;
end
if pHeight > maxPHeight
    pHeight = maxPHeight;
end

panorama = zeros([pHeight, pWidth, 3], 'like', img{1});
viewpoint = imref2d([pHeight, pWidth], [floor(min(maxX(:))), ceil(max(maxX(:)))], [floor(min(maxY(:))), ceil(max(maxY(:)))]);
% Initialize weight accumulation and result panorama
weightSum = zeros(size(panorama));
resultPanorama = zeros(size(panorama));

% Multiband Blending Setup
numBands = 5;  % Number of pyramid bands for blending
filtSize = 5;  % Filter size for the Gaussian pyramid
MBBsigma = 2;  % Sigma for Gaussian filter (for multiband blending)

% Initialize structures for multiband blending
input.bands = numBands;
input.filtSize = filtSize;
input.MBBsigma = MBBsigma;

inputImages = cell(1, numImages);
inputWeights = cell(1, numImages);

% Loop over images for warping and weight computation
for i = 1:numImages
    warpedImg = imwarp(img{i}, projective2d(HRef{i}'), 'OutputView', viewpoint);
    warpedImg = im2double(warpedImg);
    
    % Generate the weight mask for the image
    mask = imwarp(true(size(img{i}, 1), size(img{i}, 2)), projective2d(HRef{i}'), 'OutputView', viewpoint);
    mask = repmat(mask, [1, 1, size(panorama, 3)]);
    mask = double(mask);
    
    % Store warped images and their corresponding weights
    inputImages{i} = warpedImg;
    inputWeights{i} = mask;
end

% Apply multi-band blending to the set of warped images
blendedImage = multiBandBlending(input, inputImages, inputWeights);

% Display the final panorama using multiband blending
figure;
imshow(blendedImage);
title('Panorama with Multiband Blending');




% Page 6 Lecture 19 Notes 
% Here  pts - Nx2 matrix of points [(x1, y1); (x2, y2); ...]
function [ptsNorm, M] = normalisePoints(pts)
    % We get better estimates if we normalize the data
    meanPt = mean(pts,1);
    % Subtracting the means 
    distances = sqrt(sum((pts-meanPt).^2, 2)); % Euclidean distances
    avg_dist = mean(distances);
    % Computing the standard deviation equivalent to normalise distances to sqrt(2)
    std = sqrt(2) / avg_dist;
    % Constructing the normalisation transformation matrix
    M = [std, 0, -std * meanPt(1); 0, std, -std * meanPt(2); 0, 0, 1];
    % Applying the transformation to the points
    numPoints = size(pts, 1);
    ptsHomogeneous = [pts, ones(numPoints, 1)]'; % Convert to homogeneous coordinates
    normHomogeneous = M * ptsHomogeneous;    % Normalise
    ptsNorm = normHomogeneous(1:2, :)';      % Convert back to Cartesian coordinates for Nx2 matrix of normalised points
end


% Page 5 Lecture Notes Chapter 19
% For N sets of corresponding point pairs (xi,yi) and (x′i, yi′)
function H = Homography(pts1, pts2)
    % pts1 and pts2 are Nx2 matrices of corresponding points
    % Normalising points
    [pts1Norm, M1] = normalisePoints(pts1);
    [pts2Norm, M2] = normalisePoints(pts2);
    % Building 2NX9 A matrix
    numPoints = size(pts1Norm, 1);
    A = zeros(2*numPoints, 9);
    for i = 1:numPoints
        x = pts1Norm(i, 1);
        y = pts1Norm(i, 2);
        xPrime = pts2Norm(i, 1);
        yPrime = pts2Norm(i, 2);
        A(2*i-1,:) = [0, 0, 0, -x, -y, -1, yPrime*x, yPrime*y, yPrime];
        A(2*i,:)   = [x, y, 1, 0, 0, 0, -xPrime*x, -xPrime*y, -xPrime];
    end
    % Solving Ah = 0 using SVD to find smallest eigenvalue
    [~, ~, V] = svd(A);
    h = V(:, end);
    HNorm = reshape(h, [3, 3])';
    % Denormalise
    H = inv(M2) * HNorm * M1;
    H = H / H(3,3);
end

% Page 7 Lecture 19 Notes 
function [consensusSet, InliersNum] = computeInliers(H, pts1, pts2, threshold)
    % The idea is to compute inliers based on computing norm between
    % projected coordinates and mapped coordinate
    % pts1 and pts2 are Nx2 matrices of corresponding points
    % numInliers is the number of inliers
    numPoints = size(pts1, 1);
    consensusSet = false(numPoints, 1);
    distances = zeros(numPoints, 1); % Array to store distances
    % Converting points to homogeneous coordinates
    ptsHomogeneous1 = [pts1, ones(numPoints, 1)]';
    ptsHomogeneous2 = [pts2, ones(numPoints, 1)]';
    % Mapping pts1 to pts2 using Homography matrix H
    pts1ToPts2 = H * ptsHomogeneous1; 
    pts1ToPts2 = pts1ToPts2 ./ pts1ToPts2(3, :);
    projectedPts2 = pts1ToPts2(1:2, :)'; % Transposing to get normalised (x,y)
    for i = 1:numPoints
        % Compute the distance/fit like in the lecture notes
        distances(i) = norm(projectedPts2(i, :) - pts2(i, :));
    end
    % Identify inliers
    consensusSet = distances < threshold;
    InliersNum = sum(consensusSet);
end

function blendedImage = multiBandBlending(input, images, weights)
    % Parameters
    numBands = input.bands;
    filtSize = input.filtSize;
    sigma = input.MBBsigma;
    
    % Preallocate pyramids
    numImages = length(images);
    gaussianPyramids = cell(numImages, 1);
    laplacianPyramids = cell(numImages, 1);
    weightPyramids = cell(numImages, 1);

    % Construct Gaussian filter
    gaussFilter = fspecial('gaussian', filtSize, sigma);

    % Create pyramids for each image and weight
    for i = 1:numImages
        [gaussianPyramids{i}, laplacianPyramids{i}] = buildPyramids(images{i}, numBands, gaussFilter);
        weightPyramids{i} = buildWeightPyramids(weights{i}, numBands, gaussFilter);
    end

    % Initialize blended pyramid
    blendedPyramid = cell(numBands, 1);
    for b = 1:numBands
        blendedPyramid{b} = zeros(size(laplacianPyramids{1}{b}));
        for i = 1:numImages
            blendedPyramid{b} = blendedPyramid{b} + ...
                weightPyramids{i}{b} .* laplacianPyramids{i}{b};
        end
    end

    % Collapse the blended pyramid
    blendedImage = collapsePyramid(blendedPyramid, gaussFilter);
end


function [gaussPyr, lapPyr] = buildPyramids(img, numBands, gaussFilter)
    gaussPyr = cell(numBands, 1);
    lapPyr = cell(numBands, 1);
    current = img;
    
    for b = 1:numBands
        gaussPyr{b} = imfilter(current, gaussFilter, 'same');
        if b < numBands
            downsampled = imresize(gaussPyr{b}, 0.5);
            lapPyr{b} = current - imresize(downsampled, size(current, 1:2));
            current = downsampled;
        else
            lapPyr{b} = gaussPyr{b};
        end
    end
end

function weightPyr = buildWeightPyramids(weight, numBands, gaussFilter)
    weightPyr = cell(numBands, 1);
    current = weight;
    for b = 1:numBands
        weightPyr{b} = imfilter(current, gaussFilter, 'same');
        if b < numBands
            current = imresize(weightPyr{b}, 0.5);
        end
    end
end

function img = collapsePyramid(pyr, gaussFilter)
    numBands = length(pyr);
    img = pyr{end};
    for b = numBands-1:-1:1
        img = pyr{b} + imresize(img, size(pyr{b}, 1:2));
        img = imfilter(img, gaussFilter, 'same');
    end
end
