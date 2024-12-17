 %% VERTICAL PANORAMA 

clear; clc; close all;
numImages = 3;
img = cell(1, numImages);
imgGray = cell(1, numImages);

% Load and rotate images by 90 degrees counterclockwise
for i = 1:numImages
    img{i} = imread(fullfile('data/raw/vertical', sprintf('%d.jpg', 3+ i)));
    img{i} = imrotate(img{i}, 90); % Rotate image
end

% Convert images to grayscale
for i = 1:numImages
    imgGray{i} = rgb2gray(img{i});
end


% RANSAC parameters
numIterations = 1000; 
inlierThreshold = 2.5; % Threshold for error in pixels

homographies = cell(1, numImages - 1); % To store homographies

% Loop over pairs of consecutive images to detect features and match
for i = 1:numImages-1
    ptsCur = detectSURFFeatures(imgGray{i});
    ptsNext = detectSURFFeatures(imgGray{i+1});
    % Extract feature descriptors
    [featuresCur, validPtsCur] = extractFeatures(imgGray{i}, ptsCur);
    [featuresNext, validPtsNext] = extractFeatures(imgGray{i+1}, ptsNext);
    % Match features
    indexPairs = matchFeatures(featuresCur, featuresNext, 'Unique', true, 'MaxRatio', 0.8);
    matchPtsCur = validPtsCur(indexPairs(:, 1));
    matchPtsNext = validPtsNext(indexPairs(:, 2));
    % RANSAC to find homographies
    bestHomography = [];
    bestInlierNum = 0;
    bestInlierIndex = [];
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
HRef = cell(1, numImages);
HRef{mid} = eye(3);

% Compute cumulative homographies
for i = mid-1:-1:1
    H_CurToNext = homographies{i};
    HRef{i} = H_CurToNext * HRef{i+1};
end
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

panorama = zeros([pHeight, pWidth, 3], 'like', img{1});
viewpoint = imref2d([pHeight, pWidth], [floor(min(maxX(:))), ceil(max(maxX(:)))], [floor(min(maxY(:))), ceil(max(maxY(:)))]);
for i = 1:numImages
    % Img is warped into panorama coordinates
    warpedImg = imwarp(img{i}, projective2d(HRef{i}'), 'OutputView', viewpoint);
    % Mask of the warped image is made and the warped image is overlayed on
    % panorama
    mask = imwarp(true(size(img{i},1), size(img{i},2)), projective2d(HRef{i}'), 'OutputView', viewpoint);
    mask = repmat(mask, [1, 1, size(panorama, 3)]);  % Expand mask to match 3rd dimension of image

    panorama(mask) = warpedImg(mask);  % Overlay warped image onto the panorama
end
% Rotate the panorama back for vertical display
finalPanorama = imrotate(panorama, -90);

figure;
imshow(finalPanorama);
title('Vertical Panorama ');

% Function Definitions
% Function for Homography
function H = Homography(pts1, pts2)
    [pts1Norm, M1] = normalisePoints(pts1);
    [pts2Norm, M2] = normalisePoints(pts2);
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
    [~, ~, V] = svd(A);
    h = V(:, end);
    HNorm = reshape(h, [3, 3])';
    H = inv(M2) * HNorm * M1;
    H = H / H(3,3);
end

% Function for Normalising Points
function [ptsNorm, M] = normalisePoints(pts)
    meanPt = mean(pts,1);
    distances = sqrt(sum((pts-meanPt).^2, 2));
    avg_dist = mean(distances);
    std = sqrt(2) / avg_dist;
    M = [std, 0, -std * meanPt(1); 0, std, -std * meanPt(2); 0, 0, 1];
    numPoints = size(pts, 1);
    ptsHomogeneous = [pts, ones(numPoints, 1)]';
    normHomogeneous = M * ptsHomogeneous;
    ptsNorm = normHomogeneous(1:2, :)';
end

% Function to Compute Inliers
function [consensusSet, InliersNum] = computeInliers(H, pts1, pts2, threshold)
    numPoints = size(pts1, 1);
    consensusSet = false(numPoints, 1);
    distances = zeros(numPoints, 1);
    ptsHomogeneous1 = [pts1, ones(numPoints, 1)]';
    ptsHomogeneous2 = [pts2, ones(numPoints, 1)]';
    pts1ToPts2 = H * ptsHomogeneous1; 
    pts1ToPts2 = pts1ToPts2 ./ pts1ToPts2(3, :);
    projectedPts2 = pts1ToPts2(1:2, :)';
    for i = 1:numPoints
        distances(i) = norm(projectedPts2(i, :) - pts2(i, :));
    end
    consensusSet = distances < threshold;
    InliersNum = sum(consensusSet);
end


