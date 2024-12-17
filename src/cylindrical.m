

%% CYLINDRICAL PANORAMA

% Similar implementation to horizontal


clear; clc; close all;
numImages = 5;

% RANSAC parameters
numIterations = 1000;
inlierThreshold = 2.5; 
minInliers = 30; 
focalLength = 700; 
img = cell(1, numImages);
imgGray = cell(1, numImages);
for i = 1:numImages
    img{i} = imread(fullfile('data/images/Q2/sequence1/', sprintf('IMG_315%d.jpg', 3 + i)));
    if size(img{i}, 3) == 3
        imgGray{i} = rgb2gray(img{i});
    else
        imgGray{i} = img{i};
    end
end

% CONVERTING TO CYLINDRICAL PROJECTION
for i = 1:numImages
    [imgCyl{i}, maskCyl{i}] = cylindricalProjection(img{i}, focalLength);
    imgGrayCyl{i} = rgb2gray(imgCyl{i});
end

homographies = cell(1, numImages - 1);

for i = 1:numImages-1
    ptsCur = detectSURFFeatures(imgGrayCyl{i}, 'MetricThreshold', 500);
    ptsNext = detectSURFFeatures(imgGrayCyl{i+1}, 'MetricThreshold', 500);
    [featuresCur, validPtsCur] = extractFeatures(imgGrayCyl{i}, ptsCur);
    [featuresNext, validPtsNext] = extractFeatures(imgGrayCyl{i+1}, ptsNext);
    indexPairs = matchFeatures(featuresCur, featuresNext, 'Unique', true, 'MaxRatio', 0.7);
    matchedPtsCur = validPtsCur(indexPairs(:, 1));
    matchedPtsNext = validPtsNext(indexPairs(:, 2));
    pts1 = matchedPtsCur.Location;
    pts2 = matchedPtsNext.Location;
    bestInlierCount = 0;
    bestH = [];
    bestInlierIdx = [];
    for iter = 1:numIterations
        if size(pts1, 1) < 4
            break; 
        end
        idx = randperm(size(pts1, 1), 4);
        sample_pts1 = pts1(idx, :);
        sample_pts2 = pts2(idx, :);
        H = Homography(sample_pts1, sample_pts2);
        [inlierIdx, numInliers] = computeInliers(H, pts1, pts2, inlierThreshold);
        if numInliers > bestInlierCount
            bestInlierCount = numInliers;
            bestH = H;
            bestInlierIdx = inlierIdx;
        end
    end

    inlierPoints1 = matchedPtsCur(bestInlierIdx, :);
    inlierPoints2 = matchedPtsNext(bestInlierIdx, :);
    bestH = Homography(inlierPoints1.Location, inlierPoints2.Location);
    homographies{i} = bestH;
end

mid = ceil(numImages / 2);
H_to_ref = cell(1, numImages);
H_to_ref{mid} = eye(3);

for i = mid-1:-1:1
    H_i_to_iPlus1 = homographies{i};
    H_to_ref{i} = H_i_to_iPlus1 * H_to_ref{i+1};
    H_to_ref{i} = H_to_ref{i} / H_to_ref{i}(3, 3);
end

for i = mid+1:numImages
    H_iMinus1_to_i = homographies{i-1};
    H_to_ref{i} = inv(H_iMinus1_to_i) * H_to_ref{i-1};
    H_to_ref{i} = H_to_ref{i} / H_to_ref{i}(3, 3);
end
xLimits = zeros(numImages, 2);
yLimits = zeros(numImages, 2);

for i = 1:numImages
    [height, width, ~] = size(imgCyl{i});
    % Corners of the img
    corners = [1, 1; width, 1; width, height; 1, height];
    % Cumulative Homographies transforms corners and panorama limits are
    % updated
    [xTransformed, yTransformed] = transformPointsForward(projective2d(H_to_ref{i}'), corners(:,1), corners(:,2));
    xLimits(i,:) = [min(xTransformed), max(xTransformed)];
    yLimits(i,:) = [min(yTransformed), max(yTransformed)];
end

xMin = floor(min(xLimits(:)));
xMax = ceil(max(xLimits(:)));
yMin = floor(min(yLimits(:)));
yMax = ceil(max(yLimits(:)));

panoramaWidth = xMax - xMin + 1;
panoramaHeight = yMax - yMin + 1;
maxPanoramaWidth = 10000; 
maxPanoramaHeight = 5000;

if panoramaWidth > maxPanoramaWidth || panoramaHeight > maxPanoramaHeight
    widthScale = maxPanoramaWidth / panoramaWidth;
    heightScale = maxPanoramaHeight / panoramaHeight;
    scaleFactor = min(widthScale, heightScale);
    panoramaWidth = round(panoramaWidth * scaleFactor);
    panoramaHeight = round(panoramaHeight * scaleFactor);
    xMinScaled = xMin * scaleFactor;
    xMaxScaled = xMax * scaleFactor;
    yMinScaled = yMin * scaleFactor;
    yMaxScaled = yMax * scaleFactor;
    panoramaView = imref2d([panoramaHeight, panoramaWidth], [xMinScaled, xMaxScaled], [yMinScaled, yMaxScaled]);
else
    panoramaView = imref2d([panoramaHeight, panoramaWidth], [xMin, xMax], [yMin, yMax]);
end

panorama = zeros(panoramaHeight, panoramaWidth, 3, 'like', img{1});

for i = 1:numImages
    warpedImage = imwarp(imgCyl{i}, projective2d(H_to_ref{i}'), 'OutputView', panoramaView);
    warpedMask = imwarp(maskCyl{i}, projective2d(H_to_ref{i}'), 'OutputView', panoramaView);
    panorama = blendImages(panorama, warpedImage, warpedMask);
end

figure;
imshow(panorama);
title('Stitched Panorama using Cylindrical Projection');

% CYLINDRICAL PROJECTION FUNCTION 
function [imgCyl, maskCyl] = cylindricalProjection(img, f)
     [h, w, c] = size(img);
    [X, Y] = meshgrid(1:w, 1:h);
    x = (X - w/2) / f;
    y = (Y - h/2) / f;
    xCyl = sin(x);
    yCyl = y ./ cos(x);
    Xc = f * xCyl + w/2;
    Yc = f * yCyl + h/2;
    imgCyl = zeros(h, w, c, 'like', img);
    for ch = 1:c
        imgCyl(:,:,ch) = interp2(double(img(:,:,ch)), Xc, Yc, 'linear', 0);
    end
    maskCyl = interp2(double(ones(h,w)), Xc, Yc, 'linear', 0) > 0;
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

function panorama = blendImages(panorama, warpedImage, mask)
    panorama = im2double(panorama);
    warpedImage = im2double(warpedImage);
    overlap = (panorama ~= 0) & repmat(mask, [1, 1, 3]);
    % For overlapping regions, averaging the pixel values
    panorama(overlap) = (panorama(overlap) + warpedImage(overlap)) / 2; 
    % Adding new pixels for rest
    panorama(~overlap) = panorama(~overlap) + warpedImage(~overlap);
    panorama = im2uint8(panorama);
end