%% Feature Extraction

imageDatastoreReader(imageDatastore("images\train\nonsunset\"), "features\train_nonsunset.csv"); 
imageDatastoreReader(imageDatastore("images\train\sunset\"), "features\train_sunset.csv");

imageDatastoreReader(imageDatastore("images\validate\nonsunset\"), "features\val_nonsunset.csv"); 
imageDatastoreReader(imageDatastore("images\validate\sunset\"), "features\val_sunset.csv");

imageDatastoreReader(imageDatastore("images\test\nonsunset\"), "features\test_nonsunset.csv"); 
imageDatastoreReader(imageDatastore("images\test\sunset\"), "features\test_sunset.csv"); 

function features = imageDatastoreReader(datastore, filename)
% Example of using an image datastore.
    fprintf("Processing images for %s\n", filename); 

    nBlocks = 7; 
    nImages = numel(datastore.Files);
    
    features = zeros(nImages, nBlocks * nBlocks * 6); 
    row = 1;
    for i = 1:nImages 
        [img, fileinfo] = readimage(datastore, i);
        % fileinfo struct with filename and another field.

        featureVector = featureExtract(img, nBlocks);
        features(row,:) = featureVector;
        row = row + 1;
    end
    writematrix(features, filename); 
end

function featureVector = featureExtract(img, nBlocks)
    nColorChannels = 3; 
    nStatistics = 2;

    featureVector = zeros(nBlocks, nBlocks, nColorChannels * nStatistics);  

    % figure(1); 
    % imshow(img); 
    % Convert the image to LST space 
    img = rgb2lst(img); 

    % figure(2); 
    for i = 0:(nBlocks - 1)
        for j = 0:(nBlocks - 1)
            % Extract the desired block 
            block = splitImage(img, nBlocks, i, j); 
            
            % Calculate mean and std  
            featureVector(i + 1, j + 1, :) = blockStats(block); 
        end
    end

    % And now it's actually a vector 
    featureVector = permute(featureVector, [3, 1, 2]);
    featureVector = reshape(featureVector, [294, 1]); 
end

function block = splitImage(img, nBlocks, i, j)


    gridBlockHeight = floor(size(img, 1) / nBlocks); 
    gridBlockWidth = floor(size(img, 2) / nBlocks); 


    start_x = j * gridBlockWidth + 1;
    end_x =  (j + 1) * gridBlockWidth;  
    
    start_y = i * gridBlockHeight + 1; 
    end_y = (i+1) * gridBlockHeight; 
    
    block = img(start_y : end_y, start_x : end_x, :); 
    % disp( i*nBlocks + j + 1); 

    % display image chopped into blocks 
    % subplot(nBlocks, nBlocks, i*nBlocks + j + 1); 
    % imshow(block) 
    
end

function imgLST = rgb2lst(img)
    
    imgLST = zeros(size(img)); 

    imgRed = double(img(:, :, 1)); 
    imgGreen = double(img(:, :, 2)); 
    imgBlue = double(img(:, :, 3)); 

    % L Band
    imgLST(:, :, 1) = imgRed + imgGreen + imgBlue; 

    % S Band
    imgLST(:, :, 2) = imgRed - imgBlue; 
   
    % T Band 
    imgLST(:, :, 3) = imgRed - (2* imgGreen) + imgBlue; 
    
end

function features = blockStats(block)
   
    blockL = block(:, :, 1); 
    blockS = block(:, :, 2); 
    blockT = block(:, :, 3); 

    nColorChannels = 3; 
    nStatistics = 2;

    features = zeros(nColorChannels * nStatistics, 1); 

    features(1) = mean(mean(blockL));
    features(2) = std(blockL(:)); 

    features(3) = mean(mean(blockS));
    features(4) = std(blockS(:)); 

    features(5) = mean(mean(blockT));
    features(6) = std(blockT(:)); 

end