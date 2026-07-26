
% Conv-SVM is AlexNet frozen as a feature extractor (fc7, 4096-D),
% features fed to a standalone RBF SVM. 
% Conv-Transfer is AlexNet's 1000-class head replaced with a 2-class head and
% retrained by backpropagation with the conv trunk frozen.
%
% Requires rocCurve.m on the path
% Originally written to be run cell by cell in matlab online but turns out
% my CPU was fine with training this within >10 min

%% Section 0: configuration and environment
clc; clear;
rng(0);                          % folds and weight init

FEATURE_LAYER = "fc7";           % 4096-D. Alternatives: "relu7", "pool5" (9216-D)
MINI_BATCH    = 32;
MAX_EPOCHS    = 30;              % in practice I've found we meet validation criterion way before 30
LEARN_RATE    = 1e-4;
USE_AUGMENT   = false;           % true adds random horizontal flips

if ~exist('results','dir'); mkdir('results'); end
if ~exist('features','dir'); mkdir('features'); end

% machine specs I'll fill these in later 
fprintf('=== Environment ===\n');
fprintf('MATLAB           : %s\n', version);
fprintf('Platform         : %s\n', computer);
fprintf('Physical cores   : %d\n', feature('numcores'));
try
    g = gpuDevice;
    fprintf('GPU              : %s\n', g.Name);
catch
    fprintf('GPU              : none (CPU-only training)\n');
end

%% Section 1: datastores
% Labels come from folder names, so they are categorical in ALPHABETICAL
% order where nonsunset = 1, sunset = 2. Hardcoding numbers is bad and I
% hate it
trainDS = imageDatastore('images/train',    'IncludeSubfolders',true,'LabelSource','foldernames');
valDS   = imageDatastore('images/validate', 'IncludeSubfolders',true,'LabelSource','foldernames');
testDS  = imageDatastore('images/test',     'IncludeSubfolders',true,'LabelSource','foldernames');

classNames = categories(trainDS.Labels);
sunsetIdx  = find(string(classNames) == "sunset");
assert(~isempty(sunsetIdx), 'No "sunset" folder found - check image paths.');

fprintf('\n=== Data ===\n');
fprintf('train %d  validate %d  test %d\n', ...
    numel(trainDS.Files), numel(valDS.Files), numel(testDS.Files));
disp(countEachLabel(trainDS));

net = imagePretrainedNetwork("alexnet", NumClasses=numel(classNames));
inputSize = net.Layers(1).InputSize;
fprintf('AlexNet input size: %d x %d x %d\n', inputSize);

if USE_AUGMENT
    augmenter = imageDataAugmenter('RandXReflection', true);
    augTrain  = augmentedImageDatastore(inputSize(1:2), trainDS, ...
        'ColorPreprocessing','gray2rgb', 'DataAugmentation', augmenter);
else
    augTrain  = augmentedImageDatastore(inputSize(1:2), trainDS, ...
        'ColorPreprocessing','gray2rgb');
end
augVal  = augmentedImageDatastore(inputSize(1:2), valDS,  'ColorPreprocessing','gray2rgb');
augTest = augmentedImageDatastore(inputSize(1:2), testDS, 'ColorPreprocessing','gray2rgb');

% Feature extraction shouldn't participate in augmentation so I put these in a separate
% datastore
augTrainClean = augmentedImageDatastore(inputSize(1:2), trainDS, ...
    'ColorPreprocessing','gray2rgb');

% check on a handful of images before starting the real run
sanity = read(subset(augTrainClean, 1:8));
fprintf('Sanity batch: %s, first image %s\n', ...
    class(sanity), mat2str(size(sanity.input{1})));
reset(augTrainClean);

% +1 = sunset, -1 = nonsunset
yTrain = 2*(trainDS.Labels == "sunset") - 1;
yVal   = 2*(valDS.Labels   == "sunset") - 1;
yTest  = 2*(testDS.Labels  == "sunset") - 1;

%% Section 2a: Conv-SVM fc7 feature extraction
fprintf('\n=== Conv-SVM: extracting %s features ===\n', FEATURE_LAYER);

tFeat = tic;
Xtr = extractFeatures(net, augTrainClean, FEATURE_LAYER, numel(trainDS.Files));
Xva = extractFeatures(net, augVal,        FEATURE_LAYER, numel(valDS.Files));
Xte = extractFeatures(net, augTest,       FEATURE_LAYER, numel(testDS.Files));
featTime = toc(tFeat);

fprintf('Feature dimensionality: %d  (vs 294 for the LST-SVM)\n', size(Xtr,2));
fprintf('Extraction: %.1f s for %d images (%.1f ms/image)\n', ...
    featTime, numel(trainDS.Files)+numel(valDS.Files)+numel(testDS.Files), ...
    1000*featTime/(numel(trainDS.Files)+numel(valDS.Files)+numel(testDS.Files)));

save('features/alexnet_features.mat', 'Xtr','Xva','Xte', ...
     'yTrain','yVal','yTest','FEATURE_LAYER','featTime','-v7.3');

%% Section 2b: Conv-SVM 
% Conv-Transfer needs a validation set,
% so it uses the original train/validate split instead of what we did in part 1
% remember to mention this in section 4 
Xcv = [Xtr; Xva];
ycv = [yTrain; yVal];
fprintf('\n=== Conv-SVM: grid search on %d images ===\n', numel(ycv));

% fc7 activations are ReLU-sparse and 4096-D, so Part 1 sigma values are
% irrelavant so we need new ones
sigmaCoarse = logspace(0, 3, 7);      % 1 ... 1000
Ccoarse     = logspace(-1, 2, 4);     % 0.1 ... 100

[accC, svC] = gridSearchSVM(Xcv, ycv, sigmaCoarse, Ccoarse, 5);
plotGrids(sigmaCoarse, Ccoarse, accC, svC, 'coarse');

[~, k] = max(accC(:));
[ia, ib] = ind2sub(size(accC), k);
sig0 = sigmaCoarse(ib); C0 = Ccoarse(ia);
fprintf('Best coarse: sigma=%.4g C=%.4g acc=%.4f\n', sig0, C0, accC(ia,ib));

sigmaZoom = logspace(log10(sig0)-0.5, log10(sig0)+0.5, 7);
Czoom     = logspace(log10(C0)-0.5,   log10(C0)+0.5,   5);
[accZ, svZ] = gridSearchSVM(Xcv, ycv, sigmaZoom, Czoom, 5);
plotGrids(sigmaZoom, Czoom, accZ, svZ, 'zoom');

% Among hyperparameters within 0.5% of the best accuracy, take the one with
% the fewest support vectors: overfitting fix
tied = accZ >= max(accZ(:)) - 0.005;
svPick = svZ; svPick(~tied) = Inf;
[~, sel] = min(svPick(:));
[sa, sb] = ind2sub(size(accZ), sel);
bestSigmaConv = sigmaZoom(sb);
bestCConv     = Czoom(sa);
cvAccConv     = accZ(sa,sb);

fprintf('Chosen: sigma=%.4g C=%.4g  cvAcc=%.4f  SV%%=%.1f  (%d tied)\n', ...
    bestSigmaConv, bestCConv, cvAccConv, 100*svZ(sa,sb), nnz(tied));

%% Section 2c: Conv-SVM final model and scores
tTrain = tic;
convSVM = fitcsvm(Xcv, ycv, 'KernelFunction','rbf', 'Standardize',true, ...
    'ClassNames',[-1 1], 'BoxConstraint',bestCConv, 'KernelScale',bestSigmaConv);
convSvmTrainTime = toc(tTrain);

posCol = find(convSVM.ClassNames == 1);
tPred = tic;
[predConvSVM, distConvSVM] = predict(convSVM, Xte);
convSvmPredTime = toc(tPred);

convSvmScores = distConvSVM(:, posCol);
accConvSVM    = mean(predConvSVM == yTest);
svPctConv     = 100 * sum(convSVM.IsSupportVector) / numel(ycv);

fprintf('\n=== Conv-SVM results ===\n');
fprintf('Test accuracy : %.4f\n', accConvSVM);
fprintf('Support vectors: %.1f%% of %d training images\n', svPctConv, numel(ycv));
fprintf('SVM train %.1f s; predict %d images %.2f s (%.2f ms/image)\n', ...
    convSvmTrainTime, numel(yTest), convSvmPredTime, 1000*convSvmPredTime/numel(yTest));
fprintf('End-to-end per test image incl. feature extraction: %.1f ms\n', ...
    1000*(featTime/(numel(trainDS.Files)+numel(valDS.Files)+numel(testDS.Files)) ...
          + convSvmPredTime/numel(yTest)));

save('results/conv_svm.mat', 'convSvmScores','predConvSVM','accConvSVM', ...
    'bestSigmaConv','bestCConv','cvAccConv','svPctConv', ...
    'accC','svC','accZ','svZ','sigmaCoarse','Ccoarse','sigmaZoom','Czoom', ...
    'convSvmTrainTime','convSvmPredTime');

%% Section 3: Conv-Transfer, fine tuning AlexNEt
% Freeze conv1 to conv5 and retrain fc6, fc7 and the
% new fc8. Turns out this barely helps runtime but I thought I was being
% real smart by doing this oh well
fprintf('\n=== Conv-Transfer: preparing network ===\n');

tNet = imagePretrainedNetwork("alexnet", NumClasses=numel(classNames));

% The freshly initialised head needs to move faster than pretrained weights
tNet = setLearnRateFactor(tNet, "fc8/Weights", 20);
tNet = setLearnRateFactor(tNet, "fc8/Bias",    20);

% Freeze every learnable belonging to a conv layer.
frozen = strings(0);
for i = 1:size(tNet.Learnables, 1)
    lname = string(tNet.Learnables.Layer(i));
    pname = string(tNet.Learnables.Parameter(i));
    if startsWith(lname, "conv")
        tNet = setLearnRateFactor(tNet, lname + "/" + pname, 0);
        frozen(end+1) = lname + "/" + pname;   %#ok<SAGROW>
    end
end
fprintf('Froze %d learnables: %s\n', numel(frozen), strjoin(unique(frozen), ', '));

% Uncomment to eyeball the modified architecture before spending compute:
% this also wound up being largely unnecessary
% analyzeNetwork(tNet);

iterPerEpoch = floor(numel(trainDS.Files) / MINI_BATCH);
opts = trainingOptions("sgdm", ...
    InitialLearnRate  = LEARN_RATE, ...
    MiniBatchSize     = MINI_BATCH, ...
    MaxEpochs         = MAX_EPOCHS, ...
    ValidationData    = augVal, ...
    ValidationFrequency = iterPerEpoch, ...   % once per epoch, so patience is roughly equal to epochs
    ValidationPatience  = 3, ...
    Shuffle           = "every-epoch", ...
    Metrics           = "accuracy", ...       
    ExecutionEnvironment = "auto", ...
    Plots             = "training-progress", ...
    Verbose           = true);

fprintf('Training: %d iterations/epoch, max %d epochs. Capture the plot when it finishes.\n', ...
    iterPerEpoch, MAX_EPOCHS);

tTT = tic;
% LEGACY: convNet = trainNetwork(augTrain, layers, opts);
convNet = trainnet(augTrain, tNet, "crossentropy", opts);
transferTrainTime = toc(tTT);
fprintf('Conv-Transfer training: %.1f s (%.1f min)\n', ...
    transferTrainTime, transferTrainTime/60);

%% Section 4: Conv-Transfer evaluation, then all three ROCs together
tTP = tic;
% LEGACY: [lbl, probs] = classify(convNet, augTest);
probsTransfer = minibatchpredict(convNet, augTest);
transferPredTime = toc(tTP);

predTransfer     = scores2label(probsTransfer, classNames);
transferScores   = double(probsTransfer(:, sunsetIdx));   % softmax, in [0 1]
accTransfer      = mean(predTransfer == testDS.Labels);
valAccTransfer   = mean(scores2label(minibatchpredict(convNet, augVal), classNames) == valDS.Labels);

fprintf('\n=== Conv-Transfer results ===\n');
fprintf('Validation accuracy: %.4f\n', valAccTransfer);
fprintf('Test accuracy      : %.4f\n', accTransfer);
fprintf('Train %.1f s; predict %d images %.1f s (%.1f ms/image)\n', ...
    transferTrainTime, numel(yTest), transferPredTime, ...
    1000*transferPredTime/numel(yTest));

save('results/conv_transfer.mat', 'transferScores','accTransfer', ...
    'valAccTransfer','transferTrainTime','transferPredTime', ...
    'MINI_BATCH','LEARN_RATE','MAX_EPOCHS');

% Confusion matrices for the report
figure; set(gcf,'Color','w');
tiledlayout(1,2);
nexttile; confusionchart(testDS.Labels, predConvSVM_cat(predConvSVM, classNames));
title('Conv-SVM');
nexttile; confusionchart(testDS.Labels, predTransfer);
title('Conv-Transfer');

figure; clf; set(gcf,'Color','w'); hold on;
if isfile('results/lst_svm_results.mat')
    S = load('results/lst_svm_results.mat');
    lstScores = S.sunsetScore;
    [~,~,aucLST] = rocCurve(lstScores, yTest, 'LST-SVM', 0);
else
    warning('results/lst_svm_results.mat not found - LST-SVM curve omitted.');
    aucLST = NaN;
end
[~,~,aucConvSVM]  = rocCurve(convSvmScores,  yTest, 'Conv-SVM',      0);
[~,~,aucTransfer] = rocCurve(transferScores, yTest, 'Conv-Transfer', 0.5);
plot([0 1],[0 1],'k--','LineWidth',1,'DisplayName','chance');
title('Test-set ROC, all classifiers');
hold off;

fprintf('\n=== AUC summary ===\n');
fprintf('LST-SVM       %.4f\nConv-SVM      %.4f\nConv-Transfer %.4f\n', ...
    aucLST, aucConvSVM, aucTransfer);

%% Section 5
if isfile('results/picked8.mat')
    P = load('results/picked8.mat');
    names = fieldnames(P.picked);
    files = cellfun(@(f) P.picked.(f), names, 'UniformOutput', false);

    ds8    = imageDatastore(files);
    aug8   = augmentedImageDatastore(inputSize(1:2), ds8, 'ColorPreprocessing','gray2rgb');
    probs8 = minibatchpredict(convNet, aug8);

    % The Conv-SVM needs fc7 features for the same eight images
    F8      = extractFeatures(net, aug8, FEATURE_LAYER, numel(files));
    [~, d8] = predict(convSVM, F8);

    T = table(names, ...
        d8(:, posCol), ...
        probs8(:, sunsetIdx), ...
        'VariableNames', {'Category','ConvSVM_score','ConvTransfer_pSunset'});
    disp(T);
    writetable(T, 'results/eight_image_scores.csv');

    figure; set(gcf,'Color','w');
    for i = 1:numel(files)
        subplot(2,4,i);
        imshow(readimage(ds8, i));
        title(sprintf('%s\nSVM %.2f | CNN %.2f', ...
            strrep(names{i},'_','-'), d8(i,posCol), probs8(i,sunsetIdx)), ...
            'FontSize', 8);
    end
else
    warning(['results/picked8.mat not found. Re-run the corrected Part 1 ' ...
             'script with C=3, sigma=17 to regenerate the eight filenames.']);
end

%% ---------- local functions ----------

function F = extractFeatures(net, augDS, layerName, nExpected)
% Forward-pass a datastore and return an nImages-by-nFeatures single matrix.
    reset(augDS);
    % LEGACY: F = activations(net, augDS, char(layerName), 'OutputAs','rows');
    F = minibatchpredict(net, augDS, Outputs=layerName);
    F = squeeze(F);
    if size(F,1) ~= nExpected && size(F,2) == nExpected
        F = F.';          % guard against a channel-first return
    end
    F = double(reshape(F, nExpected, []));
    assert(size(F,1) == nExpected, ...
        'Feature matrix has %d rows, expected %d.', size(F,1), nExpected);
end

function [accGrid, svGrid] = gridSearchSVM(X, y, sigmas, Cs, nFolds)
% Stratified k-fold grid search recording validation accuracy and the
% support-vector fraction for every (sigma, C) cell.
    cvp     = cvpartition(y, 'KFold', nFolds);   % stratified, unlike a reshape
    accGrid = zeros(numel(Cs), numel(sigmas));
    svGrid  = zeros(numel(Cs), numel(sigmas));

    for a = 1:numel(Cs)
        for b = 1:numel(sigmas)
            accSum = 0; svSum = 0;
            for k = 1:nFolds
                tr = training(cvp, k);
                va = test(cvp, k);
                m = fitcsvm(X(tr,:), y(tr), 'KernelFunction','rbf', ...
                    'Standardize',true, 'ClassNames',[-1 1], ...
                    'BoxConstraint',Cs(a), 'KernelScale',sigmas(b));
                accSum = accSum + mean(predict(m, X(va,:)) == y(va));
                svSum  = svSum  + sum(m.IsSupportVector)/sum(tr);
            end
            accGrid(a,b) = accSum / nFolds;
            svGrid(a,b)  = svSum  / nFolds;
            fprintf('  sigma=%9.4g  C=%8.4g  acc=%.4f  SV%%=%5.1f\n', ...
                sigmas(b), Cs(a), accGrid(a,b), 100*svGrid(a,b));
        end
    end
end

function plotGrids(sigmas, Cs, acc, sv, tag)
% Two heatmaps per grid: validation accuracy and support-vector fraction.
    xl = compose('%.3g', sigmas);
    yl = compose('%.3g', Cs);

    figure; set(gcf,'Color','w');
    h = heatmap(xl, yl, round(100*acc, 2));
    h.XLabel = 'Kernel scale (sigma)';
    h.YLabel = 'Box constraint (C)';
    h.Title  = sprintf('Conv-SVM validation accuracy (%%) - %s grid', tag);

    figure; set(gcf,'Color','w');
    h = heatmap(xl, yl, round(100*sv, 1));
    h.XLabel = 'Kernel scale (sigma)';
    h.YLabel = 'Box constraint (C)';
    h.Title  = sprintf('Conv-SVM support vectors (%%) - %s grid', tag);
end

function c = predConvSVM_cat(pred, classNames)
% Map the SVM's +1/-1 output onto the categorical class names so that
% confusionchart can compare it against the datastore labels.
    c = categorical(repmat(string(classNames{1}), numel(pred), 1), string(classNames));
    c(pred == 1) = string(classNames{find(string(classNames) == "sunset")});
end