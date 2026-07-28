% Training the SVM 

rng(0); 
RUN_GRID_SEARCH = false; 

% Load in the image data 
test_nonsunset = readmatrix("features\test_nonsunset.csv"); 
test_sunset = readmatrix("features\test_sunset.csv"); 

train_nonsunset = readmatrix("features\train_nonsunset.csv"); 
train_sunset = readmatrix("features\train_sunset.csv"); 

val_nonsunset = readmatrix("features\val_nonsunset.csv"); 
val_sunset = readmatrix("features\val_sunset.csv"); 

% Concatenate the nonsunset and sunset matrices 
% Assign labels of +1 to sunsets and -1 to nonsunsets 
% Since we're doing cross-validation, I don't need a dedicated validation
% set 
X_train = cat(1, train_nonsunset, train_sunset, val_nonsunset, val_sunset); 
y_train = cat(1, ones(size(train_nonsunset, 1), 1) * -1, ones(size(train_sunset, 1), 1), ones(size(val_nonsunset, 1), 1) * -1, ones(size(val_sunset, 1), 1)); 

X_test = cat(1, test_nonsunset, test_sunset); 
y_test= cat(1, ones(size(test_nonsunset, 1), 1) * -1, ones(size(test_sunset, 1), 1)); 

C_range = 1:1:12;
sigma_range = 15:1:25; 

if RUN_GRID_SEARCH
    [C, sigma] = GridSearchCV(X_train, y_train, C_range, sigma_range); 
else
    % Hyperparameters for ease of rerunning 
    C = 3;
    sigma = 17; 
end

disp_stat("Best Box Constraint:", C); 
disp_stat("Best Kernel Scale:", sigma); 

% Train the final SVM
net = fitcsvm(X_train,y_train,'KernelFunction','rbf', 'Standardize',true, ...
    'ClassNames',[-1, 1], 'BoxConstraint', C, 'KernelScale', sigma); 

pct_sv = size(net.SupportVectors, 1) / size(X_train, 1); 
disp_stat("Percent Support Vectors:", pct_sv); 

% Predict on the test set 
[pred, dist] = predict(net, X_test); 

posCol = find(net.ClassNames == 1); 
sunsetScore = dist(:, posCol); 

if ~exist('results', 'dir')
    mkdir('results'); 
end
save('results/lst_svm_results.mat', 'sunsetScore', 'y_test', 'pred', 'C', 'sigma', 'pct_sv'); 

analyze_performance(pred, sunsetScore, y_test);

% Find the nearest and farthest images from the margin 
find_example_images(pred, sunsetScore, y_test); 

function [best_C, best_sigma] = GridSearchCV(X_train, y_train, C_grid, sigma_grid)
n_folds = 5; 
% shuffle the data
df = cat(2, X_train, double(y_train));
df = df(randperm(size(df, 1)),:); 

X_train = df(:, 1:(size(df, 2) - 1)); 
y_train = df(:, size(df,2)); 

% partition into 5 folds 
X_train = reshape(X_train', size(X_train, 2), int64(size(X_train, 1) / n_folds), n_folds); 
X_train = permute(X_train, [2,1,3]); 
y_train = reshape(y_train, int64(size(y_train, 1) / n_folds), 1, n_folds); 

acc_grid = zeros(size(C_grid, 2), size(sigma_grid, 2)); 
sv_grid = zeros(size(C_grid, 2), size(sigma_grid, 2)); 

for i = 1:size(C_grid, 2) % iterate over the values of C 
    for j = 1:size(sigma_grid, 2)% iterate over the values of sigma 

        acc_sum = 0; 
        sv_sum = 0; 

        for k = 1:n_folds % 5-fold cross validation

            % mark the ith fold as the validation set 
            X_train_folds = cat(3, X_train(:, :, 1:k-1), X_train(:, :, k+1:n_folds)); 
            X_train_folds = reshape(permute(X_train_folds,[1 3 2]), [], size(X_train, 2), 1); 
            X_valid_fold = X_train(:, :, k); 

            y_train_folds = cat(3, y_train(:, :, 1:k-1), y_train(:, :, k+1:n_folds)); 
            y_train_folds = reshape(permute(y_train_folds,[1 3 2]), [], 1, 1); 
            y_valid_fold = y_train(:, :, k); 

            % Train model 
            net = fitcsvm(X_train_folds, y_train_folds,'KernelFunction','rbf', 'Standardize',true,'ClassNames',[-1, 1], 'BoxConstraint', C_grid(1, i), 'KernelScale', sigma_grid(1, j));

            % Predict on validation fold 
            [pred, dist] = predict(net, X_valid_fold);

            % Add validation accuracy to running total 
            acc_sum = acc_sum + (sum(sum(pred == y_valid_fold)) / size(y_valid_fold, 1));
            sv_sum = sv_sum + (sum(net.IsSupportVector) / size(X_train_folds, 1)); 

        end
        % Divide running total by number of folds and save it to the grid 
        acc_grid(i, j) = acc_sum / n_folds; 
        sv_grid(i, j) = sv_sum / n_folds; 
        fprintf("C =  %d, sigma = %d, accuracy = %.4f, support vectors = %.4f\n", C_grid(1, i), sigma_grid(1, j), acc_sum / n_folds, sv_sum / n_folds); 

    end
end

figure(1); 
surf(sigma_grid, C_grid, acc_grid); % okay fine I suppose that's cooler than matplotlib
ylabel("Box Constraint (C)"); 
xlabel("Kernel Scale (sigma)"); 
zlabel("Accuracy"); 
title("Grid Search Surface")
set(gcf, 'Color', 'w'); 

figure(4); 
surf(sigma_grid, C_grid, 100 * sv_grid); 
ylabel("Box Constraint (C)"); 
xlabel("Kernel Scale (sigma)"); 
zlabel("Support Vectors (%)"); 
title("Support Vector Surface")
set(gcf, 'Color', 'w'); 

% indices of the maximum accuracy value
[best_C_idx, best_sigma_idx] = find(acc_grid == max(max(acc_grid))); 

% return the last point found. Larger I think means trending toward
% underfitting which is generally better than overfitting
best_C  = C_grid(best_C_idx(size(best_C_idx, 1))); 
best_sigma = sigma_grid(best_sigma_idx(size(best_sigma_idx, 1))); 

end

function disp_stat(label, value)
disp(" "); 
disp(label); 
disp(value); 
end

function analyze_performance(pred, sunsetScore, y_test)
    % Plot ROC
    figure(2); clf; 
    [~, ~, auc] = rocCurve(sunsetScore, y_test, 'LST-SVM', 0); 
    title("ROC Curve"); 
    disp_stat("AUC: ", auc); 
    
    % Calculate the accuracy 
    acc = sum(sum(pred == y_test)) / size(y_test, 1);  
    disp_stat("Accuracy: ", acc); 
    
    % Confusion Matrix: 
    tp = sum(sum(pred == y_test & pred == 1)); 
    fp = sum(sum(pred ~= y_test & pred == 1)); 
    fn = sum(sum(pred ~= y_test & pred == -1)); 
    tn = sum(sum(pred == y_test & pred == -1)); 
    
    confusion_matrix = [tp fn; fp tn]; 
    disp_stat("Confusion Matrix:", confusion_matrix); 
    
    % TPR and FPR 
    tpr = tp / (tp + fn); 
    fpr = fp / (fp + tn); 
    disp_stat("True Positive Rate: ", tpr); 
    disp_stat("False Positive Rate: ", fpr); 
end

function find_example_images(pred, dist, y_test)
    % y_test has 500 nonsunsets then 498 sunsets in that order 
    y_test_nonsunset = y_test(1:500);
    y_test_sunset = y_test(501:size(y_test,1));
    
    nonsunset = imageDatastore("images\test\nonsunset"); 
    sunset = imageDatastore("images\test\sunset\"); 

    min_tn_idx = 1; 
    max_tn_idx = 1; 
    min_tn_dist = max(dist); 
    max_tn_dist = min(dist); 

    min_fp_idx = 1; 
    max_fp_idx = 1; 
    min_fp_dist = max(dist); 
    max_fp_dist = min(dist); 

    min_tp_idx = 1; 
    max_tp_idx = 1; 
    min_tp_dist = max(dist); 
    max_tp_dist = 0; 

    min_fn_idx = 1; 
    max_fn_idx = 1; 
    min_fn_dist = max(dist); 
    max_fn_dist = min(dist); 

    % Nonsunsets 
    for i = 1:500      
        % Correct - True Negatives 
        if(pred(i) == y_test_nonsunset(i))
            % Farthest from margin (most negative) 
            if(dist(i) < min_tn_dist)
                min_tn_dist = dist(i); 
                min_tn_idx = i; 
            end
            % Closest to margin (least negative) 
            if(max_tn_dist < dist(i))
                max_tn_dist = dist(i);
                max_tn_idx = i; 
            end
        end
        % Incorrect - False Positives 
        if(pred(i) ~= y_test_nonsunset(i))
            % Closest to margin (least positive) 
            if(dist(i) < min_fp_dist)
                min_fp_dist = dist(i); 
                min_fp_idx = i; 
            end
            % Farthest from margin (most positive) 
            if(max_fp_dist < dist(i))
                max_fp_dist = dist(i);
                max_fp_idx = i; 
            end

        end
    end

    % Sunsets 
    for i = 1:498      
        % Correct - True Positives 
        if(pred(500 + i) == y_test_sunset(i)) % 500 + i to skip the nonsunsets 
            % Closest to margin (least positive) 
            if(dist(500 + i) < min_tp_dist)
                min_tp_dist = dist(500 + i); 
                min_tp_idx = i; 
            end
            % Farthest from margin (most positive) 
            if(max_tp_dist < dist(500 + i))
                max_tp_dist = dist(500 + i);
                max_tp_idx = i; 
            end
        end
        % Incorrect - False Negatives 
        if(pred(500 + i) ~= y_test_sunset(i))
            % Closest to margin (least negative) 
            if(dist(500 + i) < min_fn_dist)
                min_fn_dist = dist(500 + i); 
                min_fn_idx = i; 
            end
            % Farthest from margin (most negative) 
            if(max_fn_dist < dist(500 + i))
                max_fn_dist = dist(500 + i);
                max_fn_idx = i; 
            end

        end
    end

    picked = struct( ...
        'TP_far',  sunset.Files{max_tp_idx}, ...
        'TP_near', sunset.Files{min_tp_idx}, ...
        'FN_far',  sunset.Files{min_fn_idx}, ...
        'FN_near', sunset.Files{max_fn_idx}, ...
        'FP_far',  nonsunset.Files{max_fp_idx}, ...
        'FP_near', nonsunset.Files{min_fp_idx}, ...
        'TN_far',  nonsunset.Files{min_tn_idx}, ...
        'TN_near', nonsunset.Files{max_tn_idx}); 

    picked_scores = struct( ...
        'TP_far',  max_tp_dist, ...
        'TP_near', min_tp_dist, ...
        'FN_far',  min_fn_dist, ...
        'FN_near', max_fn_dist, ...
        'FP_far',  max_fp_dist, ...
        'FP_near', min_fp_dist, ...
        'TN_far',  min_tn_dist, ...
        'TN_near', max_tn_dist); 

    if ~exist('results', 'dir')
        mkdir('results'); 
    end
    save('results/picked8.mat', 'picked', 'picked_scores'); 

    category = fieldnames(picked); 
    filename = struct2cell(picked); 
    svm_score = cell2mat(struct2cell(picked_scores)); 
    disp(table(category, svm_score, filename)); 

    figure(3); 
    set(gcf, 'Color', 'w'); 

    subplot(2, 4, 1); 
    imshow(readimage(sunset, max_tp_idx)); 
    title(sprintf("TP Distance = %.2f", max_tp_dist)); 

    subplot(2, 4, 2); 
    imshow(readimage(sunset, min_tp_idx)); 
    title(sprintf("TP Distance = %.2f", min_tp_dist)); 

    subplot(2, 4, 3); 
    imshow(readimage(sunset, min_fn_idx)); 
    title(sprintf("FN Distance = %.2f", min_fn_dist)); 

    subplot(2, 4, 4); 
    imshow(readimage(sunset, max_fn_idx));
    title(sprintf("FN Distance = %.2f", max_fn_dist)); 

    subplot(2, 4, 5); 
    imshow(readimage(nonsunset, max_fp_idx)); 
    title(sprintf("FP Distance = %.2f", max_fp_dist)); 

    subplot(2, 4, 6); 
    imshow(readimage(nonsunset, min_fp_idx)); 
    title(sprintf("FP Distance = %.2f", min_fp_dist)); 

    subplot(2, 4, 7); 
    imshow(readimage(nonsunset, max_tn_idx)); 
    title(sprintf("TN Distance = %.2f", max_tn_dist)); 

    subplot(2, 4, 8); 
    imshow(readimage(nonsunset, min_tn_idx)); 
    title(sprintf("TN Distance = %.2f", min_tn_dist)); 

end