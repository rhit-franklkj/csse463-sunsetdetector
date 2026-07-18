% Training the SVM 


% Load in the image data 
test_nonsunset = readmatrix("features\test_nonsunset.csv"); 
test_sunset = readmatrix("features\test_sunset.csv"); 

train_nonsunset = readmatrix("features\train_nonsunset.csv"); 
train_sunset = readmatrix("features\train_sunset.csv"); 

val_nonsunset = readmatrix("features\val_nonsunset.csv"); 
val_sunset = readmatrix("features\val_sunset.csv"); 

% Concatenate the nonsunset and sunset matrices 
% Assign labels of +1 to sunsets and -1 to nonsunsets 
X_train = cat(1, train_nonsunset, train_sunset); 
y_train = cat(1, ones(size(train_nonsunset, 1), 1) * -1, ones(size(train_sunset, 1), 1)); 

X_val = cat(1, val_nonsunset, val_sunset); 
y_val = cat(1, ones(size(val_nonsunset, 1), 1) * -1, ones(size(val_sunset, 1), 1)); 

X_test = cat(1, test_nonsunset, test_sunset); 
y_test= cat(1, ones(size(test_nonsunset, 1), 1) * -1, ones(size(test_sunset, 1), 1)); 

C_range = 5:1:15;
sigma_range = 15:1:25; 
[C, sigma] = GridSearchCV(X_train, y_train, C_range, sigma_range); 

disp_stat("Best Box Constraint:", C); 
disp_stat("Best Kernel Scale:", sigma); 


net = fitcsvm(X_train,y_train,'KernelFunction','rbf', 'Standardize',true, ...
    'ClassNames',[-1, 1], 'BoxConstraint', C, 'KernelScale', sigma); 

pct_sv = size(net.SupportVectors, 1) / size(X_train, 1); 
disp_stat("Percent Support Vectors:", pct_sv); 

% Predict on the validation set -- I don't need it for its usual purpose so
% I can save the test set for later 
[pred, dist] = predict(net, X_val); 

analyze_performance(pred, dist, y_val);


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

for i = 1:size(C_grid, 2) % iterate over the values of C 
    for j = 1:size(sigma_grid, 2)% iterate over the values of sigma 
        fprintf("C =  %d, sigma = %d\n", C_grid(1, i), sigma_grid(1, j)); 

        acc_sum = 0; 

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

            % Predict on training set 
            [pred, dist] = predict(net, X_valid_fold);

            % Add validation accuracy to running total 
            acc_sum = acc_sum + (sum(sum(pred == y_valid_fold)) / size(y_valid_fold, 1));

        end
        % Divide running total by number of folds and save it to the grid 
        acc_grid(i, j) = acc_sum / n_folds; 
    end
end

figure(1); 
surf(acc_grid); % okay fine I suppose that's cooler than matplotlib
ylabel("Box Parameter Index"); 
xlabel("Kernel Scale Index"); 
zlabel("Accuracy"); 
title("Grid Search Surface")

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

function plot_roc(dist, y_test)
% Define range to vary the threshold over 
start = min(min(dist)); 
step = 0.001; 
stop = max(max(dist));

tpr = zeros(size(start:step:stop)); 
fpr = zeros(size(start:step:stop)); 

idx = 1; 

for i = start:step:stop
    pred = zeros(size(dist)); 

    % Make prediction based on threshold 
    pred(dist >= i) = -1; 
    pred(dist < i) = 1; 

    % Count true positives, false positives, etc. 
    tp = sum(sum(pred == y_test & pred == 1)); 
    fp = sum(sum(pred ~= y_test & pred == 1)); 
    fn = sum(sum(pred ~= y_test & pred == -1)); 
    tn = sum(sum(pred == y_test & pred == -1)); 

    % Calculate and store true positive rate and false positive rate 
    tpr(idx) = tp / (tp + fn); 
    fpr(idx) = fp / (fp + tn); 
    idx = idx + 1; 
end

% Plot the graph 
figure(2);
scatter(tpr, fpr);
xlabel("True Positive Rate"); 
ylabel("False Positive Rate"); 
title("ROC Curve")
xlim([0 1]); % Consistent axis range 
ylim([0 1]);

end

function analyze_performance(pred, dist, y_test)
    % Plot ROC
    plot_roc(dist(:, 2), y_test); 
    
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