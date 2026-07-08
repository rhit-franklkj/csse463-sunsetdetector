% stuff from lab 5: 
% might be helpful
%%%%

function [best_C, best_sigma] = GridSearchCV(X_train, y_train, C_grid, sigma_grid)
    n_folds = 5; 
    % shuffle the data
    df = cat(2, X_train, y_train);
    df = df(randperm(size(df, 1)),:); 
    X_train = df(:, 1:2); 
    y_train = df(:, 3); 

    % partition into 5 folds 
    X_train_x1 = X_train(:, 1); 
    X_train_x2 = X_train(:, 2); 
    
    X_train_x1 = reshape(X_train_x1, int64(size(X_train, 1) / n_folds), n_folds); 
    X_train_x2 = reshape(X_train_x2, int64(size(X_train, 1) / n_folds), n_folds); 

    X_train = cat(3, X_train_x1, X_train_x2); 
    X_train = permute(X_train,[1 3 2]); 

    y_train = reshape(y_train, int64(size(y_train, 1) / n_folds), 1, n_folds); 

    acc_grid = zeros(size(C_grid, 2), size(sigma_grid, 2)); 

    for i = 1:size(C_grid, 2) % iterate over the values of C 
        for j = 1:size(sigma_grid, 2)% iterate over the values of sigma 

            acc_sum = 0; 
  
            for k = 1:n_folds % 5-fold cross validation

                % mark the ith fold as the validation set 
                X_train_folds = cat(3, X_train(:, :, 1:k-1), X_train(:, :, k+1:n_folds)); 
                X_train_folds = reshape(permute(X_train_folds,[1 3 2]), [], 2, 1); 
                X_valid_fold = X_train(:, :, k); 

                y_train_folds = cat(3, y_train(:, :, 1:k-1), y_train(:, :, k+1:n_folds)); 
                y_train_folds = reshape(permute(y_train_folds,[1 3 2]), [], 1, 1); 
                y_valid_fold = y_train(:, :, k); 

                % Train model 
                net = fitcsvm(X_train_folds, y_train_folds,'KernelFunction','rbf', 'Standardize',false,'ClassNames',[-1, 1], 'BoxConstraint', C_grid(1, i), 'KernelScale', sigma_grid(1, j));

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