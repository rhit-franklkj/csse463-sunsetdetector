function [FPR, TPR, AUC] = rocCurve(scores, y, name, opThresh)

    if nargin < 3, name = ''; end
    if nargin < 4, opThresh = []; end

    scores = scores(:);
    y      = y(:);
    assert(numel(scores) == numel(y), 'scores and y must be the same length.');

    P = sum(y ==  1);
    N = sum(y == -1);
    assert(P > 0 && N > 0, 'Need both classes present to build an ROC.');

    % Sort by descending score
    [s, order] = sort(scores, 'descend');
    ySorted    = y(order);

    tp = cumsum(ySorted ==  1);
    fp = cumsum(ySorted == -1);

    % Collapse tied scores to a single point
    lastOfTie = [diff(s) ~= 0; true];
    TPR = [0; tp(lastOfTie) / P];
    FPR = [0; fp(lastOfTie) / N];

    AUC = trapz(FPR, TPR);

    if ~isempty(name)
        hold on;
        plot(FPR, TPR, '-', 'LineWidth', 1.5, ...
            'DisplayName', sprintf('%s (AUC = %.3f)', name, AUC));

        if ~isempty(opThresh)
            pred = scores >= opThresh;
            fpr0 = sum(pred & y == -1) / N;
            tpr0 = sum(pred & y ==  1) / P;
            plot(fpr0, tpr0, 'o', 'MarkerSize', 7, 'LineWidth', 1.5, ...
                'HandleVisibility', 'off');
        end

        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        axis([0 1 0 1]); axis square; grid on; box on;
        set(gcf, 'Color', 'w');   % white background
        legend('Location', 'southeast');
    end
end
