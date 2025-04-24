% Funcția pentru evaluarea modelului
function [accuracy, precision, recall, f1_score] = evaluate_model(w1, w2, w3, b1, b2, x, labels)
    % Returnează acuratețea, precizia, recall-ul și scorul F1

    % Obținerea predicțiilor pentru setul de testare
    prediction = predict(w1, w2, w3, b1, b2, x);
 
    % Conversie de la dlarray la double și binarizare pe baza pragului 0.5
    predictions_double = extractdata(prediction);
    predictions_class = double(predictions_double >= 0.5);

    % Calcularea matricei de confuzie
    confusionMatrix = confusionmat(labels, predictions_class);

    % Calcularea metricilor de performanță
    accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix, 'all'); % Acuratețe
    precision = confusionMatrix(1, 1) / sum(confusionMatrix(:, 1)); % Precizie
    recall = confusionMatrix(1, 1) / sum(confusionMatrix(1, :)); % Recall (Sensibilitate)
    f1_score = 2 * precision * recall / (precision + recall); % Scorul F1
end
% Funcția de predicție
function pred = predict(w1,w2,w3,b1,b2, x)
    Z1 = act(x*w1+b1);
    Z2 = act(Z1*w2+b2);
    Z3 = Z2*w3;
    pred = exp(Z3)./(1+exp(Z3));
end

% Funcția de activare: 23.Mish:
function f = act(x)
    f = x.*tanh(log(1+exp(x)));
end