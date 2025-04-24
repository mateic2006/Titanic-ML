% Funcția pentru antrenarea modelului cu metoda ADAM
function [w1, w2, w3, b1, b2, loss_values] = ADAM(x, labels_train,learning_rate,hiden_layers_neurons,nr_iteratii)
    % Returnează greutățile și bias-urile antrenate, precum și valorile funcției obiectiv
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;

    % Inițializarea greutăților și biasurilor rețelei
    w1 = randn(3, hiden_layers_neurons);
    w2 = randn(hiden_layers_neurons, hiden_layers_neurons);
    w3 = randn(hiden_layers_neurons, 1);
    b1 = randn(1, hiden_layers_neurons);
    b2 = randn(1, hiden_layers_neurons);

    % Conversie la format deep learning array
    w1 = dlarray(w1);
    w2 = dlarray(w2);
    w3 = dlarray(w3);
    b1 = dlarray(b1);
    b2 = dlarray(b2);

    m1 = zeros(size(w1));
    m2 = zeros(size(w2));
    m3 = zeros(size(w3));
    mb1 = zeros(size(b1));
    mb2 = zeros(size(b2));

    v1 = zeros(size(w1));
    v2 = zeros(size(w2));
    v3 = zeros(size(w3));
    vb1 = zeros(size(b1));
    vb2 = zeros(size(b2));

    loss_values = [];
    for i = 1:nr_iteratii
        [loss, dw1, dw2, dw3, db1, db2] = dlfeval(@loss_and_grad, w1, w2, w3, b1, b2, x, labels_train);

        % Actualizarea momentelor m și v
        m1 = beta1 * m1 + (1 - beta1) * dw1;
        m2 = beta1 * m2 + (1 - beta1) * dw2;
        m3 = beta1 * m3 + (1 - beta1) * dw3;
        mb1 = beta1 * mb1 + (1 - beta1) * db1;
        mb2 = beta1 * mb2 + (1 - beta1) * db2;

        v1 = beta2 * v1 + (1 - beta2) * (dw1.^2);
        v2 = beta2 * v2 + (1 - beta2) * (dw2.^2);
        v3 = beta2 * v3 + (1 - beta2) * (dw3.^2);
        vb1 = beta2 * vb1 + (1 - beta2) * (db1.^2);
        vb2 = beta2 * vb2 + (1 - beta2) * (db2.^2);

        % Corectarea momentelor pentru a compensa startul de la 0
        m1_hat = m1 / (1 - beta1^i);
        m2_hat = m2 / (1 - beta1^i);
        m3_hat = m3 / (1 - beta1^i);
        mb1_hat = mb1 / (1 - beta1^i);
        mb2_hat = mb2 / (1 - beta1^i);

        v1_hat = v1 / (1 - beta2^i);
        v2_hat = v2 / (1 - beta2^i);
        v3_hat = v3 / (1 - beta2^i);
        vb1_hat = vb1 / (1 - beta2^i);
        vb2_hat = vb2 / (1 - beta2^i);

        % Actualizarea greutăților și bias-urilor
        w1 = w1 - learning_rate * m1_hat ./ (sqrt(v1_hat) + epsilon);
        w2 = w2 - learning_rate * m2_hat ./ (sqrt(v2_hat) + epsilon);
        w3 = w3 - learning_rate * m3_hat ./ (sqrt(v3_hat) + epsilon);
        b1 = b1 - learning_rate * mb1_hat ./ (sqrt(vb1_hat) + epsilon);
        b2 = b2 - learning_rate * mb2_hat ./ (sqrt(vb2_hat) + epsilon);

        loss_values = [loss_values, double(loss)]; % Salvarea valorilor funcției obiectiv
    end
end