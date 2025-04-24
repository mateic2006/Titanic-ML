% Funcția pentru antrenarea modelului cu metoda SGD
function [w1, w2, w3, b1, b2, loss_values] = SGD(x, labels_train,learning_rate,hiden_layers_neurons,nr_iteratii)
    % Returnează greutățile și bias-urile antrenate, precum și valorile funcției obiectiv

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

    loss_values = [];
    for i = 1:nr_iteratii
        [loss, dw1, dw2, dw3, db1, db2] = dlfeval(@loss_and_grad, w1, w2, w3, b1, b2, x, labels_train);
        w1 = w1 - learning_rate * dw1;
        w2 = w2 - learning_rate * dw2;
        w3 = w3 - learning_rate * dw3;
        b1 = b1 - learning_rate * db1;
        b2 = b2 - learning_rate * db2;
        loss_values = [loss_values, double(loss)]; % Salvarea valorilor funcției obiectiv
    end
end
