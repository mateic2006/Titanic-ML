% Funcția de pierdere și gradienți
function [y, dw1, dw2,dw3, db,db2] = loss_and_grad(w1,w2,w3,b1,b2,x, labels_train)
    y = f(w1,w2,w3,b1,b2, x, labels_train);
    dw1= dlgradient(y,w1);
    dw2= dlgradient(y,w2);
    dw3= dlgradient(y,w3);
    db= dlgradient(y,b1);
    db2= dlgradient(y,b2);
end

% Funcția de pierdere (entropia încrucișată binară)
function y = f(w1,w2,w3,b1,b2,x, labels)
    Z1 = act(x*w1+b1);
    Z2 = act(Z1*w2+b2);
    Z3 = Z2*w3;
    A1 = exp(Z3)./(1+exp(Z3));
    y = mean(-(labels.*log(A1) + (1-labels).* log(1-A1)));
end

% Funcția de activare: 23.Mish:
function f = act(x)
    f = x.*tanh(log(1+exp(x)));
end