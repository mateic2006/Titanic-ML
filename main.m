function main()
    % Citirea datelor din fișierul CSV
    train = readtable('train.csv');

    % Extragerea coloanelor relevante din tabelul de date
    age = train{:,"Age"};
    sex = train{:,"Sex"};
    class = train{:,"Pclass"};
    labels = train{:,"Survived"};

    % Codificarea categorică a variabilei 'Sex'
    [~, ~, sex] = unique(sex);
    sex = sex - 1;

    % Tratarea valorilor lipsă (NaN) pentru 'Age' prin înlocuirea cu media
    mean_age = mean(age(~isnan(age)));
    age(isnan(age)) = mean_age;

    % Normalizarea variabilei 'Age'
    age = age/100;

    % Determinarea dimensiunii setului de date
    [n,~]=size(train);

    % Împărțirea setului de date în set de antrenare și set de testare (80/20)
    train_size=floor(0.8*n);
    sex_train = sex(1:train_size);
    age_train = age(1:train_size);
    class_train = class(1:train_size);
    labels_train = labels(1:train_size);

    sex_test = sex(train_size+1:end);
    age_test = age(train_size+1:end);
    class_test = class(train_size+1:end);
    labels_test = labels(train_size+1:end);

    % Construirea matricei de intrare X pentru setul de antrenare
    x = [sex_train age_train class_train];
    
    % Setarea parametrilor de antrenare
    learning_rate = 0.001;
    hiden_layers_neurons=50;
    nr_iteratii=5000;

    % Antrenarea și evaluarea modelului cu metoda SGD
    disp('Antrenarea modelului cu metoda SGD:');
    tic; % Pornirea cronometrului
    [w1_sgd, w2_sgd, w3_sgd, b1_sgd, b2_sgd, loss_values_sgd] = SGD(x, labels_train,learning_rate,hiden_layers_neurons,nr_iteratii);
    sgd_time = toc; % Oprirea cronometrului

    % Evaluarea modelului SGD pe setul de testare
    xx_sgd = [sex_test age_test class_test];
    [accuracy_sgd, precision_sgd, recall_sgd, f1_score_sgd] = evaluate_model(w1_sgd, w2_sgd, w3_sgd, b1_sgd, b2_sgd, xx_sgd, labels_test);

    % Afișarea rezultatelor SGD
    disp('Rezultate pentru metoda SGD:');
    disp(['Timp de antrenare: ', num2str(sgd_time), ' secunde']);
    disp(['Acuratețe: ', num2str(accuracy_sgd)]);
    disp(['Precizie: ', num2str(precision_sgd)]);
    disp(['Recall: ', num2str(recall_sgd)]);
    disp(['Scor F1: ', num2str(f1_score_sgd)]);

    % Antrenarea și evaluarea modelului cu metoda ADAM
    disp('Antrenarea modelului cu metoda ADAM:');
    tic; % Pornirea cronometrului
    [w1_adam, w2_adam, w3_adam, b1_adam, b2_adam, loss_values_adam] = ADAM(x, labels_train,learning_rate,hiden_layers_neurons,nr_iteratii);
    adam_time = toc; % Oprirea cronometrului

    % Evaluarea modelului ADAM pe setul de testare
    xx_adam = [sex_test age_test class_test];
    [accuracy_adam, precision_adam, recall_adam, f1_score_adam] = evaluate_model(w1_adam, w2_adam, w3_adam, b1_adam, b2_adam, xx_adam, labels_test);

    % Afișarea rezultatelor ADAM
    disp('Rezultate pentru metoda ADAM:');
    disp(['Timp de antrenare: ', num2str(adam_time), ' secunde']);
    disp(['Acuratețe: ', num2str(accuracy_adam)]);
    disp(['Precizie: ', num2str(precision_adam)]);
    disp(['Recall: ', num2str(recall_adam)]);
    disp(['Scor F1: ', num2str(f1_score_adam)]);

    % Plotarea evoluției funcției obiectiv
    figure;
    plot(loss_values_sgd, 'r-', 'LineWidth', 2);
    hold on;
    plot(loss_values_adam, 'b-', 'LineWidth', 2);
    hold off;
    xlabel('Iterații');
    ylabel('Funcția obiectiv');
    title('Evoluția funcției obiectiv');
    legend('SGD', 'ADAM');
end