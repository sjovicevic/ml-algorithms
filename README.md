# Rešavanje klasifikacionih problema
___
Cilj projekta je rešavanje problema klasifikacije [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) algoritmima koji su implementirani od nule, bez korišćenja gotovih biblioteka poput scikit-learn, TensorFlow i sl.
___
## Logistička regresija
Kod pristupa rešavanja problema logističkom regresijom, postoje dve implementacije u zavisnosti od toga da li se radi o dvoklasnom ili više klasnom modelu.
Nakon 1000 epoha i stopom učenja 0.01, dobija se tačnost algoritma od oko 96%.

![Grafik loss funkcije](docs/assets/Plot1.png)
___
## Neuralna mreža
Implementirana modularnih pristupom tako što je broj skrivenih slojeva i neurona unutar slojeva proizvoljan, kao i izbor korišćenja aktivacionih funkcija.