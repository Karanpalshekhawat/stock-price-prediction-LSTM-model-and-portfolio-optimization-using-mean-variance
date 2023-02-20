
## Neural Network model for Pricing Options and computing Implied Volatility

---
An `approach` to learn non-linear relationship between option parameters, it's price and implied volatility using Deep Neural Network

### Main Features

---

- `Black-Scholes` pricing NN model
- `Heston pricing` NN model
- `Implied Volatility` NN model

Apart from these main features, repository also contains multiple python scripts to compute option price using Heston model i.e. numerical method under simulation approach to combine both stochastic process of underlying price, and it's volatility. It also contains python script to compute implied volatility for a given option price using brent method.
### Common features across all models

--- 

- Dataset that is used to train each model was created using Latin Hypercube Sampling technique for a range of input parameters. Details about using this range of input parameter is available under `model/static_data` directory, module is present under `model/pricing/utils`. 
- Same hyper-parameters are used across all 3 models. An approach used to identify these hyper-parameters was Random Search using cross validation. Module that implements it is available under `model/pricing/core` directory.

Model inputs are present under `model/static_data` directory, the reason to choose them is explained in detail in notebooks and literature document.
Model outputs are present under `model/output` which contains trained NN models, and tuner hyper-parameters values.

### Notebooks 

---

Multiple jupyter notebooks are present under `notebooks` directory to better illustrate and explain  the flow of implementation:- 

- `Black schole pricing NN` : Creates hypothetical data, load already trained NN model, define input features, predict option price and then compare difference.
- `Heston pricing NN` : Creates hypothetical data, explain extra paramters required for pricing option using Heston model, load already trained NN model, define input features, predict option price for a different set of input parameters and then compare the difference.
- `Implied volatility ` : Creates hypothetical data, load already trained NN model, predict option price, define input features, and then compare difference.
- `Model results` : Loads already trained models, combines Heston NN pricing model and implied NN model to predict implied volatiltiy of an option. The same implied volatility is also computed by using using numerical method to compute heston price and Brent method to compute implied volatilty. Both outputs are finally compared.

### Documentation

---

Implemented  research paper `Pricing options and computing implied volatilities
using neural networks` published by Shuaiqiang Liu 1, Cornelis W. Oosterlee and Sander M.Bohte which is present under the `literature` directory.


### Dependencies

---

- Tensorflow
- Pandas
- sklearn
- Optionprice
- NumPy
- MatplotLib

