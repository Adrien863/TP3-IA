# import des différentes librairies
import clr
clr.AddReference("System")
clr.AddReference("QuantConnect.Algorithm")
clr.AddReference("QuantConnect.Common")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# Création de la classe qui est le coeur de l'algorithme.

class KerasNeuralNetworkAlgorithm(QCAlgorithm):

    # Initialisation des paramètres

    def Initialize(self):
        self.SetStartDate(2019, 3, 1)  # Date de début
        self.SetEndDate(2019, 8, 1) # Date de fin
        
        self.SetCash(100000)  # Disponibilité du portefeuille (100000$)
        
        # Définition des assets utilisés (Bitcoin, Ether, Litecoin)
        
        btc = self.AddCrypto("BTCUSD", Resolution.Hour) 
        eth = self.AddCrypto("ETHUSD",Resolution.Hour) 
        ltc = self.AddCrypto("LTCUSD",Resolution.Hour) 
        
        # Paramètres graphique pour afficher les courbes
        
        BTCplot = Chart('BTC Price')
        BTCplot.AddSeries(Series('Price', SeriesType.Candle))
        ETHplot = Chart('ETH Price')
        ETHplot.AddSeries(Series('Price', SeriesType.Candle))
        LTCplot = Chart('LTC Price')
        LTCplot.AddSeries(Series('Price', SeriesType.Candle))
        
        self.AddChart(BTCplot)
        self.AddChart(ETHplot)
        self.AddChart(LTCplot)
        
        self.symbols = [btc.Symbol,eth.Symbol,ltc.Symbol] # Variable de stockages des assets nécessaire pour la suite
        
        self.lookback = 30 # Nombre de mintes de "recul" sur les données historiques
        
        # Programmation du training du réseau de neurones (s'effectue tous les lundis)
        
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Monday), self.TimeRules.AfterMarketOpen("BTCUSD", 28), self.NetTrain) 
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Monday), self.TimeRules.AfterMarketOpen("BTCUSD", 30), self.Trade)
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Monday), self.TimeRules.AfterMarketOpen("ETHUSD", 28), self.NetTrain) 
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Monday), self.TimeRules.AfterMarketOpen("ETHUSD", 30), self.Trade)
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Monday), self.TimeRules.AfterMarketOpen("LTCUSD", 28), self.NetTrain) 
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Monday), self.TimeRules.AfterMarketOpen("LTCUSD", 30), self.Trade)
    
    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        '''
        
        self.Plot('BTC Price', 'Price', data.Bars["BTCUSD"].Close)
        self.Plot('ETH Price', 'Price', data.Bars["ETHUSD"].Close)
        self.Plot('LTC Price', 'Price', data.Bars["LTCUSD"].Close)
    
    
    def NetTrain(self):
        # Données historiques utilisées pour entraîner le modèle de resolution 
        history = self.History(self.symbols, self.lookback + 1, Resolution.Hour)
        
        # Dictionnaire pour stocker les prix des assets
        self.prices_x = {} 
        self.prices_y = {}
        
        # Dictionnaire pour stocker les prix ask/bid des assets
        self.sell_prices = {}
        self.buy_prices = {}
        
        for symbol in self.symbols:
            if not history.empty:
                # x: prédictions; y: réponse
                self.prices_x[symbol] = list(history.loc[symbol.Value]['open'])[:-1]
                self.prices_y[symbol] = list(history.loc[symbol.Value]['open'])[1:]
                
        for symbol in self.symbols:
            if symbol in self.prices_x:
                # Converti les données en tableau numpy pour correspondre au modèle Keras NN
                x_data = np.array(self.prices_x[symbol])
                y_data = np.array(self.prices_y[symbol])
                
                # Construit le réseau de neurones de la première "couche" jusqu'à la dernière
                model = Sequential()

                model.add(Dense(10, input_dim = 1))
                model.add(Activation('relu'))
                model.add(Dense(1))

                sgd = SGD(lr = 0.001) # Taux d'apprentissage = 0.001
                
                # Fonction d'erreur (Mean squared error) et méthode d'optimisation (Stochastic gradient descent)
                model.compile(loss='mse', optimizer=sgd)

                # Boucle d'entraînement convergente 
                for step in range(701):
                    # Entraîne le modèle
                    cost = model.train_on_batch(x_data, y_data)
            
            # Récupère la dernière prédicion de prix 
            y_pred_final = model.predict(y_data)[0][-1]
            
            # Suivre la tendance
            self.buy_prices[symbol] = y_pred_final + np.std(y_data)
            self.sell_prices[symbol] = y_pred_final - np.std(y_data)
        
    def Trade(self):
        ''' 
        Entrer ou sortir des positions en fonction de la relation entre le prix d'ouverture de la bougie actuelle et les prix définis par le modèle d'apprentissage.
        Liquider sa position si le prix d'ouverture est inférieur au prix de vente et acheter si le prix d'ouverture est supérieur au prix d'achat
        ''' 
        for holding in self.Portfolio.Values:
            if self.CurrentSlice[holding.Symbol].Open < self.sell_prices[holding.Symbol] and holding.Invested:
                self.Liquidate(holding.Symbol)
            
            if self.CurrentSlice[holding.Symbol].Open > self.buy_prices[holding.Symbol] and not holding.Invested:
                self.SetHoldings(holding.Symbol, 1 / len(self.symbols))
