# cartola: final project

import os
import sys
import time
import random
import argparse
import warnings
import statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _pickle as pkl

from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, RidgeCV, ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR

class Cartola:
    def __init__(self):
        pd.set_option('display.max_columns', 100)

        warnings.filterwarnings("ignore")

    def __check_scouts(self, row):
        scouts_weights = np.array([-2.0, -5.0, 3.0, 7.0, -0.5, -6.0, -2.0, 1.7, 5.0, 5.0, 1.0, 0.7, 0.5, 3.5, 8.0, -0.5, -0.3, -3.5])

        return np.sum(scouts_weights*row[self.cols_scouts])

    def __get_scouts_for_round(self, df, round_, cols_scouts):
        suffixes = ('_curr', '_prev')
        cols_current = [col + suffixes[0] for col in cols_scouts]
        cols_prev = [col + suffixes[1] for col in cols_scouts]

        df_round = df[df['Rodada'] == round_]
        if round_ == 1: return df_round

        df_round_prev = df[df['Rodada'] < round_].groupby('AtletaID', as_index=False)[cols_scouts].max()
        df_players = df_round.merge(df_round_prev, how='left', on=['AtletaID'], suffixes=suffixes)

        # if is the first round of a player, the scouts of previous rounds will be NaNs. Thus, set them to zero
        df_players.fillna(value=0, inplace=True)

        # compute the scouts
        df_players[cols_current] = df_players[cols_current].values - df_players[cols_prev].values

        # update the columns
        df_players.drop(labels=cols_prev, axis=1, inplace=True)
        df_players = df_players.rename(columns=dict(zip(cols_current, cols_scouts)))
        df_players.SG = df_players.SG.clip_lower(0)

        return df_players

    def __data_load(self):
        df = pd.read_csv('data/dados_agregados.csv')

        df_teams = pd.read_csv('data/times_ids.csv')
        df_teams = df_teams.dropna()

        return df, df_teams

    def __data_clean(self, df, cols_scouts_def, cols_scouts_atk, cols_scouts):
        # remove todas as linhas cujo scouts são NANs
        df_clean = df.dropna(how='all', subset=cols_scouts)

        # remove todas as linhas com rodada == 0
        df_clean = df_clean[df_clean['Rodada'] > 0]

        # remove técnicos e jogadores sem posição
        df_clean = df_clean[(df_clean['Posicao'] != "tec") & (~df_clean['Posicao'].isnull())]

        # remove todos os jogadores que não participaram de alguma rodada
        df_clean = df_clean[(df_clean['Participou'] == True) | (df_clean['PrecoVariacao'] != 0)]

        # altera os Status = NAN para 'Provável'
        df_clean.loc[df_clean.Status.isnull(), 'Status'] = 'Provável'

        # atualiza nomes dos jogadores sem ids e remove jogadores sem nome
        df_ids =  df.groupby('AtletaID')['Apelido'].unique()
        dict_ids = dict(zip(df_ids.index, [str(v[-1]) for v in df_ids.values]))
        dict_ids = {k:v for k,v in dict_ids.items() if v != 'nan'}
        df_clean['Apelido'] = df_clean['AtletaID'].map(dict_ids)
        df_clean = df_clean[~df_clean['Apelido'].isnull()]

        # preenche NANs restantes com zeros (verificado antes!)
        df_clean.fillna(value=0, inplace=True)

        return df_clean

    def __data_teams(self, df, df_teams):
        # do not run this cell twice!
        dict_teams_id = dict(zip(df_teams['id'], df_teams['nome.cartola']))
        dict_teams_id.update(dict(zip(df_teams['cod.older'], df_teams['nome.cartola'])))

        df['ClubeID'] = df['ClubeID'].astype(np.int).map(dict_teams_id)

        df = df.dropna()

        return df

    def __update_scouts(self, df, cols_scouts):
        # a célula abaixo cria uma dataframe com os scouts dos jogadores não acumulados.

        df_scouts      = df[df['ano'] != 2015]
        df_scouts_2015 = df[df['ano'] == 2015]

        n_rounds = df_scouts_2015['Rodada'].max()

        if np.isnan(n_rounds):
            df_scouts = df
        else:
            for i in range(1, n_rounds+1):
                df_round = self.__get_scouts_for_round(df_scouts_2015, i, cols_scouts)
                df_scouts = df_scouts.append(df_round, ignore_index=True)

        return df_scouts

    def __check_points(self, df, df_scouts):
        players_points = df_scouts.apply(self.__check_scouts, axis=1)
        errors = np.where(~np.isclose(df_scouts['Pontos'].values, players_points))[0]
        df_scouts.iloc[errors, :].tail(10)

        # remove such players with wrong pontuation (do not run twice!)
        df_scouts.reset_index(drop=True, inplace=True)
        df_scouts.drop(df.index[errors], inplace=True)

        return df_scouts

    def __dict_positions(self, to_int = True):
        dict_map = {'gol':1, 'zag':2, 'lat':3, 'mei':4, 'ata':5}

        return  dict_map if to_int else dict(zip(dict_map.values(), dict_map.keys()))

    def __dict_teams(self, teams_full, to_int = True):
        teams_map = {team:(index+1) for index, team in enumerate(teams_full)}

        return teams_map if to_int else dict(zip(teams_map.values(), teams_map.keys()))

    def __create_samples(self, df, round_train, round_pred):
        '''Create a Dataframe with players from round_train, but with 'Pontos' of round_pred'''
        df_train = df[df['ano_rodada'] == round_train]
        df_pred = df[df['ano_rodada'] == round_pred][['AtletaID', 'Pontos']]
        df_merge = df_train.merge(df_pred, on='AtletaID', suffixes=['_train', '_pred'])

        df_merge = df_merge.rename(columns={'Pontos_train':'Pontos', 'Pontos_pred':'pred'})

        return df_merge

    def __to_samples(self, df, cols_info, cols_of_interest, teams_full):
        df_samples = df[cols_info + cols_of_interest].copy()
        df_samples['ClubeID'] = df_samples['ClubeID'].map(self.__dict_teams(teams_full, to_int=True))
        df_samples['Posicao'] = df_samples['Posicao'].map(self.__dict_positions(to_int=True))
        df_samples['variable'] = df_samples['variable'].map({'home.team':1, 'away.team':2})
        df_samples.reset_index(drop=True, inplace=True)

        return df_samples

    def __predict_best_players(self, df_samples, model, n_players=11):
        samples = df_samples[df_samples.columns.difference(['AtletaID', 'Rodada', 'ano'])].values.astype(np.float64)
        pred = model.predict(samples)
        best_indexes = pred.argsort()[-n_players:]

        return df_samples.iloc[best_indexes]

    def __predict_best_players_by_position(self, df_samples, model, n_gol=5, n_zag=5, n_lat=5, n_mei=5, n_atk=5):
        df_result = pd.DataFrame(columns=df_samples.columns)

        for n_players, pos in zip([n_gol, n_zag, n_lat, n_mei, n_atk], range(1,6)):
            samples = df_samples[df_samples['Posicao'] == pos]
            df_pos = self.__predict_best_players(samples, model, n_players)
            df_result = df_result.append(df_pos)

        return df_result

    def clean(self, save=False):
        cols_scouts_def = ['CA','CV','DD','DP','FC','GC','GS','RB','SG']
        cols_scouts_atk = ['A','FD','FF','FS','FT','G','I','PE','PP']

        self.cols_scouts = cols_scouts_def + cols_scouts_atk

        df, df_teams = self.__data_load()
        df_clean     = self.__data_clean(df, cols_scouts_def, cols_scouts_atk, self.cols_scouts)
        df_clean     = self.__data_teams(df_clean, df_teams)
        df_scouts    = self.__update_scouts(df_clean, self.cols_scouts)
        df_scouts    = self.__check_points(df, df_scouts)

        df_scouts.drop_duplicates(subset=['AtletaID', 'ano'] + self.cols_scouts, keep='first', inplace=True)

        if save == True:
            df_scouts.to_csv('src/data/data_clean.csv', index=False)

        return df_scouts

    def prepare(self, df_samples, load=False, save=False):

        if load == True:
            df_samples = pd.read_csv('src/data/data_clean.csv')

        cols_of_interest = df_samples.columns.difference(['Apelido', 'Status', 'Participou', 'dia', 'mes']).values.tolist()

        cols_info = ['Rodada', 'ano']

        df_samples = df_samples[cols_of_interest]

        teams_full = pd.Series(df_samples['ClubeID'].unique()).sort_values().values

        df_samples['ClubeID']  = df_samples['ClubeID'].map(self.__dict_teams(teams_full, to_int=True))
        df_samples['Posicao']  = df_samples['Posicao'].map(self.__dict_positions(to_int=True))
        df_samples['variable'] = df_samples['variable'].map({'home.team':1, 'away.team':2})

        if save == True:
            df_samples.to_csv('src/data/data_samples.csv', index=False)

        return df_samples

    def train(self, type, df_samples, plot=False, load=False, save=False):

        if load == True:
            df_samples = pd.read_csv('src/data/data_samples.csv')

        df_samples = df_samples[df_samples.ano < 2017]

        # modificação para treinar com mais de uma temporada
        df_samples['ano_rodada'] = df_samples['Rodada'] + df_samples['ano']*100

        df_train = pd.DataFrame(data = [], columns=list(df_samples.columns) + ['pred'])

        n_rounds = df_samples['ano_rodada'].max()
        init_rounds = df_samples['ano_rodada'].min()

        for round_train, round_pred in zip(range(init_rounds, n_rounds), range(init_rounds+1, n_rounds+1)):
            df_round = self.__create_samples(df_samples, round_train, round_pred)

            if df_round.shape[0] != 0:
                df_train = df_train.append(df_round, ignore_index=True)

        samples = df_train[df_train.columns.difference(['AtletaID', 'Rodada','pred', 'ano', 'ano_rodada'])].values.astype(np.float64)
        scores  = df_train['pred'].values

        scaler = MinMaxScaler()

        samples = scaler.fit_transform(samples)

        if type == "NeuralNetwork":
            steps = [('MinMax', MinMaxScaler()), ('NN', MLPRegressor(solver='adam', activation='identity', learning_rate_init=1e-2, momentum=0.9, max_iter=2000))]
            pipe = Pipeline(steps)
            params = dict(NN__hidden_layer_sizes=[(50,50,50,50,50), (50,100,50), (50,100,100,50), (128), (128,128), (128, 128, 128)])

            model = GridSearchCV(pipe, params, scoring='neg_mean_squared_error', n_jobs=-1, cv=5, verbose=10)

            model.fit(samples, scores)

            print(model.best_params_, model.best_score_)

            pd.DataFrame(model.best_estimator_.named_steps['NN'].loss_curve_).plot()

        elif type == "RandomForest":
            samples = scaler.fit_transform(samples)

            n_estimators = [1000, 100, 10]
            param_grid = {'n_estimators': n_estimators}

            model = GridSearchCV(RandomForestRegressor(max_depth=500), param_grid, cv=5)

            model.fit(samples, scores)

            print(model.best_params_, model.best_score_)

        elif type == "BayesianRidge":
            samples = scaler.fit_transform(samples)

            alpha_1  = [1e-7, 1e-6, 1e-5]
            alpha_2  = [1e-7, 1e-6, 1e-5]
            lambda_1 = [1e-7, 1e-6, 1e-5]
            lambda_2 = [1e-7, 1e-6, 1e-5]

            param_grid = {'alpha_1': alpha_1, 'alpha_2':alpha_2, 'lambda_1':lambda_1, 'lambda_2':lambda_2}

            model = GridSearchCV(BayesianRidge(), param_grid, cv=5)

            model.fit(samples, scores)

            print(model.best_params_, model.best_score_)

        elif type == "Ridge":
            model = RidgeCV().fit(samples, scores)

            model.fit(samples, scores)

        elif type == "ElasticNet":
            samples = scaler.fit_transform(samples)

            model = ElasticNetCV().fit(samples, scores)

        elif type == "GradientBoost":
            n_estimators = [100, 50, 10]
            learning_rate = [0.1, 0.2, 0.3]
            param_grid = {'n_estimators': n_estimators, 'learning_rate': learning_rate}

            model = GridSearchCV(GradientBoostingRegressor(max_depth=3), param_grid, cv=5)

            model.fit(samples, scores)

            print(model.best_params_, model.best_score_)

        elif type == "SVR":
            C  = [100, 50, 10]
            gamma = [0.1, 0.2, 0.3]
            param_grid = {'C': C, 'gamma': gamma}

            model = GridSearchCV(SVR(), param_grid, cv=5)

            model.fit(samples, scores)

            print(model.best_params_, model.best_score_)

        if save == True:
            pkl.dump(model, open('src/data/model.pkl', 'wb'), -1)

        return model

    def play(self, df_test, model, year, load=False):

        print('\n=== year : ', year, ' ===')

        cols_info = ['Rodada', 'ano']

        list_points = []

        total_rounds = 0

        if load == True:
            df_test = pd.read_csv('src//data/data_clean.csv')

        df_test = df_test[df_test.ano == year]

        cols_of_interest = df_test.columns.difference(['Apelido', 'Status', 'Participou', 'dia', 'mes']).values.tolist()

        teams_full = pd.Series(df_test['ClubeID'].unique()).sort_values().values

        if load == True:
            model = pkl.load(open('src/data/model.pkl', 'rb'))

        for round_to_predict in range(5, 38):
            df_rodada  = df_test[(df_test['Rodada'] == (round_to_predict - 1)) & (df_test['Status'] == "Provável")]
            df_samples = self.__to_samples(df_rodada, cols_info, cols_of_interest, teams_full)

            df_players = self.__predict_best_players_by_position(df_samples, model, n_gol=1, n_zag=2, n_lat=2, n_mei=4, n_atk=2)

            time = df_rodada.iloc[df_players.index][['Apelido', 'Posicao', 'ClubeID']]

            array = time['Apelido'].tolist()

            df = df_test[(df_test['Rodada'] == (round_to_predict))]

            points = df.loc[df['Apelido'].isin(array)]['Pontos'].sum()

            list_points.append(points)

            total_rounds = total_rounds + 1

            print('round #', total_rounds + 5, ' : ', points)

        print('\ntotal points: ', sum(list_points))
        print('mean  points: ', sum(list_points)/total_rounds)
        print('stdev points: ', statistics.stdev(list_points))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='cartola project')

    args = parser.parse_args()

    cartola = Cartola()

    data_clean   = cartola.clean()
    data_samples = cartola.prepare(data_clean)

    types = ["NeuralNetwork", "RandomForest", "BayesianRidge", "Ridge", "ElasticNet", "GradientBoost", "SVR"]

    for type in types:
        model = cartola.train(type, data_samples)

        cartola.play(data_clean, model, 2017) # play 2017 season

    final_model = cartola.train(["BayesianRidge"], data_samples)

    cartola.play(data_clean, final_model, 2018) # play 2018 season

# taf!
