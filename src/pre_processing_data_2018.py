
# coding: utf-8

# In[199]:


import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import numpy as np


# In[200]:


df_samples = pd.read_csv('dados_agregados_limpos.csv')
c_treino = df_samples.columns.tolist()


# In[201]:


cols_scouts_def = ['CA','CV','DD','DP','FC','GC','GS','RB','SG'] # alphabetical order
cols_scouts_atk = ['A','FD','FF','FS','FT','G','I','PE','PP'] # alphabetical order
cols_scouts = cols_scouts_def + cols_scouts_atk

scouts_weights = np.array([-2.0, -5.0, 3.0, 7.0, -0.5, -6.0, -2.0, 1.7, 5.0, 5.0, 1.0, 0.7, 0.5, 3.5, 8.0, -0.5, -0.3, -3.5])


# In[202]:


rodadas = pd.DataFrame()
acc = 0
for x in range(1, 39):
    rodada = pd.read_csv("2018/rodada-"+str(x)+".csv")
    rodadas = pd.concat([rodadas, rodada])
    acc
    
    
rodadas = rodadas.drop(['Unnamed: 0', 'atletas.foto', 'atletas.slug', 'atletas.nome', 'atletas.clube_id'], axis=1)
rodadas['ano'] = 2018
rodadas['Participou'] = True

# remove todas as linhas cujo scouts são NANs 

rodadas.rename(columns={'atletas.apelido':'Apelido',
                        'atletas.atleta_id':'AtletaID',
                        'atletas.rodada_id':'Rodada',
                        'atletas.clube.id.full.name' : 'ClubeID',
                        'atletas.preco_num': 'Preco',
                        'atletas.variacao_num' : 'PrecoVariacao',
                        'atletas.status_id' : 'Status',
                        'atletas.media_num' : 'PontosMedia',
                        'atletas.posicao_id' : 'Posicao',
                        'atletas.pontos_num' : 'Pontos'},inplace=True)

rodadas = rodadas.dropna(how='all', subset=cols_scouts)

rodadas[cols_scouts] = rodadas[cols_scouts].fillna(value=0)


c_teste = rodadas.columns.tolist()



in_treino_not_teste = [x for x in c_treino if x not in c_teste]


in_teste_not_treino = [x for x in c_teste if x not in c_treino]

print(in_treino_not_teste)
print()
print(len(in_treino_not_teste))

rodadas


# In[168]:



def get_scouts_for_round(df, round_):
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


df_scouts = pd.DataFrame()
df_scouts_2015 = rodadas

n_rounds = df_scouts_2015['Rodada'].max()

if np.isnan(n_rounds):
    df_scouts = df_clean
else:
    for i in range(1, n_rounds+1):
        df_round = get_scouts_for_round(df_scouts_2015, i)
        print("Dimensões da rodada #{0}: {1}".format(i, df_round.shape))
        df_scouts = df_scouts.append(df_round, ignore_index=True)
    
print(df_scouts.shape)
rodadas = df_scouts


# In[169]:


def check_scouts(row):
    return np.sum(scouts_weights*row[cols_scouts])

players_points = rodadas.apply(check_scouts, axis=1)
errors = np.where(~np.isclose(rodadas['Pontos'].values, players_points))[0]
print("qtde. de jogadores com pontuação diferente dos scouts: ", errors.shape)
rodadas.iloc[errors, :].tail(10)


# In[182]:


def avg(df, nft, bft):
    ids = df_2018.AtletaID.unique()
    df[nft] = 0
    for player in ids:
        tmp = df.loc[df['AtletaID'] == player]
        rds = sorted(tmp['Rodada'])
        acc = 0
        for idx,rd in enumerate(rds):
            pt = float(df.loc[(df["AtletaID"] == player) & (df["Rodada"] == rd), bft])        
            df_2018.loc[(df["AtletaID"] == player) & (df["Rodada"] == rd), nft] = (acc + pt)/(idx+1)
            acc = acc + pt
        
    return df
    


# In[196]:


def avg5(df, nft, bft):
    ids = df_2018.AtletaID.unique()
    df[nft] = 0
    for player in ids:
        tmp = df.loc[df['AtletaID'] == player]
        rds = sorted(tmp['Rodada'])
        acc = 0
        prev = [1, 2, 3, 4, 5]
        for idx,rd in enumerate(rds):
            if idx < 5:
                prev[idx] = float(df.loc[(df["AtletaID"] == player) & (df["Rodada"] == rd), bft])
                continue
            pt = float(df.loc[(df["AtletaID"] == player) & (df["Rodada"] == rd), bft])        
            df_2018.loc[(df["AtletaID"] == player) & (df["Rodada"] == rd), nft] = sum(prev)/len(prev)
            for i in range(0,4):
                prev[i] = prev[i+1]
            prev[4] = float(df.loc[(df["AtletaID"] == player) & (df["Rodada"] == rd), bft])
        
    return df


# In[ ]:


df_2018 = rodadas

df_2018 = df_2018[df_2018.Pontos != 0]


def_2018 = avg(df_2018,'avg.Points', 'Pontos')
def_2018 = avg5(df_2018,'avg.last05', 'Pontos')

def_2018 = avg(df_2018,'avg.FS', 'FS')
def_2018 = avg5(df_2018,'avg.FS.l05', 'FS')

def_2018 = avg(df_2018,'avg.PE', 'PE')
def_2018 = avg5(df_2018,'avg.PE.l05', 'PE')

def_2018 = avg(df_2018,'avg.A', 'A')
def_2018 = avg5(df_2018,'avg.A.ls05', 'A')

def_2018 = avg(df_2018,'avg.FT', 'FT')
def_2018 = avg5(df_2018,'avg.FT.l05', 'FT')

def_2018 = avg(df_2018,'avg.FD', 'FD')
def_2018 = avg5(df_2018,'avg.FD.l05', 'FD')

def_2018 = avg(df_2018,'avg.FF', 'FF')
def_2018 = avg5(df_2018,'avg.FF.l05', 'FF')

def_2018 = avg(df_2018,'avg.G', 'G')
def_2018 = avg5(df_2018,'avg.G.l05', 'G')

def_2018 = avg(df_2018,'avg.I', 'I')
def_2018 = avg5(df_2018,'avg.I.l05', 'I')

def_2018 = avg(df_2018,'avg.PP', 'PP')
def_2018 = avg5(df_2018,'avg.PP.l05', 'PP')

def_2018 = avg(df_2018,'avg.RB', 'RB')
def_2018 = avg5(df_2018,'avg.RB.l05', 'RB')

def_2018 = avg(df_2018,'avg.FC', 'FC')
def_2018 = avg5(df_2018,'avg.FC.l05', 'FC')

def_2018 = avg(df_2018,'avg.GC', 'GC')
def_2018 = avg5(df_2018,'avg.GC.l05', 'GC')

def_2018 = avg(df_2018,'avg.CA', 'CA')
def_2018 = avg(df_2018,'avg.CV', 'CV')

def_2018 = avg(df_2018,'avg.SG', 'SG')
def_2018 = avg5(df_2018,'avg.SG.l05', 'SG')

def_2018 = avg(df_2018,'avg.DD', 'DD')
def_2018 = avg5(df_2018,'avg.DD.l05', 'DD')

def_2018 = avg(df_2018,'avg.DP', 'DP')
def_2018 = avg5(df_2018,'avg.DP.l05', 'DP')

def_2018 = avg(df_2018,'avg.GS', 'GS')
def_2018 = avg5(df_2018,'avg.GS.l05', 'GS')

# In[208]:


df_2018.loc[df_2018['AtletaID'] == 73515]

