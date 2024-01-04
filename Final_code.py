import numpy as np
import matplotlib.pyplot as plt
import math as m
import pandas as pd
import seaborn as sns
import pydotplus
import os
import sklearn.metrics as metrics

from sklearn.tree import export_text, plot_tree, DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

#ENV SETTINGS VARIABLES
_PRINT = False
_SHOW = False
_SAVE_IMAGE = False
_SAVE_CSV = True

#FUNCTION
def save(data : pd.DataFrame,name : str):
    if _SAVE_CSV : data.to_csv(os.getcwd() + '/' + name + '.csv')

#IMPORT & CLEAN DATA
data = pd.DataFrame(pd.read_csv('CFBeattendance.csv', encoding='cp1252'))
if _PRINT : print('data.head(10)',data.head(10))

#dropping data
columnsDrop = ['Date','Site','Tailgating','Conference']
data.drop(columnsDrop, axis=1, inplace=True)
if _PRINT : print('data.dtypes',data.dtypes)

#Cleaning Team and Opponent Column
data.Team = data.Team.apply(lambda x : x.replace('*',' '))
data.Team = data.Team.apply(lambda x : x[6:].lstrip() if 'No.' in x else x)
data.Opponent = data.Opponent.apply(lambda x : x.replace('*',' '))
data.Opponent = data.Opponent.apply(lambda x : x[6:].lstrip() if 'No.' in x else x)
homeTeams = data.groupby('Team').count().sort_values(by='Opponent',ascending=False)['Opponent'].index
opponentTeams = data.groupby('Opponent').count().sort_values(by='Team',ascending=False)['Team']

# How many match are they when an opponentTeam can also be a homeTeam in the dataset
sum = 0
for homeTeam in homeTeams :
    if homeTeam not in opponentTeams.index:
        if _PRINT : print(homeTeam,'is not an Opponent team')
    else :
        sum+=opponentTeams[homeTeam]
if _PRINT : print(sum,' games where the opponent is also a homeTeam in the dataset \n')

# Only keep these games
data = data[data.Opponent.apply(lambda x : x in data.Team.values)]
data = data[data.Team.apply(lambda x : x in data.Opponent.values)]

#get dummies for the team name
team_dummies = pd.get_dummies(data.Team,dtype=int)*2 + pd.get_dummies(data.Opponent,dtype=int)
data[team_dummies.columns] = team_dummies
save(data,'data')

#1.0.2 Rank ‘NR’
#Non Ranked team become 26th
data.loc[:,'Rank'] = data.Rank.apply(lambda x : int(26) if x=='NR' else int(x))
data.loc[:,'Opponent_Rank'] = data.Opponent_Rank.apply(lambda x : int(26) if x=='NR' else int(x))
# Tranform TV channels into boolean
data.loc[:,'TV'] = data['TV'].apply(lambda x : int(x=='Not on TV'))
#Int new_coach
data.loc[:,'New Coach'] = data['New Coach'].apply(int)

#Transform Time as int
if _PRINT : print('data.Time.str.contains(\'PM\')',data.Time.str.contains('PM'))
time = data.Time.str.replace(':','')
afternoonshift = data.Time.apply(lambda x : 1200 if 'PM' in x else 0)
data.Time = time.apply(lambda x : int(x.split()[0]))
data.Time = data.Time + afternoonshift

if _PRINT : print('data.Result.apply(lambda x : x.split(\' \')[0]).unique()',data.Result.apply(lambda x : x.split(' ')[0]).unique())
if _PRINT : print('Find the \'NC\' case')
if _PRINT : print('--data[data.Result.str.contains(\'NC\')]',data[data.Result.str.contains('NC')])

# Resolve the 'NC' case and Transform into boolean
data.Result = data.Result.str.replace('NC','L').apply(lambda x : x.split('␣')[0]=='W')
data.loc[:,'Result'] = data['Result'].apply(int)

data = data.drop(columns=['Team','Opponent'])
save(data,'data')

if _PRINT : print(data)

if _SHOW :
    sns.countplot(x = "Result", data = data)
    plt.title("Result")
    plt.show()
    sns.countplot(x = "Team", data = data, hue = "Result")
    plt.xticks(rotation = 45)
    plt.title("Teams")
    plt.show()

y = data['Result'] # encode edibility as 1 or 0

X = data.drop(columns = ["Result"])
save(X,'test')

if _PRINT : print(X.columns)
if _PRINT : print(X.head())
#X = pd.get_dummies(X,dtype = int)
if _PRINT : print(X.columns)
if _PRINT : print(X.head())
if _PRINT : print('X.shape ',X.shape)
if _PRINT : print(X.head())

save(X,'final')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)#random_state=2)
print('X_train.shape ',X_train.shape)
print('X_test.shape ',X_test.shape)
save(X,'final')

#TRAIN AND TEST
clfs = [DecisionTreeClassifier(random_state=0, max_depth=2),
DecisionTreeClassifier(random_state=0, max_depth=3),
DecisionTreeClassifier(random_state=0, max_depth=4),
DecisionTreeClassifier(random_state=0, max_depth=5),
DecisionTreeClassifier(random_state=0, max_depth=30),
RandomForestClassifier(criterion='gini'),
RandomForestClassifier(criterion='entropy'),
RandomForestClassifier(criterion='log_loss')
]

clfs_names = ['DT depth2',
'DT depth3',
'DT depth4',
'DT depth5',
'DT depth30',
'RF gini',
'RF entropy',
'RF log_loss']

clfs = [DecisionTreeClassifier(random_state=0, max_depth=3)]
clfs_names = ['DT depth3']
def train_test_acc(clf,training = False):
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_training = model.predict(X_train)
    if training : return accuracy_score(y_train,y_pred_training)
    return accuracy_score(y_test, y_pred)

for clf,clfs_name in zip(clfs,clfs_names):
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc = train_test_acc(clf)
    confmat = confusion_matrix(y_test, y_pred)
    print(clfs_name)
    print("Accuracy:", acc)

    # plot the confusion matrix
    if _PRINT:
        ax = plt.subplot()
        sns.heatmap(confmat,annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted labels',fontsize=15)
        ax.set_ylabel('True labels',fontsize=15)
        plt.show()
        plot_tree(model, feature_names = list(X.columns), filled = True, class_names=["lose", "win"])
        plt.show()
    # save as pdf for a high res image:
    if _SAVE_IMAGE :
        d_tree = export_graphviz(clf, feature_names = list(X.columns), filled = True, class_names=["lose", "win"])
        pydot_graph = pydotplus.graph_from_dot_data(d_tree)
        pydot_graph.write_pdf(os.getcwd() + '/' + 'football_tree.pdf')


depth = range(1,20)
criterias = ['entropy']
random_state = 42
accs = np.array([[train_test_acc(DecisionTreeClassifier(max_depth=i, criterion=criterion, random_state=np.random.seed())) for i in depth] for criterion in criterias])
accs_training = np.array([[train_test_acc(DecisionTreeClassifier(max_depth=i, criterion=criterion, random_state=np.random.seed())) for i in depth] for criterion in criterias])

for acc,acc_training in zip(accs,accs_training) :
    plt.plot(depth,acc)
    plt.plot(depth,acc_training)
plt.tight_layout()
plt.title('Accuracy depending of the Decision Tree\'s parameters')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.legend(('testing accuracy','training accuracy'))
plt.grid()
plt.show()

criterion = ['log_loss']
splitter = ['best', 'random']
max_depth = [2,3,4,5,6,7,8]
min_samples_split = [2, 5, 10, 15, 20]
min_samples_leaf = [1, 2, 3, 4]
max_features = ['sqrt', 'log2']
# Create the random grid
random_grid = {'criterion': criterion,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'splitter':splitter}
rf = DecisionTreeClassifier()
rf_random = RandomizedSearchCV(estimator = rf,
                            param_distributions = random_grid,
                            n_iter = 200,
                            cv = 3,
                            verbose=2,
                            random_state=42,
                            n_jobs = -1)
rf_random.fit(X_train, y_train)

print(train_test_acc(DecisionTreeClassifier(**rf_random.best_params_)))
rf_random.best_params_

mlp = MLPClassifier(solver='adam',
                    alpha=1e-4,
                    max_iter = 2000,
                    hidden_layer_sizes=(20,),
                    verbose=True,
                    learning_rate_init=1e-5,
                    learning_rate='adaptive',
                    n_iter_no_change = 10000)

mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)
print(accuracy_score(y_test, y_pred))

plt.plot(mlp.loss_curve_)

plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

print(accuracy_score(mlp.predict(X_train),y_train))
accuracy_score(mlp.predict(X_test),y_test)