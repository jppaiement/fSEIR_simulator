import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import getopt
import os
import random
import sys
import time
import cartopy.crs as ccrs
import seaborn as sns
import csv
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.stats import gaussian_kde
from scipy import stats
from sklearn.neighbors import KernelDensity

global VISUALIZE


#%% 

#DEFINE SIMULATION PARAMETERS

# Make a directory for the data, and change into that directory.
os.chdir(r"C:\...............................")
currenttime = time.strftime("%Y-%m-%dT%H%M%S", time.gmtime())
os.makedirs(currenttime)
os.chdir(currenttime)

# Number of simulations to run
NUM_SIMULATIONS = 5              #Number of simulation to run

#SEIR Model name
MName = "Predictive spread model simulation"

#Learning features from demographic data
used_columns = ['GDP', 'PopDen', 'HealthExp', 'TouristArrival', 'sanitaryServie_pct', 'pop_above65', 'pop_15-64', 'pop_0-14', 'perc_male', 'UrbanPop_perc', 'Pop_growth', 'TouristUSD', 'CO2Emiss', 'ChineseDiasp']

#Ranfom Forest Parameters
nestimators = 150
nmax_depth = 25
nmin_samples_split = 2
nmin_samples = 5

#Number of country demographic clusters
n = 15

#choose 1 prediction method
use_trueevol = False 
use_pred = True       #Use the predicted desease evolution from the random forest model
#choose 1 weighting method
use_demo = False         #Use demographic distance to predict next exposed node
use_weightP = True     #Use ramdom forest feature importance weight to predcit next exposed node

#SEIR input parameters       
beta =  35       #The parameter controlling how many nodes are could be infected by an infected passenger
gamma = 80       #The rate an infected recovers and moves into the resistant phase
alpha = 27     #The rate at which an exposed node becomes infective
panday = 220    #Number of day to which the pandemic spreads

#Define Simulation parameters defaults
start = 3376                    #node ID to start infection
target = 'Wuhan'                # Name of initial infected city
VISUALIZE = True                #Vizualize map during simulation
INTERNATIONAL = True            #Consider International airports 
DOMESTIC = True                 #Consider Domestic airports


#Stem propagation efforts strategy

#Vaccination Strategy
Vaccinated_Countries = ['Italy', 'Spain', 'United States of America', 'United Kingdom', 'Germany', 'France']        #List of countries to vaccinate
DELAY = 60                        #Delay in starting the vaccination program

#Flight Cancellation Strategy
strategy= 'none'               #cancellation strategy ("clustering", "betweenness", "random" or "custom" )
custom_cancel = []               #List of countrie to cancel flights
INTER_cancel = False             #Cancel international flights
DOM_cancel = False               #Cancel domestic flights
Cancel_Delay = 0                #on which step to start cancelling flights

#Stemming efforts (in pervent)
efforts = (0,)                   #Cancellation effort to apply (percentages of cancelled flights from list)


#%% PREPARE NODE DATA (impute and wieghts from ML model)

print('Building predictive model for day zero and calculation weights')
print('****************************************************************************************')

#Import data

DFimp = pd.read_csv(r".....................................................",encoding="ANSI")
print(list(DFimp.columns.values)) 

DF = DFimp.drop_duplicates(subset='Country', keep="first")

#Create Learning and Test sets
#DFlearn = DF.loc[:,6:19]
#print(DFlearn)

#columns to be used to as target
DFlearn = DF[DF['DayZero']<1000]

selected_columns = used_columns
def select_columns(data_frame, column_names):
    new_frame = data_frame.loc[:, column_names]
    return new_frame
voxetL = select_columns(DFlearn, selected_columns)


# Create Training Features and Labels Set
X = voxetL 
y = np.ravel(np.c_[DFlearn['DayZero']])

#Ipute missing
imp = IterativeImputer(missing_values=np.nan, max_iter=100, random_state=0)
Xt= pd.DataFrame(imp.fit_transform(X))
Xt.columns = used_columns

Xt['GDP'] = Xt['GDP']/10000000000
Xt['TouristArrival'] = Xt['TouristArrival']/1000000
Xt['pop_above65'] = Xt['pop_above65']/1000000
Xt['pop_15-64'] = Xt['pop_15-64']/1000000
Xt['pop_0-14'] = Xt['pop_0-14']/1000000

scaler = StandardScaler()
scaler.fit(Xt)
Xt = scaler.transform(Xt)

#bins = np.linspace(0, max(y), StratiBins)
X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.3)  

print('TEST AND LEARN COUNTS')
print('Number of observations in the training data:', len(X_train))
print('Number of observations in the test data:',len(X_test))
print('Number of observations in the target training data:',len(y_train))
print('Number of observations in the target test data:',len(y_test))
print('Number of features:', Xt.shape[1])

while True:            
    # Create Random forest Classifier & Train on data
    X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.3)
    model = RandomForestRegressor(n_estimators=nestimators, max_depth=nmax_depth, min_samples_split=nmin_samples_split, min_samples_leaf=nmin_samples, verbose=3)
    model.fit(X_train, y_train.ravel()) 
    
    train_Y = model.predict(X_train)
    test_Y = model.predict(X_test)
    
    R2 = r2_score(y_test, test_Y) 
    print('R2_score:', R2)

    if R2>=0.5:
        break

print('****************************************************************************************')
    
plot = (sns.jointplot(x=y_test, y=test_Y, kind='reg', color='blue', height=8, scatter_kws={"s": 10})
        .plot_joint(sns.kdeplot, color='blue', shade= False, alpha=0.5)
        )
plot.x = y_train
plot.y = train_Y 
plot.plot_joint(plt.scatter, marker='x', c='g', s=10, alpha=0.8)
plot.ax_marg_x.set_xlim(0, 100)
plot.ax_marg_y.set_ylim(0, 100)
x0, x1 = plot.ax_joint.get_xlim()
y0, y1 = plot.ax_joint.get_ylim()
lims = [max(x0, y0), min(x1, y1)]
plot.ax_joint.plot(lims, lims, ':k', color='red')  
    
    
Features_Importance = pd.DataFrame.from_records((list(zip(DF[selected_columns], model.feature_importances_))))
print(Features_Importance)
importances = model.feature_importances_
x_values = list(range(len(importances)))

plt.figure(figsize=(12,6))
plt.bar(x_values, importances, orientation = 'vertical')
plt.xticks(x_values, selected_columns, rotation='vertical', size=8)
plt.ylabel('Importance'); plt.xlabel('Features'); plt.title('Variable Importances Timeline Prediction');
plt.savefig("FeatureImportance.png", bbox_inches='tight', dpi=96)
plt.show()


def select_columns(data_frame, column_names):
    new_frame = data_frame.loc[:, column_names]
    return new_frame
DFpred = select_columns(DFimp, selected_columns)


print('Imputing Missing Values')
#Ipute missing
imp2 = IterativeImputer(missing_values=np.nan, max_iter=800, random_state=0)
DFpredt= pd.DataFrame(imp2.fit_transform(DFpred))
DFpredt.columns = used_columns
DFpredt['GDP'] = DFpredt['GDP']/10000000000
DFpredt['TouristArrival'] = DFpredt['TouristArrival']/1000000
DFpredt['pop_above65'] = DFpredt['pop_above65']/1000000
DFpredt['pop_15-64'] = DFpredt['pop_15-64']/1000000
DFpredt['pop_0-14'] = DFpredt['pop_0-14']/1000000
  
DFw = pd.DataFrame(DFpredt)
DFw.columns = used_columns
DFimp['predDayZero'] = model.predict(DFw)



print('Done')
print('****************************************************************************************')

#%%
print('Calculating Weights')
DFimp['weightsP'] = (abs(DFw['GDP'])*Features_Importance.loc[0,1])+(abs(DFw['PopDen'])*Features_Importance.loc[1,1])+(abs(DFw['HealthExp'])*Features_Importance.loc[2,1])+(abs(DFw['TouristArrival'])*Features_Importance.loc[3,1]+(abs(DFw['sanitaryServie_pct'])*Features_Importance.loc[4,1])+(abs(DFw['pop_above65'])*Features_Importance.loc[5,1])+(abs(DFw['perc_male'])*Features_Importance.loc[8,1])+(abs(DFw['UrbanPop_perc'])*Features_Importance.loc[9,1])+(abs(DFw['Pop_growth'])*Features_Importance.loc[10,1])+(abs(DFw['TouristUSD'])*Features_Importance.loc[11,1])+(abs(DFw['CO2Emiss'])*Features_Importance.loc[12,1])+(abs(DFw['ChineseDiasp'])*Features_Importance.loc[13,1]))
#DFimp['weightsP'] = ((abs(DFw['HealthExp'])*Features_Importance.loc[2,1])+(abs(DFw['sanitaryServie_pct'])*Features_Importance.loc[4,1])+(abs(DFw['TouristUSD'])*Features_Importance.loc[11,1]))
#DFimp['weightsP'] = (abs(DFw['TouristArrival']))
#DFimp['weightsP'] = (abs(DFw['ChineseDiasp']))

x = DFimp['weightsP'].values.astype(float).reshape(-1,1)
scale = preprocessing.MinMaxScaler()
DFimp['weightsP'] = scale.fit_transform(x,x)


plt.figure(figsize=(10,6))
DFimp.weightsP.hist(alpha=0.4, bins=30, range=[0,1])
plt.title("MWeights from RF model")
plt.xlabel("Weights")
plt.savefig("Weights of nodes.png", bbox_inches='tight', dpi=96)
plt.show()

DFnodes = (DFimp.loc[:, ['ID', 'City', 'Country', 'lat', 'lon', 'weightsP', 'DayZero', 'predDayZero']])

print('Done')
print('****************************************************************************************')

#%%

#Cluster demographic data to assign as weights

print('Clustering of demographic data')
#Create dataset for clustering (countries), impute and scale
country = DFimp.drop_duplicates(subset='Country', keep="first")

used_columns2 = ['DayZero', 'PopDen', 'HealthExp', 'TouristArrival', 'sanitaryServie_pct', 'pop_above65', 'pop_15-64', 'pop_0-14', 'perc_male', 'UrbanPop_perc', 'Pop_growth', 'TouristUSD', 'CO2Emiss','ChineseDiasp']
selected_columns2 = used_columns2
def select_columns(data_frame, column_names):
    new_frame = data_frame.loc[:, column_names]
    return new_frame
countryC = select_columns(country, selected_columns2)

countryCt= pd.DataFrame(imp2.fit_transform(countryC))
countryCt.columns = used_columns2

names = countryCt.columns
scaler = preprocessing.StandardScaler()
scaled_countryCt = scaler.fit_transform(countryCt)
scaled_countryCt = pd.DataFrame(scaled_countryCt, columns=names)

#create linkage matrix ad assign hierarchical clustering
Z = linkage(scaled_countryCt, method='ward', metric='euclidean', optimal_ordering=False)

plt.figure(figsize=(15, 8))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=3.,  # font size for the x axis labels
    )
plt.savefig("Hierarchical tree classification.png", bbox_inches='tight', dpi=200)
plt.show()


MaxK=100
clustersperf = fcluster(Z, MaxK, criterion='maxclust')
clustersperf

last = Z[-MaxK:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)

plt.figure(figsize=(10, 6))
plt.plot(idxs, last_rev)
acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.title("Clustering Performance")
plt.savefig("Hierarchical clustering sensivity.png", bbox_inches='tight', dpi=96)
plt.show()
k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
print ("clusters:", k)

#%%

#Assign clusters to aiports
country['clust'] = fcluster(Z, n, criterion='maxclust')
c=(country.loc[:,'Country'])
cl=(country.loc[:,'clust'])
cdict = dict(zip(c,cl))

DFnodes['clust'] = DFnodes['Country'].map(cdict)
convert_dict = {'ID':int, 'City':str, 'Country':str, 'lat':float, 'lon':float, 'weightsP':float, 'DayZero':float, 'predDayZero':float, 'clust':float} 
DFnodes = DFnodes.astype(convert_dict)

#Save data
DFnodes.to_csv(r'C:/.............................................', index=False, header=False, quoting=csv.QUOTE_NONNUMERIC, quotechar= '"', encoding='ANSI')

print('[Done]')

#%% DEFINITIONS


def create_network(nodes, edges):
    
    #Load Data for Nodes and Routes
    print("Creating network.")
    G = nx.DiGraph()
    
    print("\tLoading airports", end="")
    sys.stdout.flush()
    # Populate the graph with nodes.
    with open('C:/...........................................................................', 'r', encoding='ANSI') as f:
    
        for line in f.readlines():
            entries = line.replace('"',"").rstrip().split(",")
    
            G.add_node(int(entries[0]), 
                           country=entries[2],
                           name=entries[1], 
                           lat=entries[3],
                           lon=entries[4],
                           weightP=entries[5],
                           clust=entries[8],
                           predDayZero=entries[7],
                           dayzero = entries[6]
                           )
    print("\t\t\t\t\t[Done]")
        
    
    print("\tLoading routes",end="")
    # Populate the graph with edges.v
    sys.stdout.flush()
    edge_count = 0
    error_count = 0
    duplicate_count = 0
    line_num = 1
    with open('C:/........................................................................', 'r', encoding='ANSI') as f:
    
        for line in f.readlines():
            entries = line.replace('"',"").rstrip().split(",")
            try:
                if G.has_edge(int(entries[3]),int(entries[5])):
                    duplicate_count += 1
                else:
                    if line_num > 1:
                        from_vertex = int(entries[3])
                        to_vertex = int(entries[5])
                        G.add_edge(from_vertex, to_vertex )
                        G.edges[from_vertex, to_vertex]['IATAFrom'] = entries[2]
                        G.edges[from_vertex, to_vertex]['IATATo'] = entries[4]
                        edge_count += 1
            except ValueError:
                    # The value doesn't exist
                error_count += 1
                pass
            line_num += 1   
    print("\t\t\t\t\t\t[Done]")

    
    def calculate_weights(input_network):
        
        """
        Add weights to the edges of a network based on the degrees of the connecting
        verticies, and return the network.
        Args:
            input_network: A NetworkX graph object
        Returns:
            G: A weighted NetworkX graph object.
        """
        
        G = input_network.copy()
    
        # Add weights to edges
        for n in G.nodes():
            successors = list(G.successors(n))
            weights = dict()
            # Calculate the total out degree of all succs
            wP = float(G.nodes[n]["weightP"])
            predDayZero = float(G.nodes[n]["predDayZero"])
            dayzero = float(G.nodes[n]["dayzero"])
            
                    
            total_degree = 0
            for successor in successors:
                total_degree = G.out_degree(successor)
                
            # Find the weight for all possible successors
            for successor in successors:
                successor_degree = G.out_degree(successor)
                if total_degree > 0:
                    probability_of_infection = successor_degree/total_degree  
                else:
                    probability_of_infection = 0    
                weights[successor] = probability_of_infection  
        
            largest_weight = 0
            smallest_weight = 2
            for successor, weight in weights.items():
                if weight > largest_weight:
                    largest_weight = weight
                elif weight < smallest_weight:
                    smallest_weight = weight
                  
            for successor in successors:
                if largest_weight != smallest_weight:
                        relative_weight = str(((weights[successor] - smallest_weight) / (largest_weight - smallest_weight)))
                        
                else:
                    relative_weight = 0
                #print(relative_weight)
                G[n][successor]['weight'] = relative_weight
                G[n][successor]['weightP'] = wP
                G[n][successor]['predDayZero'] = predDayZero
                G[n][successor]['dayzero'] = dayzero
    
        return G
    
    

    
    # Calculate the edge weights
    print("\tCalculating edge weights",end="")
    G = calculate_weights(G)
    print("\t\t\t\t[Done]")
    
    # Limit to the first subgraph
    print("\tFinding largest subgraph",end="")
    undirected = G.to_undirected()
    subgraphs = nx.subgraph(G, undirected)
    subgraph_nodes = subgraphs.nodes()
    to_remove = list()
    for node in G.nodes():
        if node not in subgraph_nodes:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    print("\t\t\t\t[Done]")
    
    # Remove nodes without inbound edges
    print("\tRemoving isolated vertices",end="")
    indeg = G.in_degree()
    outdeg = G.out_degree()
    to_remove=[n for n, degree in indeg if (indeg[n] + outdeg[n] < 1)]  
    G.remove_nodes_from(to_remove)
    print("\t\t\t\t[Done]")
    
    
    # Add clustering data
    print("\tCalculating clustering coefficents",end="")
    cluster_network = nx.Graph(G)
    lcluster = nx.clustering(cluster_network)
    for i,j in G.edges():
        cluster_sum = lcluster[i] + lcluster[j]
        G[i][j]['cluster'] = cluster_sum
    print("\t\t\t[Done]")
    
    # Flag flights as domestic or international and remove Domestic
    print("\tCategorizing international and domestic flights",end="")
    for i,j in G.edges():
        if G.nodes[i]["country"] == G.nodes[j]["country"]:
            G[i][j]['international'] = False
        else:
            G[i][j]['international'] = True
    print("\t\t[Done]")
    
    # Calculate distance between demographics
    print("\tCalculaying demographic clusters distance",end="")
    for i,j in G.edges():
        G[i][j]['DistDemo'] = abs(float(G.nodes[i]["clust"]) - float(G.nodes[j]["clust"]))
    print("\t\t[Done]")
    
    # Remove nodes without inbound edges
    print("\tRemoving isolated vertices",end="")
    indeg = G.in_degree()
    outdeg = G.out_degree()
    to_remove=[n for n, degree in indeg if (indeg[n] + outdeg[n] < 1)]  
    G.remove_nodes_from(to_remove)
    print("\t\t\t\t[Done]")
    
    # Limit to the first subgraph
    print("\tFinding largest subgraph",end="")
    undirected = G.to_undirected()
    subgraphs = nx.subgraph(G, undirected)
    subgraph_nodes = subgraphs.nodes()
    to_remove = list()
    for node in G.nodes():
        if node not in subgraph_nodes:
            to_remove.append(node)
    G.remove_nodes_from(to_remove)
    print("\t\t\t\t[Done]")
      
    
    return G


def infection(input_network, vaccination, starts, DELAY=DELAY, Cancel_Delay=Cancel_Delay, vis = True, file_name = "sir.csv", title = MName,  RECALCULATE = False):
    
    print("Simulating infection.")
    
    network = input_network.copy()
       
    # Recalculate the weights of the network as per necessary
    
    # Open the data file
    f = open(file_name, "w")
    f.write("time, s, e, i, r\n")
    f.close()
    
    # Set the default to susceptable
    sys.stdout.flush()
    for node in network.nodes():
        network.nodes[node]["status"] =  "s"
        network.nodes[node]["color"] = "#A0C8F0"
        network.nodes[node]["age"] = 0
        
    # Assign the infected    
    #for start in starts:
    infected = start
    network.nodes[infected]["status"] = "i"
    network.nodes[infected]["color"]  = "red"
    
    if vis:
        pos = nx.spring_layout(network, scale=2)
    
    if isinstance(network,nx.DiGraph):
        in_degree = network.in_degree()[infected] 
        out_degree = network.out_degree()[infected]
        degree = in_degree + out_degree
    else:
        degree = network.degree()[infected]
    print("\t",network.nodes[infected]["name"],"[",degree,"]", " connections")
 
    #List vaccinated edges and remove   
    for i,j in network.edges():
        network[i][j]["vaccinated"] = False
        if network.nodes[i]["country"] in Vaccinated_Countries or network.nodes[j]["country"] in Vaccinated_Countries:
            network[i][j]["vaccinated"] = True
    vaccination = list(((u,v) for u,v,j in network.edges(data=True) if j['vaccinated'] == True))
            
    
    if vaccination is not None:
        print("\tVaccinated: ",Vaccinated_Countries, ": ", len(vaccination)," edges" )
    else: 
        print("\tVaccinated: None")
    
    if cancelled is not None:
        print("\tCancelled: ", len(cancelled)," edges" )
    else: 
        print("\tCancelled: None")
        
    # Iterate Vaccination and/or Cancellation through the evolution of the disease.
    for step in range(0,panday):
        # If the delay is over, vaccinate.
        # Convert the STRING! 
        if int(step) == int(DELAY):
            if vaccination is not None:
                print(DELAY,"Vaccination on step",DELAY)
                network.remove_edges_from(vaccination)
                # Recalculate the weights of the network as per necessary
                if RECALCULATE == True:
                    network = calculate_weights(network)
        if int(step) == int(Cancel_Delay): 
            if cancelled is not None:
                print("Cancellation on step",Cancel_Delay, ": ", len(cancelled), " remove flights")
                network.remove_edges_from(cancelled)
                # Recalculate the weights of the network as per necessary
                if RECALCULATE == True:
                    network = calculate_weights(network)

                    
    # Create variables to hold the outcomes as they happen
        S,E,I,R = 0,0,0,0
    
        for node in network.nodes():
            status = network.nodes[node]["status"]
            age = network.nodes[node]["age"]
            color = network.nodes[node]["color"]
            
    
            if status is "i" and age >= gamma:
                # The infected has reached its recovery time after 60 days
                network.nodes[node]["status"] = "r"
                network.nodes[node]["color"] = "purple"
                    
            if status is "e" and age >= alpha and age < gamma:
                # Exposed nodes have an incubation in average 14 days
                network.nodes[node]["status"] = "i"
                network.nodes[node]["color"] = "red"
    
            elif status is "e":
                network.nodes[node]["age"] += 1
    
            elif status is "i":
                # Propogate the infection.
                if age > alpha:
                    victims = (list(network.successors(node)))
                    number_infections = 0
                    
                    if len(victims) >= beta:
                        victims = random.sample((list(network.successors(node))), beta)
                        number_infections = 0
                    else:
                        victims = (list(network.successors(node)))
                        number_infections = 0
                    
                    for victim in victims:
                        infect_status = network.nodes[victim]["status"]
                        infect = False # Set this flag to False to start weighting.
                        rand = random.uniform(0,1)                         
                        
                        if network[node][victim]['international'] == False and random.uniform(0,1) <= float(network[node][victim]['weight']):
                            infect = True
                            number_infections+=1
                                                      
                        if use_pred == True and network[node][victim]['international'] == True :
                            if use_demo == True and network[node][victim]['DistDemo'] >= rand:
                                infect = True
                                number_infections+=1
                            if use_weightP == True and rand <= float(network[node][victim]['weightP']):
                                infect = True
                                number_infections+=1
    
                        if use_trueevol == True and network[node][victim]['international'] == True and float(network[node][victim]['dayzero'])<step:
                            if use_demo == True and network[node][victim]['DistDemo'] >= rand:
                                infect = True
                                number_infections+=1
                            if use_weightP == True and rand <= float(network[node][victim]['weightP']):
                                infect = True
                                number_infections+=1
    
                        if infect_status == "s" and infect == True:
                            network.nodes[victim]["status"] = "e"
                            network.nodes[victim]["age"] = 0
                            network.nodes[victim]["color"] = "#30cc1f"
    
            network.nodes[node]["age"] += 1
    
            # Loop twice to prevent bias.
            for node in network.nodes():
                status = network.nodes[node]["status"]
                age = network.nodes[node]["age"]
                color = network.nodes[node]["color"]
    
                if status is "s":
                    # Count those susceptable
                    S += 1
    
                if status is "e":
                    E += 1
    
                if status is "v":
                    S += 1
    
                elif status is "r":
                    R += 1
    
                elif status is "i":
                    I += 1
    
    
        print("{0}, {1}, {2}, {3}, {4}".format(step, S, E, I, R))
    
    
        printline = "{0}, {1}, {2}, {3}, {4}".format(step, S, E, I, R)
        f = open(file_name, "a")
        f.write(printline + "\n")
        f.close()
    
        print("\t"+printline)
    
        if I is 0:
            break
    
        if vis:
            #write_dot(network, title+".dot")
            visualize(network, title, pos)
            
    print("\t----------\n\tS: {0}, I: {1}, R: {2}".format(S,I,R))


    return {"Suscceptable":S,"Infected":I, "Recovered":R}


def weighted_random(weights):
    number = random.random() * sum(weights.values())
    for k,v in weights.items():
        if number <= v:
            break
        number -= v
    return k


def pad_string(integer, n):
    """
    Add "0" to the front of an interger so that the resulting string in n 
    characters long.
    Args:
        integer: The number to pad.
        n: The desired length of the string
    Returns
        string: The padded string representation of the integer.
        
    """

    string = str(integer)

    while len(string) < n:
        string = "0" + string

    return string


def visualize(network, title,pos):
    """
    Visualize the network given an array of posisitons.
    """
    print("-- Starting to Visualize --")
    
    colors = []
    colori = []
    i_edge_colors = []
    d_edge_colors = []
    default = []
    infected = []
    nstart = []
    ninfect = []
    
    for node in network.nodes():
        colorn = network.nodes[node]["color"]
        if colorn == "#A0C8F0":
            nstart.append(node)
            colors.append(network.nodes[node]["color"])
        elif colorn == "#30cc1f" or colorn == "red" or colorn == "purple":
            ninfect.append(node)
            colori.append(network.nodes[node]["color"])            
            
    for i,j in network.edges():
        color = network.nodes[i]["color"]
        if color == "#A0C8F0" or color == "#30cc1f" or color == "purple":
            color = "#A6A6A6"
            default.append((i,j))
            d_edge_colors.append(color)
        else:
            color = "red"
            infected.append((i,j))
            i_edge_colors.append(color)
            
            
    plt.figure(figsize=(30,20))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    
    #make density plot of infection
    node_positions = {node[0]: (float(node[1]['lon']), float(node[1]['lat'])) for node in network.nodes(data=True)}
    
    xp = []
    yp = []
    
    for node in network.nodes():
        infec = network.nodes[node]["status"] 
        if infec == 'i':
            xp.append(network.nodes[node]['lon'])
            yp.append(network.nodes[node]['lat'])
    
    if len(xp)>=1:
        m1, m2 = np.array(xp).astype(np.float), np.array(yp).astype(np.float)
        xmin = -180
        xmax = 180
        ymin = -90
        ymax = 90
        
        # get the density estimation 
        Xp, Yp = np.mgrid[xmin:xmax:250j, ymin:ymax:250j]
        XpYp = np.vstack([Xp.ravel(), Yp.ravel()]).T
        XpYp = np.radians(XpYp)
        values = np.column_stack((np.array(np.vstack(m1)), np.array(np.vstack(m2))))
        kernel = KernelDensity(bandwidth=0.035)
        kernel.fit(np.radians(values))
 
        #kernel = stats.gaussian_kde(values)
        Z = np.exp(kernel.score_samples(XpYp))
        Z = Z.reshape(Xp.shape)
  
        # plot the result
        cmap = plt.cm.jet
        cmap.set_under('white')
        plt.imshow(np.rot90(Z), norm = plt.Normalize(vmin=(Z.max()-(Z.max()*0.9)), vmax=Z.max()), cmap=cmap,
               extent=[xmin, xmax, ymin, ymax], alpha=0.3, interpolation = 'gaussian')
    
        
    # Fist pass - Gray lines
    nx.draw_networkx_edges(network,pos=node_positions,edgelist=default,
            width=0.005,
            edge_color=d_edge_colors,
            alpha=0.005,
            arrows=False)
   
    # Second Pass - Colored lines
    nx.draw_networkx_edges(network,pos=node_positions,edgelist=infected,
            width=0.1,
            edge_color=i_edge_colors,
            alpha=0.25,
            arrows=False)

     # first Pass - small nodes
    nx.draw_networkx_nodes(network,
            pos=node_positions,
            nodelist=nstart,
            linewidths=0.2,
            node_size=5,
            with_labels=False,
            node_color = colors)
    
#    # Second Pass - large nodes
    nx.draw_networkx_nodes(network,
            pos=node_positions,
            nodelist=ninfect,
            linewidths=0.2,
           node_size=20,
            with_labels=False,
            node_color = colori)
        
    plt.axis('off')

    number_files = str(len(os.listdir()))
    while len(number_files) < 3:
        number_files = "0" + number_files

    plt.savefig("infection-{0}.png".format(number_files),
                bbox_inches='tight', dpi=72 
            )
    plt.show()
    plt.close()


#%% BUILDING NETWORK

simulation = 0

for i in range (NUM_SIMULATIONS):
    
   for effort  in efforts: 
    
        #seed = 100
        #random.seed(seed)
        
        # Identify the script.
        print("Flight Network Disease Simulator 1.0.0")
        print("Modified by Jean-Philippe Paiement from Nicholas A. Yager and Matthew Taylor\n\n")
        
        
        #Simulation od the Pandemic
        print("Setting Simulation Parameters.")
        
        
        # Determine the parameters of the current simulation.
        args = sys.argv[1:]
        opts, args = getopt.getopt("brcsidv",["delay=","nsim="])
        
        AIRPORT_DATA = args[0]
        ROUTE_DATA = args[1]
        
        # Make a new folder for the data.
        
        subsim = (strategy + pad_string(simulation,4))
        os.makedirs(subsim)
        os.chdir(subsim)
        
        # Create the network using the command arguments.
        network = create_network(AIRPORT_DATA, ROUTE_DATA)
        
        print("\tDetermining network type.")
        
        # Determine if the graph is directed or undirected
        if isinstance(network,nx.DiGraph):
            network_type = "Directed"
        else:
            network_type = "Undirected"
        print("\t\t[Done]")
        
        print("\tCalculaing edges and verticies.")
        # Number of verticies and edges
        edges = network.number_of_edges()
        verticies = network.number_of_nodes()
        print("\t\t[Done]")
            
        # Not every vertex can lead to every other vertex.
        # Create a subgraph that can.
        print("\tTemporarily converting to undirected.")
        undirected = network.to_undirected()
        print("\t\t[Done]")
        print("\tFinding subgraphs.")
        subgraphs = [undirected.subgraph(c).copy() for c in nx.connected_components(undirected)]
        print("\t\t[Done]")
        # Find the number of vertices in the diameter of the network
        print("\tFinding network diameter.")
        #diameter = nx.diameter(subgraphs[0])
        print("\t\t[Done]")
        
        print("\tStoring network parameters")
        data_file = open("network.dat", "w")
        data_file.write("Simulation name: ")
        data_file.write("Network properties\n===============\n")
        data_file.write("Network type: {0}\n".format(network_type))
        data_file.write("Number of verticies: {0}\n".format(verticies))
        data_file.write("Number of edges: {0}\n".format(edges))
        #data_file.write("Diameter: {0}\n".format(diameter))
        
        data_file.close()
        print("\t\t[Done]")
        
        
        print("\tRemoving international and/or domestic flights")
        #Remove International and/or Domestic flights
        if INTERNATIONAL == False:
            etoi_remove = list(((u,v) for u,v,j in network.edges(data=True) if j['international']==True))
            network.remove_edges_from(etoi_remove)   
        
        if DOMESTIC == False:
            etod_remove = list(((u,v) for u,v,j in network.edges(data=True) if j['international']==False))
            network.remove_edges_from(etod_remove)
        print("\t\t[Done]")
        
        #Drawing network
        print("\tPlotting Network.")
        node_positions = {node[0]: (float(node[1]['lon']), float(node[1]['lat'])) for node in network.nodes(data=True)}
        plt.figure(figsize=(30,20))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        
        nx.draw_networkx_nodes(network, pos=node_positions, node_color = 'blue', linewidths=0.2, node_size=40, with_labels=False)
        nx.draw_networkx_edges(network,pos=node_positions, edge_color='gray', width=0.2, alpha=0.6, arrows=False)   
        plt.axis('off')
        plt.savefig("Air Transportation Network.png", bbox_inches='tight', dpi=300)
        plt.show()
        
        print("\t\t\t\t[Done]")
        
        
    #%%    RUNNING SIMULATION
        
        #Creating flight cancellation list
        print("{0} Cancellation Strategy Mode.".format(strategy) )
        
        # Generate a list a sorted list of flights to cancel based on the
        # strategy.
        
        #Build cancellation edge pool
        print("\tBuilding cancellation list")
        
        edgepoolG = network.copy()
        
        edgepool = list()
        
        if INTER_cancel == True and DOM_cancel == False:
            etoi_remove = list(((u,v) for u,v,j in network.edges(data=True) if j['international']==False))
            edgepoolG.remove_edges_from(etoi_remove)   
            edgepool = list(edgepoolG.edges(data=True))    
    
        if DOM_cancel == True and INTER_cancel == False:
            etod_remove = list(((u,v) for u,v,j in network.edges(data=True) if j['international']==True))
            edgepoolG.remove_edges_from(etod_remove)
            edgepool = list(edgepoolG.edges(data=True))
    
        else:
            edgepool = list(edgepoolG.edges(data=True))
        
        
        cancellist = list()
        if strategy != 'none':
            cancellist = list()
        
        if strategy == "random": 
            # Sort the edges randomly
            cancellist = random.sample(edgepool, len(edgepool))
        
        if strategy == "clustering":
            # Sort the edges based on the sum of the clustering coefficent.
            sorted_cluster = sorted(edgepool, key=lambda k: k[2]['cluster'], reverse=True)
            for cluster_item in sorted_cluster:
                if network[cluster_item[0]][cluster_item[1]]['cluster'] < 2:
                    if network[cluster_item[0]][cluster_item[1]]['cluster'] > 0:
                        cancellist.append((cluster_item[0], cluster_item[1]))
        
        if strategy == "betweenness":
            # Sort the edges based on weighted edge-betweenness.
            betweennesses = nx.edge_betweenness_centrality(network, weight='weight')
            cancellist = sorted(betweennesses.keys(), key=lambda k: betweennesses[k], reverse=True)
     
        elif strategy == "custom" and len(custom_cancel)>0:
            cancellist = list()
            for (u,v) in edgepoolG.edges():
                if edgepoolG.nodes[u]['country'] in custom_cancel or edgepoolG.nodes[v]['country'] in custom_cancel:
                    eremove=(u,v)
                    cancellist.append(eremove)
        
                #print(cancellist[:20])
        print(len(cancellist), " Flights available for cancellation")
        print("\t\t[Done]")
        
           
        #Open a file for this targets dataset
        #output_file = open("{0}/{0}_{1}.csv".format(strategy, pad_string(simulation,4)),"w")
        #output_file.write('"effort","total_infected, edges_closed"\n')
        
        #Running simulation
        if effort > 0:
            max_index = int((len(cancellist) * (effort)/100)-1)
            cancelled = cancellist[0:max_index]
        else:
            cancelled = None
        
        
        title = "{0} - {1}%".format(strategy, effort/100)
        results = infection(network, start, target, vis=VISUALIZE, title=title, DELAY=DELAY, Cancel_Delay=Cancel_Delay)
        total_infected = results["Infected"] + results["Recovered"]
        #output_file.write("{0},{1}\n".format(effort/100,total_infected))
                        
        #if total_infected == 1:
        #    for remaining_effort in range(effort+1):
        #        output_file.write("{0},{1}\n".format(remaining_effort/100, total_infected))
        #        break
        
        simulation += 1
        #iteration += 1
        #output_file.close()
        os.chdir(r"C:\Users\jeanphilippep\OneDrive - mirageoscience\PROJECTS\COVID-19")
        os.chdir (currenttime)


#%%

strategy= 'custom'               #cancellation strategy ("clustering", "betweenness", "random" or "custom" )
custom_cancel = ['China', 'Japan']               #List of countrie to cancel flights
INTER_cancel = True             #Cancel international flights
DOM_cancel = True               #Cancel domestic flights
Cancel_Delay = 32                #on which step to start cancelling flights

#Stemming efforts (in pervent)
efforts = (100,) 

RECALCULATE = False

args = sys.argv[1:]
opts, args = getopt.getopt("brcsidv",["delay=","nsim="])
        
AIRPORT_DATA = args[0]
ROUTE_DATA = args[1]
        
# Create the network using the command arguments.
network = create_network(AIRPORT_DATA, ROUTE_DATA)



edgepoolG = network.copy()
        
edgepool = list()

if INTER_cancel == True and DOM_cancel == False:
    etoi_remove = list(((u,v) for u,v,j in network.edges(data=True) if j['international']==False))
    edgepoolG.remove_edges_from(etoi_remove)   
    edgepool = list(edgepoolG.edges(data=True))    
    
if DOM_cancel == True and INTER_cancel == False:
    etod_remove = list(((u,v) for u,v,j in network.edges(data=True) if j['international']==True))
    edgepoolG.remove_edges_from(etod_remove)
    edgepool = list(edgepoolG.edges(data=True))
    
else:
    edgepool = list(edgepoolG.edges(data=True))


cancellist = list()
if strategy != 'none':
    cancellist = list()
        
if strategy == "random": 
# Sort the edges randomly
    cancellist = random.sample(edgepool, len(edgepool))
        
if strategy == "clustering":
# Sort the edges based on the sum of the clustering coefficent.
    sorted_cluster = sorted(edgepool, key=lambda k: k[2]['cluster'], reverse=True)
    for cluster_item in sorted_cluster:
        if network[cluster_item[0]][cluster_item[1]]['cluster'] < 2:
            if network[cluster_item[0]][cluster_item[1]]['cluster'] > 0:
                cancellist.append((cluster_item[0], cluster_item[1]))
        
if strategy == "betweenness":
# Sort the edges based on weighted edge-betweenness.
    betweennesses = nx.edge_betweenness_centrality(network, weight='weight')
    cancellist = sorted(betweennesses.keys(), key=lambda k: betweennesses[k], reverse=True)
     
elif strategy == "custom" and len(custom_cancel)>0:
    cancellist = list()
    for (u,v) in edgepoolG.edges():
        if edgepoolG.nodes[u]['country'] in custom_cancel or edgepoolG.nodes[v]['country'] in custom_cancel:
            eremove=(u,v)
            cancellist.append(eremove)

#print(cancellist[:20])
print(len(cancellist), " Flights available for cancellation")
print("\t\t[Done]")
        
        
if effort > 0:
    max_index = int((len(cancellist) * (effort)/100)-1)
    cancelled = cancellist[0:max_index]
else:
    cancelled = None
    
network.remove_edges_from(cancelled)
# Recalculate the weights of the network as per necessary
if RECALCULATE == True:
    network = calculate_weights(network)
        
node_positions = {node[0]: (float(node[1]['lon']), float(node[1]['lat'])) for node in network.nodes(data=True)}
plt.figure(figsize=(30,20))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
        
nx.draw_networkx_nodes(network, pos=node_positions, node_color = 'blue', linewidths=0.2, node_size=40, with_labels=False)
nx.draw_networkx_edges(network,pos=node_positions, edge_color='gray', width=0.2, alpha=0.6, arrows=False)   
plt.axis('off')
plt.show()
        
