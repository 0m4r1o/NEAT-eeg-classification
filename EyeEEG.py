import neat
from sklearn.metrics import matthews_corrcoef
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.signal import butter, filtfilt
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
data = pd.read_csv('EEG_Eye_State_Classification.csv')

plot= False

if plot:
	dfzscore=stats.zscore(data)
	zscorelist=np.abs(dfzscore)
	filter=(zscorelist<4).all(axis=1)
	df=data[filter]
	df=df.reset_index(drop=True)
	dfplot=df.drop(columns='eyeDetection')
	for i in dfplot.columns:
	    dfplot[i].plot(figsize=(15,2),title=i)
	    plt.show()

	windowsize=100
	for i in dfplot.columns:
	    dfplot[i].rolling(window=windowsize,center=False).mean().plot(figsize=(15,2),title=i)
	    plt.show()

	sns.histplot(data=data,x='eyeDetection')

for column in data.columns[:-1]:  # Exclude the target column
    data_mean, data_std = data[column].mean(), data[column].std()
    cut_off = data_std * 4
    lower, upper = data_mean - cut_off, data_mean + cut_off
    data = data[(data[column] >= lower) & (data[column] <= upper)]

# Reset index after removing outliers
data.reset_index(drop=True, inplace=True)
X = data.drop(columns=['eyeDetection'])  # Features
y = data['eyeDetection']  # Target

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

fs = 200  # Sampling frequency
highcut = 4.0  # Higher cutoff frequency (Hz)

# Additional EEG preprocessing steps
def highpass_filter(data, highcut, fs, order=4):
    nyquist = 0.5 * fs
    high = highcut / nyquist
    b, a = butter(order, high, btype='high')
    y = filtfilt(b, a, data)
    return y
# Apply highpass_filter to EEG signals
X_train_filtered = highpass_filter(X_train_scaled.T,  highcut, fs).T
X_test_filtered = highpass_filter(X_test_scaled.T,  highcut, fs).T

X_train = X_train_filtered
X_test = X_test_filtered
# Define the fitness function
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        y_pred_train = []
        for xi in X_train:
            output = net.activate(xi)
            y_pred_train.append(output[0] > 0.5)
        
        # Calculate MCC for the training predictions
        mcc = matthews_corrcoef(y_train, y_pred_train)*100
        # print(f'MCC = {mcc} ')
        genome.fitness = mcc

# Set up the NEAT configuration
config_path = "neat_config"  # Make sure to have a neat_config file in the same directory
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Run for up to 100 generations.
winner = p.run(eval_genomes, 100)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Evaluate the best genome on the test data
net = neat.nn.FeedForwardNetwork.create(winner, config)
y_pred_test = []
for xi in X_test:
    output = net.activate(xi)
    y_pred_test.append(output[0] > 0.5)

mcc_test = matthews_corrcoef(y_test, y_pred_test)
print(f"Test MCC: {mcc_test:.3f}")