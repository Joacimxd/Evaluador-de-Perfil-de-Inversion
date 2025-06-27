import numpy as np

def normalize_per_col(array):
    arr_min = array.min(axis=0)
    arr_max = array.max(axis=0)
    return (array - arr_min) / (arr_max - arr_min)


class KMeans():
    def __init__(self, n_means, linear = False):
        self.no_features = None
        self.n_means = n_means
        self.linear = linear
        self.means = None
        self.dataset = None
        self.labels = None

    def obtain_means(self, array, ran_means):
        means = []
        for i in range(len(array)):
            for j in range(len(array[i])):
                if np.isnan(np.array(array[i], dtype=np.float64)).any():
                    array[i][j] = ran_means[i]
            
        for i in range(len(array)):
            mean = np.mean(array[i], 0)
            means.append(mean)
            
        return means
    
    def step(self, dataset, ran_means):
        k_means = len(ran_means)
        no_features = len(ran_means[0])
        classified_samples = [[] for i in range(k_means)]
        classified_labels = [[] for i in range(k_means)]

        for i in range(dataset.shape[0]):
            distances = []
            for j in range(k_means):
                distance = 0
                for k in range(no_features):
                    distance += (ran_means[j][k] - dataset[i][k+1])**2
                distances.append(distance**(1/2))
            print(distances)
            print(np.argmin(distances))
            classified_samples[np.argmin(distances)].append(dataset[i][1:])
            classified_labels[np.argmin(distances)].append(dataset[i][0])

        return self.obtain_means(classified_samples, ran_means), classified_labels
    
    def train(self, dataset):
        self.dataset = dataset
        dataset = dataset.values
        self.no_features = dataset.shape[1]-1

        if self.linear:
            means = np.array([[i/10,i/10] for i in range(10)])  
        else:
            means = np.random.randint(0, 1000, size=(self.n_means, self.no_features))/1000

        for _ in range(50):
            means, labels = self.step(dataset, means)
        self.means = np.array(means)
        self.labels = labels