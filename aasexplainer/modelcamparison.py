import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


class AASModelComparison:
  """Compares multiple AAS models using their aasexplainers.
  
  Attributes:
    aasexplainers: A list of aasexplainers
  """

  def __init__(self, aasexplainers):
    self.exps = aasexplainers



  def plot_similarity_matrix(self, similarity_matrix):
    n = similarity_matrix.shape[0]
    
    fig, ax = plt.subplots()
    sns.heatmap(similarity_matrix, annot=True, ax=ax, cmap='viridis', fmt='.2f')
    
    ax.set_xticks(np.arange(n) + 0.5)  # Center the x-axis labels
    ax.set_yticks(np.arange(n) + 0.5)  # Center the y-axis labels
    ax.set_xticklabels(np.arange(1, n+1), ha='center')  # Center the x-axis tick labels
    ax.set_yticklabels(np.arange(1, n+1), va='center')  # Center the y-axis tick labels
    
    plt.show()

  def matrix_distances_pred(self,arrays, norm = False):
    n = len(arrays)
    length = len(arrays[0])
    
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = length - np.sum(arrays[i] == arrays[j])
            if norm:
              similarity_matrix[i, j] = similarity_matrix[i, j] / length
              #rounf to 2 decimal places
              similarity_matrix[i, j] = round(similarity_matrix[i, j], 2)
    return similarity_matrix 

#####################################################################################

  def distances_topk(self, shap1, shap2, k=10):
    n = len(shap1)
    distances = []

    for alg in range(n):
      shap_1_alg = shap1[alg]
      shap_2_alg = shap2[alg]

      distance = np.zeros(len(shap_1_alg))
      for i in range(len(shap_1_alg)):
          shap_1_alg_inst = np.abs(shap_1_alg[i])
          shap_2_alg_inst = np.abs(shap_2_alg[i])
          # take abs
          shap_1_alg_inst = np.abs(shap_1_alg_inst)
          shap_2_alg_inst = np.abs(shap_2_alg_inst)

          # get the top k features
          first_topk = np.argsort(shap_1_alg_inst)[-k:]
          second_topk = np.argsort(shap_2_alg_inst)[-k:]

          # check how many appear in both arrays
          common_elements = np.intersect1d(first_topk, second_topk)
          num_common_elements = len(common_elements)
          distance[i] = k - num_common_elements
        
      distances.append(distance)

    return distances

  def distances_euclidian(self, shap1, shap2):
    n = len(shap1)
    distances = []

    for i in range(n):
      first = shap1[i]
      second = shap2[i]
      distances.append(np.linalg.norm(second - first, axis =1))
    
    return distances 

  def matrix_distances_topk(self, k=10):
    shaps = [exp.shap_values for exp in self.exps]
    n = len(shaps)
    
    # matrix distances top K between all features
    distances_matrix = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            distances = self.distances_topk(shaps[i], shaps[j], k=k)
            # flatten list [array(1,2), array(1,2)] -> [1,2,1,2]  
            distances = np.array([item for sublist in distances for item in sublist])

            distances_matrix[i,j] = np.mean(distances)
    
    return distances_matrix
  
  def matrix_distances_euclidian(self, k=10):
    shaps = [exp.shap_values for exp in self.exps]
    n = len(shaps)
    
    # matrix distances top K between all features
    distances_matrix = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            distances = self.distances_euclidian(shaps[i], shaps[j])
            # flatten list [array(1,2), array(1,2)] -> [1,2,1,2]  
            distances = np.array([item for sublist in distances for item in sublist])

            distances_matrix[i,j] = np.mean(distances)
    
    return distances_matrix

  def plot_distances_topk(self, k=10):
    distances_matrix = self.matrix_distances_topk(k=k)
    self.plot_similarity_matrix(distances_matrix)

  def values_distances_topk(self, k=10):
    distances_matrix = self.matrix_distances_topk(k=k)
    values = [distances_matrix[i,j] for i in range(distances_matrix.shape[0]-1) for j in range(distances_matrix.shape[1]-1)  if i<j]
    # box plot horizontal
    return values
    
  def plot_distances_euclidian(self, k=10):
    distances_matrix = self.matrix_distances_euclidian(k=k)
    self.plot_similarity_matrix(distances_matrix)

  def values_distances_euclidian(self, k=10):
    distances_matrix = self.matrix_distances_euclidian(k=k)
    values = [distances_matrix[i,j] for i in range(distances_matrix.shape[0]-1) for j in range(distances_matrix.shape[1]-1)  if i<j]
    # box plot horizontal
    return values
  
  def plot_distances_pred(self, pred_scenario, norm = False):
    predictions = []
    for exp in self.exps:
      predictions.append(exp.model.predict(pred_scenario.feature_data.values))
    similarity_matrix = self.matrix_distances_pred(predictions, norm=norm)
    self.plot_similarity_matrix(similarity_matrix)

  def values_distances_pred(self, pred_scenario, norm = False):
    predictions = []
    for exp in self.exps:
      predictions.append(exp.model.predict(pred_scenario.feature_data.values))
    similarity_matrix = self.matrix_distances_pred(predictions, norm = norm)
    values = [similarity_matrix[i,j] for i in range(similarity_matrix.shape[0]-1) for j in range(similarity_matrix.shape[1]-1)  if i<j]
    return values


#############################################################################
# Following methods are not used in paper, but can be used for further analysis

  def printImportanceScatter(self, n = 10, scale_shap = False, labels = None, show = True, ax = None):
    """
    Prints importance scores for all models.

    Attributes:
      n: Number of displayed features
      scale_shap: If True important scores are normalized (add up to 1)
    """
    idxs = self.getAverageMostImportant(n)
    scenario = self.exps[0].scenario

    for i in range(len(self.exps)):
      exp = self.exps[i]
      imps = exp.imp[idxs]
      n = len(imps)
      
      if scale_shap:
        sum = np.sum(exp.imp)
        imps = imps  / sum

      names =  [scenario.features[idx] for idx in idxs]
      if labels is not None:
        label = labels[i]
      else:
        label = exp.model.name+ ' ' + str(int(exp.model.get_stats()["par10"]))  

      if ax is None:
        plt.scatter(imps,names, label = label)
        plt.xlabel('feature importance')
        plt.ylabel("top " + str(n) + " features")
      else:
        ax.scatter(imps,names, label = label)
        ax.set_xlabel('feature importance')
        ax.get_yaxis().set_ticks([])
        ax.set_ylabel("top " + str(n) + " features")
        ax.legend()

    if show:
      plt.show()

  def printGraphs(self):
    """
    Prints distribution graphs for all models. 
    """
    for exp in self.exps:
      exp.printShapGraph()
    plt.show()

  def printShapDistances(self):
    """
    Prints Euclidian distance between all importance vectors.
    """
    vectors = [exp.imp for exp in self.exps]
    # Calculate the Euclidean distance between each pair of vectors
    dist_matrix = np.zeros((len(vectors), len(vectors)))
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            dist_matrix[i, j] = np.linalg.norm(vectors[i] - vectors[j])

    # Create a heatmap of the distance matrix
    return sns.heatmap(dist_matrix, annot=True, cmap='viridis', fmt='.2f')

  def printShapDistancesTopN(self,n=10, ax = None):
    """
    Prints common TopN features distance between all importance vectors.
    """
    vectors = [exp.imp for exp in self.exps]
    # Calculate the Euclidean distance between each pair of vectors
    dist_matrix = np.zeros((len(vectors), len(vectors)))
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            #dist_matrix[i, j] = np.linalg.norm(vectors[i] - vectors[j])
            g1 = set(self.exps[i].get_top_feature_names(n=n))
            g2 =  set(self.exps[j].get_top_feature_names(n=n))
            dist_matrix[i, j] = len(g1.intersection(g2))

    # Create a heatmap of the distance matrix
    return sns.heatmap(dist_matrix, annot=True, cmap='viridis', fmt='.2f', 
                vmin = 0, vmax=n,xticklabels=range(1,len(vectors)+1),
                yticklabels=range(1,len(vectors)+1), ax = ax)


  def printShapDistancesSpear(self,n=10, ax = None):
    """
    Prints Spearman correlation distance between all importance vectors.
    """
    vectors = [exp.imp for exp in self.exps]
    # Calculate the Euclidean distance between each pair of vectors
    dist_matrix = np.zeros((len(vectors), len(vectors)))
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            #dist_matrix[i, j] = np.linalg.norm(vectors[i] - vectors[j])
            g1 = set(self.exps[i].get_top_feature_names(n=n))
            g2 =  set(self.exps[j].get_top_feature_names(n=n))
            dist_matrix[i, j] = spearmanr(self.exps[i].imp,self.exps[j].imp).correlation

    # Create a heatmap of the distance matrix
    return sns.heatmap(dist_matrix, annot=True, cmap='viridis', fmt='.2f', 
                vmin = -1, vmax=1,xticklabels=range(1,len(vectors)+1),
                yticklabels=range(1,len(vectors)+1), ax = ax)
  
  def getAverageMostImportant(self, n = 10):
    """
    Most important features on average over all models.
    """
    sum_shap_values = np.zeros(self.exps[0].imp.shape)
    scenario = self.exps[0].scenario

    for exp in self.exps:
      sum_shap_values += exp.imp
    topn = n
    idxs = np.argsort(sum_shap_values)[-topn:] [::-1]
    #names =  [scenario.features[idx] for idx in idxs]
    return idxs

  def getNormalizedImps(self):
    """
    Normalized importance scores (add up to 1)
    """
    imps = []
    for exp in self.exps:
      sum = np.sum(exp.imp)
      imps.append( exp.imp  / sum)
    return imps


  def printCorrShapGap(self,label = False):
    """
    Prints correlation between euclidian distance and gap closed
    """
    gc_distances = []
    shap_distances = []
    names = []
    for a in range(len(self.exps)):
      expa = self.exps[a]
      for b in range(a+1,len(self.exps)):
        expb = self.exps[b]
        shap_distances.append(self.distance_shap_values(expa.imp,expb.imp))
        gc_distances.append(np.abs(expa.model.get_stats()["gap_closed"]- expb.model.get_stats()["gap_closed"]))
        names.append(str(a+1) + '/' + str(b+1))
    fig, ax = plt.subplots()
    ax.scatter(shap_distances, gc_distances)
    plt.ylabel('gap closed difference')
    plt.xlabel('distance between shap vectors')
    if label:
      for i, txt in enumerate(names):
          ax.annotate(txt, (shap_distances[i], gc_distances[i]))
    plt.show()
    print(spearmanr(shap_distances,gc_distances))

  def printCorrShapTop(self,label = False):
    """
    Prints correlation between topN common features  and gap closed
    """
    gc_distances = []
    shap_distances = []
    names = []
    for a in range(len(self.exps)):
      expa = self.exps[a]
      for b in range(a+1,len(self.exps)):
        expb = self.exps[b]
        shap_distances.append(self.distance_shap_values_topn(a,b))
        gc_distances.append(np.abs(expa.model.get_stats()["gap_closed"]- expb.model.get_stats()["gap_closed"]))
        #gc_distances.append(self.distance_shap_values_topn(a,b))
        names.append(str(a+1) + '/' + str(b+1))
    fig, ax = plt.subplots()
    ax.scatter(shap_distances, gc_distances)
    plt.ylabel('gap closed difference')
    plt.xlabel('distance between shap vectors')
    if label:
      for i, txt in enumerate(names):
          ax.annotate(txt, (shap_distances[i], gc_distances[i]))
    plt.show()
    print(spearmanr(shap_distances,gc_distances))

  def printCorrShapSpear(self,label = False):
    """
    Prints corrrelation between spearman correlation and gap closed.
    """
    gc_distances = []
    shap_distances = []
    names = []
    for a in range(len(self.exps)):
      expa = self.exps[a]
      for b in range(a+1,len(self.exps)):
        expb = self.exps[b]
        shap_distances.append(spearmanr(expa.imp,expb.imp).correlation)
        gc_distances.append(np.abs(expa.model.get_stats()["gap_closed"]- expb.model.get_stats()["gap_closed"]))
        names.append(str(a+1) + '/' + str(b+1))
    fig, ax = plt.subplots()
    ax.scatter(shap_distances, gc_distances)
    plt.ylabel('gap closed difference')
    plt.xlabel('distance between shap vectors')
    if label:
      for i, txt in enumerate(names):
          ax.annotate(txt, (shap_distances[i], gc_distances[i]))
    plt.show()
    print(spearmanr(shap_distances,gc_distances))

  def printCorrShapPred(self, pred_set, ref_set, shap_norm = False, distance="euk"):
    """
    Prints correlation between shap distances and similarity of predictions on the test set.
    """
    pred_set = pred_set.feature_data.values
    ref_set = ref_set.feature_data.values
    pred_distances = []
    shap_distances = []
    train_pred_distances = []
    for a in range(len(self.exps)):
      expa = self.exps[a]
      for b in range(a+1,len(self.exps)):
        expb = self.exps[b]
        s1 = expa.getImportanceScore(normalize = shap_norm)
        s2 = expb.getImportanceScore(normalize = shap_norm)
        
        if distance == "top":
          shap_distances.append(self.distance_shap_values_topn(a,b))
        else:
          shap_distances.append(self.distance_shap_values(s1,s2))
        pred_distances.append(np.count_nonzero(expa.model.predict(pred_set)-expb.model.predict(pred_set)))
        train_pred_distances.append(np.count_nonzero(expa.model.predict(ref_set)-expb.model.predict(ref_set)))
        #names.append(str(a+1) + '/' + str(b+1))

    plt.scatter(shap_distances, pred_distances)
    plt.xlabel("Shap distance")
    plt.ylabel("Different predictions on test set")
    plt.show()
    print("Correlation:")
    print(spearmanr(shap_distances,pred_distances)) 

  def distance_shap_values(self,shap1, shap2):
    """
    Euclidian distance.
    """
    return np.linalg.norm(shap1-shap2)
  
  def distance_shap_values_topn(self,i, j,n=10):
    """
    TopN common features
    """
    g1 = set(self.exps[i].get_top_feature_names(n=n))
    g2 =  set(self.exps[j].get_top_feature_names(n=n))
    return n-len(g1.intersection(g2))