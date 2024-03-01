#AASExplainer

import numpy as np  
import matplotlib.pyplot as plt
import os
import shap


class AASExplainer:
  """Explains an automated algorithm selection model.
  
  Attributes:
    scenario: An ASlibScenario containing the instances that should be explained
    model: An AAS model that has the methods predict and predict_proba
    shap_values: The Shap values explaining the instances from scenario 
  """

  def __init__(self, scenario, model, shap_values = None, background_scenario = None,background=None, path_shap = None):
    """Initializes Explainer for given scenario and model.
    
    Attributes:
      scenario: An ASlibScenario containing the samples that should be explained
      model: An AAS model that has the methods predict and predict_proba 
      shap_values: A numpy array containing the shap_values
      background_scenario: An ASlibScenario providing instances for creation of background dataset
      background: A numpy array with instances for feature removal
      path_shap: A path to store/load shap_values

    """
    self.scenario = scenario
    self.model = model

    # Get general scenario data
    self.X = self.scenario.feature_data.values
    self.performance = self.scenario.performance_data.values
    self.y = np.argmin(self.performance, axis = 1)
    self.y_pred = self.model.predict(self.X)

    # Load background dataset
    self.background = background
    if self.background is None:
      self.setBackgroundDataset(background_scenario)

    # Load Shap_values
    self.shap_values = shap_values
    if self.shap_values == None:
      self.shap_values = self.get_shap_values(path_shap = path_shap)

    explainer = shap.KernelExplainer(self.model.predict_proba, 
          self.background, output_names = self.scenario.algorithms)
    self.base_values = explainer.expected_value

    # Get Importance score
    self.importance_score = self.getImportanceScore()
    self.imp = self.importance_score

  def setBackgroundDataset(self, background_scenario,sample = True,use_kmeans = True,n_background = 20):
    """
    Sets background dataset used for feature removal based on a scenario.
    
    Arguments:
      background_scenario: An ASlibScenario providing instances for creation of background dataset
      sample: If True a subset of background_scenario will be used (speed up)
      use_kmeans: If True kmeans is used to sumarize data
      n_background: Size of background dataset
    """
    if background_scenario is None:
      self.background = self.X
    elif not sample or background_scenario.feature_data.values.shape[0] <= n_background:
      self.background = background_scenario.feature_data.values
    else:
      if use_kmeans:
        self.background = shap.kmeans(background_scenario.feature_data.values,n_background)
      else:
        self.background = shap.sample(background_scenario.feature_data.values,n_background)

  def get_shap_values(self, path_shap = None):
    """
    Computes or loads the Shapley values
    """
    if path_shap != None:
      if os.path.exists(path_shap):
        shap_values = np.load(path_shap)
        shap_values = [row for row in shap_values]
        return shap_values
      else: 
        vec = self.scenario.feature_data.values
        explainer = shap.KernelExplainer(self.model.predict_proba, 
          self.background, output_names = self.scenario.algorithms)
        self.shap_values = explainer.shap_values(vec)

        directory = os.path.dirname(path_shap)
        if not os.path.exists(directory):
          os.makedirs(directory)
        np.save(path_shap, self.shap_values)
        
        return self.shap_values
    else: 
      print("Please add a path to store the Shap_values")



  def printReport(self):
    """
    Prints multiple explanations.
    """
    self.printBarplot()
    self.printShapGraph()
    self.printDependencePlot(self.get_top_algorithm_names()[:2], self.get_top_feature_names()[:2])
    self.printSelectedvsBest(self,idx = self.getBiggestErrorIdx())


  def printBarplot(self, n = 15, plot_size= (15,10),show=True):
    """
    Prints feature importance scores.
    
    Arguments:
      n: Number of shown features
    """
    shap.summary_plot(self.shap_values, feature_names = self.scenario.features, class_names=self.scenario.algorithms, plot_size = plot_size, max_display=n, show=show)

  def printSummaryplot(self, n = 15, plot_size= (15,10),show=True):
    """
    Prints a summary plot including all shap values for all algorithms.
    
    Arguments:
      n: Number of shown features
    """
    shap_alg = np.concatenate(self.shap_values)
    shap.summary_plot(shap_alg, feature_names = self.scenario.features, max_display=n, show=show, plot_size = plot_size)

  def printSummaryOneAlg(self, algorithm, n = 15, plot_size= (15,10),show=True):
    """
    Prints a summary plot including all shap values for one given algorithm.
    
    Arguments:
      algorithm: Index of algorithm of interest
      n: Number of shown features
    """
    shap_alg = self.shap_values[algorithm]
    shap.summary_plot(shap_alg,features = self.X, feature_names = self.scenario.features, max_display=n, show=show, plot_size = plot_size)

  def printSummaryPlotSelected(self, n = 15, plot_size= (15,10), show = True):
    """
    Prints a summary plot including all shap values for the selected algorithms.
    
    Arguments:
      n: Number of shown features
    """
    selected_shap = self.getSelectedShap()
    shap.summary_plot(selected_shap, feature_names = self.scenario.features, 
      class_names=self.scenario.algorithms, plot_size = plot_size, max_display=n,
      show=True)

  def printShapGraph(self,scale_shap = False,scale_feature = False,cutoff = False,cutoff_value = 50):
    """
    Prints a graph showing the distribution of the shap values over the features.

    Arguments:
      shap_scale: If True the values add up to 1
      scale_feature: If True x-axis goes from 0 to 1
      cutoff: If True features are only displayes until cutoff_value
      cutoff_value: Number of features displayed
    """
    shap_values_mean = np.abs(self.shap_values)
    shap_values_mean = np.mean(shap_values_mean, axis = 0)
    shap_values_mean = np.mean(shap_values_mean, axis = 0)

    shap_values_sorted = sorted(shap_values_mean, reverse=True)
    if cutoff and len(shap_values_sorted) > cutoff_value:
      shap_values_sorted = shap_values_sorted[:cutoff_value]
    n = len(shap_values_sorted)
    if scale_feature:
      steps = np.linspace(0, 1, n)
      plt.xlabel("x most important feature / # features")
    else: 
      steps = range(n)
      plt.xlabel('x most important feature')
    plt.ylabel('feature importance')  

    if scale_shap:
      sum = np.sum(shap_values_sorted)
      shap_values_sorted = shap_values_sorted  / sum
      plt.ylabel('relative feature importance') 

    plt.plot(steps,shap_values_sorted, label = self.scenario.scenario)
    plt.legend()


  def printDependencePlot(self, algs, features,show =True):
    """
    Prints all dependence plots for given algorithms and features.

    Arguments: 
      algs: List of algorithm names
      features: List of feature names
    """
    for feature in features:
      for alg in algs:
        alg_idx = self.scenario.algorithms.index(alg)
        feature_idx = self.scenario.features.index(feature)

        print(alg)
        print(feature)

        shap.dependence_plot(feature_idx, self.shap_values[alg_idx], self.X, feature_names=self.scenario.features, interaction_index = None,show=show)


  def printSelectedvsBest(self,idx,scale=1,show = True):
    """
    Prints decision plot for best and actual prediction

    Arguments:
      idx: Index of instance
      scale: Displayed number are scaled by this factor (use for small numbers)
    """
    best_alg_idx = self.y[idx]
    pred_alg_idx = self.y_pred[idx]
    
    shap_values_y = np.array(self.shap_values)[best_alg_idx,idx]
    shap_values_y_pred = np.array(self.shap_values)[pred_alg_idx,idx]
    shap_values_multi = [[shap_values_y], [shap_values_y_pred]]

    print("Instance with worst selection: {}".format(idx))
    print("best selection: {}, performance: {}".format(self.scenario.algorithms[best_alg_idx], self.performance[idx, self.y[idx]]))
    print("model selection: {}, performance: {}".format(self.scenario.algorithms[pred_alg_idx], self.performance[idx, self.y_pred[idx]]))

    base_values = self.base_values
    shap.multioutput_decision_plot([base_values[best_alg_idx],base_values[pred_alg_idx]], 
                                   shap_values_multi, row_index=0, feature_names = self.scenario.features,show=show,
                                   legend_labels=["best: "+self.scenario.algorithms[best_alg_idx],"selected: " + self.scenario.algorithms[pred_alg_idx]])
    if show:
      print("best selection")
      self.printWaterfall(idx,best_alg_idx,scale=scale)
      explanation = shap.Explanation(shap_values_y, base_values = base_values[best_alg_idx])
      self.printForcePlot(idx, best_alg_idx)
      print("model selection")
      self.printWaterfall(idx,pred_alg_idx,scale=scale)
      self.printForcePlot(idx,pred_alg_idx)
  
  def printWaterfall(self,idx,alg,scale =1, show = True):
    """
    Prints waterfall plot for given index and algorithm

    Arguments:
      idx: Index of instance
      alg: Index of algorithm
      scale: Displayed number are scaled by this factor (use for small numbers)
    """
    shap_values = np.array(self.shap_values)[alg,idx]
    explanation = shap.Explanation(shap_values*scale, base_values = self.base_values[alg]*scale, feature_names=self.scenario.features)
    shap.waterfall_plot(explanation,show = show)

  def printForcePlot(self,idx,alg, show = True,plot_size = (10,3)):
    """
    Prints force plot for given index and algorithm

    Arguments:
      idx: Index of instance
      alg: Index of algorithm
    """
    shap_values_y = np.array(self.shap_values)[alg,idx]
    shap.force_plot(self.base_values[alg], shap_values_y, #self.X[idx:idx+1,:],
    feature_names = self.scenario.features,matplotlib=True, show = show, figsize = plot_size) 


  def getSelectedShap(self):
    selected_shap = np.zeros(self.shap_values[0].shape)
    for i in range(selected_shap.shape[0]):
      selected_shap[i] = self.shap_values[self.y_pred[i]][i]
    return selected_shap

  def getImportanceScore(self, normalize = False):
    shap_abs = np.abs(self.shap_values) #absolute values
    shap_mean = np.mean(shap_abs, axis = 0) #average over algorithms
    importance_score = np.mean(shap_mean, axis = 0) #average over instances

    if normalize:
      sum = np.sum(importance_score)
      importance_score =  importance_score  / sum
    return importance_score

  def getBiggestErrorIdx(self, n=1):
    y_performance = (self.performance[np.arange(len(self.y)),self.y])
    y_pred_performance = (self.performance[np.arange(len(self.y_pred)),self.y_pred])
    
    performance_diff = y_pred_performance - y_performance
    #max_diff_idx = np.argmax(performance_diff)
    max_diff_idx = np.argsort(performance_diff)[-n]
    return max_diff_idx


  def get_top_feature_names(self, n = 10):
    shap_values_abs = np.abs(self.shap_values)
    shap_values_mean = np.mean([np.abs(array) for array in shap_values_abs], axis=0)
    shap_values_mean = np.mean(shap_values_mean, axis = 0)
    #idx = np.argmax(shap_values_mean)
    #idxs =  np.argpartition(shap_values_mean, -topn)[-topn:]
    idxs = np.argsort(shap_values_mean,)[-n:] [::-1]
    features = [self.scenario.features[idx] for idx in idxs]
    return features

  def get_top_algorithm_names(self):
    predictions = self.y_pred
    counts = np.bincount(predictions)
    idxs = np.argsort(counts)[::-1]
    algorithms = [self.scenario.algorithms[idx] for idx in idxs]
    return algorithms

  def get_top_algorithm(self):
    predictions = self.y_pred
    counts = np.bincount(predictions)
    idxs = np.argsort(counts)[::-1]
    return idxs

  
