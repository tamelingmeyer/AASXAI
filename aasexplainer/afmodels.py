# This class uses code taken from autofolio
import logging
import pickle
import numpy as np
import pandas as pd
from aslib_scenario.aslib_scenario import ASlibScenario

# validation
from autofolio.validation.validate import Validator, Stats
from autofolio.autofolio import AutoFolio
import copy

AAS_PATH = './'

class AFModel:
  """
  Gerneral class to load and use autofolio models for aasexplainer.
  """
  path_model_folder_default = AAS_PATH + 'Models/'
  path_scenario_folder_default = AAS_PATH + 'aslib_data/'
  
  def __init__(self,scenario_str, fold=0, scenario_lib=None, name = None, path_model=None, path_scenario=None,path_model_folder= 
    path_model_folder_default,path_scenario_folder=path_scenario_folder_default):
    self.name = name
    if name == None:
      self.name = scenario_str
    self.scenario_str = scenario_str
    self.fold = fold
    self.path_model = path_model
    self.path_scenario = path_scenario

    #path managment
    if self.path_model == None:
      self.path_model = path_model_folder + scenario_str + "/" + scenario_str + "fold" + str(fold) + ".pkl"
    self.path_shap = self.path_model + "shap.npy"

    if self.path_scenario == None:
       self.path_scenario = path_scenario_folder + scenario_str + "/" 
    
    if scenario_lib == None:
      self.scenario = load_scenario(path_scenario)
    else:
      self.scenario = scenario_lib

    #load model from file extracted from autofolio
    self.af = AutoFolio()
    with open(self.path_model, "br") as fp:
        scenario, feature_pre_pipeline, pre_solver, selector, config = pickle.load(
            fp)
    for fpp in feature_pre_pipeline:
        fpp.logger = logging.getLogger("Feature Preprocessing")
    if pre_solver:
        pre_solver.logger = logging.getLogger("Aspeed PreSolving")
    selector.logger = logging.getLogger("Selector")

    #save model components
    self.afscenario = scenario
    self.feature_pre_pipeline = feature_pre_pipeline
    self.pre_solver = pre_solver
    self.selector = selector
    self.config = config
    self.alg_to_idx = {element: index for index, element in enumerate(scenario.algorithms)}
    self.af = AutoFolio()

  def transform(self,X):
    """
    Applies feature preprocessing on X.
    """
    pred_scenario = ASlibScenario()
    n = X.shape[0]
    pred_scenario.feature_data = pd.DataFrame(X, index = range(n), columns=self.scenario.feature_data.columns)
    pred_scenario.instances = range(n)
    for t in self.feature_pre_pipeline:
      pred_scenario = t.transform(pred_scenario)
    features = pred_scenario.feature_data.values
    features = self.selector.normalizer.transform(features)
    return features

  def get_stats(self, test_scenario = None,train_scenario = None):
    #get test and train scenario
    if test_scenario==None or train_scenario==None:
      test_scenario,train_scenario = self.scenario.get_split(self.fold)
    else:
      test_scenario = copy.copy(test_scenario)
      train_scenario = copy.copy(train_scenario)

    #predict schedules additionally sets used feature group in test_scenario
    schedules = self.af.predict(
      test_scenario, self.config, self.feature_pre_pipeline, self.pre_solver, self.selector)

    val = Validator()
    if self.scenario.performance_type[0] == "runtime":
      stats = val.validate_runtime(
          schedules=schedules, test_scenario=test_scenario, train_scenario=train_scenario)

    return self.get_scores_from_stats(stats)

  def get_scores_from_stats(self,stats):
    scores = {}
    timeouts = stats.timeouts - stats.unsolvable
    par1 = stats.par1 - (stats.unsolvable * stats.runtime_cutoff)
    par10 = stats.par10 - (stats.unsolvable * stats.runtime_cutoff * 10)
    oracle = stats.oracle - (stats.unsolvable * stats.runtime_cutoff * 10)
    sbs = stats.sbs - (stats.unsolvable * stats.runtime_cutoff * 10)
            
    n_samples = timeouts + stats.solved
    scores["par1"]= (par1 / n_samples)
    scores["par10"] = (par10 / n_samples)
    scores["oracle"] = (oracle / n_samples)
    scores["sbs"] = (sbs / n_samples)
    scores["gap_closed"]= 1 - ( par10 - oracle) / (sbs - oracle) 
    return scores
  
 

class AFModelMulti(AFModel):
  """
  Loads and uses multiclass classification autofolio model
  """
  path_model_folder_default = AAS_PATH +'Models/'
  path_scenario_folder_default = AAS_PATH +'aslib_data/'
  
  def __init__(self,scenario_str, fold=0, name = None, path_model = None, path_scenario=None, path_model_folder= 
    path_model_folder_default,path_scenario_folder=path_scenario_folder_default):

    super().__init__(scenario_str, fold=fold, name = name, path_model = path_model, path_scenario=path_scenario, path_model_folder= 
    path_model_folder,path_scenario_folder=path_scenario_folder)

    self.model = self.selector.classifier.model

    #fix missing output
    algs = range(len(self.scenario.algorithms))
    y_set = set(np.argmin(self.scenario.performance_data.values, axis = 1))
    self.missing_algs = [idx for idx in algs if idx not in y_set]

  def predict_proba(self,X):
    Z = self.transform(X)
    pred = self.model.predict_proba(Z)
    return np.insert(pred, self.missing_algs,0, axis = 1)  

  def predict(self,X):
    Z = self.transform(X)
    return self.model.predict(Z) 


class AFModelPair(AFModel):
  """
  Loads and uses pairwise classification autofolio model
  """
  path_model_folder_default = AAS_PATH +'Models/'
  path_scenario_folder_default = AAS_PATH +'aslib_data/'
  
  def __init__(self,scenario_str, fold=0, scenario_lib=None, name = None, path_model = None, path_scenario=None, path_model_folder= 
    path_model_folder_default,path_scenario_folder=path_scenario_folder_default):

    super().__init__(scenario_str, fold=fold,scenario_lib=scenario_lib,name = name, path_model = path_model, path_scenario=path_scenario, path_model_folder= 
    path_model_folder,path_scenario_folder=path_scenario_folder)

    #self.model = self.selector.classifier.model

    self.normalizer = self.selector.normalizer

    self.classifiers = self.selector.classifiers


  def predict(self,X):
    Z = self.transform(X)
    n_algos = len(self.scenario.algorithms)
    scores = np.zeros((X.shape[0], n_algos))
    clf_indx = 0
    for i in range(n_algos):
        for j in range(i + 1, n_algos):
            clf = self.classifiers[clf_indx]
            Y = clf.predict(Z)
            scores[Y == 1, i] += 1
            scores[Y == 0, j] += 1
            clf_indx += 1

    algo_indx = np.argmax(scores, axis=1)
    return algo_indx  

  def predict_proba(self,X):
    return self.predict_proba_vote(X)

  def predict_proba_vote(self,X):
    Z = self.transform(X)
    n_algos = len(self.scenario.algorithms)
    scores = np.zeros((X.shape[0], n_algos))
    clf_indx = 0
    for i in range(n_algos):
        for j in range(i + 1, n_algos):
            clf = self.classifiers[clf_indx]
            Y = clf.predict(Z)
            scores[Y == 1, i] += 1
            scores[Y == 0, j] += 1
            clf_indx += 1

    prob = 2/(n_algos*(n_algos-1))*scores
    #algo_indx = np.argmax(scores, axis=1)
    return prob


class AFModelPairRegression(AFModel):
  """
  Loads and uses pairwise regression autofolio model
  """
  path_model_folder_default = AAS_PATH +'Models/'
  path_scenario_folder_default = AAS_PATH +'aslib_data/'
  
  def __init__(self,scenario_str, fold=0,scenario_lib=None, name = None, path_model = None, path_scenario=None, path_model_folder= 
    path_model_folder_default,path_scenario_folder=path_scenario_folder_default):

    super().__init__(scenario_str, fold=fold,scenario_lib=scenario_lib, name = name, path_model = path_model, path_scenario=path_scenario, path_model_folder= 
    path_model_folder,path_scenario_folder=path_scenario_folder)

    self.regressors = self.selector.regressors


  def predict(self,X):
    Z = self.transform(X)
    n_algos = len(self.scenario.algorithms)
    scores = np.zeros((X.shape[0], n_algos))
    reg_indx = 0
    for i in range(n_algos):
        for j in range(i + 1, n_algos):
            reg = self.regressors[reg_indx]
            Y = reg.predict(Z)
            scores[:, i] += Y
            scores[:, j] += -1 * Y
            reg_indx += 1

    algo_indx = np.argmin(scores, axis=1)
    return algo_indx  

  def predict_proba(self,X):
    return self.predict_proba_vote(X)

  def predict_proba_vote(self,X):
    Z = self.transform(X)
    n_algos = len(self.scenario.algorithms)
    scores = np.zeros((X.shape[0], n_algos))
    reg_indx = 0
    for i in range(n_algos):
        for j in range(i + 1, n_algos):
            reg = self.regressors[reg_indx]
            Y = reg.predict(Z)
            scores[Y < 0, i] += 1
            scores[Y >=0, j] += 1
            reg_indx += 1

    prob = 2/(n_algos*(n_algos-1))*scores
    return prob
  
  
  def transform(self,X):
    """
    Applies feature preprocessing on X.
    """
    pred_scenario = ASlibScenario()
    n = X.shape[0]
    #pred_scenario.feature_data = pd.DataFrame(X, index = range(n), columns=self.scenario.features)
    pred_scenario.feature_data = pd.DataFrame(X, index = range(n), columns=self.scenario.feature_data.columns)
    pred_scenario.instances = range(n)
    for t in self.feature_pre_pipeline:
      pred_scenario = t.transform(pred_scenario)
    features = pred_scenario.feature_data.values
    return features
  

def load_scenario(scenario_path):
  path = scenario_path
  scenario = ASlibScenario()
  scenario.read_scenario(path)
  return scenario

def get_shap_scenario(scenario_lib, features_per_split):
  new_scenario = copy.copy(scenario_lib)
  feature_data_list = []

  # Iterate over each fold
  for fold in range(1, 11):
      test_scenario,train_scenario = scenario_lib.get_split(fold)
      instances = test_scenario.feature_data.head(features_per_split)
      feature_data_list.append(instances)

  # Set the feature data of the new scenario to the list of instances
  new_scenario.feature_data = pd.concat(feature_data_list)
  shap_scenario = new_scenario
  return shap_scenario

def load_model_auto(path_model, path_scenario, scenario_str):
  model = AFModel(scenario_str, 1, path_model=path_model, path_scenario=path_scenario)

  if type(model.selector).__name__ == 'PairwiseClassifier':
      model = AFModelPair(scenario_str, 1, path_model=path_model, path_scenario=path_scenario)
  elif type(model.selector).__name__ == 'PairwiseRegression':
      model = AFModelPairRegression(scenario_str, 1, path_model=path_model, path_scenario=path_scenario)

  return model

def load_model(path_model,scenario_lib, scenario_str):
  model = AFModel(scenario_str, path_model=path_model, scenario_lib=scenario_lib)

  if type(model.selector).__name__ == 'PairwiseClassifier':
      model = AFModelPair(scenario_str, 1, path_model=path_model,  scenario_lib=scenario_lib)
  elif type(model.selector).__name__ == 'PairwiseRegression':
      model = AFModelPairRegression(scenario_str, 1, path_model=path_model, scenario_lib=scenario_lib)

  return model
