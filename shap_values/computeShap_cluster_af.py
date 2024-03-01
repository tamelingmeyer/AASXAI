

import argparse
from aasexplainer.aasexplainer import AASExplainer
from aasexplainer.afmodels import AFModelPair
from aasexplainer.afmodels import AFModelPairRegression
from aasexplainer.afmodels import AFModel
from aasexplainer.afmodels import load_model_auto
from aasexplainer.afmodels import load_scenario

from path_handler import *
import copy
import pandas as pd

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--fold", type=int, help="Number of folds")
  #parser.add_argument("--path_model", type=str, help="Path to file")
  parser.add_argument("--scenario", type=str, help="Scenario string")
  args = parser.parse_args()

  fold = args.fold
  #path_model = args.path_model
  scenario_str = args.scenario
  exp_name = "Final200"
  features_per_split = 20

  path_model = get_model_path(scenario_str, fold)
  path_scenario = get_scenario_path(scenario_str)
  path_shap = get_shap_path(scenario_str, fold, exp_name)

  scenario_lib = load_scenario(path_scenario)
  test_scenario,train_scenario = scenario_lib.get_split(fold)
  #shap_scenario,_  = scenario_lib.get_split(1)

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

  model = load_model_auto(path_model, path_scenario, scenario_str)

  exp = AASExplainer(shap_scenario, model, background_scenario = scenario_lib, path_shap = path_shap)

  
if __name__ == "__main__":
  main()
