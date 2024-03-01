from aasexplainer.afmodels import load_model_auto
from aasexplainer.afmodels import load_model
from aasexplainer.afmodels import load_scenario



Model_PATH = './models/'

Scenario_PATH = './aslib_data/'

Shap_PATH = './shap_values/'


def set_paths(model_path, scenario_path, shap_path):
    global Model_PATH
    global Scenario_PATH
    global Shap_PATH
    Model_PATH = model_path
    Scenario_PATH = scenario_path
    Shap_PATH = shap_path


def get_model_path(scenario, fold):
    return Model_PATH + scenario + "/" + str(fold) + ".pkl"

def get_scenario_path(scenario):
    return Scenario_PATH + scenario + "/"

def get_shap_path(scenario, fold, experiment_name):
    return Shap_PATH + experiment_name + "/" +  scenario + "/" + str(fold) + "_shap.npy"

def full_load(scenario_str, fold):
    path_model = get_model_path(scenario_str, fold)
    path_scenario = get_scenario_path(scenario_str)

    scenario_lib = load_scenario(path_scenario)
    test_scenario,train_scenario = scenario_lib.get_split(fold)

    model = load_model_auto(path_model, path_scenario, scenario_str)

    return model, test_scenario, train_scenario

def load_model_only(scenario_str, fold, scenario_lib):
    path_model = get_model_path(scenario_str, fold)
    model = load_model(path_model,scenario_lib, scenario_str)
    return model