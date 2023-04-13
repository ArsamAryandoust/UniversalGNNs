import argparse
from train import train_single, train_mutual, train_baselines
from loader import load_datasets, load_multidatasets, load_encoders, load_graphbuilders, load_regressors
import yaml

def parse_arguments() -> argparse.Namespace:
    """
    Parses the command line arguments passed to the program
    """
    parser = argparse.ArgumentParser(prog="UniversalGNNs",
                                     description="""Train models on 
                        ClimART, UberMovement and/or BuildingElectricity datasets.
                        
                        Can train baselines such as Random forests and MLPs or the main model which consists on
                        a universal GNN that uses an encoder to get a common dimentional representation of the 
                        data and a regressor to solve the problems.

                        First specify the dataset(s) to use, then the experiments to run on them.
                        """)
    # datasets
    parser.add_argument("-all_datasets", help="add all the available datasets", action="store_true")
    parser.add_argument("-climart", help="add the ClimART dataset", action="store_true")
    parser.add_argument("-uber", help="add the UberMovement dataset", action="store_true")
    parser.add_argument("-BE", help="add the BuildingElectricity dataset", action="store_true")

    # models
    parser.add_argument("--RF", help="Train a Random Forest on the specified datasets.", action="store_true")
    parser.add_argument("--GB", help="Train a Gradient Boosting model on the specified datasets.", action="store_true")
    parser.add_argument("--MLP", help="Train an MLP on the specified datasets.", action="store_true")
    parser.add_argument("--train_single", help="Train the datasets on the UniversalGNN model one at a time.", action="store_true")
    parser.add_argument("--train_mutual",
                        help="Train the datasets on the UniversalGNN models all together.",
                        action="store_true")
    
    # logger
    parser.add_argument("--nolog", help="Disable run logging (currently on wandb).", action="store_true")

    parser.add_argument("--test", help="This is a test run: disable logging and models are trained only for a limited amount of steps.", action="store_true")

    args = parser.parse_args()

    # do some checks for  validity of args
    if not (args.all_datasets or args.climart or args.uber or args.BE):
        print("Must select at least one dataset!")
        exit(1)
    if args.RF or args.GB or args.MLP:
        args.baselines = True
    else:
        args.baselines = False

    if not (args.baselines or args.train_single or args.train_mutual):
        choice = input("No model selected. Are you sure you just want to load the datasets (y/n)?")
        while choice.lower() != "y" and choice.lower() != "n":
            choice = input("Only load datasets? Type 'y' or 'n'.")
        if choice.lower() == "n":
            exit(1)

    if args.test:
        args.nolog = True
    args.log_run = not args.nolog

    return args

def set_test_config(config: dict) -> dict:
    # mlp
    config["baselines"]["mlp"]["epochs"] = 1
    config["baselines"]["mlp"]["batches_per_epoch"] = 10

    # encoders
    config["encoders"]["max_steps"] = 10

    # single
    config["train_single"]["epochs"] = 1
    config["train_single"]["batches_per_epoch"] = 10
    config["train_single"]["max_steps"] = 10

    # mutual
    config["train_mutual"]["epochs"] = 1
    config["train_mutual"]["batches_per_epoch"] = 10
    return config


if __name__ == "__main__":

    args = parse_arguments()
    with open("config.yaml", "r") as configfile:
        config = yaml.safe_load(configfile)

    # override some configs if the test flag is set
    if args.test:
        config = set_test_config(config)

    datasets = load_datasets(vars(args), config["force_node_level"])

    # if we want to run the baselines then do it
    if args.baselines:
        train_baselines(config["baselines"], datasets, args.RF, args.GB, args.MLP, args.log_run)

    # if we want to use the GNN we need to load the encoders, graphbuilders and regressors
    if args.train_single:
        graphbuilders_dict = load_graphbuilders(config["graphbuilders"], datasets)
        autoencoders_dict = load_encoders(config["encoders"], datasets, graphbuilders_dict, args.log_run)
        regressors_dict = load_regressors(config["regressors"], datasets)
        train_single(config, datasets, autoencoders_dict, graphbuilders_dict, regressors_dict, args.log_run)
    
    if args.train_mutual:
        graphbuilders_dict = load_graphbuilders(config["graphbuilders"], datasets)
        autoencoders_dict = load_encoders(config["encoders"], datasets, graphbuilders_dict, args.log_run)
        regressors_dict = load_regressors(config["regressors"], datasets)
        # must first load the multidataset data loaders
        data_loaders = load_multidatasets(config["train_mutual"], datasets)
        train_mutual(config, data_loaders, autoencoders_dict, graphbuilders_dict, regressors_dict, args.log_run)


# climart 946
# BE 1110