import argparse
from train import train_single, train_universal, train_baselines
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
    parser.add_argument("-climart", help="add the ClimART dataset", action="store_true")
    parser.add_argument("-uber", help="add the UberMovement dataset", action="store_true")
    parser.add_argument("-BE", help="add the BuildingElectricity dataset", action="store_true")

    # models
    parser.add_argument("--RF", help="Train a Random Forest on the specified datasets.", action="store_true")
    parser.add_argument("--GB", help="Train a Gradient Boosting model on the specified datasets.", action="store_true")
    parser.add_argument("--MLP", help="Train an MLP on the specified datasets.", action="store_true")
    parser.add_argument("--train_single", help="Train the datasets on the UniversalGNN model one at a time.", action="store_true")
    parser.add_argument("--train_universal",
                        help="Train the datasets on the UniversalGNN models all together.",
                        action="store_true")

    args = parser.parse_args()

    # do some checks for  validity of args
    if not (args.climart or args.uber or args.BE):
        print("Must select at least one dataset!")
        exit(1)
    if args.RF or args.GB or args.MLP:
        args.baselines = True
    else:
        args.baselines = False

    if not (args.baselines or args.train_single or args.train_universal):
        print("Must select one model to train!")
        exit(1)

    return args


if __name__ == "__main__":

    args = parse_arguments()
    with open("config.yaml", "r") as configfile:
        config = yaml.safe_load(configfile)

    datasets = load_datasets(args)

    # if we want to run the baselines then do it
    if args.baselines:
        train_baselines(config["baselines"], datasets, args.RF, args.GB, args.MLP)

    # if we want to use the GNN we need to load the encoders, graphbuilders and regressors
    if args.train_single or args.train_universal:
        graphbuilders_dict = load_graphbuilders(config["graphbuilders"], datasets)
        autoencoders_dict = load_encoders(config["encoders"], datasets, graphbuilders_dict)
        regressors_dict = load_regressors(config["regressors"], datasets)
        if args.train_single:
            train_single(config["train_single"], datasets, autoencoders_dict, graphbuilders_dict, regressors_dict)
        if args.train_universal:
            # must first load the multidataset data loaders
            data_loaders = load_multidatasets(config["train_universal"], datasets)
            train_universal(config["train_universal"], data_loaders, autoencoders_dict, graphbuilders_dict, regressors_dict)
