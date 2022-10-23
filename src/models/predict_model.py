# -*- coding: utf-8 -*-
from cgi import test
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import pickle
from src.data.preprocess import *
import src.config as cfg
import os


@click.command()
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('input_catboost_model', type=click.Path())
@click.argument('input_sklearn_model', type=click.Path())
@click.argument('output_predictions_filepath', type=click.Path())

def main(input_data_filepath, input_catboost_model, input_sklearn_model, output_predictions_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # load data
    df = pd.read_pickle(input_data_filepath)

    sklearn_model = pickle.load(open(input_sklearn_model, 'rb'))
    catboost_model = pickle.load(open(input_catboost_model, 'rb'))

    catboost_prediction = catboost_model.predict(df)
    sklearn_prediction = sklearn_model.predict(df)

    df_pred_sc = pd.DataFrame(sklearn_prediction, columns = cfg.TARGET_COLS, index=df.index)
    df_pred_cb = pd.DataFrame(catboost_prediction, columns = cfg.TARGET_COLS, index=df.index)
    
    
    if not os.path.isdir("reports/inference"):
        os.makedirs("reports/inference")
        
    df_pred_sc.to_csv(output_predictions_filepath + 'sklearn_pred.csv')
    df_pred_cb.to_csv(output_predictions_filepath + 'catboost_pred.csv')




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()