import tensorflow as tf
import pandas as pd
import numpy as np

COLUMN_NAMES = [ 'Cancer_Type', 'Tumor_Sample_ID', 'Gene_Name',
                 'Chromosome', 'Start_Position', 'End_Position',
                 'Variant_Type', 'Reference_Allele','Tumor_Allele' ]

LABELS = [ 'BRCA', 'COADREAD', 'GBM', 'LUAD', 'OV', 'UCEC' ]

def load_data(data_path):
    print('load data from: ' + data_path)

    dataset = pd.read_csv(data_path) # total dataset
    samples = dataset.groupby('Tumor_Sample_ID').size().reset_index(name='Count')

    dataset_size = len(dataset)
    samples_size = len(samples)

    return samples, dataset, samples_size, dataset_size

def main(argv):
    file_train = argv[1]
    file_test = argv[2]

    # train
    # Load data
    train_samples, train_dataset, samples_size, dataset_size = load_data(file_train)
    
    # shuffle train sample
    shuffle_train_samples = train_samples.sample(frac=1)
    #print shuffle_train_samples

    for index, sample in shuffle_train_samples.iterrows():
        # read a sample
        sample_id, batch_seqlen = sample['Tumor_Sample_ID'], sample['Count']
        train_batch = train_dataset[train_dataset['Tumor_Sample_ID'] == sample_id]
        cancer_type = train_batch.iloc[0]['Cancer_Type']
        #print sample_id
        #print batch_seqlen
        #print train_batch
        #print cancer_type

        # shuffle batch
        batch = train_batch.sample(frac=1)

        # make feature column
        # convert 'Tumor_Sample_ID' to a embedded tensor

        # train 6 models for each cancer type
        model_brca = None
        model_coadread = None
        model_gbm = None
        model_luad = None
        model_ov = None
        model_ucec = None
        models = { "BRCA":model_brca, 
                   "COADREAD":model_coadread, 
                   "GBM":model_gbm, 
                   "LUAD":model_luad, 
                   "OV":model_ov, 
                   "UCEC":model_ucec }
        for cancer in models.keys():
            # make label in binary
            if cancer_type == cancer:
                print cancer
            else:
                print models[cancer]
            # train each model

        # train ensemble model
        #ensemble_nn(models)
        exit()

    exit()
    # test

    print result   

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
