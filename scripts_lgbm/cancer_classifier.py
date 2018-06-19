import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import datetime
#from multiprocessing import Pool

#HEADER: Cancer_Type,Tumor_Sample_ID,Gene_Name,Chromosome,Start_Position,End_Position,Variant_Type,Reference_Allele,Tumor_Allele

def get_idx(ndarr, target):
    return np.where(ndarr == target)[0][0]

def create_dataset(csv_file, Cancer_Type, Gene_Name):
    # create dataset
    #print "[timestamp] begin create_dataset: " + str(datetime.now())
    csv = pd.read_csv(csv_file)
    samples = csv.Tumor_Sample_ID.unique()
    num_samples = len(samples)
    col_names = ['Tumor_Sample_ID', 'Cancer_Type']
    for i in range(0, len(Gene_Name)):
        col_names.append(str(i))
    dataset = pd.DataFrame(index=range(0, num_samples), columns=col_names)
    for i in range(0, num_samples):
         dataset.at[i, 'Tumor_Sample_ID'] = samples[i]
    # fill label of dataset
    #print "[timestamp] begin fill_label: " + str(datetime.now())
    for i in range(0, len(dataset)):
        idx = csv.index[csv['Tumor_Sample_ID'] == dataset.at[i, 'Tumor_Sample_ID']][0]
        cancer_type = csv.at[idx, 'Cancer_Type']
        cancer_type_idx = get_idx(Cancer_Type, cancer_type)
        dataset.at[i, 'Cancer_Type'] = cancer_type_idx + 0.5 #tmp
    # fill features of dataset
    #print "[timestamp] begin fill_features: " + str(datetime.now())
    for i in range(0, len(csv)):
        sample = csv.at[i, 'Tumor_Sample_ID']
        sample_idx = dataset[dataset['Tumor_Sample_ID'] == sample].index[0]
        gene = csv.at[i, 'Gene_Name']
        gene_idx = get_idx(Gene_Name, gene)
        dataset.at[sample_idx, str(gene_idx)] = 1
    dataset.fillna(0, inplace=True)
    #print "[timestamp] end create_dataset: " + str(datetime.now())
    


    return dataset

def main():
    # read the entire data to create gene index
    #print "[timestamp] begin read all_csv: " + str(datetime.now())
    all_csv = pd.read_csv('TCGA_6_Cancer_Type_Mutation_List.csv')
    Cancer_Type = all_csv.Cancer_Type.unique()
    Gene_Name = all_csv.Gene_Name.unique()
    #print "[timestamp] end read all_csv: " + str(datetime.now())

    #tmp
    #train_dataset = create_dataset('TCGA_6_Cancer_Type_Mutation_List_train.csv', Cancer_Type, Gene_Name)
    #test_dataset = create_dataset('TCGA_6_Cancer_Type_Mutation_List_test.csv', Cancer_Type, Gene_Name)
    #train_dataset.to_csv('train_dataset.csv', sep=',')
    #test_dataset.to_csv('test_dataset.csv', sep=',')
    #train_dataset.to_csv('train_dataset_1-6.csv', sep=',')
    #test_dataset.to_csv('test_dataset_1-6.csv', sep=',')
    #train_dataset.to_csv('train_dataset_0.5-5.5.csv', sep=',')
    #test_dataset.to_csv('test_dataset_0.5-5.5.csv', sep=',')
    #exit()
    train_dataset = pd.read_csv('train_dataset.csv')
    test_dataset = pd.read_csv('test_dataset.csv')
    #train_dataset = pd.read_csv('train_dataset_1-6.csv')
    #test_dataset = pd.read_csv('test_dataset_1-6.csv')
    #train_dataset = pd.read_csv('train_dataset_0.5-5.5.csv')
    #test_dataset = pd.read_csv('test_dataset_0.5-5.5.csv')

    train_y = train_dataset['Cancer_Type'].values
    cols = ['Tumor_Sample_ID', 'Cancer_Type']
    train_x = train_dataset.drop(cols, axis = 1).values

    test_y = test_dataset['Cancer_Type'].values
    cols = ['Tumor_Sample_ID', 'Cancer_Type']
    test_x = test_dataset.drop(cols, axis = 1).values

    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(train_x, train_y)

    #params = {
    #    'task': 'train', 
    #    'boosting_type': 'gbdt', 
    #    'objective': 'multiclass', 
    #    'metric': 'multi_logloss', 
    #    'num_class': 6, 
    #    'metric_freq': 1, 
    #    'is_training_metric': 'true', 
    #    'max_bin': 255, 
    #    #'early_stopping': 10, 
    #    'num_trees': 100, 
    #    'learning_rate': 0.05, 
    #    'num_leaves': 31, 
    #    'max_depth': 30
    #}
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 6,
        'num_leaves': 255, 
        'max_depth': 8,
        'min_child_samples': 200,
        'subsample': 0.9,
        'subsample_freq': 1,
        'colsample_bytree': 0.5,
        'min_child_weight': 0,
        'subsample_for_bin': 1000000,
        'min_split_gain': 0,
        'reg_lambda': 0,
        'verbose': 0
    }

    #print "[timestamp] begin training: " + str(datetime.now())
    gbm = lgb.train(params, 
            lgb_train, 
            num_boost_round=50)
    #print "[timestamp] end training: " + str(datetime.now())

    print "Save model..."
    gbm.save_model('model.txt')

    pred_y = gbm.predict(test_x, num_iteration=gbm.best_iteration)
    for i in range(0, len(pred_y)):
        print pred_y[i]
    #print('The rmse of prediction is:', mean_squared_error(test_y, pred_y) ** 0.5)

    
    
if __name__ == "__main__":
    main()
