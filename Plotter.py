import pandas as pd
import matplotlib.pyplot as plt

root_dit = './logs 40%/'

root_folder = {
    'ERM': {
        'MAE' : 'ERM-Brain_cancer-MAE',
        'MSE' : 'ERM-Brain_cancer-MSE',
        'Harmonized': 'Harmonized-ERM-Brain_cancer-MAE'
    },
    'IRM': {
        'MAE' : 'IRM-Brain_cancer-MAE',
        'MSE' : 'IRM-Brain_cancer-MSE',
        'Harmonized': 'Harmonized-IRM-Brain_cancer-MAE'

    },
    'deepCORAL': {
        'MAE' : 'deepCORAL-Brain_cancer-MAE',
        'MSE' : 'deepCORAL-Brain_cancer-MSE',
        'Harmonized': 'Harmonized-deepCORAL-Brain_cancer-MAE'
    },

    'DANN': {
        'MAE' : 'DANN-Brain_cancer-MAE',
        'Harmonized': 'Harmonized-DANN-Brain_cancer-MAE'

    },
    'PsuedoLabel':{
        'MAE' : 'PsuedoLabel-Brain_cancer-MAE',
        'Harmonized': 'Harmonized-PseudoLabel-Brain_cancer-MAE'

    },
    # 'groupDRO': {
    #     'MAE' : 'groupDRO-Brain_cancer-MAE',
    #     'Harmonized': 'Harmonized-groupDRO-Brain_cancer-MAE'

    # },
    'NoisyStudent': {
        'MAE' : 'NoisyStudent-Brain_cancer-MAE',
        'Harmonized': 'Harmonized-NoisyStudent-Brain_cancer-MAE'

    },

}
# Create plots for each dataset (train, validation, test)
def PlotMAE():
    # List of CSV files for different methods and datasets
    csv_files = {
        'train': {
            # 'ERM': 'ERM-Brain_cancer-MAE',
            # 'IRM': 'IRM-Brain_cancer-MAE',
            # 'deepCORAL': 'deepCORAL-Brain_cancer-MAE',
            # 'DANN': 'DANN-Brain_cancer-MAE',
            # 'PsuedoLabel': 'PsuedoLabel-Brain_cancer-MAE',
            'Harmonized IRM': 'Harmonized-IRM-Brain_cancer-MAE',
            'Harmonized ERM': 'Harmonized-ERM-Brain_cancer-MAE',
            'Harmonized deepCORAL': 'Harmonized-deepCORAL-Brain_cancer-MAE',
            'Harmonized DANN': 'Harmonized-DANN-Brain_cancer-MAE',
            'Harmonized PseudoLabel': 'Harmonized-PseudoLabel-Brain_cancer-MAE',
            # 'NoisyStudent': 'NoisyStudent-Brain_cancer-MAE',
            # 'groupDRO': 'groupDRO-Brain_cancer-MAE',

        },
        'validation': {
            # 'ERM': 'ERM-Brain_cancer-MAE',
            # 'IRM': 'IRM-Brain_cancer-MAE',
            # 'deepCORAL': 'deepCORAL-Brain_cancer-MAE',
            # 'DANN': 'DANN-Brain_cancer-MAE',
            # 'PsuedoLabel': 'PsuedoLabel-Brain_cancer-MAE',
            'Harmonized IRM': 'Harmonized-IRM-Brain_cancer-MAE',
            'Harmonized ERM': 'Harmonized-ERM-Brain_cancer-MAE',
            'Harmonized deepCORAL': 'Harmonized-deepCORAL-Brain_cancer-MAE',
            'Harmonized DANN': 'Harmonized-DANN-Brain_cancer-MAE',
            'Harmonized PseudoLabel': 'Harmonized-PseudoLabel-Brain_cancer-MAE',

            # 'NoisyStudent': 'NoisyStudent-Brain_cancer-MAE',
            # 'groupDRO': 'groupDRO-Brain_cancer-MAE',

        },
        'test': {
            # 'ERM': 'ERM-Brain_cancer-MAE',
            # 'IRM': 'IRM-Brain_cancer-MAE',
            # 'deepCORAL': 'deepCORAL-Brain_cancer-MAE',
            # 'DANN': 'DANN-Brain_cancer-MAE',
            # 'PsuedoLabel': 'PsuedoLabel-Brain_cancer-MAE',
            'Harmonized IRM': 'Harmonized-IRM-Brain_cancer-MAE',
            'Harmonized ERM': 'Harmonized-ERM-Brain_cancer-MAE',
            'Harmonized deepCORAL': 'Harmonized-deepCORAL-Brain_cancer-MAE',
            'Harmonized DANN': 'Harmonized-DANN-Brain_cancer-MAE',
            'Harmonized PseudoLabel': 'Harmonized-PseudoLabel-Brain_cancer-MAE',

            # 'NoisyStudent': 'NoisyStudent-Brain_cancer-MAE',
            # 'groupDRO': 'groupDRO-Brain_cancer-MAE',


        }
    }
    for dataset, method_files in csv_files.items():
        plt.figure(figsize=(10, 6))
        epoch_avg_mae = {dt: {} for dt in csv_files}
        for method, csv_file_path in method_files.items():
            dataset_str = dataset
            if dataset == 'validation':
                dataset_str = 'val'
            file_path = root_dit+csv_file_path+'/'+dataset_str+'_algo.csv'
            data = pd.read_csv(file_path)
            avg_mae = data['mae_avg']
            epochs = data['epoch']
            
            

            if dataset == 'train':
                avg_mae_by_epoch = avg_mae.groupby(epochs).mean()
                epoch_avg_mae[dataset] = avg_mae_by_epoch  
                plt.plot(epochs.drop_duplicates(), epoch_avg_mae[dataset], label=method) 
            else:                         
                plt.plot(epochs, avg_mae, label=method)
            print(epoch_avg_mae)

        # default_data = pd.read_csv('training_results.csv')
        # if dataset == 'train':
        #     default_avg_auc = default_data['train_auc']
        #     default_epoch = default_data['epoch']
        #     plt.plot(default_epoch, default_avg_auc, label='default')
        # elif dataset == 'validation':
        #     default_avg_auc = default_data['validation_auc']
        #     default_epoch = default_data['epoch']
            # plt.plot(default_epoch, default_avg_auc, label='default')
        plt.title(f'{dataset.capitalize()} - Average MAE Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Average MAE')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Save the plot to a PDF file
        pdf_file_name = f'{dataset}_mae_plot.pdf'
        plt.savefig('./figures/'+pdf_file_name)
        plt.close()  # Close the current plot before moving to the next
        
        print(f'Saved {pdf_file_name}')
        
def PlotLoss():
    # List of CSV files for different methods and datasets
    csv_files = {
        'train': {
            'ERM': 'ERM-Brain_cancer-MSE',
            'IRM': 'IRM-Brain_cancer-MSE',
            'deepCORAL': 'deepCORAL-Brain_cancer-MSE',

        },
        'validation': {
            'ERM': 'ERM-Brain_cancer-MSE',
            'IRM': 'IRM-Brain_cancer-MSE',
            'deepCORAL': 'deepCORAL-Brain_cancer-MSE',

        },
        'test': {
            'ERM': 'ERM-Brain_cancer-MSE',
            'IRM': 'IRM-Brain_cancer-MSE',
            'deepCORAL': 'deepCORAL-Brain_cancer-MSE',

        }
    }
    for dataset, method_files in csv_files.items():
        plt.figure(figsize=(10, 6))
        epoch_avg_loss = {dt: {} for dt in csv_files}
        for method, csv_file_path in method_files.items():
            dataset_str = dataset
            if dataset == 'validation':
                dataset_str = 'val'
            file_path = root_dit+csv_file_path+'/'+dataset_str+'_algo.csv'
            data = pd.read_csv(file_path)
            avg_loss = data['loss_avg']
            epochs = data['epoch']
            
            if dataset == 'train':
                avg_loss_by_epoch = avg_loss.groupby(epochs).mean()
                epoch_avg_loss[dataset] = avg_loss_by_epoch  
                plt.plot(epochs.drop_duplicates(), epoch_avg_loss[dataset], label=method)
            else:
                plt.plot(epochs, avg_loss, label=method)
        
        default_data = pd.read_csv('training_results.csv')
        if dataset == 'train':
            default_avg_loss = default_data['train_loss']
            default_epoch = default_data['epoch']
            plt.plot(default_epoch, default_avg_loss, label='default')
        elif dataset == 'validation':
            default_avg_loss = default_data['validation_loss']
            default_epoch = default_data['epoch']
            plt.plot(default_epoch, default_avg_loss, label='default')
        plt.title(f'{dataset.capitalize()} - Average Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Save the plot to a PDF file
        pdf_file_name = f'{dataset}_mse_plot.pdf'
        plt.savefig('./figures/'+pdf_file_name)
        plt.close()  # Close the current plot before moving to the next
        
        print(f'Saved {pdf_file_name}')


if __name__=='__main__':
    PlotMAE()
    PlotLoss()
    print("Done!")