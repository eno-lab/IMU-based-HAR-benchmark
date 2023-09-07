# title       :results
# description :Script that generates results for the paper
# author      :Ronald Mutegeki
# date        :20210203
# version     :1.0
# usage       :provide the values to be plotted before running results.py.
# notes       :Use his script along with the result_file.txt file which contains the results for iSPLInception.
#              Also don't forget to change the dataset name and where the result images should be saved.
import matplotlib.pyplot as plt
import pandas as pd


def plot_performance(dataframe, img_path="images"):
    plt.figure()
    ax = dataframe.plot(x='Epochs', secondary_y=['loss', 'val_loss'], color=['b', 'r', 'g', 'orange'])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(70, 101)
    ax.right_ax.set_ylabel("Loss")
    ax.legend(loc='upper right')
    ax.right_ax.legend(loc='lower right')
    plt.savefig(f"{img_path}/ucihar_ispl_inception.png", bbox_inches='tight')
    plt.show()


def compare(acc, loss, img_path="images"):
    plt.close()
    acc.plot(kind='bar', title='Accuracy comparison graph')
    plt.savefig(f"{img_path}/accuracy.png", bbox_inches='tight')
    plt.show()
    loss.plot(kind='bar', title='Loss comparison graph')
    plt.savefig(f"{img_path}/loss.png", bbox_inches='tight')
    plt.show()

    combined = pd.DataFrame()
    combined['Accuracy'] = acc
    combined['Loss'] = loss
    ax = combined.plot.bar(secondary_y=['Loss'], color=['orange', 'C0'], title="Model Performance on PAMAP2 Dataset")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(70, combined['Accuracy'].max() + 10)
    ax.right_ax.set_ylabel("Loss")
    ax.right_ax.set_ylim(0, combined['Loss'].max() + 0.1)
    ax.axhline(y=100, linestyle=':', color='orange')
    ax.axhline(y=combined['Accuracy'].max(), linestyle=':', color='green', linewidth=0.5)
    ax.axhline(y=combined['Accuracy'].min(), linestyle=':', color='red', linewidth=0.5)
    ax.right_ax.axhline(y=combined['Loss'].min(), linestyle=':', color='green', linewidth=0.5)
    ax.right_ax.axhline(y=combined['Loss'].max(), linestyle=':', color='red', linewidth=0.5)
    plt.savefig(f"{img_path}/pamap2_performance.png", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # Specify the file with the results
    df = pd.read_csv('result_file.txt', delimiter=',', header=0)
    df_100 = df.drop(columns='lr').copy()
    df_100['accuracy'] = df['accuracy'] * 100
    df_100['val_accuracy'] = df['val_accuracy'] * 100
    df_100 = df_100.round(decimals=2)
    plot_performance(df_100, "images")

    # Compare the various models on this dataset and plot them
    accuracies = pd.Series({'CNN': 85.788113,
                            'CNN_LSTM': 88.3720934,
                            'vLSTM': 85.529715,
                            'sLSTM': 86.976743,
                            'BiLSTM': 86.976743,
                            'iSPLInception': 89.095604,
                            })
    losses = pd.Series({'CNN': 0.664733,
                        'CNN_LSTM': 0.617437,
                        'vLSTM': 0.613304,
                        'sLSTM': 0.776437,
                        'BiLSTM': 0.491776,
                        'iSPLInception': 0.432280,
                        })

    compare(accuracies, losses, "images")
