import importlib
from datetime import datetime
from time import time

import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# Util Imports
from utils import *

from datareader import gen_datareader

# ................................................................................................ #
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ................................................................................................ #

import argparse  

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', required=True)
parser.add_argument('--model_name', required=True)
parser.add_argument('--ispl_datareader', action='store_true')
parser.add_argument('--class_weight', action='store_true')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--patience', type=int, default=50)
args = parser.parse_args()

total_prediction = []
total_true = []

epochs = args.epochs
batch_size = args.batch_size
patience = args.patience

# for debug
# tf.debugging.experimental.enable_dump_debug_info( #     "/tmp/tfdbg2_logdir",
#     tensor_debug_mode="FULL_HEALTH",
#     circular_buffer_size=-1)

datasets = eval(args.datasets) 
model_name = args.model_name

print("===============")
print(f"{datasets=}")
print(f"{model_name=}")
print("===============")

# Training time
training_id = f'{time()*1000:.0f}'

for dataset in datasets:

    dataset_origin = dataset.split('_')[0]
    file_prefix = f"{training_id}_{model_name}_{dataset}"

    if args.ispl_datareader:
        X_train, y_train, X_val, y_val, X_test, y_test, labels, n_classes = load_dataset(dataset)
        file_prefix = f"{file_prefix}_ispl-datareader"
    else:
        dr = gen_datareader(dataset)
        X_train, y_train, X_val, y_val, X_test, y_test, labels, n_classes = dr.gen_ispl_style_set()

    out_loss, out_activ = get_loss_and_activation(dataset)

    # Models to run
    try:
        # Loss and acc
        METRICS = [
            'accuracy'
        ]

        # Report Generation
        report_name = os.path.join("reports", dataset_origin, f"{file_prefix}_report.txt")
        os.makedirs(os.path.dirname(report_name), exist_ok=True)

        # Path to images
        img_path = os.path.join("images", dataset_origin, file_prefix)
        os.makedirs(img_path, exist_ok=True)

        pd.set_option('display.max_rows', 400)
        pd.set_option('display.max_columns', 200)

        # Display the input data signal

        with open(report_name, "w") as report:
            print("################# Information #################")
            print(f"Dataset: {dataset}")
            print(f"Training_id: {training_id}")
            print(f"Epochs: {epochs}")
            print(f"Train:{X_train.shape} | {y_train.shape}")
            print(f"Validation:{X_val.shape} | {y_val.shape}")
            print(f"Test:{X_test.shape} | {y_test.shape}")

            print(f"Models being trained : {model_name}")

            report.write(f"This is the report for the {dataset} Dataset\n\n")

            report.write("Data Distribution: \n\n")
            report.write(f"Train:  X -> {X_train.shape} Class count -> {list(np.bincount(y_train.argmax(1)))} \n\n"
                         f"{pd.DataFrame(y_train.mean(axis=0) * 100, index=labels, columns=['frequency'])}\n\n")
            report.write(f"Validation:  X -> {X_val.shape} Class count -> {list(np.bincount(y_val.argmax(1)))} \n\n"
                         f"{pd.DataFrame(y_val.mean(axis=0) * 100, index=labels, columns=['frequency'])}\n\n")
            report.write(f"Test:  X -> {X_test.shape} Class count -> {list(np.bincount(y_test.argmax(1)))} \n\n"
                         f"{pd.DataFrame(y_test.mean(axis=0) * 100, index=labels, columns=['frequency'])}\n\n")

            start = time()

            print('###############################################################################')
            print(f"Training {model_name} : {datetime.now()}")
            print('###############################################################################')
            log_dir = os.path.abspath(os.path.join('logs', 'fit', dataset, file_prefix))
            #save_name = os.path.abspath(os.path.join('trained_models', dataset, f"{file_prefix}_tf")) # too slow
            save_name = os.path.abspath(os.path.join('trained_models', dataset, f"{file_prefix}.h5")) # faster

            input_shape = X_train.shape

            model = None

            try:
                mod = importlib.import_module(f'models.{model_name}')
                model, hyperparameters  = eval(f'mod.gen_preconfiged_model(input_shape, n_classes, out_loss, out_activ, dataset, metrics=METRICS)')
            except Exception as e:
                raise NotImplementedError(f'The model {model_name} is not implemented enough yet: {e}')

            model.summary()

            try:
                # Train and evaluate the current model on the dataset. Save the trained models and histories
                model, history = evaluate_model(model, X_train, y_train, X_val, y_val, patience=patience,
                                                _epochs=epochs, _save_name=save_name, _log_dir=log_dir,
                                                no_weight = args.class_weight)

                scores = model.evaluate(X_test, y_test, verbose=1)


                print('###############################################################################')
            except Exception as er:
                print(f"######################## Oh Man! An error occurred. #########################\n{er}")
                import traceback
                print(repr(traceback.format_exception(ex)))

            # **************** Time to summarize the model's performance ****************
            report.write(f"{model_name} Model : {datetime.now()}\n")
            # Training History
            report.write(f"Model History \n{pd.DataFrame(history.history)}\n\n")
            model_str = []
            model.summary(print_fn=lambda x: model_str.append(x))
            report.write("\n".join(model_str))
            report.write('\n\n')
            report.write("+++Hyperparameters+++\n")
            report.write(f"Number of Epochs: {epochs}\n")
            report.write(f"Batch Size: {batch_size}\n")
            for key, item in hyperparameters.items():
                report.write(f"{key.replace('_', ' ').capitalize()}: {item}\n")
            report.write('\n\n')
            report.write(f"Test Accuracy: {scores[1] * 100}\n")
            report.write(f"Test Loss: {scores[0]}\n")
            report.write('\n\n\n')

            # let's plot our training history
            plot_metrics(history, model_name, dataset, os.path.join(img_path, f"{file_prefix}_history.png"))

            # Results for each model
            predictions = model.predict(X_test).argmax(1)

            classificationReport = classification_report(y_test.argmax(1), predictions,
                                                         labels=np.arange(n_classes), 
                                                         target_names=labels,
                                                         digits=4,
                                                         )
            
            print(classificationReport)
            report.write(f"Classification Report\n{classificationReport}\n\n\n")

            print("Confusion Matrix:")
            cm = confusion_matrix(y_test.argmax(axis=1), predictions, labels=np.arange(n_classes))
            print(cm)
            report.write(f"Confusion Matrix\n{cm}\n\n\n")

            print("Normalised Confusion Matrix: True")
            normalised_confusion_matrix = np.around(confusion_matrix(y_test.argmax(axis=1),
                                                                     predictions,
                                                                     labels=np.arange(n_classes),
                                                                     normalize='true') * 100, decimals=2)
            print(normalised_confusion_matrix)
            normalised_confusion_matrix_df = pd.DataFrame(normalised_confusion_matrix, index=labels, columns=labels)
            report.write(f"Normalised Confusion Matrix: True\n{normalised_confusion_matrix_df}\n\n\n")
            sns.heatmap(normalised_confusion_matrix_df, cmap='rainbow')
            plt.title("Confusion matrix\n(normalised to % of total test data)")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(os.path.join(img_path, f"{file_prefix}_confusion_matrix.png"), bbox_inches='tight')
            plt.show()

            text = f"Finished working on: {model_name} at: {datetime.now()} -> {time() - start}"
            print(text)
            report.write("{text}\n\n")

            total_prediction.append(predictions)
            total_true.append(y_test.argmax(axis=1))

    except Exception as ex:
        print("Caught an exception: ", ex)
        import traceback
        print(repr(traceback.format_exception(ex)))
        continue

    print('**Finished**')


total_report_name = os.path.join('reports', dataset_origin, f"{file_prefix}_total_report.txt")
os.makedirs(os.path.dirname(total_report_name), exist_ok=True)

total_prediction = np.concatenate(total_prediction)
total_true = np.concatenate(total_true)
with open(total_report_name, "w") as report:

    classificationReport = classification_report(total_true, total_prediction,
                                                 labels=np.arange(n_classes),
                                                 target_names=labels,
                                                 digits=4,
                                                 )

    print(classificationReport)
    report.write(f"Classification Report\n{classificationReport}\n\n\n")

    print("Confusion Matrix:")
    cm = confusion_matrix(total_true, total_prediction, labels=np.arange(n_classes))
    print(cm)
    report.write(f"Confusion Matrix\n{cm}\n\n\n")

    print("Normalised Confusion Matrix: True")
    normalised_confusion_matrix = np.around(confusion_matrix(total_true,
                                                             total_prediction,
                                                             labels=np.arange(n_classes),
                                                             normalize='true') * 100, decimals=2)
    print(normalised_confusion_matrix)
    normalised_confusion_matrix_df = pd.DataFrame(normalised_confusion_matrix, index=labels, columns=labels)
    report.write(f"Normalised Confusion Matrix: True\n{normalised_confusion_matrix_df}\n\n\n")

