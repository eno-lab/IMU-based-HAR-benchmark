import os
import gc
import sys
import types
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
from datetime import datetime
from time import time

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

from datareader import gen_datareader

import argparse  

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', required=True)
parser.add_argument('--model_name', required=True)
parser.add_argument('--ispl_datareader', action='store_true')
parser.add_argument('--class_weight', action='store_true')
parser.add_argument('--epochs', type=int, default=350)
parser.add_argument('--boot_strap_epochs', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--shuffle_on_train', action='store_true')
parser.add_argument('--lr_magnif', type=float, default=1)
parser.add_argument('--lr_magnif_on_plateau', type=float, default=0.8)
parser.add_argument('--lr_auto_adjust_based_bs', action='store_true')
parser.add_argument('--mixed_precision', default=None)
parser.add_argument('--pretrained_model', default=None)
parser.add_argument('--two_pass', action='store_true')
parser.add_argument('--skip_train', action='store_true')
parser.add_argument('--best_selection_metrics', default='mf1')
parser.add_argument('--optuna', action='store_true')
parser.add_argument('--optuna_study_suffix', default='')
parser.add_argument('--optuna_num_of_trial', type=int, default=10)
parser.add_argument('--downsampling_ignore_rate', type=float, default=0)
args = parser.parse_args()

import optuna
import logging
import sys
# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

total_prediction = []
total_true = []

epochs = args.epochs
batch_size = args.batch_size
patience = args.patience
shuffle_on_train = args.shuffle_on_train
lr_magnif = args.lr_magnif
downsampling_ignore_rate = args.downsampling_ignore_rate
if not (0<= downsampling_ignore_rate < 1):
    sys.exit(f"Invalid downsampling_ignore_rate, {downsampling_ignore_rate}. Valid range is 0<= rate < 1.")
    
pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 200)

# it is not suitable for many case.
# not recommended
if args.lr_auto_adjust_based_bs and batch_size != 64:
    # Learning Rates as a Function of Batch Size: A Random Matrix Theory Approach to Neural Network Training
    # Diego Granziol, Stefan Zohren, Stephen Roberts
    # for Adam
    lr_magnif *= np.sqrt(batch_size)/np.sqrt(64) # it seems to big

datasets = eval(args.datasets) 
model_name = args.model_name
training_id = f'{time()*1000:.0f}'


print("===============")
print(f"{datasets=}")
print(f"{model_name=}")
print("===============")

# ...................................................................................#
# init framework
# ...................................................................................#
mod = importlib.import_module(f'models.{model_name}')
framework_name  = mod.get_dnn_framework_name()
if framework_name == 'tensorflow':
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # ...................................................................................#
    # for debug
    # tf.debugging.experimental.enable_dump_debug_info( #     "/tmp/tfdbg2_logdir",
    #     tensor_debug_mode="FULL_HEALTH",
    #     circular_buffer_size=-1)
    # ...................................................................................#

    if args.mixed_precision is not None:
        tf.keras.mixed_precision.set_global_policy(args.mixed_precision)

    if 'evaluate_model' not in globals():
        from utils import evaluate_model
    if 'plot_metrics' not in globals():
        from utils import plot_metrics

elif framework_name == 'pytorch':
    import torch
    from torch import nn
    import torchinfo        # summary generate module
    
    if (torch.cuda.is_available()):
        torch_device = 'cuda'
    else:
        torch_device = 'cpu'

    if 'plot_metrics' not in globals():
        from utils import plot_metrics

    from torch_utils import save_model

    from utils import calc_class_weight

else:
    raise NotImplementedError("Invalid DNN framework is specified. {framework_name=}")
# ...................................................................................#

# ...................................................................................#
if args.ispl_datareader:
    from utils import load_dataset, get_loss_and_activation
# ...................................................................................#

for dataset in datasets:

    dataset_origin = dataset.split('-')[0]
    file_prefix = f"{training_id}_{model_name}_{dataset}"

    if args.optuna:
        study_name = f'{model_name}_{dataset}_{args.optuna_study_suffix}'  # Unique identifier of the study.
        storage_name = f"sqlite:///{study_name}.db"
        study = optuna.create_study(directions=["maximize", "minimize", "maximize"],
                                    study_name=study_name, 
                                    storage=storage_name, 
                                    #pruner=optuna.pruners.HyperbandPruner(), 
                                    load_if_exists=True)

    if args.ispl_datareader:
        X_train, y_train, X_val, y_val, X_test, y_test, labels, n_classes = load_dataset(dataset)
        out_loss, out_activ = get_loss_and_activation(dataset)
        file_prefix = f"{file_prefix}_ispl-datareader"
    else:
        dr = gen_datareader(dataset)
        X_train, y_train, X_val, y_val, X_test, y_test, labels, n_classes = dr.gen_ispl_style_set()
        out_loss, out_activ = dr.out_loss, dr.out_activ


    if downsampling_ignore_rate > 0:
        def downsampling(X, y):
            assert X.shape[0] == y.shape[0]
            _len = X.shape[0]
            _ix = np.round(np.linspace(0, _len-1, round(_len*(1-downsampling_ignore_rate)))).astype(int)
            return X[_ix], y[_ix]
        X_train, y_train = downsampling(X_train, y_train)
        X_val, y_val = downsampling(X_val, y_val)
        X_test, y_test = downsampling(X_test, y_test)

    _file_prefix = file_prefix

    # Models to run
    try:
        # Loss and acc
        METRICS = [
            'accuracy'
        ]

        def objective(trial):

            if trial is not None:
                file_prefix = f'{_file_prefix}_trial-{trial.number}'
            else:
                file_prefix = f'{_file_prefix}'

            # Report Generation
            report_name = os.path.join("reports", dataset_origin, f"{file_prefix}_report.txt")
            os.makedirs(os.path.dirname(report_name), exist_ok=True)

            # Path to images
            img_path = os.path.join("images", dataset_origin, file_prefix)
            os.makedirs(img_path, exist_ok=True)

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

                report.write('################### Args ###################\n')
                _vars = vars(args)
                for _k in _vars:
                    report.write(f'{_k}: {_vars[_k]}\n')
                report.write('############################################\n\n')

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
                if framework_name == 'tensorflow':
                    #save_name = os.path.abspath(os.path.join('trained_models', dataset, f"{file_prefix}_tf")) # too slow
                    save_name = os.path.abspath(os.path.join('trained_models', dataset, f"{file_prefix}.h5")) # faster
                elif framework_name == 'pytorch':
                    save_name = os.path.abspath(os.path.join('trained_models', dataset, f"{file_prefix}.pth")) # faster

                input_shape = X_train.shape

                model = None

                for pass_n, clw in enumerate([False, args.class_weight] if args.two_pass else [args.class_weight]):
                    pass_n +=1

                    if pass_n == 2 and model is not None:
                        del model
                    try:
                        if args.optuna:
                            hyperparameters  = mod.get_optim_config(dataset, trial, lr_magnif=lr_magnif)
                        else:
                            hyperparameters  = mod.get_config(dataset, lr_magnif=lr_magnif)

                        if pass_n == 1:
                            if args.pretrained_model is not None:
                                model = tf.keras.saving.load_model(args.pretrained_model)
                            else:
                                model = mod.gen_model(input_shape, n_classes, out_loss, out_activ, METRICS, hyperparameters)
                        elif pass_n == 2:
                            model = tf.keras.saving.load_model(best_model_weight_path)

                        for key, item in hyperparameters.items():
                            print(f"{key.replace('_', ' ').capitalize()}: {item}")
                    except Exception as e:
                        raise NotImplementedError(f'The model {model_name} is not implemented enough yet: {e}')


                    if args.skip_train:
                        best_model_path = None
                        history = None
                    else:
                        #--------------------------------------------------------------#
                        # Training
                        #--------------------------------------------------------------#
                        if framework_name == 'tensorflow':
                            try:
                                # Train and evaluate the current model on the dataset. Save the trained models and histories
                                model, history, best_model_path = evaluate_model(model, X_train, y_train, X_val, y_val, patience=patience, 
                                                                boot_strap_epochs = args.boot_strap_epochs,
                                                                _epochs=epochs, _save_name=save_name, _log_dir=log_dir,
                                                                shuffle_on_train = shuffle_on_train if pass_n == 1 else True,
                                                                batch_size = batch_size,
                                                                no_weight = not clw,
                                                                lr_magnif_on_plateau = args.lr_magnif_on_plateau)


                                print('###############################################################################')
                            except Exception as e:
                                print(f"######################## Oh Man! An error occurred. #########################\n{e}")
                                import traceback
                                #print(repr(traceback.format_exception(e) # for 3.10 > python version
                                print(repr(traceback.format_exception(None, e, e.__traceback__)))

                        elif framework_name == 'pytorch':

                            # TODO: add Tensorbord support

                            model:torch.nn.Module
                            optim:torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
                            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=args.lr_magnif_on_plateau, patience=10, min_lr=1e-7)

                            class TrialModel(nn.Module):
                                def __init__(self, base, out_loss, out_activ, class_weight=None) -> None:
                                    super().__init__()
                                    self.base = base
                                    if out_loss == 'categorical_crossentropy':
                                        if(class_weight is not None):
                                            _w = torch.Tensor(class_weight)
                                        else:
                                            _w = None
                                        self.lossfunc_train:torch.nn.BCELoss = torch.nn.BCELoss(weight=_w).to(torch_device)
                                        self.lossfunc_test:torch.nn.BCELoss = torch.nn.BCELoss(weight=None).to(torch_device)
                                    else:
                                        raise NotImplementedError(f'This loss func option [{out_loss}] is not implemented by pytorch mode.')
                                    
                                    if out_activ == 'softmax':
                                        self.activfunc = nn.Softmax().to(torch_device)
                                    else:
                                        raise NotImplementedError(f'This activation func option [{out_activ}] is not implemented by pytorch mode.')

                                def forward(self, x, y):
                                    pred = self.activfunc(self.base(x))
                                    if self.training:
                                        loss = self.lossfunc_train(pred, y)
                                    else:
                                        loss = self.lossfunc_test(pred, y)
                                    #loss *= y_weight
                                    return pred, loss
                            
                            try_model = TrialModel(model, out_loss, out_activ, calc_class_weight(y_train) if not clw else None)

                            def summarize_loss(loss):
                                return torch.sum(loss).item()
                            
                            def calc_acc(pred, target):
                                return (torch.count_nonzero(torch.argmax(pred, dim=1) == torch.argmax(target, dim=1)) / pred.shape[0]).item()

                            # make dataset into torch.Tensor
                            X_tr = torch.Tensor(X_train).to(torch_device)
                            y_tr = torch.Tensor(y_train).to(torch_device)
                            X_tes = torch.Tensor(X_val).to(torch_device)
                            y_tes = torch.Tensor(y_val).to(torch_device)

                            # initialize keras like history object
                            history = types.SimpleNamespace()
                            history.epoch = list()
                            history.history = dict()
                            hist = history.history
                            for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
                                hist[key] = []

                            # initialize best model saving valiables 
                            best_model_monitor = 'val_loss'
                            best_model_path = None
                            best_model_idx = 0

                            # set early stopping config
                            early_stop_monitor = 'val_loss'
                            early_stop_min_delta = 0
                            early_stop_patience = patience
                            # set early stopping valiables
                            early_stop_no_implovement_count = 0
                            early_stop_target_idx = 0

                            for epoch in range(0, epochs):
                                try_model.to(torch_device)

                                print(f'Epoch: {epoch}', end='')
                                history.epoch.append(epoch)

                                # train
                                try_model.train()
                                try_model.zero_grad()
                                pred, loss = try_model(X_tr, y_tr)

                                # logging train result
                                hist['loss'].append(summarize_loss(loss))
                                hist['accuracy'].append(calc_acc(pred, y_tr))

                                loss.backward()
                                optim.step()

                                # validate
                                try_model.eval()
                                pred, loss = try_model(X_tes, y_tes)

                                # logging validation result
                                hist['val_loss'].append(summarize_loss(loss))
                                hist['val_accuracy'].append(calc_acc(pred, y_tes))

                                # end one epoch
                                # reduce optimizer lr if required
                                scheduler.step(hist['loss'][-1])

                                # display epoch result
                                print(f', train_loss: {hist["loss"][-1]:.3g}, train_acc: {hist["accuracy"][-1]:.3g}, test_loss: {hist["val_loss"][-1]:.3g}, test_acc: {hist["val_accuracy"][-1]:.3g}, lr: {scheduler.get_last_lr()}')

                                # save best model
                                if(hist[best_model_monitor][best_model_idx] > hist[best_model_monitor][-1]):
                                    save_model(model, save_name)
                                    best_model_path = save_name
                                    best_model_idx = epoch

                                # early stopping
                                if (hist[early_stop_monitor][early_stop_target_idx] - hist[early_stop_monitor][-1]) > early_stop_min_delta:
                                    early_stop_target_idx = epoch
                                    early_stop_no_implovement_count = 0
                                else:
                                    early_stop_no_implovement_count += 1
                                    if early_stop_no_implovement_count > early_stop_patience and epochs >= args.boot_strap_epochs:
                                        break

                        else:
                            raise NotImplementedError("Invalid DNN framework is specified. {framework_name=}")
                        #--------------------------------------------------------------#


                    # **************** Time to summarize the model's performance ****************
                    report.write(f"{model_name} Model : {datetime.now()}\n")
                    # Training History
                    if history is not None:
                        report.write(f"Model History \n{pd.DataFrame(history.history)}\n\n")
                    model_str = []
                    if framework_name == 'tensorflow':
                        model.summary(print_fn=lambda x: model_str.append(x))
                    if framework_name == 'pytorch':
                        model_str.append(repr(torchinfo.summary(model, input_size=X_train.shape)))

                    report.write("\n".join(model_str))
                    report.write('\n\n')
                    report.write("+++Hyperparameters+++\n")
                    report.write(f"Number of Epochs: {epochs}\n")
                    report.write(f"Batch Size: {batch_size}\n")

                    # let's plot our training history
                    if history is not None:
                        plot_metrics(history, model_name, dataset, os.path.join(img_path, f"{file_prefix}_history.png"))

                    for key, item in hyperparameters.items():
                        report.write(f"{key.replace('_', ' ').capitalize()}: {item}\n")

                    best_model_weight_path = None
                    best_score = None
                    for model_type in ["last_model", "best_model"]:
                        if model_type == 'best_model':
                            if best_model_path is None:
                                continue

                            if framework_name == 'tensorflow':
                                model.load_weights(best_model_path) 
                            elif framework_name == 'pytorch':
                                model.load_state_dict(torch.load(best_model_path))
                            else:
                                raise NotImplementedError("Invalid DNN framework is specified. {framework_name=}")

                        print('###############################################################################')
                        print(model_type)
                        print('###############################################################################')

                        if args.two_pass:
                            _save_model_name = f'{file_prefix}_pass-{pass_n}_{model_type}'
                        else:
                            _save_model_name = f'{file_prefix}_{model_type}'

                        #--------------------------------------------------------------#
                        # save trained model and get scores
                        #--------------------------------------------------------------#
                        if framework_name == 'tensorflow':
                            _save_model_path = os.path.abspath(os.path.join('trained_models', dataset, f"{_save_model_name}.h5")) # h5 is fast

                            if not args.skip_train:
                                model.save(_save_model_path) 

                            # Results for each model
                            # TODO check: is it a cause of "CUDA_ERROR_ILLEGAL_ADDRESS"? should we reload the model from the pass?
                            scores = model.evaluate(X_test, y_test, verbose=1)
                            predictions = model.predict(X_test).argmax(1)
                        elif framework_name == 'pytorch':
                            _save_model_path = os.path.abspath(os.path.join('trained_models', dataset, f"{_save_model_name}.pth")) # h5 is fast

                            if not args.skip_train:
                                save_model(model, _save_model_path)

                            model.to(torch_device)
                            X_tes = torch.Tensor(X_test).to(torch_device)
                            y_tes = torch.Tensor(y_test).to(torch_device)

                            try_model = TrialModel(model, out_loss, out_activ)
                            try_model.eval()
                            pred, loss = try_model(X_tes, y_tes)

                            scores = [summarize_loss(loss), calc_acc(pred, y_tes)]
                            predictions = torch.argmax(pred, dim=1).cpu().detach().numpy()
                        else:
                            raise NotImplementedError("Invalid DNN framework is specified. {framework_name=}")
                        #--------------------------------------------------------------#

                        report.write('###############################################################################\n')
                        report.write(model_type)
                        report.write('###############################################################################\n')

                        report.write('\n\n')
                        report.write(f"Test Accuracy: {scores[1] * 100}\n")
                        report.write(f"Test Loss: {scores[0]}\n")
                        report.write('\n\n\n')

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

                        ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap=plt.cm.Blues,)
                        plt.grid(False)
                        #sns.heatmap(normalised_confusion_matrix_df, cmap='rainbow')
                        #plt.title("Confusion matrix\n(normalised to % of total test data)")
                        #plt.ylabel('True label')
                        #plt.xlabel('Predicted label')
                        plt.savefig(os.path.join(img_path, f"{file_prefix}_{model_type}_confusion_matrix.png"), bbox_inches='tight')

                        clr = classification_report(y_test.argmax(1), predictions,
                                                    labels=np.unique(y_test.argmax(1)), output_dict=True, zero_division=0.0)

                        pass_score = {'mf1': clr['macro avg']['f1-score'], 
                                'wf1': clr['weighted avg']['f1-score'],
                                'acc': scores[1],
                                'loss': scores[0]}

                        if best_model_weight_path is None or best_score[args.best_selection_metrics] < pass_score[args.best_selection_metrics]:
                            best_score = pass_score
                            best_model_weight_path = _save_model_path

                    text = f"Finished working on: {model_name} at: {datetime.now()} -> {time() - start}"
                    print(text)
                    report.write(f"{text}\n\n")

                    if not args.optuna:
                        total_prediction.append(predictions)
                        total_true.append(y_test.argmax(axis=1))

                    if model is not None:
                        del model

                    if framework_name == 'tensorflow':
                        tf.keras.backend.clear_session()
                    elif framework_name == 'pytorch':
                        # NOTE: Release pytorch resource here if it is required.
                        pass
                    else:
                        raise NotImplementedError("Invalid DNN framework is specified. {framework_name=}")
                    gc.collect()

                    return best_score['mf1'], best_score['loss'], best_score['acc']

        if args.optuna:
            study.optimize(objective, n_trials=args.optuna_num_of_trial)
            # TODO add best scors for total_prediction and total_true to support optuna optim for losocv, etc.
            # but is it necessary?
        else:
            objective(None)

    except Exception as ex:
        print("Caught an exception: ", ex)
        import traceback
        print(repr(traceback.format_exception(ex)))
        continue

    print('**Finished**')

if not args.optuna and len(datasets) > 1:
    total_report_name = os.path.join('reports', dataset_origin, f"{training_id}_{model_name}_total_report.txt")
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


