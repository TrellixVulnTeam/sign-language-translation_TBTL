import os
import pandas as pd
import matplotlib.pyplot as plt

log_path = '/nas1/yjun/slt/slt/log/'
saving_path = '/nas1/yjun/slt/slt/experiment_results/'

def load_log(log_path):
    lines = []
    with open(log_path, 'r') as f:
        for line in f:
            lines.append((line))
    return lines

try:
    df = pd.read_csv(saving_path + 'test_result.csv')
except:
    df = pd.DataFrame(columns=['start_day', 'train_no', 'eval_metric', 'learning_rate', 'batch_size', 'early_stopping_patience', 'weight_decay', 'ee_dropout', 'e_dropout', 'de_dropout', 'd_dropout', 'Epoch', 'Step', 'WER', 'BLEU-4'])

df_new = df.copy()
train_list = sorted([int(x.split('.')[0].split('train')[1]) for x in os.listdir(log_path)])

for number in train_list:
    train_no = f'train{number}'
    if train_no not in list(df['train_no']):
        print(f'appending {train_no}')
        log = load_log(log_path + train_no + '.log')
        start_day, eval_metric, learning_rate, batch_size, early_stopping_patience, weight_decay, ee_dropout, e_dropout, de_dropout, d_dropout= '', '', '', '', '', '', '', '', '', ''
        cfg_done = False
        for line in log:
            if line.startswith('20') and not start_day:
                start_day = line[:10]
            elif 'cfg.training.eval_metric' in line and not eval_metric:
                eval_metric = line.split(': ')[-1].split('\n')[0]
            elif 'cfg.training.learning_rate' in line and not learning_rate:
                learning_rate = line.split(': ')[-1].split('\n')[0]
                print(learning_rate)
            elif 'cfg.training.batch_size' in line and not batch_size:
                batch_size = line.split(': ')[-1].split('\n')[0]
            elif 'cfg.training.early_stopping_patience' in line and not early_stopping_patience:
                early_stopping_patience = line.split(': ')[-1].split('\n')[0]
            elif 'cfg.training.weight_decay' in line and not weight_decay:
                weight_decay = line.split(': ')[-1].split('\n')[0]
            elif 'cfg.model.encoder.embeddings.dropout' in line and not ee_dropout:
                ee_dropout = line.split(': ')[-1].split('\n')[0]
            elif 'cfg.model.encoder.dropout' in line and not e_dropout:
                e_dropout = line.split(': ')[-1].split('\n')[0]
            elif 'cfg.model.decoder.embeddings.dropout' in line and not de_dropout:
                de_dropout = line.split(': ')[-1].split('\n')[0]
            elif 'cfg.model.decoder.dropout' in line and not d_dropout:
                d_dropout = line.split(': ')[-1].split('\n')[0]
                cfg_done = True
            elif cfg_done:
                break
        epoch = [int(x.split('Epoch:')[1].split(' ')[1]) for x in log if 'Epoch: ' in x]
        step = [int(x.split('Step:')[1].split(']')[0]) for x in log if 'Step: ' in x]
        assert len(epoch) == len(step)
        # epoch_step = [f'{epoch[i]}_{step[i]}' for i in range(len(epoch))]

        wer = [float(x.split('\t')[1].split(' ')[1]) for x in log if '\tWER ' in x]
        bleu4 = [float(x.split('\t')[1].split(' ')[1]) for x in log if '\tBLEU-4 ' in x]
        plt.clf()
        plt.plot(step, wer[:len(step)])
        plt.title(train_no + '_wer')
        plt.xlabel('step')
        plt.ylabel('wer')
        plt.savefig(saving_path + 'training_curve/' + train_no + '_wer')

        plt.clf()
        plt.plot(step, bleu4[:len(step)])
        plt.title(train_no + '_bleu4')
        plt.xlabel('step')
        plt.ylabel('bleu4')
        plt.savefig(saving_path + 'training_curve/' + train_no + '_bleu4')

        Epoch, Step, WER, BLEU = max(epoch), max(step), wer[-1], bleu4[-1]
        record = [start_day, train_no, eval_metric, learning_rate, batch_size, early_stopping_patience, weight_decay, ee_dropout, e_dropout, de_dropout, d_dropout, Epoch, Step, WER, BLEU]
        df_new.loc[len(df_new)] = record
df_new.to_csv(saving_path + 'test_result.csv', index=False)
