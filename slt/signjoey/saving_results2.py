import os
import pandas as pd
import matplotlib.pyplot as plt

saving_path = '/nas1/yjun/slt/slt/experiment_results/'
log_dir = '/nas1/yjun/slt/slt/saved_model/'

def load_log(log_path):
    lines = []
    with open(log_path, 'r') as f:
        for line in f:
            lines.append((line))
    return lines

try:
    df = pd.read_csv(saving_path + 'test_result.csv')
except:
    df = pd.DataFrame(columns=['start_day', 'train_no', 'recognition_loss_weight', 'translation_loss_weight', 'load_model', 'Epoch', 'Step', 'WER', 'BLEU-4'])

df_new = df.copy()
train_models = sorted([int(x.split('sign_model')[1]) for x in os.listdir(log_dir)])
for number in train_models:
    if number==90: continue
    train_no = f'train{number}'
    if train_no not in list(df['train_no']):
        print(f'appending {train_no}')
        log = load_log(log_dir + f'/sign_model{number}/train.log')
        start_day, recognition_loss_weight, translation_loss_weight, load_model = '', '', '', ''
        cfg_done = False
        for line in log:
            if line.startswith('20') and not start_day:
                start_day = line[:10]
            elif 'cfg.training.recognition_loss_weight' in line and not recognition_loss_weight:
                recognition_loss_weight = line.split(': ')[-1].split('\n')[0]
            elif 'cfg.training.translation_loss_weight' in line and not translation_loss_weight:
                translation_loss_weight = line.split(': ')[-1].split('\n')[0]
            elif 'cfg.training.load_model' in line and not load_model:
                load_model = line.split(': ')[-1].split('\n')[0]
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
        record = [start_day, train_no, recognition_loss_weight, translation_loss_weight, load_model, Epoch, Step, WER, BLEU]
        df_new.loc[len(df_new)] = record
df_new.to_csv(saving_path + 'test_result.csv', index=False)
