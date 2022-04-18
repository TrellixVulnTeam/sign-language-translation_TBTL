import matplotlib.pyplot as plt

TRIALNUMBER = ''
model_path = f'/nas1/yjun/slt/slt/saved_model/sign_model{TRIALNUMBER}/'

def load_log(log_path):
    lines = []
    with open(log_path, 'r') as f:
        for line in f:
            lines.append((line))
    return lines

log = load_log(model_path + 'train.log')

epoch = [int(x.split('Epoch:')[1].split(' ')[1]) for x in log if 'Epoch: ' in x]
step = [int(x.split('Step:')[1].split(']')[0]) for x in log if 'Step: ' in x]
assert len(epoch) == len(step)
# epoch_step = [f'{epoch[i]}_{step[i]}' for i in range(len(epoch))]

wer = [float(x.split('\t')[1].split(' ')[1]) for x in log if '\tWER ' in x]
bleu4 = [float(x.split('\t')[1].split(' ')[1]) for x in log if '\tBLEU-4 ' in x]

plt.plot(step, wer[:len(step)])
plt.savefig(model_path + f'wer curve{TRIALNUMBER}')

plt.clf()
plt.plot(step, bleu4[:len(step)])
plt.savefig(model_path + f'bleu4 curve{TRIALNUMBER}')
# #print(experiment_metrics)
# #print(experiment_metrics[4999]['iteration'])
# # plt.ylim(0.15, 0.3)
# # plt.xlim(right=40000)
# plt.plot(
#     [x['iteration'] for x in experiment_metrics if 'total_loss' in x], 
#     [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])
# plt.plot(
#     [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
#     [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
# plt.legend(['total_loss', 'validation_loss'], loc='upper right')

# plt.show()

# val_loss_list = [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x]
# iter_list = [x['iteration'] for x in experiment_metrics if 'validation_loss' in x]
# min_val_loss = min([x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
# min_val_loss_idx = val_loss_list.index(min_val_loss)
# print('iter : ', iter_list[min_val_loss_idx], 'val_loss ; ', min_val_loss) # 9299 9399