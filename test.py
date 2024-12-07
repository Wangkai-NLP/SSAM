import matplotlib.pyplot as plt
import torch


def plot_heatmap(tensor, labels, save_as, show=True):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    cmap_pos = plt.get_cmap('Reds')
    cmap_neg = plt.get_cmap('Blues')

    pos_mask = tensor > 0
    neg_mask = tensor < 0
    pos_tensor = tensor.clone()
    pos_tensor[neg_mask] = 0
    pos_tensor = pos_tensor / pos_tensor.max()

    neg_tensor = tensor.clone()
    neg_tensor[pos_mask] = 0
    neg_tensor = -neg_tensor / -neg_tensor.min()

    pos_im = axs[0].imshow(pos_tensor.detach().numpy(), cmap=cmap_pos, alpha=0.5, vmin=0, vmax=1, origin='upper')
    axs[0].set_xticks(range(tensor.shape[1]))
    axs[0].set_yticks(range(tensor.shape[0]))
    axs[0].set_xticklabels(labels)
    axs[0].set_yticklabels(labels)
    axs[0].set_title('Positive values')
    axs[0].tick_params(axis='x', labelsize=8, labelrotation=90)
    axs[0].tick_params(axis='y', labelsize=8)
    # plt.xticks(rotation=90, fontsize=8)

    neg_im = axs[1].imshow(neg_tensor.detach().numpy(), cmap=cmap_neg, alpha=0.5, vmin=0, vmax=1, origin='upper')
    axs[1].set_xticks(range(tensor.shape[1]))
    axs[1].set_yticks(range(tensor.shape[0]))
    axs[1].set_xticklabels(labels)
    axs[1].set_yticklabels(labels)
    axs[1].set_title('Negative values')
    axs[1].tick_params(axis='x', labelsize=8, labelrotation=90)
    axs[1].tick_params(axis='y', labelsize=8)

    # plt.xticks(rotation=90, fontsize=8)

    # fig.suptitle(title)

    if save_as:
        plt.savefig(save_as, format=save_as.split('.')[-1], dpi=300)
    if show:
        plt.show()



from model.model import CNNNer
from model.model import CNNNer
from model.metrics import NERMetric
from data.ner_pipe import SpanNerPipe
from data.padder import Torch3DMatrixPadder
import numpy as np
import torch
from fastNLP import cache_results, prepare_torch_dataloader
from fastNLP import print
from fastNLP import Trainer, Evaluator
from fastNLP import TorchGradClipCallback, MoreEvaluateCallback
from fastNLP import FitlogCallback, RichCallback, Callback
from fastNLP import SortedSampler, BucketedBatchSampler
from fastNLP import TorchWarmupCallback
import fitlog


@cache_results('caches/ner_caches.pkl', _refresh=False)
def get_data(dataset_name, model_name):
    # 以下是我们自己的数据
    if dataset_name == 'ace2004':
        paths = 'preprocess/outputs/ace2004'
    elif dataset_name == 'ace2005':
        paths = 'preprocess/outputs/ace2005'
    elif dataset_name == 'genia':
        paths = 'preprocess/outputs/genia'
    elif dataset_name == 'genia_w2ner':
        paths = 'preprocess/outputs/genia_w2ner'
    elif dataset_name == 'ace05':
        paths = 'preprocess/outputs/ace05'
    elif dataset_name == 'ace_2005':
        paths = 'preprocess/outputs/ace_2005'
    else:
        raise RuntimeError("Does not support.")
    pipe = SpanNerPipe(model_name=model_name)
    dl = pipe.process_from_file(paths)

    return dl, pipe.matrix_segs

dl, matrix_segs = get_data('ace_2005', 'roberta-large')

def densify(x):
    x = x.todense().astype(np.float32)
    return x


dl.apply_field(densify, field_name='matrix', new_field_name='matrix', progress_bar='Densify')

dls = {}
for name, ds in dl.iter_datasets():
    ds.set_pad('matrix', pad_fn=Torch3DMatrixPadder(pad_val=ds.collator.input_fields['matrix']['pad_val'],
                                                    num_class=matrix_segs['ent'],
                                                    batch_size=1))


    _dl = prepare_torch_dataloader(ds, batch_size=1, num_workers=0,
                                       batch_sampler=BucketedBatchSampler(ds, 'input_ids',
                                                                          batch_size=1,
                                                                          num_batch_per_bucket=30),
                                       pin_memory=True, shuffle=True)

    # else:
    #     _dl = prepare_torch_dataloader(ds, batch_size=1, num_workers=0,
    #                                    sampler=SortedSampler(ds, 'input_ids'), pin_memory=True, shuffle=True)
    dls[name] = _dl

model = CNNNer('roberta-large', num_ner_tag=matrix_segs['ent'], cnn_dim=120, biaffine_size=200,
               size_embed_dim=25, logit_drop=0.1,
               kernel_size=3, n_head=5, cnn_depth=2)

model.load_state_dict(torch.load('ace05.pkl'))
model.eval()

from transformers import AutoTokenizer
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

tokenizer = AutoTokenizer.from_pretrained('roberta-large')

print(tokenizer.tokenize("Later this month Indonesian President Megawati Sukarnoputri travels to Moscow to seek the Kremlin 's help in modernizing her increasingly obsolete 300,000 - member armed forces ."))

while True:
    pos = int(input('位置>>'))
    for idx, i in enumerate(dls.get('test')):
        if idx != pos:
            continue
        if i['word_len'][0] > 20 or i['word_len'][0] < 5:
            continue
        result = model(input_ids=i['input_ids'], bpe_len=i['bpe_len'], indexes=i['indexes'], matrix=i['matrix'])

        words = tokenizer.convert_ids_to_tokens(i['input_ids'][0].tolist())[1:-1]
        words = tokenizer.convert_tokens_to_string(words).split(' ')[1:]
        plot_heatmap(result['attention'][0].squeeze(), words, None)
        plot_heatmap(result['attention'][1].squeeze(), words, None)
        plot_heatmap(result['attention'][2].squeeze(), words, None)

        save_p = input('保存那一张(图片/n)')
        if save_p != 'n':
            save_p = int(save_p)
            plot_heatmap(result['attention'][save_p].squeeze(), words, f'{pos}-{save_p}.eps', False)
        break
    # print(idx)
