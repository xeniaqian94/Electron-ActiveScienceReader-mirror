'''
-*- coding: utf-8 -*-
Copyright (C) 2019/2/14 
Author: Xin Qian

AllenNLP Source code location: /anaconda3/lib/python3.6/site-packages/allennlp/data/fields/__init__.py

Note: the namespace where label_num are generated is *very* confusing to me on line `vocab.get_vocab_size('labels')`

Usage:

(Only test on dev set) python reproducibility_classifier_train.py --small_test
(Use glove embedding) python reproducibility_classifier_train.py --small_test --glove
(Use glove real testing) python reproducibility_classifier_train.py --glove

(Use glove on participant-info new annotation)
'''

from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
import allennlp
from allennlp.common import Params
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, LabelField, MetadataField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
import pandas as pd
import argparse

# print(allennlp.data.fields.__file__)
from tqdm import tqdm

torch.manual_seed(1)


class ReproducibilityClaimDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.is_train = True

    def switch_to_test(self):
        self.is_train = False

    def text_to_instance(self, tokens: List[Token], tag: str = None, sent_id: str = None) -> Instance:
        '''

        :param tokens:
        :param tag: an integer on whether it is reproducibility claim
        :return:
        '''
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        if tag:
            label_field = LabelField(tag)

            # If we design some novel label_namespace, carefully prevent writing code such as,
            #  label_field = LabelField(tag, label_namespace='labewwwls')
            # Warning will be like allennlp.data.fields.label_field -   Your label namespace was 'labewwwls'. \
            # We recommend you use a namespace ending with 'labels' or 'tags', \
            # so we don't add UNK and PAD tokens by default to your vocabulary.  See documentation for `non_padded_namespaces` \
            # parameter in Vocabulary.
            fields["label"] = label_field
        fields["metadata"] = MetadataField({"sent_id": sent_id})

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:

        df = pd.read_csv(file_path)

        for idx, row in df.iterrows():
            text = row['text'].strip().split()
            if not self.is_train and pd.isnull(row['label']):  # dev split
                yield self.text_to_instance([Token(word) for word in text], "0", sent_id=row['sent_id'])
            elif self.is_train and not pd.isnull(row['label']):  # train split
                yield self.text_to_instance([Token(word) for word in text], str(int(row[
                                                                                        'label'])), sent_id=row[
                    'sent_id'])  # Based on https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/stanford_sentiment_tree_bank.py


class LSTMClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder

        # input(vocab.print_statistics())
        # Wow, this is dominated by the LabelField's default namespace, see label_field.py where label_namespace: str = 'labels',

        self.linear = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))

        # input(self.linear.weight.shape)
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.f1_measure = F1Measure(2)
        self.softmax = torch.nn.Softmax(
            dim=1)  # softmax over the last output dimension output=Tensor(batch_size, label_size)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None, metadata=None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)
        # input("logits shape"+str(logits.shape))
        output = {"logits": logits, "softmax": self.softmax(logits)}
        if label is not None:
            # input(logits.shape)
            # input(label.shape)
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1_measure = self.f1_measure.get_metric(reset)
        return {'accuracy': self.accuracy.get_metric(reset),
                'precision': precision,
                'recall': recall,
                'f1_measure': f1_measure}


def main():

    parser = argparse.ArgumentParser(description='Input, output and other configurations')

    # Old eval on general RC
    # parser.add_argument('--csv_path', type=str,
    #                     default="/Users/xinq/Desktop/lit-review/de-contextualize/output/reproducibility_sentence_output_annotated_Xin_021319.csv")
    parser.add_argument('--csv_path', type=str,
                        default="/Users/xinq/Desktop/lit-review/de-contextualize/output/evaluate_RC_021919/reproducibility_sentence_output_to_annotate_021919_randomized-Xin.csv")


    # parser.add_argument('--output', type=str, default="../output/reproducibility_sentence_output_to_annotate_new.csv")
    # parser.add_argument('--no_extract_candidates', dest='extract_candidates', action='store_false', default=True)
    parser.add_argument('--csv_test_path', type=str,
                        default="/Users/xinq/Desktop/lit-review/de-contextualize/output/reproducibility_sentence.csv")
    parser.add_argument('--csv_out_path', type=str,
                        default="/Users/xinq/Desktop/lit-review/de-contextualize/output/evaluate_RC_021919/reproducibility_sentence_scored.csv")
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--glove', dest='glove', action='store_true', default=False)
    parser.add_argument('--small_test', dest='small_test', action='store_true', default=False)
    parser.add_argument('--model_path',type=str,default="../model/model.th")
    parser.add_argument('--vocab_path',type=str,default="../model/vocab.th")
    parser.add_argument('--embedding_path',type=str,default="../model/embedding.th")
    parser.add_argument('--no_test', dest='no_test', action='store_true',default=False)
    # parser.add_argument('--split', type=int, default=0)

    args = parser.parse_args()

    reader = ReproducibilityClaimDatasetReader()
    train_dataset = reader.read(args.csv_path)
    reader.switch_to_test()
    ## Note: we implemented train/dev split (over the single annotation files that we have)
    ## Note (cont.) such that unlabelled are automatically considered as dev_dataset.
    dev_dataset = reader.read(args.csv_path)  # Using the same path here
    if args.small_test or args.no_test:
        test_dataset = dev_dataset
    else:
        test_dataset = reader.read(args.csv_test_path)  # The test set contains all sentence from 100 CHI 2018 papers

    vocab = Vocabulary.from_instances(train_dataset + dev_dataset, min_count={'tokens': 3})
    # input(vocab._non_padded_namespaces) ## Still confused!!
    # print(vocab.get_index_to_token_vocabulary("tokens")) ##  Output is like {0: '@@PADDING@@', 1: '@@UNKNOWN@@', 2: 'the', 3: 'to', 4: 'of', 5: 'and', 6: 'a', 7: 'in', 8: 'that', 9: 'for', 10: 'with'
    # print(vocab.__dict__)
    print("Namespaces of vocab are", vocab._token_to_index.keys())

    # input("Get label_idx from label "+str(vocab.get_token_index("2","labels"))+str(type(vocab.get_token_index("2","labels"))))
    # input("Get label_idx from label "+str(vocab.get_token_index("1","labels")))
    # input("Get label_idx from label "+str(vocab.get_token_index("0","labels")))
    # input()

    print(vocab.get_vocab_size("tokens"), "vocab.get_vocab_size(tokens")
    print(vocab.__dict__['_token_to_index'].__dict__['_non_padded_namespaces'])
    print(vocab.__dict__['_token_to_index'].__dict__['_padded_function'])
    print(vocab.__dict__['_padding_token'])
    print(vocab.__dict__['_oov_token'])
    # input()

    EMBEDDING_DIM = args.embedding_dim if not args.glove else 100
    HIDDEN_DIM = args.hidden_dim

    # TODO: switch to Glove for now!? (worked on 022119)

    # If you go back to where we defined our DatasetReader, the default parameters included a single index called "tokens", \
    # so our mapping just needs an embedding corresponding to that index.
    # token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
    #                             embedding_dim=EMBEDDING_DIM)

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)

    if args.glove:
        params = Params({"pretrained_file": "/Users/xinq/Downloads/glove/glove.6B." + str(EMBEDDING_DIM) + "d" + ".txt",
                         "embedding_dim": EMBEDDING_DIM})
        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=EMBEDDING_DIM).from_params(
            vocab=vocab, params=params)


        #                             pretrained_file="/Users/xinq/Downloads/glove/glove.6B." + str(
        #                                 EMBEDDING_DIM) + "d" + ".txt")

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    # torch.save(word_embeddings,open("../model/toy","wb"))
    # word_embeddings=torch.load(open("../model/toy","rb"))

    lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True)) # batch_size * seqlen * embedding/hidden
    model = LSTMClassifier(word_embeddings, lstm, vocab)

    # TODO: implement self-attention based on paper: (efficiency is also important!)

    # TODO: Option A: biattention (biattentive classifier)
    #  # Compute biattention. This is a special case since the inputs are the same.
    #         attention_logits = encoded_tokens.bmm(encoded_tokens.permute(0, 2, 1).contiguous()) # https://pytorch.org/docs/stable/torch.html#torch.bmm
    #         attention_weights = util.masked_softmax(attention_logits, text_mask)
    #        TODO: confirm where is text_mask -> text_mask = util.get_text_field_mask(tokens).float()
    #         encoded_text = util.weighted_sum(encoded_tokens, attention_weights) # function https://github.com/allenai/allennlp/blob/6d8da97312bfbde05a41558668ff63d92a9928e9/allennlp/nn/util.py#L530

    # TODO: Option B: Bilinear attention
    # Bilinear matrix attention  (对吗???)  ``X W Y^T + b``. W=weight
    #         intermediate = torch.matmul(matrix_1.unsqueeze(1), weight)
    #         final = torch.matmul(intermediate, matrix_2.unsqueeze(1).transpose(2, 3))
    #         return self._activation(final.squeeze(1) + self._bias)
    #

    # TODO (cont.) a structured self-attentive sentence embedding https://arxiv.org/pdf/1703.03130.pdf

    # optimizer = optim.SGD(model.parameters(), lr=0.1)
    # optimizer=optim.Adam(model.parameters,lr=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3,
                           weight_decay=1e-5)  # current setting that coverges on train: optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])  # sort by num_tokens
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      # validation_dataset=None, #
                      validation_dataset=train_dataset,
                      patience=10,
                      num_epochs=15)  # 10  # seems that w/ Glove 20 will be better...
    trainer.train()

    predictor = SentenceTaggerPredictor(model,
                                        dataset_reader=reader)  # SentenceTagger shares the same logic as sentence classification predictor

    '''
    allennlp/allennlp/commands/predict.py
    
    The ``predict`` subcommand allows you to make bulk JSON-to-JSON
    or dataset to JSON predictions using a trained model and its
    :class:`~allennlp.service.predictors.predictor.Predictor` wrapper.
    '''

    if not args.no_test:
        sents = []
        delimiter = "pdf_"

        # for line in open(args.csv_test_path)
        for instance in tqdm(test_dataset):  # Loop over every single instance on test_dataset
            # print(instance.fields['tokens']['tokens'].__dict__)
            # print((instance.fields['tokens'][0].__dict__)) # NOTE: stop here
            # input()
            prediction = predictor.predict_instance(instance)
            # logits = prediction['logits']
            # print(logits)
            softmax = prediction['softmax']
            # print(softmax)
            # input()
            # label_id = np.argmax(logits)
            pos_label_idx = vocab.get_token_index("2",
                                                  "labels")  # getting the corresponding dimension integer idx for label "2"
            pos_score = softmax[pos_label_idx]
            # print("metadata for this instance",instance.fields['metadata']['sent_id'],type(instance.fields['metadata']['sent_id']))
            # print(str(instance.fields['tokens']))
            # print(instance.fields['tokens'].get_text())
            # input()

            sents.append({"paperID": instance.fields['metadata']['sent_id'].split(delimiter)[0], "sent_pos": int(
                instance.fields['metadata']['sent_id'].split(delimiter)[1]), "text": instance.fields['tokens'].get_text(),
                          "pos_score": float(pos_score)})

        # write output into a .csv file. Takes about 2 mins
        df = pd.DataFrame(sents)

        # TODO: change the sort_values criteria when we generate the eval plot
        # df = df.sort_values(by=['paperID', 'pos_score'], ascending=False)
        df = df.sort_values(by=['pos_score'], ascending=False)
        df.to_csv(args.csv_out_path)

    # print("label_id=np.argmax(logits)", pos_label_idx, model.vocab.get_token_from_index(label_id, 'labels'))

    # print(instance.__dict__)
    # print(type(instance))

    # logits = predictor.predict("We allow participants to speak out loud.")['logits']
    # label_id=np.argmax(logits)
    # print("label_id=np.argmax(logits)",label_id, model.vocab.get_token_from_index(label_id, 'labels'))

    # tag_ids = np.argmax(tag_logits, axis=-1)
    # print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

    # # Here's how to save the model.
    with open(args.model_path, 'wb') as f:
        torch.save(model.state_dict(), f)

    vocab.save_to_files(args.vocab_path)
    torch.save(word_embeddings,open(args.embedding_path,"wb"))
    # word_embeddings=torch.load(open("../model/toy","rb"))


# And here's how to reload the model.




# tag_logits2 = predictor2.predict("The dog ate the apple")['tag_logits']
# assert tag_logits2 == tag_logits

if __name__== "__main__":
  main()
