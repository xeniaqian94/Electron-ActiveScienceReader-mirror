'''
-*- coding: utf-8 -*-
Copyright (C) 2019/2/24 
Author: Xin Qian

This scripts is the back-end logic to generate the MegaCog in-PDF reproducibility-claim info.
The input is a PDF, the output is a JSON file that lists top RC sentences and its tie back to parsed highlighting box (parsed sentences).

Usage:
        source activate snorkel // activate the env
        python parse-PDF-generate-top-RC-as-JSON-match-to-textToDisplay.py --pdf_dir ../paper/MattsPdfLibrary/soylent-uist2010/  (this is the pdf dir to be parsed)

TODO: suppress warning info e.g. 02/26/2019 11:43:50 - INFO - pdfminer.pdfinterp

NOTE: the bash script to ls all pdf files in a library then run all PDFs in Matt's library

    for filename in ../paper/MattsPdfLibrary/*; do python parse-PDF-generate-top-RC-as-JSON-match-to-textToDisplay.py --pdf_dir $filename"/"; done

    Output is something like below, ...

    ../paper/MattsPdfLibrary/1804.02445/
    ../paper/MattsPdfLibrary/2012_Poverty Impedes Cognitive Function_Science_4703a3c0-2b36-11e9-96c3-bf329bf8f6d9/

'''
import logging
import copy

logging.getLogger("requests").setLevel(logging.WARNING)

import argparse
import re
import torch
import unicodedata
from allennlp.data import Vocabulary
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.predictors import SentenceTaggerPredictor
from tqdm import tqdm
import pandas as pd
from reproducibility_classifier_train import LSTMClassifier, ReproducibilityClaimDatasetReader

parser = argparse.ArgumentParser(description='Input, output and other configurations')
parser.add_argument('--pdf_dir', type=str, default="../paper/MattsPdfLibrary/colusso-HCI_TS_Model-CHI2019/")
parser.add_argument('--embedding_dim', type=int, default=100)  # 100 for glove; 128 if not glove
parser.add_argument('--hidden_dim', type=int, default=128)

parser.add_argument('--model_path', type=str, default="../model/model.th")  # pre-trained models
parser.add_argument('--vocab_path', type=str,
                    default="../model/vocab.th")  # vocabulary mapping (bundled w/ pre-trained models)
parser.add_argument('--embedding_path', type=str,
                    default="../model/embedding.th")  # again, bundled w/ pre-trained models

parser.add_argument('--topn', type=int, default=10)
parser.add_argument('--threshold', type=float, default=0.3)

args = parser.parse_args()

# Note: This is how MattPDFLibrary's naming convention
args.pdf_in_path = args.pdf_dir + args.pdf_dir.split("/")[-2] + ".pdf"
args.csv_tmp_path = args.pdf_in_path.replace(".pdf", ".csv")
args.csv_out_path = args.pdf_in_path.replace(".pdf", ".scored.csv")

from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter

import PyPDF2
from spacy.lang.en import English
from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams


def normalize(text, normalize=True):
    # Note: this is the best *normalization* effort we could reach at this point... after many trials and errors
    if normalize:
        text = unicodedata.normalize('NFKD', text)
    return text.replace("‘", "'").replace("’", "'").replace("“", "\"").replace("”", "\"").replace("\n", " ").replace(
        "\t", " "). \
        encode('utf-8', errors="ignore").decode('utf-8', errors="ignore").strip("-")
    # we strip end dash to accomodate line break


def get_document_sents(path, until_references=False):
    '''
    TODO: get_document_sents to be changed from raw PDF to concatenate JSON files
    :param path:
    :param until_references:
    :return:
    '''
    pdf = PyPDF2.PdfFileReader(open(path, "rb"))
    fp = open(path, 'rb')
    num_of_pages = pdf.getNumPages()
    print("num_of_pages", num_of_pages)

    nlp = English()  # just the language with no model
    sbd = nlp.create_pipe('sentencizer')  # or: nlp.create_pipe('sbd')
    nlp.add_pipe(sbd)
    doc_text = ""
    sentences = []

    for i in range(num_of_pages):
        inside = [i]
        pagenos = set(inside)
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'
        laparams = LAParams()

        for param in ("all_texts", "detect_vertical", "word_margin", "char_margin", "line_margin", "boxes_flow"):
            paramv = locals().get(param, None)
            if paramv is not None:
                setattr(laparams, param, paramv)

        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True

        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                      check_extractable=True):
            interpreter.process_page(page)
            text = retstr.getvalue()

            # text = text.decode("ascii","replace")

            ### next, parse text into a sequence of sentences w/ Spacy
            # https://spacy.io/usage/linguistic-features search "sentencizer"
            # nlp.add_pipe(nlp.create_pipe('sentencizer'))
            doc_text += str(text)

    doc = nlp(str(doc_text))
    for sent in doc.sents:
        raw_sent = (re.sub('\s+', ' ', sent.text.replace("\n", " ")).strip())
        sentences += [raw_sent]

    total_len = len(sentences)
    print(len(sentences), "len (sentences) in this document")

    try:
        if until_references:
            for idx in range(total_len):

                # Handle line break!!
                sentences[idx] = re.sub(r"([a-zA-Z]+)- ([a-zA-Z]+)", r'\g<1>\g<2>', sentences[idx])

                if (re.search("(^|\s)REFERENCES", sentences[idx]) is not None \
                    or re.search("^acknowledg(e)*ment", sentences[idx].lower()) is not None or re.search(
                            "ACKNOWLEDG(E)*MENT", sentences[idx]) is not None) and idx > int(
                    total_len * 0.5):
                    break
            if idx == total_len - 1:
                print("[WARNING] Never encountered references!!! ")
                return sentences
            sentences = sentences[:idx]

        ### TODO: data cleaning, remove sentence of permission, page header paragraph
    except Exception as e:
        print(str(e))
        return sentences

    fp.close()

    return sentences


def raw_pdf_to_csv(pdf_in_path, csv_tmp_path):
    '''
    This function is adapted from ../data/prepare-reproducibility-corpus.py
    PDFMiner and spacy parses a PDF into sentences, saves to csv_out_path
    '''

    # def remove_ligature(sent):  # maybe normalize by NFKD???
    #     return sent.replace('ﬁ', 'fi').replace('ﬂ', 'fl')

    sents = get_document_sents(pdf_in_path, until_references=True)

    # TODO: consider change get_document_sents from textToDisplay too!!!

    if sents == None:
        raise Exception("sents are None!")

    sents_csv_entry = []

    # Adapted from reproducibility_matcher.py output_all_sents()
    for sent_pos, sent in enumerate(sents):
        normalized_text = normalize(sent, normalize=True)
        sents_csv_entry.append({"sent_id": str(pdf_in_path.split("/")[-1]) + "_" + str(sent_pos),
                                "text": normalized_text
                                })
    df = pd.DataFrame(sents_csv_entry)
    df.to_csv(csv_tmp_path, columns=['sent_id', 'text', 'label'])
    return


raw_pdf_to_csv(args.pdf_in_path, args.csv_tmp_path)
print("raw_pdf_to_csv finished!")
# input("raw_pdf_to_csv finished! See if csv has many wierd codec characters")

lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(args.embedding_dim, args.hidden_dim, batch_first=True))
vocab = Vocabulary.from_files(args.vocab_path)
word_embeddings = torch.load(open(args.embedding_path, "rb"))
model = LSTMClassifier(word_embeddings, lstm, vocab)
with open(args.model_path, 'rb') as f:
    model.load_state_dict(torch.load(f))

sents = []
delimiter = "pdf_"

reader = ReproducibilityClaimDatasetReader()
reader.switch_to_test()
test_dataset = reader.read(args.csv_tmp_path)

# for line in open(args.csv_test_path)
predictor = SentenceTaggerPredictor(model,
                                    dataset_reader=reader)  # SentenceTagger shares the same logic as sentence classification predictor
for instance in tqdm(test_dataset):  # Loop over every single instance on test_dataset

    prediction = predictor.predict_instance(instance)
    softmax = prediction['softmax']

    pos_label_idx = vocab.get_token_index("2",
                                          "labels")
    pos_score = softmax[pos_label_idx]
    sents.append({"paperID": instance.fields['metadata']['sent_id'].split(delimiter)[0], "sent_pos": int(
        instance.fields['metadata']['sent_id'].split(delimiter)[1]), "text": instance.fields['tokens'].get_text(),
                  "pos_score": float(pos_score)})

# write output into a .csv file. Takes about 2 mins
df = pd.DataFrame(sents)

# TODO: change the sort_values criteria when we generate the eval plot
# df = df.sort_values(by=['paperID', 'pos_score'], ascending=False)
df = df.sort_values(by=['pos_score'], ascending=False)

mask = (df['text'].str.len() > 10)
df = df.loc[mask]

df.to_csv(args.csv_out_path)

'''

Next, assemble the two JSON files to support *fuzzy string match,* and output in below format

meta => metadata in the pdf
outline => section names in the pdf
textToDisplay => what pdfs store but converted to json objects (you want this)

========================
metadataToHighlight.json format (all page union)

{
    "note":"Below are a list of (key, value) for metadata. \
            Each key is the metadata type, the value is a list of \
            top-scored sentences for that metadata type. \
            These sentences were parsed and concatenated with an external tool (spacy). ",
    "participant_detail":[   // one type of metadata type
    {"text":"This participant has 10 teeth".  // the sentence
    "score": 0.9999                           // the likelihood/score
    },
    
    {"text":"This participant has 5 teeth".
    "score": 0.9595
    },
    
    ]
}

========================
metadataToHighlight-page000X.json   // mapped to corresponding fields in textToDisplay-page000X.json (id, str, etc.)

{
  "pageNumber": 1,
  "metadataToHighlight": [
    {"id": "0001-0000",
    "substr_tohighlight": "Extracting Scientific Figures", //Sometimes, the text span goes across two sentences and we only \
                                                want to highlight one of them...That's why we have substr here.
    "str": "Extracting Scientific Figures with",
    }
  ]
}

'''

import json

with open(args.pdf_dir + "metadataToHighlight.json", "w") as fout:
    metadataToHighlight = dict()
    metadataToHighlight['note'] = "Below are a list of (key, value) for metadata. \
    Each key is the metadata type, the value is a list of \
    top-scored sentences for that metadata type. \
    These sentences were parsed and concatenated with an external tool (spacy). "
    participant_detail = list()
    top_10_text = list()
    for idx, row in df.head(n=args.topn).iterrows():  # we take topn sentences for participant_detail
        score = float(row['pos_score'])
        if score < args.threshold:  # if below threshold, we discard
            continue
        participant_detail += [{
            "text": row['text'],
            "score": score,
        }]
        top_10_text += [row['text']]
    metadataToHighlight['participant_detail'] = participant_detail
    json.dump(metadataToHighlight, fout, indent=4)

print("top_10_text are", top_10_text)
# loop over textToDisplay-pageXXXX.json files, read in each file
'''
whose format is

{
  "pageNumber": 8,
  "text": [
    {
      "str": "interested in design, or for lack of incentive. For P15,",
      "dir": "ltr",
      "width": 218.98791059999994,
      "height": 10.161852,
      "transform": [
        10.161852,
        0,
        0,
        9.9626,
        53.798,
        689.385
      ], ...
'''

# test case: top_10_text=["something okay do not get actionable information from researchers (P7, P11,","that can facilitate translation.","Great competition between Jordan Beck and Hamid R. Ekbia.","see you in  2018.", "The Theory-Practice Gap As default is good"]
# for each text segment, split the sentences, and
# see each sentence-segment is a fuzzy sub-string match with any of the top 10 setences (top_10_text)
import re, os

nlp = English()  # just the language with no model
sbd = nlp.create_pipe('sentencizer')  # or: nlp.create_pipe('sbd')
nlp.add_pipe(sbd)
for f in os.listdir(args.pdf_dir):
    if re.match('textToDisplay-page*', f):  # loop over all textToDisplay files
        textToDisplay = json.load(open(args.pdf_dir + f, "rb"))
        metadataToHighlight = dict()
        metadataToHighlight['pageNumber'] = textToDisplay['pageNumber']
        participant_detail = list()
        for text_segment in textToDisplay['text']:
            raw_str = (text_segment['str'])
            doc = nlp(str(raw_str))
            this_text_segment_highlighted = False
            for sent in doc.sents:  # sentence-segment
                if this_text_segment_highlighted:
                    break
                for top_text in top_10_text:  ## TODO: something need to be changed here!!

                    # call find_near_matches() with the sub-sequence you're looking for, \
                    # the sequence to search, and the matching parameters:
                    # if len(find_near_matches(sent.text,top_text,max_deletions=2, max_insertions=2, max_substitutions=2))>0:
                    #     print(sent.text, "WITHIN!!",top_text)

                    # TODO: modify this rule, may not be accurate for our intended fuzzy string match
                    substr_text_normalized = normalize(str(sent.text).lower())
                    top_text_normalized = normalize(top_text.lower())
                    # if text_segment['id']=="0002-0081":
                    #     print(substr_text_normalized)
                    #     print(top_text_normalized)
                    #     input("SANITY CHECK")
                    if substr_text_normalized in top_text_normalized and len(substr_text_normalized) > 0.3 * len(
                            top_text_normalized):
                        # startOffset = raw_str.index(sent.text)
                        # endOffset = startOffset + len(sent.text)
                        # participant_detail.append({
                        #     "id": text_segment['id'],
                        #     "substr_tohighlight": sent.text,
                        #     "str": text_segment['str'],
                        #     "startOffset": startOffset,
                        #     "endOffset": endOffset,
                        #     "note": "opening bracket (open-ended) on endOffset"
                        # })
                        # break
                        text_segment_copy = copy.deepcopy(text_segment)
                        text_segment_copy['corresponding_sentence_normalized'] = top_text_normalized
                        text_segment_copy['substr_text_normalized'] = substr_text_normalized
                        participant_detail.append(text_segment_copy)
                        this_text_segment_highlighted = True
                        break

        # TODO: extend each participant_detail to previous and subsequent text segments (before lunch)
        textToDisplay['text'] = sorted(list(textToDisplay['text']), key=lambda item: int(item['id'].split("-")[1]))
        new_participant_detail_to_append = dict()

        for participant_detail_item in participant_detail:
            substr_text_normalized = participant_detail_item['substr_text_normalized']

            top_text_normalized = participant_detail_item['corresponding_sentence_normalized']
            central_idx = int(participant_detail_item['id'].split("-")[1])

            stop_loop = False
            for i in range(central_idx - 1, -1, -1):
                if stop_loop:
                    break
                raw_str = textToDisplay['text'][i]['str']
                doc = nlp(str(raw_str))
                this_text_segment_highlighted = False
                *_,last_sent=doc.sents
                # for sent in doc.sents:  # loop over sentence-segment, ideally should be doc.sents[::-1] but should be fine semantically (every textToDisplay containing multiple sentences is not the thing we want to expand)
                this_substr_text_normalized = normalize(str(last_sent.text).lower())
                # print("[i=]", i, "[this_substr_text_normalized]", this_substr_text_normalized,
                #       "[substr_text_normalized]", substr_text_normalized, "[concatenated]",
                #       this_substr_text_normalized+ substr_text_normalized, "[original]", top_text_normalized)

                is_match=(re.search(this_substr_text_normalized+"[ ]*"+substr_text_normalized,top_text_normalized))
                if is_match:
                # if this_substr_text_normalized + substr_text_normalized in top_text_normalized:
                    substr_text_normalized = is_match.group()
                    new_participant_detail_to_append[i] = textToDisplay['text'][i]
                else:
                    stop_loop = True
                    break

            substr_text_normalized = participant_detail_item['substr_text_normalized']

            stop_loop = False
            for i in range(central_idx + 1, len(textToDisplay['text']), 1):
                if stop_loop:
                    break
                raw_str = textToDisplay['text'][i]['str']
                doc = nlp(str(raw_str))
                this_text_segment_highlighted = False
                for sent in doc.sents:  # sentence-segment
                    this_substr_text_normalized = normalize(str(sent.text).lower())
                    # print("[i=]", i, "[this_substr_text_normalized]", this_substr_text_normalized,
                    #       "[substr_text_normalized]", substr_text_normalized, "[concatenated]",
                    #       substr_text_normalized + this_substr_text_normalized, "[original]", top_text_normalized)

                    is_match = (
                        re.search(substr_text_normalized + "[ ]*" + this_substr_text_normalized, top_text_normalized))
                    if is_match:
                        # if this_substr_text_normalized + substr_text_normalized in top_text_normalized:
                        substr_text_normalized = is_match.group()
                        new_participant_detail_to_append[i] = textToDisplay['text'][i]
                    else:
                        stop_loop = True
                        break

        participant_detail += list(new_participant_detail_to_append.values())

        deduplicated_participant_detail=list()
        participant_detail_unique_id=set()
        for item in participant_detail:
            if item["id"] not in participant_detail_unique_id:
                participant_detail_unique_id.add(item["id"])
                deduplicated_participant_detail.append(item)

        metadataToHighlight["text"] = deduplicated_participant_detail
        json.dump(metadataToHighlight, open(args.pdf_dir + "metadataToHighlight-" + f.split("-")[1], "w"), indent=4)
