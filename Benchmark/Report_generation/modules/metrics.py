from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from bert_score import BERTScorer
from bert_score import score
from transformers import AutoModel, AutoTokenizer


bertscorer = BERTScorer(
    model_type="microsoft/BiomedVLP-CXR-BERT-specialized",
    num_layers=12,
    all_layers=True
    # trust_remote_code=True
)

def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        # (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    eval_res = {}
    eval_res['METEOR'] = 0.0
    # Compute score for each metric

    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score

    pred_reports = []
    gt_reports = []

    for k in gts.keys():
        pred_reports.append("".join(res[k]).strip())
        gt_reports.append("".join(gts[k]).strip())

    P, R, F1 = bertscorer.score(gt_reports, pred_reports)
    f1_last_layer = F1[-1].mean().item()
    eval_res['Bert_Score'] = f1_last_layer
    return eval_res
