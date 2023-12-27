import evaluate
from loguru import logger

bleu = evaluate.load('bleu')

def evaluate_bleu(data_ref, data_sys):
    global bleu
    bleu_score = 0
    for i in range(len(data_sys)):
        try:
            output_metric = bleu.compute(predictions=data_sys[i], references=data_ref[i])
        except Exception as e:
            logger.warning(e)
            output_metric = {'bleu':0}
        bleu_score+=output_metric['bleu']
        avg_bleu=bleu_score/len(data_sys)

    return avg_bleu
