def evaluate_bleu(data_ref, data_sys):
    bleu_score=0
    for i in range(len(data_sys)):
        try:
            output_metric = corpus_bleu([data_ref[i]],[data_sys[i]],smoothing_function=SmoothingFunction().method1)
        except Exception as e:
            print('Error in calculate bleu score')
            output_metric = 0
        bleu_score+=output_metric
        avg_bleu=bleu_score/len(data_sys)

    return avg_bleu
