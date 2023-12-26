import os
import numpy as np
import torch
import random

from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

#from modeling_t5 import MyT5ForConditionalGeneration as MyT5
from transformers import T5ForConditionalGeneration as MyT5


from data import VNHistoryDataset,VNHistoryDataLoader
#from data import evaluate_bleu
from tqdm import tqdm, trange
import json


def run(args, logger):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Finetune on hVNHistoryDataset
    train_dataset = VNHistoryDataset(logger, args, args.train_file, tokenizer, "train")
    dev_dataset = VNHistoryDataset(logger, args, args.predict_file, tokenizer, "val")
    train_dataloader = VNHistoryDataLoader(args, train_dataset, "train")
    dev_dataloader = VNHistoryDataLoader(args, dev_dataset, "dev")

    if args.do_train:
        # Load model parameters
        model = MyT5.from_pretrained(args.model_path)

        print('model parameters: ', model.num_parameters())

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        if not args.no_lr_decay:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=t_total)
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=0,
                                                        num_training_steps=1000000)


        train(args, logger, model, train_dataloader, dev_dataloader, optimizer, scheduler, tokenizer)
    if args.do_predict:
        # Inference on the test set
        checkpoint = args.output_dir
        model = MyT5.from_pretrained(checkpoint)
        logger.info("Loading checkpoint from {}".format(checkpoint))
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()
        ems = inference(model, dev_dataloader, tokenizer, args, logger, save_predictions=True)
        logger.info("%s on %s data: %.4f" % (dev_dataloader.dataset.metric, dev_dataloader.dataset.data_type, ems))

def train(args, logger, model, train_dataloader, dev_dataloader, optimizer, scheduler, tokenizer):
    model.train()
    global_step = 0
    wait_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training = False

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    logger.info("Starting training!")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:
            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            if global_step == 1:
                for tmp_id in range(9):
                    print(batch[tmp_id])

            loss = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3])

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training = True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            # Gradient accumulation
            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()  # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()

            # Print loss and evaluate on the valid set
            if global_step % args.eval_period == 0:
                model.eval()
                curr_em = inference(model if args.n_gpu == 1 else model.module, dev_dataloader, tokenizer, args, logger)
                logger.info("Step %d Train loss %.2f Learning rate %.2e %s %.2f%% on epoch=%d" % (
                    global_step,
                    np.mean(train_losses),
                    scheduler.get_lr()[0],
                    dev_dataloader.dataset.metric,
                    curr_em * 100,
                    epoch))
                train_losses = []
                if best_accuracy < curr_em:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(args.output_dir)
                    logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" %
                                (dev_dataloader.dataset.metric, best_accuracy * 100.0, curr_em * 100.0, epoch, global_step))
                    best_accuracy = curr_em
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
                model.train()
        if stop_training:
            break


def inference(model, dev_dataloader, tokenizer, args, logger, save_predictions=False):
    predictions = []
    # Inference on the test set
    for i, batch in enumerate(dev_dataloader):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 input_node_ids=batch[4],
                                 input_edge_ids=batch[5],
                                 node_length=batch[6],
                                 edge_length=batch[7],
                                 adj_matrix=batch[8],
                                 num_beams=args.num_beams,
                                 length_penalty=args.length_penalty,
                                 max_length=args.max_output_length,
                                 early_stopping=True,)
        # Convert ids to tokens
        for input_, output in zip(batch[0], outputs):
            pred = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=args.clean_up_spaces)
            predictions.append(pred.strip())

    # Save the generated results
    if save_predictions:
        save_path = os.path.join(args.output_dir, "{}predictions.txt".format(args.prefix))
        with open(save_path, "w") as f:
            for pred in predictions:
                f.write(pred + '\n')
        logger.info("Saved prediction in {}".format(save_path))

    data_ref = [data_ele['text'] for data_ele in dev_dataloader.dataset.data]
    assert len(predictions) == len(data_ref)
    #return evaluate_bleu(data_ref=data_ref, data_sys=predictions)
