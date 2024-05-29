"""
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
"""

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from pcgrad import PCGrad
from famo import FAMO

from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data,
)

from evaluation import (
    model_eval_sst,
    model_eval_multitask,
    model_eval_test_multitask,
    model_val_sts,
    model_val_para,
)

from grad_surgery import hmpcgrad

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5
N_TASKS = 3


class MultitaskBERT(nn.Module):
    """
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    """

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == "last-linear-layer":
                param.requires_grad = False
            elif config.fine_tune_mode == "full-model":
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        self.sentiment_af = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.predict_paraphrase_af = nn.Linear(2 * BERT_HIDDEN_SIZE, 2)
        self.predict_similarity_af = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        self.dropout_layer = nn.Dropout(config.hidden_dropout_prob)
        self.cos = torch.nn.CosineSimilarity(dim=1)

    def forward(self, input_ids, attention_mask):
        "Takes a batch of sentences and produces embeddings for them."
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).

        state = self.bert.forward(input_ids, attention_mask)
        sequence_output = state["last_hidden_state"]
        avg_hidden = torch.mean(sequence_output[:, 1:], dim=1)
        # avg_hidden = torch.cat((avg_hidden, state["pooler_output"]), dim=-1)
        return state["pooler_output"], avg_hidden

    def predict_sentiment(self, input_ids, attention_mask, sst_labels=None):
        """Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        """

        if args.reg == "default":
            hidden, _ = self.forward(input_ids, attention_mask)
            hidden = self.dropout_layer(hidden)
            logits = self.sentiment_af(hidden)
            return logits
        elif args.reg == "smart":

            def evalfn(embed):
                hidden, _ = self.forward(embed, attention_mask=attention_mask)
                hidden = self.dropout_layer(hidden)
                logits = self.sentiment_af(hidden)
                return logits

            smart_loss_fn = SMARTLoss(
                eval_fn=evalfn, loss_fn=kl_loss, loss_last_fn=sym_kl_loss
            )
            # Compute initial (unperturbed) state
            logits = evalfn(input_ids)
            sst_loss = (
                F.cross_entropy(logits, sst_labels.view(-1), reduction="sum")
                / args.batch_size[0]
            )
            # @TODO investigate this weight
            smart_loss = smart_loss_fn(input_ids, logits)
            sst_loss += 0.02 * smart_loss
            return logits, sst_loss

    def predict_paraphrase(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        """
        hidden1, _ = self.forward(input_ids_1, attention_mask_1)
        hidden2, _ = self.forward(input_ids_2, attention_mask_2)
        hidden = torch.cat((hidden1, hidden2), dim=-1)
        hidden = self.dropout_layer(hidden)
        logits = self.predict_paraphrase_af(hidden)
        return logits

    def predict_similarity(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        """
        _, avg_hidden1 = self.forward(input_ids_1, attention_mask_1)
        _, avg_hidden2 = self.forward(input_ids_2, attention_mask_2)
        # calculate cosine similarity
        sim_score = self.cos(avg_hidden1, avg_hidden2)
        return sim_score


def save_model(model, optimizer, args, config, filepath):
    if args.optimizer != "pcgrad":
        save_info = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "args": args,
            "model_config": config,
            "system_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.random.get_rng_state(),
        }
    else:
        save_info = {
            "model": model.state_dict(),
            "args": args,
            "model_config": config,
            "system_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.random.get_rng_state(),
        }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    """Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    """
    device = torch.device(f"cuda:{args.gpuid}") if args.use_gpu else torch.device("cpu")
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, split="train"
    )
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split="train"
    )

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    print(args.batch_size, args.loss_ratio, args.fine_tune_mode)

    sst_train_dataloader = DataLoader(
        sst_train_data,
        shuffle=True,
        batch_size=args.batch_size[0],
        collate_fn=sst_train_data.collate_fn,
    )
    sst_dev_dataloader = DataLoader(
        sst_dev_data,
        shuffle=False,
        batch_size=args.batch_size[0],
        collate_fn=sst_dev_data.collate_fn,
    )

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(
        para_train_data,
        shuffle=True,
        batch_size=args.batch_size[1],
        collate_fn=para_train_data.collate_fn,
    )
    para_dev_dataloader = DataLoader(
        para_dev_data,
        shuffle=False,
        batch_size=args.batch_size[1],
        collate_fn=para_dev_data.collate_fn,
    )

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sts_train_dataloader = DataLoader(
        sts_train_data,
        shuffle=True,
        batch_size=args.batch_size[2],
        collate_fn=sts_train_data.collate_fn,
    )
    sts_dev_dataloader = DataLoader(
        sts_dev_data,
        shuffle=False,
        batch_size=args.batch_size[2],
        collate_fn=sts_dev_data.collate_fn,
    )

    loss_ratio = np.array(args.loss_ratio)
    data_lengths = np.array(
        [
            len(sst_train_dataloader),
            len(para_train_dataloader),
            len(sts_train_dataloader),
        ]
    )
    max_data_length = np.max(data_lengths[np.nonzero(loss_ratio > 0)])
    min_data_length = np.min(data_lengths[np.nonzero(loss_ratio > 0)])
    avg_data_length = int(np.mean(data_lengths[np.nonzero(loss_ratio > 0)]))
    print(data_lengths, avg_data_length)

    # Init model.
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "num_labels": num_labels,
        "hidden_size": 768,
        "data_dir": ".",
        "fine_tune_mode": args.fine_tune_mode,
    }

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    if args.model_path:
        saved = torch.load(args.model_path)
        model.load_state_dict(saved["model"])
    model = model.to(device)

    lr = args.lr
    if args.optimizer == "pcgrad":
        print("-- using pcgrad --")
        optimizer = PCGrad(AdamW(model.parameters(), lr=lr))
    elif args.optimizer == "famo":
        print("-- using famo --")
        weight_opt = FAMO(n_tasks=N_TASKS, device=device)
        optimizer = AdamW(model.parameters(), lr=lr)
    elif args.optimizer == "default":
        print("-- using default optimizer --")
        optimizer = AdamW(model.parameters(), lr=lr)
    elif args.optimizer == "hmpcgrad":
        print("-- using homemade pcgrad optimizer --")
        optimizer = AdamW(model.parameters(), lr=lr)

    best_dev_acc = 0

    # tracking loss for round robin
    loss_choice = None

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        sst_iterator = iter(sst_train_dataloader)
        sts_iterator = iter(sts_train_dataloader)
        para_iterator = iter(para_train_dataloader)
        for _ in tqdm(
            range(avg_data_length), desc=f"train-{epoch}", disable=TQDM_DISABLE
        ):
            # for batch in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            ## calculate loss of sst
            if loss_ratio[0] > 0:
                try:
                    sst_batch = next(sst_iterator)
                except StopIteration:
                    sst_iterator = iter(sst_train_dataloader)
                    sst_batch = next(sst_iterator)

                sst_ids, sst_mask, sst_labels = (
                    sst_batch["token_ids"],
                    sst_batch["attention_mask"],
                    sst_batch["labels"],
                )
                sst_ids = sst_ids.to(device)
                sst_mask = sst_mask.to(device)
                sst_labels = sst_labels.to(device)
                if args.reg == "default":
                    sst_logits = model.predict_sentiment(sst_ids, sst_mask)
                    sst_loss = (
                        F.cross_entropy(
                            sst_logits, sst_labels.view(-1), reduction="sum"
                        )
                        / args.batch_size[0]
                    )
                elif args.reg == "smart":
                    sst_logits, sst_loss = model.predict_sentiment(
                        sst_ids, sst_mask, sst_labels
                    )
            else:
                sst_loss = 0
            ## calculate loss of para
            if loss_ratio[1] > 0:
                try:
                    para_batch = next(para_iterator)
                except StopIteration:
                    para_iterator = iter(para_train_dataloader)
                    para_batch = next(para_iterator)

                para_ids1, para_mask1, para_ids2, para_mask2, para_labels = (
                    para_batch["token_ids_1"],
                    para_batch["attention_mask_1"],
                    para_batch["token_ids_2"],
                    para_batch["attention_mask_2"],
                    para_batch["labels"],
                )
                para_ids1 = para_ids1.to(device)
                para_mask1 = para_mask1.to(device)
                para_ids2 = para_ids2.to(device)
                para_mask2 = para_mask2.to(device)
                para_labels = para_labels.to(device)
                para_logits = model.predict_paraphrase(
                    para_ids1, para_mask1, para_ids2, para_mask2
                )
                para_loss = (
                    F.cross_entropy(para_logits, para_labels.view(-1), reduction="sum")
                    / args.batch_size[1]
                )
            else:
                para_loss = 0

            ## calculate loss of sts
            if loss_ratio[2] > 0:
                try:
                    sts_batch = next(sts_iterator)
                except StopIteration:
                    sts_iterator = iter(sts_train_dataloader)
                    sts_batch = next(sts_iterator)

                sts_ids1, sts_mask1, sts_ids2, sts_mask2, sts_labels = (
                    sts_batch["token_ids_1"],
                    sts_batch["attention_mask_1"],
                    sts_batch["token_ids_2"],
                    sts_batch["attention_mask_2"],
                    sts_batch["labels"],
                )
                sts_ids1 = sts_ids1.to(device)
                sts_mask1 = sts_mask1.to(device)
                sts_ids2 = sts_ids2.to(device)
                sts_mask2 = sts_mask2.to(device)
                sts_labels = sts_labels.float() / 5
                sts_labels = sts_labels.to(device)

                sts_score = model.predict_similarity(
                    sts_ids1, sts_mask1, sts_ids2, sts_mask2
                )
                sts_loss = (
                    F.mse_loss(sts_score.view(-1), sts_labels.view(-1), reduction="sum")
                    / args.batch_size[2]
                )
            else:
                sts_loss = 0

            if args.optimizer == "pcgrad":
                losses = [sst_loss, para_loss, sts_loss]
                optimizer.pc_backward(losses)
                optimizer.step()
                avg_loss = (sst_loss + para_loss + sts_loss) / 3
                train_loss = avg_loss.item()

            elif args.optimizer == "famo":
                loss = torch.stack([sst_loss, para_loss, sts_loss], dim=0)
                optimizer.zero_grad()
                weight_opt.backward(loss)
                optimizer.step()
                with torch.no_grad():
                    # new_loss = torch.tensor([sst_loss, para_loss, sts_loss])
                    new_sst_logits = model.predict_sentiment(sst_ids, sst_mask)
                    new_sst_loss = (
                        F.cross_entropy(
                            new_sst_logits, sst_labels.view(-1), reduction="sum"
                        )
                        / args.batch_size[0]
                    )
                    new_para_logits = model.predict_paraphrase(
                        para_ids1, para_mask1, para_ids2, para_mask2
                    )
                    new_para_loss = (
                        F.cross_entropy(
                            new_para_logits, para_labels.view(-1), reduction="sum"
                        )
                        / args.batch_size[1]
                    )
                    new_sts_score = model.predict_similarity(
                        sts_ids1, sts_mask1, sts_ids2, sts_mask2
                    )
                    new_sts_loss = (
                        F.mse_loss(
                            new_sts_score.view(-1), sts_labels.view(-1), reduction="sum"
                        )
                        / args.batch_size[2]
                    )
                    new_loss = torch.tensor(
                        [new_sst_loss, new_para_loss, new_sts_loss], device=device
                    )
                    weight_opt.update(new_loss)
                train_loss = loss.mean().item()

            elif args.optimizer == "default":
                ## calculate total loss and backpropagate
                if args.default_opti_loss == "total":
                    loss = (
                        sst_loss * loss_ratio[0]
                        + para_loss * loss_ratio[1]
                        + sts_loss * loss_ratio[2]
                    ) / np.sum(loss_ratio)
                elif args.default_opti_loss == "roundrobin":
                    if loss_choice == None or loss_choice == "sts":
                        loss = sst_loss
                        loss_choice = "sst"
                    elif loss_choice == "sst":
                        loss = para_loss
                        loss_choice = "para"
                    elif loss_choice == "para":
                        loss = sts_loss
                        loss_choice = "sts"
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
                train_loss = train_loss / (num_batches)
            elif args.optimizer == "hmpcgrad":
                optimizer.zero_grad()
                hmpcgrad(model, [sst_loss, para_loss, sts_loss])
                optimizer.step()
                avg_loss = (sst_loss + para_loss + sts_loss) / 3
                train_loss = avg_loss.item()

        ##### Evaluate sts
        # sts_dev_cor, *_ = model_val_sts(sts_dev_dataloader, model, device)
        # if sts_dev_cor > best_dev_acc:
        #     save_model(model, optimizer, args, config, args.filepath)

        ##### Evaluate sst
        # sst_train_acc, sst_train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # sst_dev_acc, sst_dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)

        ##### Evaluate para
        # para_dev_acc, *_ = model_val_para(sts_dev_dataloader, model, device)
        # if para_dev_acc > best_dev_acc:
        #     save_model(model, optimizer, args, config, args.filepath)
        # with open("training_record_para_sts.csv", "a") as f:
        #     f.write(f"{para_dev_acc},{sts_dev_cor}\n")

        ##### Evaluate multitask
        # sst_train_acc, _, _, para_train_acc, _, _, sts_train_corr, *_ = (
        #     model_eval_multitask(
        #         sst_train_dataloader,
        #         para_train_dataloader,
        #         sts_train_dataloader,
        #         model,
        #         device,
        #     )
        # )
        sst_train_acc, para_train_acc, sts_train_corr = 0, 0, 0
        sst_dev_acc, _, _, para_dev_acc, _, _, sts_dev_corr, *_ = model_eval_multitask(
            sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device
        )
        with open("training_record_dev_acc.csv", "a") as f:
            f.write(f"{sst_dev_acc},{para_dev_acc},{sts_dev_corr}\n")

        perfs = np.array([sst_dev_acc, para_dev_acc, sts_dev_corr])

        if np.mean(perfs[np.nonzero(loss_ratio > 0)]) > best_dev_acc:
            best_dev_acc = np.mean(perfs[np.nonzero(loss_ratio > 0)])
            save_model(model, optimizer, args, config, args.filepath)

        print(
            f"Epoch {epoch}: train loss :: {train_loss :.3f}, sst loss :: {sst_loss :.3f}, para loss :: {para_loss :.3f}, sts loss :: {sts_loss :.3f},\
              sst train acc :: {sst_train_acc :.3f}, \
              sst dev acc :: {sst_dev_acc :.3f}\
              para train acc :: {para_train_acc:.3f}, \
              para dev acc :: {para_dev_acc:.3f}, \
              sts train corr :: {sts_train_corr :.3f},\
              sts dev corr :: {sts_dev_corr:.3f}"
        )


def test_multitask(args):
    """Test and save predictions on the dev and test sets of all three tasks."""
    with torch.no_grad():
        device = (
            torch.device(f"cuda:{args.gpuid}") if args.use_gpu else torch.device("cpu")
        )
        saved = torch.load(args.filepath)
        config = saved["model_config"]

        model = MultitaskBERT(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels, para_test_data, sts_test_data = load_multitask_data(
            args.sst_test, args.para_test, args.sts_test, split="test"
        )

        sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(
            args.sst_dev, args.para_dev, args.sts_dev, split="dev"
        )

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(
            sst_test_data,
            shuffle=True,
            batch_size=args.batch_size[0],
            collate_fn=sst_test_data.collate_fn,
        )
        sst_dev_dataloader = DataLoader(
            sst_dev_data,
            shuffle=False,
            batch_size=args.batch_size[0],
            collate_fn=sst_dev_data.collate_fn,
        )

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(
            para_test_data,
            shuffle=True,
            batch_size=args.batch_size[1],
            collate_fn=para_test_data.collate_fn,
        )
        para_dev_dataloader = DataLoader(
            para_dev_data,
            shuffle=False,
            batch_size=args.batch_size[1],
            collate_fn=para_dev_data.collate_fn,
        )

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(
            sts_test_data,
            shuffle=True,
            batch_size=args.batch_size[2],
            collate_fn=sts_test_data.collate_fn,
        )
        sts_dev_dataloader = DataLoader(
            sts_dev_data,
            shuffle=False,
            batch_size=args.batch_size[2],
            collate_fn=sts_dev_data.collate_fn,
        )

        (
            dev_sentiment_accuracy,
            dev_sst_y_pred,
            dev_sst_sent_ids,
            dev_paraphrase_accuracy,
            dev_para_y_pred,
            dev_para_sent_ids,
            dev_sts_corr,
            dev_sts_y_pred,
            dev_sts_sent_ids,
        ) = model_eval_multitask(
            sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device
        )

        (
            test_sst_y_pred,
            test_sst_sent_ids,
            test_para_y_pred,
            test_para_sent_ids,
            test_sts_y_pred,
            test_sts_sent_ids,
        ) = model_eval_test_multitask(
            sst_test_dataloader,
            para_test_dataloader,
            sts_test_dataloader,
            model,
            device,
        )

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")
            f.write(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")
            f.write(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")
            f.write(f"dev sts corr :: {dev_sts_corr :.3f}")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")
    parser.add_argument("--optimizer", type=str, default="default")

    parser.add_argument("--default_opti_loss", type=str, default="total")

    parser.add_argument("--reg", type=str, default="default")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--fine-tune-mode",
        type=str,
        help="last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well",
        choices=("last-linear-layer", "full-model"),
        default="full-model",
    )
    parser.add_argument("--use_gpu", action="store_true")

    parser.add_argument("--gpuid", type=int, default=0)

    parser.add_argument("--prediction_out", type=str, default="predictions/")

    parser.add_argument(
        "--batch_size",
        help="pass a single int or a list of 3 int for batch size of sst, para and sts. sst: 64, cfimdb: 8 can fit a 12GB GPU",
        type=int,
        nargs="+",
        default=[8, 8, 8],
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    # Add other args
    parser.add_argument(
        "--loss_ratio",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        help="ratio of sst, para, sts loss in joint loss",
    )
    parser.add_argument("--model_path", default="")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"{args.prediction_out}{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt"  # Save path.
    args.sst_dev_out = f"{args.prediction_out}sst-dev-output.csv"
    args.sst_test_out = f"{args.prediction_out}sst-test-output.csv"
    args.para_dev_out = f"{args.prediction_out}para-dev-output.csv"
    args.para_test_out = f"{args.prediction_out}para-test-output.csv"
    args.sts_dev_out = f"{args.prediction_out}sts-dev-output.csv"
    args.sts_test_out = f"{args.prediction_out}sts-test-output.csv"
    if len(args.batch_size) != 3:
        args.batch_size = args.batch_size + [args.batch_size[-1]] * (
            3 - len(args.batch_size)
        )

    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
