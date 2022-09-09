import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
)

import lang_to_sem_loader
from model import ArmansSuperDuperLSTM


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    return lang_to_sem_loader.get_loaders(input_path=args.in_data_fn, batch_size=7, shuffle=True, debug=args.debug)


def setup_model(args, metadata):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #
    model = ArmansSuperDuperLSTM(args.embedding_dim, args.hidden_dim, args.vocab_size, metadata["max_actions"], metadata["max_objects"])
    return model


def setup_optimizer(args, model):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    action_criterion = torch.nn.CrossEntropyLoss()
    target_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    return action_criterion, target_criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    action_criterion,
    target_criterion,
    device,
    training=True,
):
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # print(labels)
        # put model inputs to device
        inputs = inputs.to(device)
        actions = labels[0].to(device)
        objects = labels[1].to(device)
        # print("ACTIONS ", actions.size())
        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out = model(inputs)

        # calculate the action and target prediction loss
        # NOTE: we assume that labels is a tensor of size Bx2 where labels[:, 0] is the
        # action label and labels[:, 1] is the target label
        action_loss = action_criterion(
            actions_out.squeeze(), actions.float())
        target_loss = target_criterion(
            targets_out.squeeze(), objects.float())

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)

        actions_ = actions.argmax(-1)
        objects_ = objects.argmax(-1)
        # print("ACTIONS OUT ", actions_out.size())
        # print("ACTION PREDS ", action_preds_.size())
        # print("ACTIONS_ ", actions_.size())

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().numpy())
        target_preds.extend(target_preds_.cpu().numpy())
        action_labels.extend(actions_.cpu().numpy())
        target_labels.extend(objects_.cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_action_loss, epoch_target_loss, action_acc, target_acc


def validate(
    args, model, loader, optimizer, action_criterion, target_criterion, device
):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():

        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    train_action_losses = []
    train_target_losses = []
    train_action_accs = []
    train_target_accs = []

    val_action_losses = []
    val_target_losses = []
    val_action_accs = []
    val_target_accs = []

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
        train_action_losses.append(train_action_loss)
        train_target_losses.append(train_target_loss)
        train_action_accs.append(train_action_acc)
        train_target_accs.append(train_target_acc)

        # some logging
        print(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        print(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            val_action_losses.append(val_action_loss)
            val_target_losses.append(val_target_loss)
            val_action_accs.append(val_action_acc)
            val_target_accs.append(val_target_acc)

            print(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            print(
                f"val action acc : {val_action_acc} | val target losaccs: {val_target_acc}"
            )

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #

    plt.figure(figsize=(10, 3))
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
    # Train
    ax1.plot(train_action_losses)
    ax2.plot(train_target_losses)
    ax3.plot(train_action_accs)
    ax4.plot(train_target_accs)
    ax1.title.set_text("train_action_losses")
    ax2.title.set_text("train_target_losses")
    ax3.title.set_text("train_action_accs")
    ax4.title.set_text("train_target_accs")

    # Valid
    ax5.plot(val_action_losses)
    ax6.plot(val_target_losses)
    ax7.plot(val_action_accs)
    ax8.plot(val_target_accs)
    ax5.title.set_text("val_action_losses")
    ax6.title.set_text("val_target_losses")
    ax7.title.set_text("val_action_accs")
    ax8.title.set_text("val_target_accs")

    plt.savefig("results.png")


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, metadata = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, metadata)
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(
        args, model)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(
            args, model, loaders, optimizer, action_criterion, target_criterion, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument(
        "--val_every", type=int, default=5, help="number of epochs between every eval loop"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="size of the embedding of each word in the model"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="size of the hidden state produced/consumed by the LSTM"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=1000, help="number of tokens in our vocabulary (including pad, start, end, unk)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-1, help="the learning rate for the optimizer"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()
    args.debug = True
    print(f"DEBUG {args.debug}")

    main(args)
