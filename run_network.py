import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argh

import time
import numpy as np
import torch

rand_seed: int = 21
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
np.random.seed(rand_seed)

from torch.utils.tensorboard import SummaryWriter

from make_network import make_network
from get_the_data import get_the_data
from loss_function import loss_function
from make_optimize import make_optimize


def main(
    lr_initial_nnmf: float = 0.01,
    lr_initial_cnn: float = 0.001,
    lr_initial_cnn_top: float = 0.001,
    lr_initial_cnn_skip: float = 0.001,
    lr_initial_norm: float = 0.001,
    iterations: int = 20,
    use_nnmf: bool = True,
    dataset: str = "CIFAR10",  # "CIFAR10", "FashionMNIST", "MNIST"
    enable_onoff: bool = False,
    local_learning_all: bool = False,
    local_learning_0: bool = False,
    local_learning_1: bool = False,
    local_learning_2: bool = False,
    local_learning_3: bool = False,
    local_learning_kl: bool = False,
    max_pool: bool = False,
    only_print_network: bool = False,
    use_identity: bool = False,
    da_auto_mode: bool = False,
) -> None:

    if local_learning_all:
        local_learning_0 = True
        local_learning_1 = True
        local_learning_2 = True
        local_learning_3 = True
    lr_limit: float = 1e-9

    if use_identity:
        use_nnmf = True

    torch_device: torch.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    torch.set_default_dtype(torch.float32)

    # Some parameters
    batch_size_train: int = 50#0
    batch_size_test: int = 50#0
    number_of_epoch: int = 5000

    if use_nnmf:
        prefix: str = "nnmf"
    else:
        prefix = "cnn"

    loss_mode: int = 0
    loss_coeffs_mse: float = 0.5
    loss_coeffs_kldiv: float = 1.0
    print(
        "loss_mode: ",
        loss_mode,
        "loss_coeffs_mse: ",
        loss_coeffs_mse,
        "loss_coeffs_kldiv: ",
        loss_coeffs_kldiv,
    )

    if dataset == "MNIST" or dataset == "FashionMNIST":
        input_number_of_channel: int = 1
        input_dim_x: int = 24
        input_dim_y: int = 24
    else:
        input_number_of_channel = 3
        input_dim_x = 28
        input_dim_y = 28

    train_dataloader, test_dataloader, train_processing_chain, test_processing_chain = (
        get_the_data(
            dataset,
            batch_size_train,
            batch_size_test,
            torch_device,
            input_dim_x,
            input_dim_y,
            flip_p=0.5,
            jitter_brightness=0.5,
            jitter_contrast=0.1,
            jitter_saturation=0.1,
            jitter_hue=0.15,
            da_auto_mode=da_auto_mode,
        )
    )

    (
        network,
        parameters,
        name_list,
    ) = make_network(
        use_nnmf=use_nnmf,
        input_dim_x=input_dim_x,
        input_dim_y=input_dim_y,
        input_number_of_channel=input_number_of_channel,
        iterations=iterations,
        enable_onoff=enable_onoff,
        local_learning=[
            local_learning_0,
            local_learning_1,
            local_learning_2,
            local_learning_3,
        ],
        local_learning_kl=local_learning_kl,
        max_pool=max_pool,
        torch_device=torch_device,
        use_identity=use_identity,
    )

    print(network)

    print()
    print("Information about used parameters:")
    number_of_parameter: int = 0
    for i, parameter_list in enumerate(parameters):
        count_parameter: int = 0
        for parameter_element in parameter_list:
            count_parameter += parameter_element.numel()
        print(f"{name_list[i]}: {count_parameter}")
        number_of_parameter += count_parameter
    print(f"total number of parameter: {number_of_parameter}")

    if only_print_network:
        exit()

    (
        optimizers,
        lr_schedulers,
    ) = make_optimize(
        parameters=parameters,
        lr_initial=[
            lr_initial_cnn_top,
            lr_initial_cnn_skip,
            lr_initial_cnn,
            lr_initial_nnmf,
            lr_initial_norm,
        ],
    )

    my_string: str = "_lr_"
    for i in range(0, len(lr_schedulers)):
        if lr_schedulers[i] is not None:
            my_string += f"{lr_schedulers[i].get_last_lr()[0]:.4e}_"  # type: ignore
        else:
            my_string += "-_"

    default_path: str = (
        f"{prefix}_iter{iterations}{my_string}0{local_learning_0}_1{local_learning_1}_2{local_learning_2}_3{local_learning_3}_kl{local_learning_kl}_max{max_pool}"
    )
    log_dir: str = f"log_{default_path}"

    tb = SummaryWriter(log_dir=log_dir)

    for epoch_id in range(0, number_of_epoch):
        print()
        print(f"Epoch: {epoch_id}")
        t_start: float = time.perf_counter()

        train_loss: float = 0.0
        train_correct: int = 0
        train_number: int = 0
        test_correct: int = 0
        test_number: int = 0

        # Switch the network into training mode
        network.train()

        # This runs in total for one epoch split up into mini-batches
        for image, target in train_dataloader:

            # Clean the gradient
            for i in range(0, len(optimizers)):
                if optimizers[i] is not None:
                    optimizers[i].zero_grad()  # type: ignore

            output = network(train_processing_chain(image))

            loss = loss_function(
                h=output,
                labels=target,
                number_of_output_neurons=output.shape[1],
                loss_mode=loss_mode,
                loss_coeffs_mse=loss_coeffs_mse,
                loss_coeffs_kldiv=loss_coeffs_kldiv,
            )

            assert loss is not None
            train_loss += loss.item()
            train_correct += (output.argmax(dim=1) == target).sum().cpu().numpy()
            train_number += target.shape[0]

            # Calculate backprop
            loss.backward()

            # Update the parameter
            # Clean the gradient
            for i in range(0, len(optimizers)):
                if optimizers[i] is not None:
                    optimizers[i].step()  # type: ignore

        perfomance_train_correct: float = 100.0 * train_correct / train_number
        # Update the learning rate
        for i in range(0, len(lr_schedulers)):
            if lr_schedulers[i] is not None:
                lr_schedulers[i].step(train_loss)  # type: ignore

        my_string = "Actual lr: "
        for i in range(0, len(lr_schedulers)):
            if lr_schedulers[i] is not None:
                my_string += f" {lr_schedulers[i].get_last_lr()[0]:.4e} "  # type: ignore
            else:
                my_string += " --- "

        print(my_string)
        t_training: float = time.perf_counter()

        # Switch the network into evalution mode
        network.eval()

        with torch.no_grad():

            for image, target in test_dataloader:
                output = network(test_processing_chain(image))

                test_correct += (output.argmax(dim=1) == target).sum().cpu().numpy()
                test_number += target.shape[0]

        t_testing = time.perf_counter()

        perfomance_test_correct: float = 100.0 * test_correct / test_number

        tb.add_scalar("Train Loss", train_loss / float(train_number), epoch_id)
        tb.add_scalar("Train Number Correct", train_correct, epoch_id)
        tb.add_scalar("Test Number Correct", test_correct, epoch_id)

        print(
            f"Training: Loss={train_loss / float(train_number):.5f} Correct={perfomance_train_correct:.2f}%"
        )
        print(f"Testing: Correct={perfomance_test_correct:.2f}%")
        print(
            f"Time: Training={(t_training - t_start):.1f}sec, Testing={(t_testing - t_training):.1f}sec"
        )

        tb.flush()

        lr_check: list[float] = []
        for i in range(0, len(lr_schedulers)):
            if lr_schedulers[i] is not None:
                lr_check.append(lr_schedulers[i].get_last_lr()[0])  # type: ignore

        lr_check_max = float(torch.tensor(lr_check).max())

        if lr_check_max < lr_limit:
            torch.save(network, f"Model_{default_path}.pt")
            tb.close()
            print("Done (lr_limit)")
            return

    torch.save(network, f"Model_{default_path}.pt")
    print()

    tb.close()
    print("Done (loop end)")

    return


if __name__ == "__main__":
    argh.dispatch_command(main)
