import torch
from SplitOnOffLayer import SplitOnOffLayer
from append_block import append_block
from L1NormLayer import L1NormLayer
from NNMF2d import NNMF2d
from append_parameter import append_parameter


def make_network(
    use_nnmf: bool,
    input_dim_x: int,
    input_dim_y: int,
    input_number_of_channel: int,
    iterations: int,
    torch_device: torch.device,
    epsilon: bool | None = None,
    positive_function_type: int = 0,
    beta: float | None = None,
    # Conv:
    number_of_output_channels: list[int] = [32, 64, 96, 10],
    kernel_size_conv: list[tuple[int, int]] = [
        (5, 5),
        (5, 5),
        (-1, -1),  # Take the whole input image x and y size
        (1, 1),
    ],
    stride_conv: list[tuple[int, int]] = [
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
    ],
    padding_conv: list[tuple[int, int]] = [
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ],
    dilation_conv: list[tuple[int, int]] = [
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
    ],
    # Pool:
    kernel_size_pool: list[tuple[int, int]] = [
        (2, 2),
        (2, 2),
        (-1, -1),  # No pooling layer
        (-1, -1),  # No pooling layer
    ],
    stride_pool: list[tuple[int, int]] = [
        (2, 2),
        (2, 2),
        (-1, -1),
        (-1, -1),
    ],
    padding_pool: list[tuple[int, int]] = [
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ],
    dilation_pool: list[tuple[int, int]] = [
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
    ],
    local_learning: list[bool] = [False, False, False, False],
    local_learning_kl: bool = True,
    max_pool: bool = True,
    enable_onoff: bool = False,
    use_identity: bool = False,
) -> tuple[
    torch.nn.Sequential,
    list[list[torch.nn.parameter.Parameter]],
    list[str],
]:

    assert len(number_of_output_channels) == len(kernel_size_conv)
    assert len(number_of_output_channels) == len(stride_conv)
    assert len(number_of_output_channels) == len(padding_conv)
    assert len(number_of_output_channels) == len(dilation_conv)
    assert len(number_of_output_channels) == len(kernel_size_pool)
    assert len(number_of_output_channels) == len(stride_pool)
    assert len(number_of_output_channels) == len(padding_pool)
    assert len(number_of_output_channels) == len(dilation_pool)
    assert len(number_of_output_channels) == len(local_learning)

    if enable_onoff:
        input_number_of_channel *= 2

    parameter_cnn_top: list[torch.nn.parameter.Parameter] = []
    parameter_cnn_skip: list[torch.nn.parameter.Parameter] = []
    parameter_cnn: list[torch.nn.parameter.Parameter] = []
    parameter_nnmf: list[torch.nn.parameter.Parameter] = []
    parameter_norm: list[torch.nn.parameter.Parameter] = []

    test_image = torch.ones(
        (1, input_number_of_channel, input_dim_x, input_dim_y), device=torch_device
    )

    network = torch.nn.Sequential()
    network = network.to(torch_device)

    if enable_onoff:
        network.append(SplitOnOffLayer())
        test_image = network[-1](test_image)

    for block_id in range(0, len(number_of_output_channels)):

        test_image = append_block(
            network=network,
            out_channels=number_of_output_channels[block_id],
            test_image=test_image,
            dilation=dilation_conv[block_id],
            padding=padding_conv[block_id],
            stride=stride_conv[block_id],
            kernel_size=kernel_size_conv[block_id],
            epsilon=epsilon,
            positive_function_type=positive_function_type,
            beta=beta,
            iterations=iterations,
            local_learning=local_learning[block_id],
            local_learning_kl=local_learning_kl,
            torch_device=torch_device,
            parameter_cnn_top=parameter_cnn_top,
            parameter_cnn_skip=parameter_cnn_skip,
            parameter_cnn=parameter_cnn,
            parameter_nnmf=parameter_nnmf,
            parameter_norm=parameter_norm,
            use_nnmf=use_nnmf,
            use_identity=use_identity,
        )

        if (kernel_size_pool[block_id][0] > 0) and (kernel_size_pool[block_id][1] > 0):
            network.append(torch.nn.ReLU())
            test_image = network[-1](test_image)

            mock_output = (
                torch.nn.functional.conv2d(
                    torch.zeros(
                        1,
                        1,
                        test_image.shape[2],
                        test_image.shape[3],
                    ),
                    torch.zeros((1, 1, 2, 2)),
                    stride=(2, 2),
                    padding=(0, 0),
                    dilation=(1, 1),
                )
                .squeeze(0)
                .squeeze(0)
            )

            network.append(
                torch.nn.Unfold(
                    kernel_size=(2, 2),
                    stride=(2, 2),
                    padding=(0, 0),
                    dilation=(1, 1),
                )
            )
            test_image = network[-1](test_image)

            network.append(
                torch.nn.Fold(
                    output_size=mock_output.shape,
                    kernel_size=(1, 1),
                    dilation=1,
                    padding=0,
                    stride=1,
                )
            )
            test_image = network[-1](test_image)

            network.append(L1NormLayer())
            test_image = network[-1](test_image)

            network.append(
                NNMF2d(
                    in_channels=test_image.shape[1],
                    out_channels=test_image.shape[1] // 4,
                    epsilon=epsilon,
                    positive_function_type=positive_function_type,
                    beta=beta,
                    iterations=iterations,
                    local_learning=False,
                    local_learning_kl=False,
                ).to(torch_device)
            )

            test_image = network[-1](test_image)
            append_parameter(module=network[-1], parameter_list=parameter_nnmf)

            network.append(
                torch.nn.BatchNorm2d(
                    num_features=test_image.shape[1],
                    device=torch_device,
                    momentum=0.1,
                    track_running_stats=False,
                )
            )
            test_image = network[-1](test_image)
            append_parameter(module=network[-1], parameter_list=parameter_norm)

    network.append(torch.nn.Softmax(dim=1))
    test_image = network[-1](test_image)

    network.append(torch.nn.Flatten())
    test_image = network[-1](test_image)

    parameters: list[list[torch.nn.parameter.Parameter]] = [
        parameter_cnn_top,
        parameter_cnn_skip,
        parameter_cnn,
        parameter_nnmf,
        parameter_norm,
    ]

    name_list: list[str] = [
        "cnn_top",
        "cnn_skip",
        "cnn",
        "nnmf",
        "batchnorm2d",
    ]

    return (
        network,
        parameters,
        name_list,
    )
