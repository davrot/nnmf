import torch
from L1NormLayer import L1NormLayer
from NNMF2d import NNMF2d
from append_parameter import append_parameter


def append_block(
    network: torch.nn.Sequential,
    out_channels: int,
    test_image: torch.Tensor,
    parameter_cnn_top: list[torch.nn.parameter.Parameter],
    parameter_cnn_skip: list[torch.nn.parameter.Parameter],
    parameter_cnn: list[torch.nn.parameter.Parameter],
    parameter_nnmf: list[torch.nn.parameter.Parameter],
    parameter_norm: list[torch.nn.parameter.Parameter],
    torch_device: torch.device,
    dilation: tuple[int, int] | int = 1,
    padding: tuple[int, int] | int = 0,
    stride: tuple[int, int] | int = 1,
    kernel_size: tuple[int, int] = (5, 5),
    epsilon: float | None = None,
    positive_function_type: int = 0,
    beta: float | None = None,
    iterations: int = 20,
    local_learning: bool = False,
    local_learning_kl: bool = False,
    use_nnmf: bool = True,
    use_identity: bool = False,
    momentum: float = 0.1,
    track_running_stats: bool = False,
) -> torch.Tensor:

    kernel_size_internal: list[int] = [kernel_size[-2], kernel_size[-1]]

    if kernel_size[0] < 1:
        kernel_size_internal[0] = test_image.shape[-2]

    if kernel_size[1] < 1:
        kernel_size_internal[1] = test_image.shape[-1]

    # Main

    network.append(torch.nn.ReLU())
    test_image = network[-1](test_image)

    # I need the output size
    mock_output = (
        torch.nn.functional.conv2d(
            torch.zeros(
                1,
                1,
                test_image.shape[2],
                test_image.shape[3],
            ),
            torch.zeros((1, 1, kernel_size_internal[0], kernel_size_internal[1])),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        .squeeze(0)
        .squeeze(0)
    )
    network.append(
        torch.nn.Unfold(
            kernel_size=(kernel_size_internal[-2], kernel_size_internal[-1]),
            dilation=dilation,
            padding=padding,
            stride=stride,
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
            out_channels=out_channels,
            epsilon=epsilon,
            positive_function_type=positive_function_type,
            beta=beta,
            iterations=iterations,
            local_learning=local_learning,
            local_learning_kl=local_learning_kl,
        ).to(torch_device)
    )
    test_image = network[-1](test_image)
    append_parameter(module=network[-1], parameter_list=parameter_nnmf)

    if (test_image.shape[-1] > 1) or (test_image.shape[-2] > 1):
        network.append(
            torch.nn.BatchNorm2d(
                num_features=test_image.shape[1],
                momentum=momentum,
                track_running_stats=track_running_stats,
                device=torch_device,
            )
        )
        test_image = network[-1](test_image)
        append_parameter(module=network[-1], parameter_list=parameter_norm)

    network.append(
        torch.nn.Conv2d(
            in_channels=test_image.shape[1],
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
            device=torch_device,
        )
    )
    # Init the cnn top layers 1x1 conv2d layers
    for name, param in network[-1].named_parameters():
        with torch.no_grad():
            if name == "bias":
                param.data *= 0
            if name == "weight":
                assert param.shape[-2] == 1
                assert param.shape[-1] == 1
                param[: param.shape[0], : param.shape[0], 0, 0] = torch.eye(
                    param.shape[0], dtype=param.dtype, device=param.device
                )
                param[param.shape[0] :, :, 0, 0] = 0
                param[:, param.shape[0] :, 0, 0] = 0

    test_image = network[-1](test_image)
    append_parameter(module=network[-1], parameter_list=parameter_cnn_top)

    if (test_image.shape[-1] > 1) or (test_image.shape[-2] > 1):
        network.append(
            torch.nn.BatchNorm2d(
                num_features=test_image.shape[1],
                device=torch_device,
                momentum=momentum,
                track_running_stats=track_running_stats,
            )
        )
        test_image = network[-1](test_image)
        append_parameter(module=network[-1], parameter_list=parameter_norm)

    return test_image
