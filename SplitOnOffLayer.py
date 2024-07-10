import torch


class SplitOnOffLayer(torch.nn.Module):

    def __init__(
        self,
    ) -> None:
        super().__init__()

    ####################################################################
    # Forward                                                          #
    ####################################################################

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 4

        temp = input - 0.5
        temp_a = torch.nn.functional.relu(temp)
        temp_b = torch.nn.functional.relu(-temp)
        output = torch.cat((temp_a, temp_b), dim=1)

        return output
