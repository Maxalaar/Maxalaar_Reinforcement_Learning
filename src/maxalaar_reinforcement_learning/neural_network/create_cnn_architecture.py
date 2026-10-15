from typing import List, Tuple
from torch import nn

def create_cnn_architecture(
    in_channels: int,
    configuration_cnn: List[Tuple[int, int, int]],
    activation_function_class: nn.Module = nn.ReLU,
    use_normalization: bool = False,
) -> nn.Sequential:
    """
    Crée une architecture CNN basée sur une liste de configurations.

    Args:
        in_channels: Nombre de canaux en entrée.
        configuration_cnn: Liste de tuples (out_channels, kernel_size, stride).
        activation_function_class: Fonction d'activation à utiliser après chaque couche Conv2d.
        use_normalization: Indique si on doit ajouter une couche de normalisation après chaque Conv2d.

    Returns:
        Un modèle nn.Sequential.
    """
    layers = []
    current_in_channels = in_channels
    for out_channels, kernel_size, stride in configuration_cnn:
        layers.append(nn.Conv2d(current_in_channels, out_channels, kernel_size, stride))
        if use_normalization:
            # On utilise GroupNorm qui fonctionne même avec un batch de taille 1.
            # Choisissez un nombre de groupes qui divise out_channels, ici on choisit 8 par défaut,
            # tout en s'assurant que le nombre de groupes ne dépasse pas out_channels.
            num_groups = min(8, out_channels)
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_channels))
        layers.append(activation_function_class())
        current_in_channels = out_channels
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)
