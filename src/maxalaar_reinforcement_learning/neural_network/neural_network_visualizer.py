import torch
import torch.nn as nn
import tkinter as tk


class NeuralNetworkVisualizer:
    def __init__(
            self,
            model,
            width=800,
            height=600,
            neuron_radius=20,
            max_connections_per_layer=None,
            window_title: str = 'Neural Network Visualizer',
            batch_index_to_visualize: int = 0,
    ):
        self.model = model
        self.width = width
        self.height = height
        self.neuron_radius = neuron_radius
        self.max_connections_per_layer = max_connections_per_layer
        self.batch_index_to_visualize = batch_index_to_visualize

        self.window = tk.Toplevel()
        self.window.title(window_title)
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height, bg='white')
        self.canvas.pack()

        self.layers = self.extract_layers(model)
        self.linear_layers = [module for module in model if isinstance(module, nn.Linear)]
        self.positions = self.compute_positions()

        self.draw_network()
        self.window.update()

        self._activations = {}
        self.hook_handles = []

        if not self.linear_layers:
            return

        # Register hooks
        handle = self.linear_layers[0].register_forward_pre_hook(
            lambda module, input: self._save_activation(0, input[0]))
        self.hook_handles.append(handle)

        for i in range(1, len(self.linear_layers)):
            layer = self.linear_layers[i]
            handle = layer.register_forward_pre_hook(
                lambda module, input, idx=i: self._save_activation(idx, input[0]))
            self.hook_handles.append(handle)

        last_layer_idx = len(self.linear_layers) - 1
        handle = self.linear_layers[last_layer_idx].register_forward_hook(
            lambda module, input, output: self._save_activation(last_layer_idx + 1, output))
        self.hook_handles.append(handle)

    def extract_layers(self, model):
        sizes = []
        for module in model:
            if isinstance(module, nn.Linear):
                if not sizes:
                    sizes.append(module.in_features)
                sizes.append(module.out_features)
        return sizes

    def compute_positions(self):
        positions = []
        n_layers = len(self.layers)
        x_spacing = self.width / (n_layers - 1) if n_layers > 1 else self.width / 2
        for i, n_neurons in enumerate(self.layers):
            x = i * x_spacing
            y_spacing = self.height / (n_neurons + 1)
            layer_positions = [(x, (j + 1) * y_spacing) for j in range(n_neurons)]
            positions.append(layer_positions)
        return positions

    def draw_network(self):
        self.neuron_items = []
        self.connection_items = []

        for layer in self.positions:
            neuron_ids = [self.create_neuron(x, y) for (x, y) in layer]
            self.neuron_items.append(neuron_ids)

        for i in range(len(self.positions) - 1):
            connection_layer = []
            weights = self.linear_layers[i].weight.data if i < len(self.linear_layers) else None
            for j, (x1, y1) in enumerate(self.positions[i]):
                for k, (x2, y2) in enumerate(self.positions[i + 1]):
                    color = self.weight_to_color(weights[k, j].item()) if weights is not None else 'grey'
                    line_id = self.create_connection(x1, y1, x2, y2, color)
                    connection_layer.append(line_id)
            self.connection_items.append(connection_layer)

    def create_neuron(self, x, y):
        return self.canvas.create_oval(
            x - self.neuron_radius, y - self.neuron_radius,
            x + self.neuron_radius, y + self.neuron_radius,
            fill='grey', outline='black'
        )

    def create_connection(self, x1, y1, x2, y2, color):
        return self.canvas.create_line(
            x1 + self.neuron_radius, y1,
            x2 - self.neuron_radius, y2,
            fill=color
        )

    def _save_activation(self, idx, tensor):
        act = tensor.detach().cpu()[self.batch_index_to_visualize] if tensor.dim() > 1 else tensor.detach().cpu()
        self._activations[idx] = act  # Store as tensor

    # def _save_activation(self, idx, tensor):
    #     # Extract and flatten activation tensors
    #     if tensor.dim() > 1:
    #         act = tensor.detach().cpu()[0]  # Get first element from batch
    #     else:
    #         act = tensor.detach().cpu()
    #     act = act.squeeze()  # Remove all singleton dimensions
    #     self._activations[idx] = act

    def activations_to_colors(self, activations_tensor):
        max_val = 1.0
        clamped = torch.clamp(activations_tensor, -max_val, max_val)
        negative_mask = clamped < 0
        t_neg = (clamped + max_val) / max_val
        red_neg = green_neg = (255 * t_neg).to(torch.uint8)
        blue_neg = torch.full_like(red_neg, 255, dtype=torch.uint8)

        positive_mask = ~negative_mask
        t_pos = clamped / max_val
        red_pos = torch.full_like(t_pos, 255, dtype=torch.uint8)
        green_blue = (255 * (1 - t_pos)).to(torch.uint8)
        green_pos = blue_pos = green_blue

        red = torch.where(negative_mask, red_neg, red_pos)
        green = torch.where(negative_mask, green_neg, green_pos)
        blue = torch.where(negative_mask, blue_neg, blue_pos)

        colors = []
        for r, g, b in zip(red.tolist(), green.tolist(), blue.tolist()):
            colors.append(f'#{r:02x}{g:02x}{b:02x}')
        return colors

    def weight_to_color(self, weight):
        return self.activation_to_color(weight)

    def activation_to_color(self, activation):
        act_tensor = torch.tensor([activation], dtype=torch.float32)
        return self.activations_to_colors(act_tensor)[0]

    def update(self):
        # Update neurons
        for layer_idx, neuron_ids in enumerate(self.neuron_items):
            if layer_idx in self._activations:
                activations = self._activations[layer_idx]
                if isinstance(activations, torch.Tensor):
                    colors = self.activations_to_colors(activations)
                    for j, neuron_id in enumerate(neuron_ids):
                        self.canvas.itemconfig(neuron_id, fill=colors[j])

        # Update connections
        for layer_idx in range(len(self.connection_items)):
            if layer_idx >= len(self.linear_layers):
                continue
            linear_layer = self.linear_layers[layer_idx]
            weights = linear_layer.weight.data.cpu()
            activations = self._activations.get(layer_idx)
            if activations is None:
                continue

            if not isinstance(activations, torch.Tensor):
                activations = torch.tensor(activations)

            # Vectorized strength calculation
            with torch.no_grad():
                # strengths = activations * weights.T
                # strengths_flat = strengths.flatten()
                # max_abs = strengths_flat.abs().max().item() or 1.0
                # norm_strengths = strengths_flat / max_abs
                # colors = self.activations_to_colors(norm_strengths)
                # thicknesses = (1 + 3 * norm_strengths.abs()).tolist()

                strengths = activations.unsqueeze(1) * weights.T
                strengths_flat = strengths.flatten()
                max_abs = strengths_flat.abs().max().item() or 1.0
                norm_strengths = strengths_flat / max_abs
                colors = self.activations_to_colors(norm_strengths)
                thicknesses = (1 + 3 * norm_strengths.abs()).tolist()

                top_connections = None
                if self.max_connections_per_layer is not None:
                    k = min(self.max_connections_per_layer, len(strengths_flat))
                    if k > 0:
                        _, top_indices = torch.topk(strengths_flat.abs(), k)
                        top_connections = set(top_indices.tolist())

                connection_layer = self.connection_items[layer_idx]
                for c, line_id in enumerate(connection_layer):
                    if top_connections is not None and c not in top_connections:
                        self.canvas.itemconfig(line_id, state='hidden')
                    else:
                        self.canvas.itemconfig(
                            line_id,
                            state='normal',
                            fill=colors[c],
                            width=thicknesses[c]
                        )

        # Single UI update after all changes
        self.window.update_idletasks()
        self.window.update()

    def start(self):
        self.window.mainloop()


# Example usage
if __name__ == '__main__':
    model = nn.Sequential(
        nn.Linear(4, 5),
        nn.ReLU(),
        nn.Linear(5, 3)
    )
    vis = NeuralNetworkVisualizer(model, max_connections_per_layer=10)


    def update_loop():
        inp = torch.randn(1, 4)
        _ = model(inp)
        vis.update()
        vis.window.after(33, update_loop)  # ~30 FPS


    update_loop()
    vis.start()