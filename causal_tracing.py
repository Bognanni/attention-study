import torch
import torch.nn as nn

from metrics_utils import FaithfulnessEvaluator

class HookManager:
    """
    Safely manages PyTorch forward hooks for Activation Patching.
    Ensures hooks are automatically cleared to prevent memory leaks.
    """
    def __init__(self):
        # Dictionary to store cached activations for each module
        self.activations = {}
        # List to keep track of registered hooks for cleanup
        self.handles = []
        
    def get_caching_hook(self, name):
        def hook(module, input, output):
            """
            Caches the output of the module during the clean run.
            """
            if isinstance(output, tuple):
                self.activations[name] = output[0].clone().detach()
            else:
                self.activations[name] = output.clone().detach()
        return hook
        
    def get_patching_hook(self, name, patch_pos):
        def hook(module, input, output):
            """
            Patches the output of the module at the specified position with the cached activation.
            """
            if isinstance(output, tuple):
                patched = output[0].clone()
                patched[:, patch_pos, :] = self.activations[name][:, patch_pos, :]
                return (patched,) + output[1:]
            else:
                patched = output.clone()
                patched[:, patch_pos, :] = self.activations[name][:, patch_pos, :]
                return patched
        return hook
        
    def clear_hooks(self):
        """
        Removes all registered hooks to prevent memory leaks.
        """
        for h in self.handles:
            h.remove()
        self.handles = []


class ActivationPatcher:
    """
    Orchestrates the 2D Module-Level Activation Patching protocol.
    """
    def __init__(self, model):
        self.model = model
        self.manager = HookManager()
        self.modules = self._get_target_modules()
        
    def _get_target_modules(self):
        """
        Identifies and returns a dictionary of target modules for patching.
        """
        modules = {}
        for l, block in enumerate(self.model.transformer_blocks):
            modules[f'L{l}_MHA'] = block.multihead_attention
            modules[f'L{l}_FFN'] = block.dense2
            modules[f'L{l}_Residual'] = block
        return modules

    def run_patching_protocol(self, input_seq, target_item, pad_token_id, corrupt_positions, valid_idx, epsilon=1e-3):
        """
        Executes Clean -> Corrupt -> Restore for specified vital items, tracing the flow
        across all valid token positions (2D Causal Tracing).
        Returns: {corrupt_pos: {module_name: tensor_of_restoration_scores_for_all_positions}}
        """
        seq_2d = input_seq.unsqueeze(0)
        module_names = list(self.modules.keys())
        
        # Clean run (Cache all internal states)
        for name, mod in self.modules.items():
            self.manager.handles.append(mod.register_forward_hook(self.manager.get_caching_hook(name)))
            
        with torch.no_grad():
            clean_logits = FaithfulnessEvaluator.get_logits(self.model, seq_2d)
            clean_logit = clean_logits[0, target_item].item()
            
        self.manager.clear_hooks()
        
        results = {}
        
        # Corrupt and restore loops (only for the highly causal positions to save compute)
        for corrupt_pos in corrupt_positions:
            results[corrupt_pos] = {m: torch.zeros(len(input_seq), device=input_seq.device) for m in module_names}
            
            corrupted_seq = input_seq.clone()
            # Replace with a random valid token instead of pad_token_id so the transformer mask doesn't zero it out
            random_token = torch.randint(1, self.model.num_items + 1, (1,), device=input_seq.device).item()
            corrupted_seq[corrupt_pos] = random_token
            corrupted_seq_2d = corrupted_seq.unsqueeze(0)
            
            with torch.no_grad():
                corrupt_logits = FaithfulnessEvaluator.get_logits(self.model, corrupted_seq_2d)
                corrupt_logit = corrupt_logits[0, target_item].item()

            base_drop = clean_logit - corrupt_logit
            
            # Mathematical Safety Filter (if the item wasn't actually causal, we skip the restoration runs)
            if base_drop < epsilon:
                continue
                
            # 2D Patching: trace the information flow across ALL valid positions
            for patch_pos in valid_idx:
                for name, mod in self.modules.items():
                    h = mod.register_forward_hook(self.manager.get_patching_hook(name, patch_pos))
                    try:
                        with torch.no_grad():
                            patch_logits = FaithfulnessEvaluator.get_logits(self.model, corrupted_seq_2d)
                            patch_logit = patch_logits[0, target_item].item()
                            
                        restoration_score = (patch_logit - corrupt_logit) / base_drop
                        results[corrupt_pos][name][patch_pos] = restoration_score
                    finally:
                        h.remove()
                        
        return results, module_names