import torch
import torch.nn as nn

from interpretability_eval import FaithfulnessEvaluator

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
        
    def get_patching_hook(self, name, position):
        def hook(module, input, output):
            """
            Patches the output of the module at the specified position with the cached activation.
            """
            if isinstance(output, tuple):
                patched = output[0].clone()
                patched[:, position, :] = self.activations[name][:, position, :]
                return (patched,) + output[1:]
            else:
                patched = output.clone()
                patched[:, position, :] = self.activations[name][:, position, :]
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
    Orchestrates the 3-step Module-Level Activation Patching protocol.
    """
    def __init__(self, model):
        self.model = model
        self.manager = HookManager()
        self.modules = self._get_target_modules()
        
    def _get_target_modules(self):
        """
        Identifies and returns a dictionary of target modules for patching.
        Keys are module names, values are the corresponding nn.Module objects.
        """
        modules = {}
        for l, block in enumerate(self.model.transformer_blocks):
            modules[f'L{l}_MHA'] = block.multihead_attention
            modules[f'L{l}_FFN'] = block.dense2
            # The residual stream output is the final LayerNorm of the block
            # Actually, to patch the residual stream, we can patch the block output itself
            modules[f'L{l}_Residual'] = block
        return modules

    def run_patching_protocol(self, input_seq, target_item, pad_token_id, epsilon=1e-3):
        """
        Executes Clean -> Corrupt -> Restore for all valid items.
        Returns: {module_name: [restoration_scores_for_each_item]}
        """
        valid_idx = (input_seq != pad_token_id).nonzero(as_tuple=True)[0]
        seq_2d = input_seq.unsqueeze(0)
        
        module_names = list(self.modules.keys())
        results = {m: torch.zeros(len(input_seq), device=input_seq.device) for m in module_names}
        
        # Clean run (Cache all internal states)
        for name, mod in self.modules.items():
            self.manager.handles.append(mod.register_forward_hook(self.manager.get_caching_hook(name)))
            
        with torch.no_grad():
            clean_logits = FaithfulnessEvaluator.get_logits(self.model, seq_2d)
            # Single logit for the target item
            clean_logit = clean_logits[0, target_item].item()
            
        self.manager.clear_hooks()
        
        # Corrupt and restore loops
        for pos in valid_idx:
            corrupted_seq = input_seq.clone()
            corrupted_seq[pos] = pad_token_id
            corrupted_seq_2d = corrupted_seq.unsqueeze(0)
            
            with torch.no_grad():
                corrupt_logits = FaithfulnessEvaluator.get_logits(self.model, corrupted_seq_2d)
                corrupt_logit = corrupt_logits[0, target_item].item()

            # Once computed the logit associated with the target item, we can compute the logit drop
            base_drop = clean_logit - corrupt_logit
            
            # Mathematical Safety Filter (if the item wasn't actually causal, we skip the restoration runs)
            if base_drop < epsilon:
                for name in module_names:
                    results[name][pos] = 0.0
                continue
                
            # Restoration runs
            for name, mod in self.modules.items():
                h = mod.register_forward_hook(self.manager.get_patching_hook(name, pos))
                try:
                    with torch.no_grad():
                        patch_logits = FaithfulnessEvaluator.get_logits(self.model, corrupted_seq_2d)
                        patch_logit = patch_logits[0, target_item].item()
                        
                    restoration_score = (patch_logit - corrupt_logit) / base_drop
                    results[name][pos] = restoration_score
                finally:
                    h.remove()
                    
        return results, module_names, valid_idx