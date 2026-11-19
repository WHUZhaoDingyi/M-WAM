"""Versioned Gemma cache: no silent use of legacy T5 features."""
import json
from .text_cache import CachedTextContextEncoder

ENCODER = 'google/gemma-2-2b-it'
REVISION = '299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8'
SCHEMA = 'pixeldit_t2i_gemma2_chi_last_hidden_300_v1'

class GemmaCachedTextContextEncoder(CachedTextContextEncoder):
    preserve_context_mask = True

    def __init__(self, cache_dir):
        super().__init__(cache_dir, context_len=300, hidden_size=2304)
        receipt = json.loads((self.cache_dir / 'encoder_contract.json').read_text())
        for k,v in dict(schema_version=SCHEMA, encoder=ENCODER, revision=REVISION,
                        max_length=300, hidden_size=2304).items():
            if receipt.get(k) != v:
                raise ValueError(f'Wrong Gemma cache contract: {k}')
        self.encoder_contract = receipt


    def cache_path(self, prompt):
        return self.cache_dir / f'{self.prompt_key(prompt)}.gemma2_len300.pt'
