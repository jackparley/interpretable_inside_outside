"""Compute and cache the three latent probes at four depth-12 checkpoints."""
import gc
import pickle
from pathlib import Path
import probe_stage2_16384 as shared

STEPS = (512, 16384, 262144, 2097152)
WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT = WORKSPACE / 'second_third_last_probe_checkpoint_results_depth12.pkl'

def main():
    with (WORKSPACE / 'second_third_last_probe_checkpoint_results.pkl').open('rb') as handle:
        original = pickle.load(handle)
    names = ('second_last_level1', 'third_last_level1', 'third_last_level2')
    metadata = {name: original['probe_metadata'][name] for name in names}
    paths = {step: WORKSPACE / f'run_L3_v8_bin2_ter16_P16777216_lr0.001_l20.03_bs128_depth12_demb_256_nheads_16_arr0_t{step}.pt' for step in STEPS}
    bundle = None
    if OUTPUT.exists():
        with OUTPUT.open('rb') as handle:
            bundle = pickle.load(handle)
        for key in ('grammar', 'dataset', 'probe_C'):
            if bundle[key] != original[key]:
                raise ValueError(f'Cached experiment differs: {key}')
        if bundle['probe_metadata'] != metadata or bundle['architecture']['num_layers'] != 12 or bundle['architecture']['embedding_dim'] != 256:
            raise ValueError('Cached probe definitions or architecture differ')
    pending = [step for step in STEPS if bundle is None or step not in bundle['results']]
    for step in pending:
        if not paths[step].is_file():
            raise FileNotFoundError(paths[step])
    if not pending:
        print(f'All depth-12 stages already saved: {OUTPUT}', flush=True)
        return
    grammar, dataset = original['grammar'], original['dataset']
    shared.V, shared.L = grammar['V'], grammar['L']
    m2, m3 = grammar['BIN'], grammar['BIN'] * shared.V
    shared.rules = shared.sample_padded_rules(shared.V, shared.V, m2, m3, 2, 3, shared.L, grammar['SEED_RULES'])
    shared.ternary_probs = {level: shared.torch.cat([shared.torch.zeros(m2), shared.torch.full((m3,), 1/m3)]) for level in range(shared.L)}
    train, train_latents = shared.generate_fully_ternary(dataset['n_train'], dataset['train_seed'])
    test, test_latents = shared.generate_fully_ternary(dataset['n_test'], dataset['test_seed'])
    for step in pending:
        print(f'Checkpoint {step} on {shared.DEVICE}', flush=True)
        saved = shared.torch.load(paths[step], map_location='cpu', weights_only=False)
        state = {key.removeprefix('_orig_mod.'): value for key, value in saved['model'].items()}
        arch = shared.infer_architecture(state, 16)
        if (arch['num_layers'], arch['embedding_dim'], arch['block_size']) != (12, 256, dataset['input_length']):
            raise ValueError(f'Unexpected architecture: {arch}')
        if int(saved.get('step', step)) != step:
            raise ValueError(f'Checkpoint step mismatch: {step}')
        if bundle is None:
            bundle = {key: original[key] for key in ('grammar', 'dataset', 'probe_C', 'chance_accuracy')}
            bundle.update(architecture=arch, layer_labels=['emb']+[f'L{i}' for i in range(1,13)], probe_metadata=metadata, results={})
        elif arch != bundle['architecture']:
            raise ValueError('Checkpoint architectures differ')
        model = shared.models.CLM(**arch)
        model.load_state_dict(state)
        model.to(shared.DEVICE).eval()
        del saved, state
        positions = {name: config['position'] for name, config in metadata.items()}
        train_acts = shared.extract_multiple_position_activations(model, train[:, :dataset['input_length']], positions, 64)
        test_acts = shared.extract_multiple_position_activations(model, test[:, :dataset['input_length']], positions, 64)
        result = {'checkpoint': str(paths[step]), 'saved_step': step, 'probes': {}}
        for name, config in metadata.items():
            print(f'{step}: {name}', flush=True)
            index = shared.L - config['level']
            result['probes'][name] = shared.fit_layerwise_probes(train_acts.pop(name), train_latents[index][:, config['node']].numpy(), test_acts.pop(name), test_latents[index][:, config['node']].numpy(), C=bundle['probe_C'])
        bundle['results'][step] = result
        temporary = OUTPUT.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            pickle.dump(bundle, handle)
        temporary.replace(OUTPUT)
        print(f'Saved checkpoint {step}: {OUTPUT}', flush=True)
        del model, train_acts, test_acts
        gc.collect()
        if shared.DEVICE == 'mps': shared.torch.mps.empty_cache()
        elif shared.DEVICE == 'cuda': shared.torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
