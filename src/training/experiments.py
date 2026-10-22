"""Train, cache, sample, and evaluate with persisted data/checkpoint lineage."""
import json
from pathlib import Path
import hashlib
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.models.vqvae import VQVAE3D, PatchDiscriminator3D
from src.models.diffusion import UNet3D, DDPM
from src.metrics import reconstruction_metrics
from src.utils.seed import set_seed
from src.utils.logging import write_json
from src.utils.integrity import confined_path, sha256_file
from .common import (TrainingDataset, device_for, fingerprint, save_checkpoint,
                      read_checkpoint, fresh_output, validate_output)
from .protocol import evaluation_context, evaluation_attempt
from src.utils.logging import atomic_json
from src.utils.cohort import verify_cohort
from .statistics import bootstrap_interval
from .diagnostics import foreground_metrics
from src.utils.artifacts import reference
from .state import ModelEMA, rng_state, restore_rng, precision, scheduler_for
from .latents import pack_latents, LatentCache


def checkpoint_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(c):
    set_seed(c['seed'])
    torch.set_num_threads(c.get('threads', 4))
    return device_for(c.get('device', 'auto'))


def loader_for(dataset, c, epoch=0):
    return DataLoader(dataset, batch_size=c['batch_size'], shuffle=True,
                      generator=torch.Generator().manual_seed(c['seed']+epoch), num_workers=0)


def load_vq(path, device):
    ck = read_checkpoint(path, 'vq')
    model = VQVAE3D(**ck['model_config']).to(device)
    model.load_state_dict(ck['model'])
    return model.eval(), ck


def train_vq(root, output, c, resume=False):
    device = setup(c)
    train = TrainingDataset(root, 'train', c['shape'], c.get('limit'))
    validation = TrainingDataset(root, 'validation', c['shape'], c.get('limit'))
    if not len(train):
        raise ValueError('Training split is empty')
    if not c.get('smoke_test', False) and not len(validation):
        raise ValueError('A research training run requires a nonempty validation split')
    cohort = verify_cohort(root)
    data_fingerprint = cohort.fingerprint
    records = cohort.records()
    out = fresh_output(output, resume)
    model = VQVAE3D(**c['vq']).to(device)
    if not resume and c.get('initialize_codebook', False):
        # Only training features initialize the codebook, never validation inputs.
        with torch.no_grad():
            first = torch.stack([train[i][0] for i in range(min(c['batch_size'], len(train)))])
            features = model.encoder(first.to(device)).movedim(1, -1).reshape(-1, c['vq']['latent_channels'])
            model.quantizer.initialize(features)
    optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'])
    gan = c.get('gan_weight', 0) > 0
    discriminator = PatchDiscriminator3D(c['vq']['base']).to(device) if gan else None
    dopt = torch.optim.Adam(discriminator.parameters(), lr=c['learning_rate']) if discriminator is not None else None
    accumulation = c.get('gradient_accumulation', 1)
    if gan and accumulation != 1:
        raise ValueError('VQ-GAN uses separate optimizers and currently requires gradient_accumulation=1')
    amp, dtype, scaler = precision(c, device)
    scheduler = scheduler_for(optimizer, c)
    history, start = [], 0
    if resume:
        ck = read_checkpoint(out/'last.pt', 'vq')
        if ck.get('inference_only'):
            raise ValueError('Compact inference exports cannot resume training')
        old, new = dict(ck['config']), dict(c)
        old.pop('epochs'); new.pop('epochs')
        if ck['dataset_fingerprint'] != data_fingerprint or old != new:
            raise ValueError('Resume requires identical dataset/config except total epochs')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        if discriminator is not None and dopt is not None:
            discriminator.load_state_dict(ck['discriminator']); dopt.load_state_dict(ck['d_optimizer'])
        history, start = ck['history'], ck['epoch']
        if ck.get('scaler'):
            scaler.load_state_dict(ck['scaler'])
        if scheduler and ck.get('scheduler'):
            scheduler.load_state_dict(ck['scheduler'])
        restore_rng(ck, device)
    for epoch in range(start, c['epochs']):
        epoch_start = time.perf_counter()
        model.train()
        values, updates = [], 0
        loader = loader_for(train, c, epoch)
        batches = min(len(loader), c['steps_per_epoch'])
        optimizer.zero_grad(set_to_none=True)
        for step, (x, _) in enumerate(loader):
            if step >= batches:
                break
            x = x.to(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                reconstruction, quantization, perplexity = model(x)
                reconstruction_loss = F.l1_loss(reconstruction, x)
                loss = reconstruction_loss+quantization
                dloss = torch.zeros((), device=device)
                if discriminator is not None and dopt is not None and epoch >= c.get('gan_start_epoch', 1):
                    dopt.zero_grad(set_to_none=True)
                    dloss = (F.relu(1-discriminator(x)).mean()+F.relu(1+discriminator(reconstruction.detach())).mean())/2
                    if not torch.isfinite(dloss):
                        raise ValueError('Nonfinite discriminator loss')
                    scaler.scale(dloss).backward(); scaler.unscale_(dopt)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1., error_if_nonfinite=True)
                    scaler.step(dopt); scaler.update()
                    discriminator.requires_grad_(False)
                    loss = loss-c['gan_weight']*discriminator(reconstruction).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite VQ training loss')
            group_size = min(accumulation, batches-(step//accumulation)*accumulation)
            scaler.scale(loss/group_size).backward()
            if (step+1) % accumulation == 0 or step+1 == batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                updates += 1
            if discriminator is not None:
                discriminator.requires_grad_(True)
            values.append([loss.item(), reconstruction_loss.item(), quantization.item(), perplexity.item(), dloss.item()])
        model.eval()
        validation_loss = []
        with torch.no_grad():
            for x, _ in DataLoader(validation, batch_size=c['batch_size']):
                x = x.to(device)
                validation_loss.append(F.l1_loss(model(x)[0], x).item())
        means = np.mean(values, axis=0).tolist()
        record = dict(epoch=epoch+1, loss=means[0], reconstruction_l1=means[1], vq_loss=means[2],
            optimizer_steps=updates, micro_batches=len(values), learning_rate=optimizer.param_groups[0]['lr'],
            duration_seconds=time.perf_counter()-epoch_start,
            codebook_perplexity=means[3], discriminator_loss=means[4],
            validation_l1=float(np.mean(validation_loss)) if validation_loss else None)
        history.append(record)
        if scheduler:
            scheduler.step(record['validation_l1'] if validation_loss else record['reconstruction_l1'])
        save_checkpoint(out/'last.pt', dict(kind='vq', model=model.state_dict(), model_config=model.config,
            optimizer=optimizer.state_dict(), discriminator=discriminator.state_dict() if discriminator is not None else None,
            d_optimizer=dopt.state_dict() if dopt is not None else None, **rng_state(),
            scaler=scaler.state_dict(), scheduler=scheduler.state_dict() if scheduler else None,
            config=c, epoch=epoch+1, history=history, dataset_fingerprint=data_fingerprint,
            train_ids=train.ids, shape=c['shape'], smoke_test=c.get('smoke_test', False),
            cohort_records=records))
        write_json(out/'history.json', history)
        if validation_loss and record['validation_l1'] <= min(r['validation_l1'] for r in history):
            save_checkpoint(out/'best.pt', read_checkpoint(out/'last.pt', 'vq'))
        selected = 'best.pt' if c.get('select_best', False) and (out/'best.pt').exists() else 'last.pt'
        write_json(out/'run.json', dict(kind='vq', smoke_test=c.get('smoke_test', False),
            completed_epochs=epoch+1, train_count=len(train), validation_count=len(validation),
            data_root=reference(root, out), shape=c['shape'], dataset_fingerprint=data_fingerprint,
            checkpoint_file=selected, checkpoint_sha256=sha256_file(out/selected),
            validation_status='Internal validation; anatomical review pending'))
        print(f'VQ epoch {epoch+1}: train L1={record["reconstruction_l1"]:.5f}; '
              f'validation L1={record["validation_l1"]}; perplexity={record["codebook_perplexity"]:.2f}', flush=True)
        patience = c.get('early_stopping_patience', 0)
        if patience and validation_loss and len(history)-1-min(range(len(history)), key=lambda i: history[i]['validation_l1']) >= patience:
            break
    return str(out/('best.pt' if c.get('select_best', False) and (out/'best.pt').exists() else 'last.pt'))


@torch.no_grad()
def cache_latents(root, vq_path, output, device='auto'):
    device = device_for(device)
    model, ck = load_vq(vq_path, device)
    cohort = verify_cohort(root)
    if ck['dataset_fingerprint'] != cohort.fingerprint:
        raise ValueError('VQ checkpoint was trained on a different processed dataset')
    out = fresh_output(output)
    rows, train_tensors = [], []
    for split in ('train', 'validation', 'test'):
        dataset = TrainingDataset(root, split, ck['shape'])
        (out/split).mkdir()
        for i, row in enumerate(dataset.rows):
            if split == 'train' and row['scan_id'] not in ck['train_ids']:
                continue
            x, labels = dataset[i]
            latent = model.encode(x[None].to(device))[0][0].cpu()
            relative = f'{split}/{row["scan_id"]}.pt'
            torch.save(latent, out/relative)
            rows.append(dict(scan_id=row['scan_id'], split=split, file=relative, labels=labels.tolist(),
                             sha256=sha256_file(out/relative)))
            if split == 'train':
                train_tensors.append((float(latent.double().sum()), float(latent.double().square().sum()), latent.numel()))
    if not train_tensors:
        raise ValueError('No training latents')
    sums, squares, count = np.array(train_tensors).sum(0)
    mean, std = sums/count, max(squares/count-(sums/count)**2, 1e-12)**.5
    packed = pack_latents(out, rows)
    write_json(out/'manifest.json', dict(rows=rows, mean=float(mean), std=float(std), packed=packed,
        normalization_fit='train_only', latent_shape=list(latent.shape), vq_path=reference(vq_path, out),
        vq_sha256=checkpoint_hash(vq_path), shape=ck['shape'], train_ids=ck['train_ids'],
        smoke_test=ck['smoke_test'], dataset_fingerprint=ck['dataset_fingerprint'],
        cohort_records=cohort.records()))
    return str(out/'manifest.json')


def train_diffusion(latent_dir, output, c, resume=False):
    device = setup(c)
    latent_dir = Path(latent_dir)
    manifest = json.loads((latent_dir/'manifest.json').read_text())
    for row in manifest['rows']:
        if not row.get('sha256') or sha256_file(confined_path(latent_dir, row['file'])) != row['sha256']:
            raise ValueError('Latent cache integrity is missing or invalid; regenerate the cache')
    if not np.isfinite([manifest['mean'], manifest['std']]).all() or manifest['std'] <= 0:
        raise ValueError('Invalid latent normalization statistics')
    train = [r for r in manifest['rows'] if r['split'] == 'train']
    conditional = c.get('conditional', False)
    if conditional and any(len(r['labels']) != 3 or any(v not in (0, 1) for v in r['labels']) for r in manifest['rows']):
        raise ValueError('Conditional diffusion requires genuine MRNet label vectors')
    if not train:
        raise ValueError('Training latents are empty')
    if not c.get('smoke_test', False) and (manifest['smoke_test'] or not any(r['split'] == 'validation' for r in manifest['rows'])):
        raise ValueError('Research diffusion requires a research VQ checkpoint and validation latents')
    dropout = c.get('label_dropout', 0.)
    config = dict(channels=manifest['latent_shape'][0], base=c['unet_base'], label_dim=3 if conditional else 0,
                  attention=c.get('attention', False), attention_heads=c.get('attention_heads', 4),
                  channel_mult=c.get('channel_mult'), allow_null_labels=conditional and dropout > 0)
    model = UNet3D(**config).to(device)
    schedule = DDPM(c['diffusion_steps'], c.get('prediction_type', 'epsilon')).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'])
    ema = ModelEMA(model, c['ema_decay']) if c.get('ema_decay') else None
    amp, dtype, scaler = precision(c, device)
    scheduler = scheduler_for(optimizer, c)
    cache = LatentCache(latent_dir, manifest)
    out = fresh_output(output, resume)
    history, start = [], 0
    if resume:
        ck = read_checkpoint(out/'last.pt', 'diffusion')
        if ck.get('inference_only'):
            raise ValueError('Compact inference exports cannot resume training')
        old, new = dict(ck['config']), dict(c)
        old.pop('epochs'); new.pop('epochs')
        old_manifest, new_manifest = dict(ck['latent_manifest']), dict(manifest)
        old_manifest.pop('vq_path', None); new_manifest.pop('vq_path', None)
        if old != new or old_manifest != new_manifest:
            raise ValueError('Resume requires the same config and latent manifest')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        history, start = ck['history'], ck['epoch']
        if ema:
            ema.load_state_dict(ck['ema'], device)
        if ck.get('scaler'):
            scaler.load_state_dict(ck['scaler'])
        if scheduler and ck.get('scheduler'):
            scheduler.load_state_dict(ck['scheduler'])
        restore_rng(ck, device)
    for epoch in range(start, c['epochs']):
        epoch_start = time.perf_counter()
        model.train(); losses = []
        order = np.random.default_rng(c['seed']+epoch).permutation(len(train))
        for step in range(c['steps_per_epoch']):
            optimizer.zero_grad(set_to_none=True)
            accumulation = c.get('gradient_accumulation', 1)
            for micro in range(accumulation):
                batch_index = step*accumulation+micro
                batch = [train[int(order[(batch_index*c['batch_size']+j) % len(order)])] for j in range(c['batch_size'])]
                z = cache.batch(batch).to(device)
                z = (z-manifest['mean'])/manifest['std']
                labels = torch.tensor([r['labels'] for r in batch], device=device) if conditional else None
                mask = torch.rand(len(batch), device=device) >= dropout if conditional and dropout else None
                t = torch.randint(schedule.steps, (len(batch),), device=device)
                noise = torch.randn_like(z)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                    prediction = model(schedule.add_noise(z, t, noise), t, labels, condition_mask=mask)
                    loss = F.mse_loss(prediction, schedule.training_target(z, t, noise))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite diffusion loss')
                scaler.scale(loss/accumulation).backward()
                losses.append(loss.item())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update()
            if ema:
                ema.update(model)
        model.eval()
        validation_losses, ema_losses = [], []
        strata: list[list[float]] = [[] for _ in range(4)]
        ema_model = ema.evaluation_model(model) if ema else None
        validation_rng = torch.Generator(device=device).manual_seed(c['seed']+100000)
        with torch.no_grad():
            for row in manifest['rows']:
                if row['split'] != 'validation':
                    continue
                z = cache.tensor(row)[None].to(device)
                z = (z-manifest['mean'])/manifest['std']
                y = torch.tensor([row['labels']], device=device) if conditional else None
                draws = c.get('validation_draws', 1)
                if draws > schedule.steps:
                    raise ValueError('validation_draws cannot exceed diffusion_steps')
                for draw in range(draws):
                    t = torch.randint(draw*schedule.steps//draws, (draw+1)*schedule.steps//draws,
                                      (1,), generator=validation_rng, device=device)
                    noise = torch.randn(z.shape, generator=validation_rng, device=device)
                    noisy = schedule.add_noise(z, t, noise)
                    target = schedule.training_target(z, t, noise)
                    with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                        value = F.mse_loss(model(noisy, t, y), target).item()
                        if ema_model:
                            ema_losses.append(F.mse_loss(ema_model(noisy, t, y), target).item())
                    validation_losses.append(value)
                    strata[min(3, int(t)*4//schedule.steps)].append(value)
        selection = float(np.mean(ema_losses or validation_losses)) if validation_losses else float(np.mean(losses))
        history.append(dict(epoch=epoch+1, noise_mse=float(np.mean(losses)),
            optimizer_steps=c['steps_per_epoch'], micro_batches=len(losses), duration_seconds=time.perf_counter()-epoch_start,
            learning_rate=optimizer.param_groups[0]['lr'], selection_noise_mse=selection,
            validation_ema_mse=float(np.mean(ema_losses)) if ema_losses else None,
            validation_noise_by_quartile=[float(np.mean(v)) if v else None for v in strata],
            validation_noise_mse=float(np.mean(validation_losses)) if validation_losses else None))
        if scheduler:
            scheduler.step(selection)
        save_checkpoint(out/'last.pt', dict(kind='diffusion', model=model.state_dict(), model_config=config,
            optimizer=optimizer.state_dict(), **rng_state(), config=c, epoch=epoch+1,
            ema=ema.state_dict() if ema else None, ema_model=ema.shadow if ema else None,
            weights_for_inference='ema' if ema else 'raw', scaler=scaler.state_dict(),
            scheduler=scheduler.state_dict() if scheduler else None,
            history=history, latent_manifest=manifest, smoke_test=c.get('smoke_test', False) or manifest['smoke_test']))
        write_json(out/'history.json', history)
        if validation_losses and selection <= min(r.get('selection_noise_mse', r['validation_noise_mse']) for r in history):
            save_checkpoint(out/'best.pt', read_checkpoint(out/'last.pt', 'diffusion'))
        selected = 'best.pt' if c.get('select_best', False) and (out/'best.pt').exists() else 'last.pt'
        write_json(out/'run.json', dict(kind='diffusion', smoke_test=c.get('smoke_test', False),
            completed_epochs=epoch+1, checkpoint_file=selected, checkpoint_sha256=sha256_file(out/selected),
            train_count=len(train), validation_count=len(validation_losses)))
        print(f'Diffusion epoch {epoch+1}: MSE={history[-1]["noise_mse"]:.5f}', flush=True)
        patience = c.get('early_stopping_patience', 0)
        if patience and validation_losses and len(history)-1-min(range(len(history)), key=lambda i: history[i].get('selection_noise_mse', history[i]['validation_noise_mse'])) >= patience:
            break
    cache.close()
    return str(out/('best.pt' if c.get('select_best', False) and (out/'best.pt').exists() else 'last.pt'))


@torch.no_grad()
def generate(vq_path, diffusion_path, output, count=4, seed=42, labels=None, device='auto',
             sampler='ddpm', sampling_steps=None, guidance_scale=1., eta=0., weights='auto'):
    if count < 1:
        raise ValueError('count must be positive')
    set_seed(seed)
    device = device_for(device)
    vq, _ = load_vq(vq_path, device)
    ck = read_checkpoint(diffusion_path, 'diffusion')
    manifest = ck['latent_manifest']
    if manifest['vq_sha256'] != checkpoint_hash(vq_path):
        raise ValueError('Diffusion latents belong to a different VQ checkpoint')
    conditional = ck['model_config']['label_dim'] > 0
    if conditional and (labels is None or len(labels) != 3 or any(v not in (0, 1) for v in labels)):
        raise ValueError('Provide three binary conditioning labels: abnormal, acl, meniscus')
    if not conditional and labels is not None:
        raise ValueError('An unconditional model cannot generate labeled augmentation data')
    model = UNet3D(**ck['model_config']).to(device)
    selected_weights = ck.get('weights_for_inference', 'raw') if weights == 'auto' else weights
    if selected_weights not in ('raw', 'ema') or (selected_weights == 'ema' and not ck.get('ema_model')):
        raise ValueError('Requested inference weights are unavailable')
    model.load_state_dict(ck['ema_model'] if selected_weights == 'ema' else ck['model'])
    schedule = DDPM(ck['config']['diffusion_steps'], ck['config'].get('prediction_type', 'epsilon')).to(device)
    embeddings = vq.quantizer.embedding.weight
    lower = ((embeddings.amin(0)-manifest['mean'])/manifest['std']).view(1, -1, 1, 1, 1)
    upper = ((embeddings.amax(0)-manifest['mean'])/manifest['std']).view(1, -1, 1, 1, 1)
    out = fresh_output(output)
    rows = []
    for i in range(count):
        y = torch.tensor([labels], device=device) if conditional else None
        z, sampling_diagnostics = schedule.sample(model, (1, *manifest['latent_shape']), y, seed+i,
            clip_range=(lower, upper), method=sampler, sampling_steps=sampling_steps,
            guidance_scale=guidance_scale, eta=eta, return_diagnostics=True)
        z = z*manifest['std']+manifest['mean']
        if not torch.isfinite(z).all():
            raise ValueError('Denoising produced nonfinite latents; generation cannot be accepted')
        x = vq.decode(z, quantize=True)[0].cpu().numpy()
        quantized = vq.quantizer(z)[0]
        code_counts = torch.bincount(vq.quantizer.last_indices.flatten(), minlength=embeddings.shape[0]).cpu().tolist()
        if not np.isfinite(x).all():
            raise ValueError('Generated tensor is nonfinite')
        filename = f'synthetic_{i:04d}.npy'
        np.save(out/filename, x, allow_pickle=False)
        latent_file = f'latent_{i:04d}.npy'
        np.save(out/latent_file, z[0].cpu().numpy(), allow_pickle=False)
        rows.append(dict(file=filename, latent_file=latent_file, seed=seed+i, conditioning_labels=labels,
            sha256=sha256_file(out/filename), latent_sha256=sha256_file(out/latent_file),
            quantization_mse=float((z-quantized).square().mean()), code_counts=code_counts,
            sampling_diagnostics=sampling_diagnostics,
            latent_std=float(z.std()), latent_abs_max=float(z.abs().max())))
    write_json(out/'manifest.json', dict(rows=rows, synthetic=True, smoke_test=ck['smoke_test'],
        sampler='DDPM posterior with predicted-clean-latent clipping to trained codebook channel bounds',
        normalized_latent_bounds=dict(lower=lower.flatten().tolist(), upper=upper.flatten().tolist()),
        sampling=dict(method=sampler, sampling_steps=sampling_steps or schedule.steps,
                      training_steps=schedule.steps, prediction_type=schedule.prediction_type,
                      guidance_scale=guidance_scale, eta=eta, weights=selected_weights,
                      classifier_free_guidance_trained=ck['model_config'].get('allow_null_labels', False),
                      device=device.type, exported_weight_source=ck.get('exported_weight_source')),
        conditioning_training_support=sum(r['split'] == 'train' and r.get('labels') == labels for r in manifest['rows']) if labels is not None else None,
        vq_checkpoint=reference(vq_path, out), diffusion_checkpoint=reference(diffusion_path, out),
        label_semantics='Requested conditioning, not verified pathology',
        vq_sha256=checkpoint_hash(vq_path), diffusion_sha256=checkpoint_hash(diffusion_path),
        generator_train_ids=manifest['train_ids'], dataset_fingerprint=manifest['dataset_fingerprint'],
        cohort_records=manifest.get('cohort_records', []), shape=manifest['shape']))
    return str(out/'manifest.json')


@torch.no_grad()
def evaluate_reconstruction(root, checkpoint, output, split='validation', device='auto',
                            training_root=None, protocol=None):
    device = device_for(device)
    model, ck = load_vq(checkpoint, device)
    context = evaluation_context(root, ck, split, training_root)
    data = TrainingDataset(root, split, ck['shape'])
    if not len(data):
        raise ValueError(f'{split} is empty; no reconstruction results can be reported')
    out = validate_output(output)
    checkpoint_sha256 = sha256_file(checkpoint)
    with evaluation_attempt(protocol, context, split, out, checkpoint_sha256) as frozen:
        fresh_output(out)
        rows = []
        for i, row in enumerate(data.rows):
            x, _ = data[i]
            restored = model(x[None].to(device))[0][0].cpu().numpy()
            target = confined_path(out, f'{row["scan_id"]}.npy')
            np.save(target, restored)
            metrics = reconstruction_metrics(x[0].numpy(), restored[0], 2.)
            metrics = {k: v if np.isfinite(v) else None for k, v in metrics.items()}
            rows.append(dict(scan_id=row['scan_id'], patient_id=row.get('patient_id') or row['scan_id'],
                file=target.name, sha256=sha256_file(target), **metrics,
                **foreground_metrics(x[0].numpy(), restored[0])))
        if fingerprint(root) != context['evaluation_dataset_fingerprint']:
            raise ValueError('Evaluation dataset changed during inference')
        report = dict(split=split, smoke_test=ck['smoke_test'], rows=rows,
            input_dataset=reference(root, out), shape=ck['shape'], checkpoint_sha256=checkpoint_sha256,
            dataset_fingerprint=context['evaluation_dataset_fingerprint'], evaluation_context=context,
            confidence_interval_unit='patient' if all(r.get('patient_id') for r in data.rows) else 'exam; patient identities unavailable',
            confidence_intervals={k: bootstrap_interval([r[k] for r in rows if r[k] is not None],
                groups=[r['patient_id'] for r in rows if r[k] is not None])
                for k in ('mse', 'mae', 'psnr', 'ssim') if any(r[k] is not None for r in rows)},
            means={k: float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None
                   for k in ('mse', 'mae', 'psnr', 'ssim')},
            anatomical_preservation='Requires paired visual review; these metrics alone do not establish preservation')
        if frozen:
            report['frozen_protocol'] = frozen
        atomic_json(out/'report.json', report)
    return report
