# ============================================================
#  SECTION 14: UNIFIED TRAINING LOOP
#  Works for CRNN, AST, and HuBERT with identical WandB metrics
# ============================================================
 
def _prepare_input(x, model_name):
    """
    Squeeze the channel/batch dimension so each model gets
    the shape its forward() expects.
 
    CRNN    : (B, 1, 128, 500) → no change needed
    AST     : (B, 1, 1024, 128) → squeeze → (B, 1024, 128)
    HuBERT  : (B, 1, 160000) → squeeze → (B, 160000)
    """
    if model_name in ("AST", "HuBERT"):
        return x.squeeze(1)
    return x
 
 
def _forward(model, x, model_name):
    """
    Unified forward pass — handles different output structures.
    AST returns a SequenceClassifierOutput; CRNN/HuBERT return tensors.
    """
    out = model(x) if model_name != "AST" else model(input_values=x)
    return out.logits if hasattr(out, "logits") else out
 
 
def train_one_epoch(model, loader, optimizer, scheduler,
                    criterion, scaler, device,
                    model_name, epoch, accum_steps=1):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    optimizer.zero_grad()
 
    pbar = tqdm(loader, desc=f"[{model_name}] Ep{epoch:02d} TRAIN", leave=False)
 
    for step, (x, labels) in enumerate(pbar):
        x      = _prepare_input(x, model_name).to(device)
        labels = labels.to(device)
 
        with autocast():
            logits = _forward(model, x, model_name)
            loss   = criterion(logits, labels) / accum_steps
 
        scaler.scale(loss).backward()
 
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
 
        preds = logits.detach().argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=f"{loss.item()*accum_steps:.4f}")
 
    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, macro_f1, accuracy
 
 
@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, model_name, epoch):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
 
    pbar = tqdm(loader, desc=f"[{model_name}] Ep{epoch:02d} VAL  ", leave=False)
 
    for x, labels in pbar:
        x      = _prepare_input(x, model_name).to(device)
        labels = labels.to(device)
        with autocast():
            logits = _forward(model, x, model_name)
            loss   = criterion(logits, labels)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        total_loss += loss.item()
 
    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    accuracy = accuracy_score(all_labels, all_preds)
 
    # Per-class breakdown for diagnosis
    per_class_f1 = f1_score(all_labels, all_preds, average=None, labels=list(range(10)))
    print(f"\n   Per-class F1 [{model_name}]:")
    for i, v in enumerate(per_class_f1):
        bar = "█" * int(v * 20)
        print(f"   {CFG.IDX2GENRE[i]:12s} {bar:<20s} {v:.3f}")
 
    return avg_loss, macro_f1, accuracy, per_class_f1
 
 
def run_two_phase_training(
        model, model_name,
        train_loader, val_loader,
        phase1_epochs, phase2_epochs,
        phase1_lr, phase2_lr,
        save_path, kaggle_handle,
        noise_mixer,
        accum_steps=1,
        phase2_unfreeze_fn=None,
        class_weights  = None, 
        phase2_optimizer_fn=None):
    """
    Generic 2-phase trainer used for all 3 models.
 
    Phase 1: Frozen backbone (or full CRNN) — high LR, fast head training
    Phase 2: Full fine-tuning — low LR, gradient accumulation
 
    All WandB keys are IDENTICAL across models so runs can be compared
    side-by-side in WandB's grouping/comparison UI.
    """
    device    = CFG.DEVICE
    criterion = nn.CrossEntropyLoss(weight = class_weights,
                                    label_smoothing=0.1)
    scaler    = GradScaler()
    best_f1   = 0.0
    best_state = None
 
    total_epochs = phase1_epochs + phase2_epochs

    
    wandb.init(
        project = "23f3001800-t12026",
        name    = f"{model_name}-run",
        config  = {
            "model_name"    : model_name,
            "phase1_epochs" : phase1_epochs,
            "phase2_epochs" : phase2_epochs,
            "phase1_lr"     : phase1_lr,
            "phase2_lr"     : phase2_lr,
            "accum_steps"   : accum_steps,
            "sample_rate"   : CFG.SAMPLE_RATE,
            "clip_duration" : CFG.CLIP_DURATION,
            "augmentations" : "stem_mix+time_stretch+pitch_shift+"
                              "esc50_noise+specaugment",
        }
    )
 
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = GradScaler()
    best_f1   = 0.0
 
    # ╔══════════════════════════════════════════╗
    # ║ PHASE 1 — Head training (or full CRNN)  ║
    # ╚══════════════════════════════════════════╝
    print(f"\n{'='*55}")
    print(f"  [{model_name}] PHASE 1 — {phase1_epochs} epochs")
    print(f"{'='*55}")
 
    optimizer_p1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=phase1_lr, weight_decay=0.01
    )
    total_steps_p1 = phase1_epochs * len(train_loader)
    scheduler_p1   = CosineAnnealingLR(optimizer_p1,
                                        T_max=total_steps_p1, eta_min=1e-6)
 
    wandb.watch(model, log="gradients", log_freq=100)
    best_state = None
 
    for epoch in range(1, phase1_epochs + 1):
        tr_loss, tr_f1, tr_acc = train_one_epoch(
            model, train_loader, optimizer_p1, scheduler_p1,
            criterion, scaler, device, model_name, epoch
        )
        val_loss, val_f1, val_acc, per_cls = validate_one_epoch(
            model, val_loader, criterion, device, model_name, epoch
        )
 
        # ── Identical WandB keys for all 3 models ─────────
        wandb.log({
            "model"       : model_name,
            "epoch"       : epoch,
            "phase"       : 1,
            "train_loss"  : tr_loss,
            "train_f1"    : tr_f1,
            "train_acc"   : tr_acc,
            "val_loss"    : val_loss,
            "val_f1"      : val_f1,
            "val_acc"     : val_acc,
            "overfit_gap" : tr_f1 - val_f1,
            **{f"val_f1_{CFG.IDX2GENRE[i]}": v for i, v in enumerate(per_cls)}
        })
 
        print(f"\n  [{model_name}] Ph1 Ep{epoch:02d} | "
              f"TR f1={tr_f1:.4f} acc={tr_acc:.4f} | "
              f"VAL f1={val_f1:.4f} acc={val_acc:.4f} | "
              f"gap={tr_f1-val_f1:.4f}")
 
        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            print(f"  💾 Best Phase 1 | val_f1={best_f1:.4f}")
            wandb.run.summary["best_ph1_val_f1"] = best_f1
 
    # ╔══════════════════════════════════════════╗
    # ║ PHASE 2 — Full fine-tuning              ║
    # ╚══════════════════════════════════════════╝
    print(f"\n{'='*55}")
    print(f"  [{model_name}] PHASE 2 — {phase2_epochs} epochs")
    print(f"{'='*55}")
 
    # Unfreeze backbone if provided
    if phase2_unfreeze_fn is not None:
        phase2_unfreeze_fn(model)
    else:
        for p in model.parameters():
            p.requires_grad = True
 
    # Load best Phase 1 weights as starting point
    model.load_state_dict(best_state)
    model = model.to(device)
    print(f"  ✅ Loaded best Ph1 weights (f1={best_f1:.4f})")
 
    # Use LLRD optimizer if provided, else standard AdamW
    if phase2_optimizer_fn is not None:
        optimizer_p2 = phase2_optimizer_fn(model, base_lr=phase2_lr)
    else:
        optimizer_p2 = torch.optim.AdamW(
            model.parameters(), lr=phase2_lr, weight_decay=0.01
        )
 
    total_steps_p2 = phase2_epochs * len(train_loader)
    scheduler_p2   = CosineAnnealingLR(optimizer_p2,
                                        T_max=total_steps_p2, eta_min=1e-7)
 
    no_improve = 0
    EARLY_STOP_PAT = 7
 
    for epoch in range(1, phase2_epochs + 1):
        
        noise_mixer.step_epoch(
            current_epoch = phase1_epochs + epoch,
            total_epochs  = total_epochs
        )
        
        tr_loss, tr_f1, tr_acc = train_one_epoch(
            model, train_loader, optimizer_p2, scheduler_p2,
            criterion, scaler, device, model_name, epoch,
            accum_steps=accum_steps
        )
        val_loss, val_f1, val_acc, per_cls = validate_one_epoch(
            model, val_loader, criterion, device, model_name, epoch
        )
 
        wandb.log({
            "model"       : model_name,
            "epoch"       : phase1_epochs + epoch,
            "phase"       : 2,
            "train_loss"  : tr_loss,
            "train_f1"    : tr_f1,
            "train_acc"   : tr_acc,
            "val_loss"    : val_loss,
            "val_f1"      : val_f1,
            "val_acc"     : val_acc,
            "overfit_gap" : tr_f1 - val_f1,
            **{f"val_f1_{CFG.IDX2GENRE[i]}": v for i, v in enumerate(per_cls)}
        })
 
        print(f"\n  [{model_name}] Ph2 Ep{epoch:02d} | "
              f"TR f1={tr_f1:.4f} | "
              f"VAL f1={val_f1:.4f} | "
              f"gap={tr_f1-val_f1:.4f}")
 
        if tr_f1 - val_f1 > 0.15:
            print(f"  ⚠️  OVERFITTING DETECTED — gap={tr_f1-val_f1:.3f}")
            wandb.alert(title="Overfitting",
                        text=f"[{model_name}] Ep{epoch}: gap={tr_f1-val_f1:.3f}")
 
        if val_f1 > best_f1:
            best_f1    = val_f1
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            wandb.run.summary["best_val_f1"]  = best_f1
            wandb.run.summary["best_val_acc"] = val_acc
            print(f"  💾 Best model saved! val_f1={best_f1:.4f}")
        else:
            no_improve += 1
            print(f"  ⏳ No improvement {no_improve}/{EARLY_STOP_PAT}")
            if no_improve >= EARLY_STOP_PAT:
                print(f"  🛑 Early stopping at epoch {epoch}")
                break
 
    # Reload best weights
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"\n🏆 [{model_name}] Training complete | Best val F1: {best_f1:.4f}")
 
    # Upload to KaggleHub
    save_dir = str(Path(save_path).parent)
    print(f"📤 Uploading {model_name} → {kaggle_handle}")
    kagglehub.model_upload(handle=kaggle_handle, local_model_dir=save_dir)
    print(f"✅ Uploaded → {kaggle_handle}")
 
    wandb.finish()
    return model, best_f1
 
