import numpy as np
import time
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from src.preprocessing.typeformer_preprocess import typeformer_preprocess
from src.preprocessing.kdprint_preprocess import KDPrintPreprocessor
from src.model.typeformer_wrapper import TypeFormerWrapper
from src.evaluation.metrics import compute_user_metrics, aggregate_results
from src.evaluation.global_threshold import GlobalThresholdEstimator
from src.evaluation.adaptive_threshold import AdaptiveThresholdEstimator

def _collect_val_scores(val_users: List[np.ndarray], model: TypeFormerWrapper, 
                        E: int, use_kdprint: bool, buffer_size: int, L: int):
    all_gen, all_imp = [], []
    print("Collecting validation scores for global threshold fitting...")
    
    for i, user_sessions in enumerate(tqdm(val_users)):
        enrol_raw = user_sessions[:E]
        verify_raw = user_sessions[E:E+5]
        
        # Taking 1 impostor session from other validation users
        impostor_raw = [val_users[j][0] for j in range(len(val_users)) if j != i]
        
        if use_kdprint:
            prep = KDPrintPreprocessor(buffer_size=buffer_size, seq_len=L)
            enrol_proc = prep.fit_transform(enrol_raw, use_buffer=True)
            verify_proc = [prep.transform(s, use_buffer=False) for s in verify_raw]
            impostor_proc = [prep.transform(s, use_buffer=False) for s in impostor_raw]
        else:
            enrol_proc = typeformer_preprocess(enrol_raw, seq_len=L)
            verify_proc = typeformer_preprocess(verify_raw, seq_len=L)
            impostor_proc = typeformer_preprocess(impostor_raw, seq_len=L)
            
        z_enrol = model.extract_embeddings(enrol_proc)
        z_verify = model.extract_embeddings(verify_proc)
        z_impostor = model.extract_embeddings(impostor_proc)
        
        z_mean = z_enrol.mean(axis=0)
        
        all_gen.extend([np.linalg.norm(z - z_mean) for z in z_verify])
        all_imp.extend([np.linalg.norm(z - z_mean) for z in z_impostor])
        
    return all_gen, all_imp

def _extract_val_embeddings(val_users: List[np.ndarray], model: TypeFormerWrapper, 
                            E: int, use_kdprint: bool, buffer_size: int, L: int) -> Dict:
    """Extract embeddings for all validation users (for k optimization in adaptive threshold)."""
    val_data = {}
    print("Extracting validation embeddings for adaptive threshold k optimization...")
    
    for i, user_sessions in enumerate(tqdm(val_users)):
        enrol_raw = user_sessions[:E]
        verify_raw = user_sessions[E:E+5]
        
        impostor_raw = [val_users[j][0] for j in range(len(val_users)) if j != i]
        
        if use_kdprint:
            prep = KDPrintPreprocessor(buffer_size=buffer_size, seq_len=L)
            enrol_proc = prep.fit_transform(enrol_raw, use_buffer=True)
            verify_proc = [prep.transform(s, use_buffer=False) for s in verify_raw]
            impostor_proc = [prep.transform(s, use_buffer=False) for s in impostor_raw]
        else:
            enrol_proc = typeformer_preprocess(enrol_raw, seq_len=L)
            verify_proc = typeformer_preprocess(verify_raw, seq_len=L)
            impostor_proc = typeformer_preprocess(impostor_raw, seq_len=L)
            
        z_enrol = model.extract_embeddings(enrol_proc)
        z_verify = model.extract_embeddings(verify_proc)
        z_impostor = model.extract_embeddings(impostor_proc)
        
        val_data[str(i)] = {
            'enrol': z_enrol,
            'genuine': z_verify,
            'impostor': z_impostor
        }
    return val_data


def run_single_experiment(
    config_name: str,
    model: TypeFormerWrapper,
    eval_users: List[np.ndarray],
    val_users: List[np.ndarray],
    E: int,
    L: int = 50,
    use_kdprint: bool = False,
    use_adaptive: bool = False,
    buffer_size: int = 5,
    output_dir: str = 'results/',
    verbose: bool = True
) -> Dict:
    start_time = time.time()
    output_path = Path(output_dir) / f"{config_name}_E{E}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {config_name} | E={E} | L={L} | KDPrint={use_kdprint} | Adaptive={use_adaptive}")
        print(f"{'='*60}")
        
    # [1] Fit threshold on validation set
    global_T = None
    if use_adaptive:
        threshold_estimator = AdaptiveThresholdEstimator(k=2.0)
        val_data = _extract_val_embeddings(val_users, model, E, use_kdprint, buffer_size, L)
        best_k, val_eer = threshold_estimator.optimize_k(val_data)
        if verbose:
            print(f"  Adaptive threshold fitted: optimal k = {best_k:.2f}, Val EER = {val_eer*100:.4f}%")
    else:
        threshold_estimator = GlobalThresholdEstimator()
        all_gen, all_imp = _collect_val_scores(val_users, model, E, use_kdprint, buffer_size, L)
        global_T = threshold_estimator.fit(all_gen, all_imp)
        if verbose:
            print(f"  Global threshold fitted = {global_T:.4f}")
        
    # [2] Evaluate on test set
    if verbose:
        print(f"\nEvaluating on {len(eval_users)} test users...")
        
    all_user_metrics = []
    
    for i, user_sessions in enumerate(tqdm(eval_users, disable=not verbose)):
        enrol_raw = user_sessions[:E]
        verify_raw = user_sessions[E:E+5]
        
        impostor_raw = [eval_users[j][0] for j in range(len(eval_users)) if j != i][:999]  # 999 impostors max
        
        if use_kdprint:
            prep = KDPrintPreprocessor(buffer_size=buffer_size, seq_len=L)
            enrol_proc = prep.fit_transform(enrol_raw, use_buffer=True)
            verify_proc = [prep.transform(s, use_buffer=False) for s in verify_raw]
            impostor_proc = [prep.transform(s, use_buffer=False) for s in impostor_raw]
        else:
            enrol_proc = typeformer_preprocess(enrol_raw, seq_len=L)
            verify_proc = typeformer_preprocess(verify_raw, seq_len=L)
            impostor_proc = typeformer_preprocess(impostor_raw, seq_len=L)
            
        z_enrol = model.extract_embeddings(enrol_proc)
        z_verify = model.extract_embeddings(verify_proc)
        z_impostor = model.extract_embeddings(impostor_proc)
        
        z_mean = z_enrol.mean(axis=0)
        
        genuine_scores = [float(np.linalg.norm(z - z_mean)) for z in z_verify]
        impostor_scores = [float(np.linalg.norm(z - z_mean)) for z in z_impostor]
        
        metrics = compute_user_metrics(genuine_scores, impostor_scores)
        metrics['user_idx'] = i
        
        if use_adaptive:
            template = threshold_estimator.estimate_user_threshold(z_enrol)
            metrics['T_user'] = float(template['T_user'])
            metrics['mu_user'] = float(template['mu_user'])
            metrics['sigma_user'] = float(template['sigma_user'])
        else:
            metrics['T_global'] = float(global_T)
        
        all_user_metrics.append(metrics)
        
    # [3] Aggregate and save
    aggregated = aggregate_results(all_user_metrics)
    aggregated['config'] = config_name
    aggregated['E'] = E
    aggregated['runtime_seconds'] = time.time() - start_time
    
    if use_adaptive:
        aggregated['adaptive_k'] = float(threshold_estimator.k)
    else:
        aggregated['global_threshold'] = float(global_T)
    
    with open(output_path / 'per_user_metrics.json', 'w') as f:
        json.dump(all_user_metrics, f, indent=2)
        
    with open(output_path / 'aggregated.json', 'w') as f:
        json.dump(aggregated, f, indent=2)
        
    if verbose:
        print(f"\nResults for {config_name} E={E}:")
        print(f"  Average EER: {aggregated['avg_eer']*100:.4f}%")
        print(f"  Runtime: {aggregated['runtime_seconds']:.1f}s")
        
    return aggregated
