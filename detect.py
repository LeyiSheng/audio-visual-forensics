import torch
import sys
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import json
import pandas as pd
from fake_celeb_dataset import FakeAVceleb
from model import MP_AViT, MP_av_feature_AViT
from subprocess import call
from backbone.select_backbone import select_backbone
from torch.optim import Adam
from config_deepfake import load_opts, save_opts
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data.sampler import RandomSampler, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Optional, Union
from audio_process import AudioEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import math
import h5py
import os
import glob
import time
from deep_fake_data import prepocess_video
import logging
from load_audio import wav2filterbanks, wave2input
from torch.utils.tensorboard import SummaryWriter
from transformer_component import transformer_decoder
import pickle
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger
from tqdm.contrib.logging import logging_redirect_tqdm
from emotion_module import EmotionModels, compute_emotion_inconsistency

opts = load_opts()
device = opts.device
#local_rank = opts.local_rank
#torch.cuda.set_device()
#torch.distributed.init_process_group(backend='nccl')
device = torch.device(device)

with open('pca.pkl', 'rb') as pickle_file:
    pca = pickle.load(pickle_file)

def _infer_label_from_path(p: str) -> Optional[int]:
    """Infer binary label from path tokens.
    Returns 1 for fake, 0 for real, or None if unknown.
    """
    s = str(p).lower()
    # Positive (fake) indicators
    pos_tokens = [
        'fake', 'deepfake', 'dfdc', 'forged', 'synthesis', 'synth',
        'manipulated', 'edited', 'swap', 'faceswap', 'reenact', 'ai_fake',
        'generated', 'gen', 'spoof'
    ]
    # Negative (real) indicators
    neg_tokens = [
        'real', 'genuine', 'authentic', 'original', 'pristine', 'live'
    ]
    pos = any(tok in s for tok in pos_tokens)
    neg = any(tok in s for tok in neg_tokens)
    if pos and not neg:
        return 1
    if neg and not pos:
        return 0
    return None

def get_logger(filename, verbosity=1, name=__name__):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Also log to stdout instead of the default stderr
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

# class Testset_new(Dataset):
#     def __init__(self, h5_file, fake_type, max_len, data_list):
#         super(Testset_new, self).__init__()
#         self.h5_file = h5_file
#         self.max_len = max_len
#         self.fake_type = fake_type
#         self.data_list = data_list
#     def __len__(self):
#         with h5py.File(self.h5_file, 'r') as f:
#             return len(f[self.fake_type])
#     def __getitem__(self, index):
#         with h5py.File(self.h5_file, 'r') as f:
#             data_total = f[self.fake_type][str(index)][:, :]
#             data_total = torch.from_numpy(data_total)
#             data = data_total
#             mask = torch.zeros(self.max_len)
#             if data.shape[0] < self.max_len:
#                 pad_len = self.max_len - data.shape[0]
#                 mask[data.shape[0]:] = 1.0
#                 data = F.pad(data, (0, 0, 0, pad_len))
#             else:
#                 start = 0
#                 data = data[start:start+self.max_len, :]
#                 pad_len = 0
#         return (data, mask, pad_len)

class network(nn.Module):
    def __init__(self, vis_enc, aud_enc, transformer):
        super().__init__()
        self.vis_enc = vis_enc
        self.aud_enc = aud_enc
        self.transformer = transformer

    def forward(self, video, audio, phase=0, train=True):
        if train:
            if phase == 0:
                vid_emb = self.vis_enc(video)
                batch_size,c,t, h, w= vid_emb.shape
                vid_emb = vid_emb[:, None]
                vid_emb = vid_emb.expand(-1, batch_size, -1, -1, -1, -1)
                vid_emb = vid_emb.reshape(-1, c, t, h, w)
                aud_emb = self.aud_enc(audio)
                batch_size, c_aud, t_aud = aud_emb.shape
                aud_emb = aud_emb[None, :]
                aud_emb = aud_emb.expand(batch_size, -1, -1, -1).reshape(-1, c_aud, t_aud)
                cls_emb = self.transformer(vid_emb, aud_emb)
            elif phase == 1:
                vid_emb = self.vis_enc(video)
                batch_size,c,t, h, w= vid_emb.shape
                vid_emb = vid_emb[:, None]
                vid_emb = vid_emb.expand(-1, opts.number_sample, -1, -1, -1, -1)
                vid_emb = vid_emb.reshape(-1, c, t, h, w)
                aud_emb = self.aud_enc(audio)
                batch_size, c_aud, t_aud = aud_emb.shape
                aud_emb_new = torch.zeros_like(aud_emb)
                aud_emb_new = aud_emb_new[None, :]
                aud_emb_new = aud_emb_new.expand(opts.number_sample, -1, -1, -1).reshape(-1, c_aud, t_aud)
                num_sample = opts.number_sample
                if batch_size == num_sample*(opts.bs2):
                    for k in range(opts.bs2):
                        aud_emb_new[k*num_sample*num_sample:(k+1)*num_sample*num_sample] = (aud_emb[k*num_sample:(k+1)*num_sample][None, :]).expand(opts.number_sample, -1, -1, -1).reshape(-1, c_aud, t_aud)
                else:
                    bs2 = int(batch_size / num_sample)
                    assert batch_size == bs2 * num_sample
                    for k in range(bs2):
                        aud_emb_new[k*num_sample*num_sample:(k+1)*num_sample*num_sample] = (aud_emb[k*num_sample:(k+1)*num_sample][None, :]).expand(opts.number_sample, -1, -1, -1).reshape(-1, c_aud, t_aud)
                aud_emb = aud_emb_new
                cls_emb = self.transformer(vid_emb, aud_emb)
            
        else:
            vid_emb = self.vis_enc(video)
            aud_emb = self.aud_enc(audio)
            cls_emb = self.transformer(vid_emb, aud_emb)
        return cls_emb

def test2(dist_model, avfeature_model, loader, dist_reg_model, avfeature_reg_model, max_len=50, emotion: Optional[EmotionModels] = None):
    output_dir = opts.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger_path = os.path.join(output_dir, 'output.log')
    score_file_path = os.path.join(output_dir, 'testing_scores.npy')
    logger = get_logger(logger_path)
    logger.info('Start testing!')
    score_list = []
    with logging_redirect_tqdm():
        with tqdm(total=len(loader), position=0, leave=False, colour='green', ncols=150, file=sys.stdout) as pbar:
            for nm, aud_vis in enumerate(loader):
                video_set = aud_vis['video']
                audio_set = aud_vis['audio']
                path_for_detect = aud_vis['sample']
                #print(path_for_detect)
                pbar.set_postfix(data_path = path_for_detect)
                time_len = video_set.shape[2]
                #predict_set = np.zeros((time_len - 5 + 1, 31))
                if (time_len -5 +1) < max_len:
                    max_seq_len = time_len - 5 + 1
                else:
                    max_seq_len = max_len
                predict_set = np.zeros((max_seq_len, 31))
                predict_set_avfeature = np.zeros((max_seq_len, 31))
                #real_result.append(1)
                #fake_result.append(1)
                #for k in tqdm(range(time_len - 5 + 1)):
                emo_accum = [] if (emotion is not None and emotion.ok and opts.emotion_weight > 0) else None
                for k in tqdm(range(max_seq_len), position=1, leave=False, colour='red', ncols=80, file=sys.stdout):
                    #video_set = torch.permute(video_set, (0, 2, 1, 3, 4))
                    video = video_set[:, :, k:k+5, :, :]
                    #video = video / 255.0
                    audio = audio_set[:, (k+15-15)*opts.aud_fact:(k+5+15+15)*opts.aud_fact]
                    '''
                    audio = aud_vis['audio'].to(device)
                    audio, _, _, _ = wav2filterbanks(audio)
                    audio = audio.permute([0, 2, 1])[:, None]
                    video = aud_vis['video'].to(device)
                    '''
                    dist_model.eval()
                    avfeature_model.eval()
                    with torch.no_grad():
                        batch_size = video.shape[0]
                        b, c ,t, h, w = video.shape
                        video = video[:, None]
                        video = video.repeat(1,31 , 1, 1, 1, 1).reshape(-1, c, t, h, w).to(device)
                        audio_list = []
                        for j in range(batch_size):
                            for i in range(31):
                                audio_list.append(audio[j:j+1, i*opts.aud_fact:(i+5)*opts.aud_fact])
                        audio = torch.cat(audio_list, dim=0).to(device)
                        #audio = wave2input(audio, device=device)
                        audio, _, _, _ = wav2filterbanks(audio.to(device), device=device)
                        audio = audio.permute([0, 2, 1])[:, None]
                        score = dist_model(video, audio, train=False)
                        avfeature = avfeature_model(video, audio, train=False)[15]
                        score = score.reshape(batch_size, 31)
                        avfeature = avfeature.cpu().numpy()[None, ...]
                        avfeature = pca.transform(avfeature)
                        # Optional: emotion inconsistency using pretrained models on center frame + audio slice
                        emo_kl = None
                        if emotion is not None and emotion.ok and opts.emotion_weight > 0:
                            try:
                                # center frame of the 5-frame clip, batch first sample
                                center_idx = min(t // 2, t - 1)
                                # original un-moved video_set is on CPU, use it for extracting frame
                                frame_np = video_set[0, :, k + center_idx, :, :].permute(1, 2, 0).cpu().numpy()
                                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                                # center aligned audio slice (15 offset)
                                aud_wave = audio_set[0, 15*opts.aud_fact:(15+5)*opts.aud_fact].cpu().numpy()
                                emo_kl = compute_emotion_inconsistency(emotion, frame_np, aud_wave, sr=opts.sample_rate)
                            except Exception as _:
                                emo_kl = None
                        #predict = torch.argmax(score, dim=1)
                        #distribution[0, predict.item()] += 1
                        #real_result[-1] = real_result[-1] * real_distribution[0, predict.item()]
                        #fake_result[-1] = fake_result[-1] * fake_distribution[0, predict.item()]
                        #predict = torch.abs(predict - 15)
                        #predict_set.append(predict.item())
                        predict = score.squeeze(0).cpu().numpy()
                        if emo_kl is not None and opts.emotion_weight > 0 and emo_accum is not None:
                            emo_accum.append(emo_kl)
                        if emo_kl is not None and opts.emotion_weight > 0:
                            # Increase distribution entropy if emotion is inconsistent by adding penalty to negative log-likelihood proxy
                            # Here we simply store the base distribution; emotion penalty is added later as a scalar to prob.
                            pass
                        predict_set[k] = predict
                        predict_set_avfeature[k] = avfeature
                    #print('--------------------computing score for video-------------------')
                dist_reg_model.eval()
                avfeature_reg_model.eval()
                mask = torch.zeros(max_len)
                predict_set = torch.from_numpy(predict_set)
                predict_set_avfeature = torch.from_numpy(predict_set_avfeature)
                criterion = nn.KLDivLoss(reduce=False)
                criterion_av_feature = nn.MSELoss(reduction="none")
                if predict_set.shape[0] < max_len:
                    pad_len = max_len - predict_set.shape[0]
                    mask[predict_set.shape[0]:] = 1.0
                    seq = F.pad(predict_set, (0, 0, 0, pad_len))
                    seq_avfeature = F.pad(predict_set_avfeature, (0, 0, 0, pad_len))
                else:
                    start = 0
                    seq = predict_set[start:start+max_len, :]
                    seq_avfeature = predict_set_avfeature[start:start+max_len, :]
                    pad_len = 0
                seq = seq[None, :, :]
                seq_avfeature = seq_avfeature[None, :, :]
                seq = seq.to(device)
                seq_avfeature = seq_avfeature.to(device)
                mask = mask.to(device)
                mask = mask[None, :]
                with torch.no_grad():
                    target = seq[:, 1:, :]
                    target = nn.functional.softmax(target.float(), dim=2)
                    target_av_feature = seq_avfeature[:, 1:, :]
                    target_av_feature = F.normalize(target_av_feature, p=2.0, dim=2)
                    input = seq[:, :-1, :]
                    input_av_feature = seq_avfeature[:, :-1, :]
                    input_av_feature = F.normalize(input_av_feature, p=2.0, dim=2)
                    input_mask_ = mask[:, :-1]
                    logit= dist_reg_model(input.float(), input_mask_)
                    logit_avfeature = avfeature_reg_model(input_av_feature.float(), input_mask_)
                    logit = nn.functional.log_softmax(logit, dim=2)
                    prob_total = criterion(logit, target)
                    prob_total_avfeature = criterion_av_feature(logit_avfeature, target_av_feature)
                    prob = prob_total[0, :(max_len - pad_len -1)]
                    prob_avfeature = prob_total_avfeature[0, :(max_len - pad_len -1)]
                    prob = torch.sum(prob, dim=1)
                    prob = torch.mean(prob)
                    prob_avfeature = torch.sum(prob_avfeature, dim=1)
                    prob_avfeature = torch.mean(prob_avfeature)
                    prob = (opts.lam)*prob_avfeature + prob
                    # Add global emotion inconsistency penalty averaged over windows (if enabled)
                    if 'emo_accum' in locals() and emo_accum:
                        emo_mean = float(np.mean(emo_accum))
                        prob = prob + opts.emotion_weight * torch.tensor(emo_mean, device=prob.device)
                #tqdm.write("The score of this video is {} ".format(prob.item()))
                logger.info("The score of this video is {} ".format(prob.item()))
                score_list.append(prob.item())
                pbar.update(1)
            np.save(score_file_path, np.array(score_list))
            logger.info('Finished!')
    return np.array(score_list)
            


def main():
    fake_distribution = np.zeros(31)
    vis_enc, _ = select_backbone(network='r18')
    aud_enc = AudioEncoder()
    #lrs2_distribution = np.zeros(31)
    Transformer = MP_AViT(image_size=14, patch_size=0, num_classes=1, dim=512, depth=3, heads=4, mlp_dim=512,  dim_head=128, dropout=0.1, emb_dropout=0.1, max_visual_len=5, max_audio_len=4)
    avfeature_Transformer = MP_av_feature_AViT(image_size=14, patch_size=0, num_classes=1, dim=512, depth=3, heads=4, mlp_dim=512,  dim_head=128, dropout=0.1, emb_dropout=0.1, max_visual_len=5, max_audio_len=4)
    sync_model = network(vis_enc=vis_enc, aud_enc=aud_enc, transformer=Transformer)
    avfeature_sync_model = network(vis_enc=vis_enc, aud_enc=aud_enc, transformer=avfeature_Transformer)
    #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Expect sync_model.pth to be downloaded as per README
    sync_model_weight = torch.load('sync_model.pth', map_location=device)
    sync_model.load_state_dict(sync_model_weight)
    avfeature_sync_model.load_state_dict(sync_model_weight)
    sync_model.to(device)
    avfeature_sync_model.to(device)
    #model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    dist_regressive_model = transformer_decoder(input_dim_old=31, input_dim=256, compress_factor=1, num_heads=16, dropout_prob=0.1, max_len=49, layers=2)
    avfeature_regressive_model = transformer_decoder(input_dim_old=31, input_dim=256, compress_factor=1, num_heads=16, dropout_prob=0.1, max_len=49, layers=2)
    reg_model_weight = torch.load('dist_regressive_model.pth', map_location=device)
    avfeature_reg_model_weight = torch.load('avfeature_regressive_model.pth', map_location=device)
    dist_regressive_model.load_state_dict(reg_model_weight)
    dist_regressive_model.to(device)
    avfeature_regressive_model.load_state_dict(avfeature_reg_model_weight)
    avfeature_regressive_model.to(device)
    # optional: emotion models
    emotion = None
    if getattr(opts, 'use_emotion', False) and opts.emotion_weight > 0:
        emotion = EmotionModels(
            visual_model_name=opts.emotion_visual_model,
            audio_model_name=opts.emotion_audio_model,
            device=device,
            local_files_only=True,
        )

    labels = None  # binary labels for evaluation
    label_mode = 'binary'  # 'binary' or 'both_fake_positive'
    paths_for_dataset = None
    if opts.test_video_path is not None:
        if opts.test_video_path.split('.')[-1].lower() == 'mp4':
            paths_for_dataset = [opts.test_video_path]
        elif opts.test_video_path.split('.')[-1].lower() == 'csv':
            # CSV support: expect a column for paths and optionally labels
            df = pd.read_csv(opts.test_video_path)
            # guess path column
            path_cols = [c for c in df.columns if str(c).lower() in ['path','video_path','video','filepath','file','mp4','mp4_path']]
            if len(path_cols) == 0:
                # fallback to first column
                path_col = df.columns[0]
            else:
                path_col = path_cols[0]
            paths_for_dataset = df[path_col].astype(str).tolist()
            # try dual-label columns for both-fake logic
            dual_pairs = [
                ('v_fake','a_fake'),
                ('video_fake','audio_fake'),
                ('fake_video','fake_audio'),
                ('v_label','a_label'),
            ]
            found_dual = None
            lower_cols = {str(c).lower(): c for c in df.columns}
            for a,b in dual_pairs:
                if a in lower_cols and b in lower_cols:
                    found_dual = (lower_cols[a], lower_cols[b])
                    break
            if found_dual is not None:
                va = df[found_dual[0]].astype(float).fillna(0).astype(int)
                aa = df[found_dual[1]].astype(float).fillna(0).astype(int)
                labels = ((va==1) & (aa==1)).astype(int).to_numpy()
                label_mode = 'both_fake_positive'
            else:
                # single label columns
                cand = None
                for name in ['label','y','target','fake','is_fake','both_fake']:
                    if name in lower_cols:
                        cand = lower_cols[name]
                        break
                if cand is not None:
                    ser = df[cand]
                    # map common string tokens
                    def map_token(v):
                        try:
                            f = float(v)
                            return 1 if int(f)==1 else 0
                        except Exception:
                            s = str(v).strip().lower()
                            if s in ['1','true','t','yes','y','fake','pos','positive','both','bothfake','both_fake','av','va']:
                                return 1
                            if s in ['0','false','f','no','n','real','neg','negative']:
                                return 0
                            return None
                    mapped = ser.map(map_token)
                    if mapped.notna().all():
                        labels = mapped.astype(int).to_numpy()
                # else: leave labels=None
        else:
            with open(opts.test_video_path) as file:
                raw_lines = [l.strip() for l in file.readlines() if len(l.strip()) > 0]
            paths_for_dataset = []
            parsed_labels = []
            for line in raw_lines:
                parts = line.split()
                # Default: whole line is path
                path = line
                y = None
                # Try to parse from the end: support two-label format "<path> <v_fake> <a_fake>"
                if len(parts) >= 3:
                    last2 = parts[-2:]
                    try:
                        v_fake = int(float(last2[0]))
                        a_fake = int(float(last2[1]))
                        if v_fake in (0,1) and a_fake in (0,1):
                            label_mode = 'both_fake_positive'
                            y = 1 if (v_fake == 1 and a_fake == 1) else 0
                            path = ' '.join(parts[:-2])
                    except Exception:
                        pass
                # If not two-label, try single label at end
                if y is None and len(parts) >= 2:
                    lab_token = parts[-1]
                    # Recognize tokens
                    tok = str(lab_token).lower()
                    if tok in ['1','0']:
                        y = int(tok)
                        path = ' '.join(parts[:-1])
                    elif tok in ['fake','pos','positive','true']:
                        y = 1
                        path = ' '.join(parts[:-1])
                    elif tok in ['real','neg','negative','false']:
                        y = 0
                        path = ' '.join(parts[:-1])
                    elif tok in ['both','bothfake','av','va','both_fake']:
                        # mark as positive only when both modalities are fake
                        label_mode = 'both_fake_positive'
                        y = 1
                        path = ' '.join(parts[:-1])
                paths_for_dataset.append(path)
                parsed_labels.append(y)
            # if we got labels for all samples, use them
            if len(parsed_labels) == len(paths_for_dataset) and len(parsed_labels) > 0 and all(l is not None for l in parsed_labels):
                labels = np.array(parsed_labels, dtype=int)
        # If labels not provided, try inferring from path tokens
        if labels is None and paths_for_dataset is not None and len(paths_for_dataset) > 0:
            inferred = [_infer_label_from_path(p) for p in paths_for_dataset]
            if all(l is not None for l in inferred):
                labels = np.array(inferred, dtype=int)
                label_mode = 'binary'
        test_video = FakeAVceleb(paths_for_dataset, opts.resize, opts.fps, opts.sample_rate, vid_len=opts.vid_len, phase=0, train=False, number_sample=1, lrs2=False, need_shift=False, lrs3=False, kodf=False, lavdf=False, robustness=False, test=True)
    loader_test = DataLoader(test_video, batch_size=opts.bs, num_workers=opts.n_workers, shuffle=False)
    scores = test2(sync_model, avfeature_sync_model, loader_test, dist_reg_model=dist_regressive_model, avfeature_reg_model=avfeature_regressive_model, max_len=opts.max_len, emotion=emotion)

    # If labels are provided, compute AUC and ACC (1=fake, higher score => more fake)
    if labels is not None and len(labels) == len(scores):
        output_dir = opts.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger_path = os.path.join(output_dir, 'output.log')
        met_path = os.path.join(output_dir, 'metrics.json')
        logger = get_logger(logger_path)
        try:
            auc = float(roc_auc_score(labels, scores))
        except Exception:
            auc = None
        # choose threshold maximizing accuracy over ROC thresholds
        fpr, tpr, thr = roc_curve(labels, scores)
        acc_best, thr_best = None, None
        if thr is not None and len(thr) > 0:
            accs = []
            for th in thr:
                if np.isfinite(th):
                    pred = (scores >= th).astype(int)
                    accs.append(accuracy_score(labels, pred))
                else:
                    accs.append(-1.0)
            best_idx = int(np.argmax(accs)) if len(accs) > 0 else 0
            acc_best = float(accs[best_idx]) if len(accs) > 0 else None
            thr_best = float(thr[best_idx]) if len(thr) > 0 else None
        metrics = {
            'num_samples': int(len(scores)),
            'auc': auc,
            'acc_best': acc_best,
            'thr_best': thr_best,
            'label_mode': label_mode,
            'pos_count': int(labels.sum()),
        }
        with open(met_path, 'w') as fw:
            json.dump(metrics, fw, indent=2)
        if label_mode == 'both_fake_positive':
            logger.info(f"[both-fake positive] AUC: {auc}, ACC(best): {acc_best} @ thr={thr_best}")
        else:
            logger.info(f"AUC: {auc}, ACC(best): {acc_best} @ thr={thr_best}")



if __name__ == '__main__':
    main()
