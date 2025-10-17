import argparse
import configargparse


def save_opts(args, fn):
    with open(fn, 'w') as fw:
        for items in vars(args):
            fw.write('%s %s\n' % (items, vars(args)[items]))


def load_opts():
    parser = argparse.ArgumentParser()

    # --- general

    parser.add_argument('--output_dir',
                        type=str,
                        default="./save",
                        help='Path for saving results')

    parser.add_argument('--n_workers',
                        type=int,
                        default=0,
                        help='Num data workers')

    parser.add_argument('--bs', type=int, default=1, help='Batch size')
    
    parser.add_argument('--bs2', type=int, default=1, help='this is a hyperparameter during training')

                        
    parser.add_argument('--vid_len',
                        type=int,
                        default=5,
                        help='length of video in frames')

    parser.add_argument('--aud_fact',
                        type=int,
                        default=640,
                        help='the value of sample rate of audio divided by sample rate of video')


    # --- video
    parser.add_argument('--resize',
                        default=224,
                        type=int,
                        help='Scale input video to that resolution')
    parser.add_argument('--fps', type=int, default=25, help='Video input fps')

    # --- audio
    parser.add_argument('--sample_rate', type=int, default=16000, help='')



    # -- distributed
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--world_size', type=int,default=8)
    parser.add_argument('--epochs_0', type=int,default=50)
    parser.add_argument('--epochs_1', type=int,default=90)
    # -- deepfake
    parser.add_argument('--test_video_path', type=str, default='', help='Testing video full path')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--lam", type=float, default=0)

    # -- emotion consistency (optional)
    parser.add_argument('--use_emotion', action='store_true', help='Enable audio-visual emotion consistency scoring')
    parser.add_argument('--emotion_weight', type=float, default=0.0, help='Weight for emotion inconsistency term')
    parser.add_argument('--emotion_visual_model', type=str, default='MahmoudWSegni/swin-tiny-patch4-window7-224-finetuned-face-emotion-v12', help='HF repo id for visual emotion model')
    parser.add_argument('--emotion_audio_model', type=str, default='audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim', help='HF repo id for audio emotion model')
    args = parser.parse_args()
    return args
