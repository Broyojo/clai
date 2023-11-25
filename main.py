import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        prog="clai",
        description="Command Line AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    get_global_args(parser)

    tasks_subparser = parser.add_subparsers(title="tasks", dest="task")

    get_transcribe_args(tasks_subparser)

    return parser.parse_args()

def get_global_args(parser):
    # Model
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=False,
        help="Name or path of model to use"
    )
    
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        required=False,
        help="Use less CPU memory",
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        required=False,
        help="Use GPU",
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        required=False,
        help="Use CPU",
    )
    
def get_transcribe_args(tasks_subparser):
    transcribe_parser = tasks_subparser.add_parser(
        name="transcribe",
        description="Transcribe audio to text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General
    
    transcribe_parser.add_argument(
        "input",
        type=str,
        help="Input audio path",
    )
    
    transcribe_parser.add_argument(
        "output",
        type=str,
        help="Output text path",
    )
        
    transcribe_parser.add_argument(
        "--chunk_batch_size",
        required=False,
        type=int,
        default=24,
        help="Size of parallel audio chunk batches you want to compute",
    )
    
    transcribe_parser.add_argument(
        "--chunk_length",
        required=False,
        type=float,
        default=30.0,
        help="Length of each chunk in seconds",
    )
    
    transcribe_parser.add_argument(
        "--timestamps",
        required=False,
        type=str,
        default=False,
        choices=["chunk", "word"],
        help="What kind of timestamps to use",
    )
    
    transcribe_parser.add_argument(
        "--language",
        required=False,
        type=str,
        help="Language of the audio",
    )
    
    transcribe_parser.add_argument(
        "--translate",
        required=False,
        type=str,
        help="Language to translate the audio to",
    )
    
    # Diarization
    
    diarize_group = transcribe_parser.add_argument_group("Diarization Options")

    diarize_group.add_argument(
        "--diarize",
        action="store_true",
        required=False,
        help="Diarize the audio",
    )

    diarize_group.add_argument(
        "--num_speakers",
        type=int,
        required=False,
        help="Number of speakers to diarize",
    )
    
    diarize_group.add_argument(
        "--min_speakers",
        type=int,
        required=False,
        help="Minimum number of speakers to diarize",
    )
    
    diarize_group.add_argument(
        "--max_speakers",
        type=int,
        required=False,
        help="Maximum number of speakers to diarize",
    )
    
    # VAD
    
    vad_group = transcribe_parser.add_argument_group("VAD Options")
    
    vad_group.add_argument(
        "--vad",
        action="store_true",
        required=False,
        help="Perform voice activity detection",
    )
    
    vad_group.add_argument(
        "--no_speech_threshold",
        type=float,
        required=False,
        default=0.6,
        help="Threshold for no speech",
    )

def main(args):
    print(args)
    if args.gpu:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    elif args.cpu:
        device = "cpu"
    else:
        # in general: cuda > cpu > mps
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        args.model, 
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    ).to(device)
    print(model.device)

if __name__ == "__main__":
    args = parse_args()
    import torch
    from transformers import AutoModel
    main(args)