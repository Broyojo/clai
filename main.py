import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        prog="clai",
        description="Command Line AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    #################### Misc ####################
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        required=False,
        help="Verbose mode",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="Debug mode",
    )
    
    #################### Global Arguments ####################
    
    global_args = parser.add_argument_group("Global Arguments")
    
    global_args.add_argument(
        "--model",
        type=str,
        required=False,
        help="Name or path of model to use",
    )
    
    global_args.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        required=False,
        help="Use less CPU memory",
    )
    
    global_args.add_argument(
        "--cpu",
        action="store_true",
        required=False,
        help="Use CPU",
    )
    
    global_args.add_argument(
        "--gpu",
        action="store_true",
        required=False,
        help="Use GPU",
    )
    
    global_args.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=1,
        help="Batch size",
    )
    
    global_args.add_argument(
        "--compute_type",
        type=str,
        required=False,
        default="float32",
        choices=["float32", "float16", "float4", "int8"],
        help="Compute type",
    )
    
    global_args.add_argument(
        "--use_better_transformer",
        action="store_true",
        required=False,
        help="Use Better Transformer",
    )
    
    global_args.add_argument(
        "--use_flash_attention",
        action="store_true",
        required=False,
        help="Use Flash Attention",
    )
    
    global_args.add_argument(
        "--use_flash_attention_2",
        action="store_true",
        required=False,
        help="Use Flash Attention 2",
    )
    
    global_args.add_argument(
        "--compile",
        action="store_true",
        required=False,
        help="Compile model",
    )
    
    #################### Tasks ####################
    
    tasks = parser.add_mutually_exclusive_group()
    
    tasks.add_argument(
        "--transcribe",
        action="store_true",
        required=False,
        help="Perform transcription"
    )
    
    #################### Transcribe Task ####################
    
    transcribe_args = parser.add_argument_group("Transcribe Task")
    
    transcribe_args.add_argument(
        "--input",
        type=str,
        required=False,
        help="Input audio path/directory",
    )
    
    transcribe_args.add_argument(
        "--chunk_batch_size",
        type=int,
        required=False,
        default=24,
        help="Size of parallel audio chunk batches you want to compute",
    )
    
    transcribe_args.add_argument(
        "--chunk_length",
        required=False,
        type=float,
        default=30.0,
        help="Length of each chunk in seconds",
    )
    
    transcribe_args.add_argument(
        "--timestamps",
        required=False,
        type=str,
        default=False,
        choices=["chunk", "word"],
        help="What kind of timestamps to use",
    )
    
    transcribe_args.add_argument(
        "--language",
        required=False,
        type=str,
        help="Language of the audio",
    )
    
    transcribe_args.add_argument(
        "--translate",
        required=False,
        type=str,
        help="Language to translate the audio to",
    )
    
    transcribe_args.add_argument(
        "--diarize",
        action="store_true",
        required=False,
        help="Diarize the audio",
    )
    
    transcribe_args.add_argument(
        "--num_speakers",
        type=int,
        required=False,
        help="Number of speakers to diarize",
    )
    
    transcribe_args.add_argument(
        "--min_speakers",
        type=int,
        required=False,
        help="Minimum number of speakers to diarize",
    )
    
    transcribe_args.add_argument(
        "--max_speakers",
        type=int,
        required=False,
        help="Maximum number of speakers to diarize",
    )
    
    transcribe_args.add_argument(
        "--vad",
        action="store_true",
        required=False,
        help="Perform voice activity detection",
    )
    
    transcribe_args.add_argument(
        "--no_speech_threshold",
        type=float,
        required=False,
        default=0.6,
        help="Threshold for no speech",
    )
    
    return parser.parse_args()

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