# clai
Command line artificial intelligence

Goal: Make using AI models very easy to run from the command line.

## General Command Structure

```
input -> clai [options] -> output
```

input: one or more files, streams, directories, urls, prompts
options: option for the command
output: one or more files, streams

By default, the tool tries to dynamically find the optimal parameters for any specific run. This includes batch size, 

For more optimization, you can specify the optimization level you want, which may start doing things like quantization.

## Command Documentation

Global Options:
- `--model` Specifies the model that should be used.
- `--batch_size [n]` Specifies how many samples the model should be run on at once.
- `--num_gpu [n]` Specifies the number of GPUs that should be used.
- `--cpu` Enables CPU inference.
- `--gpu` Enables GPU inference.
- `--mps` Enables MPS inference.
- `--device [cpu, gpu, mps]` Set device to use.
- `--compute_type [auto, int8, float16, etc.]` Set compute type
- `--quantize [4bit, 8bit, etc.]` Quantize model, (cuda only). (there are more complex quantization options btw)
- `--better_transformer` Use Better Transformer
- `--flash_attention` Use flash attention (cuda only).
- `--flash_attention_2` Use flash attention 2 (cuda only).
- `--low_cpu_mem_usage` Enable low cpu memory usage.
- `--ipex` Use Intel IPEX optimizations (intel only).
- `--compile` Run torch compile.
- `--verbose` Enable verbose output
- `--temperature`
- `--top_p`
- `--top_k`
- `--greedy`
- `--sample`
- `--beam_size`
- `--num_outputs`
- `--seed`
- `--prompt`

Tasks:
- `--transcribe` or `--stt`
  - `--chunk_batch_size`
  - `--diarize`
    - `--min_speakers`
    - `--max_speakers`
    - `--num_speakers`
  - `--timestamps [chunk, word]`
  - `--chunk_size [seconds]`
  - `--language [en, ch, etc.]`
  - `--translate [en, ch, etc.]`
  - `--vad`
    - `--no_speech_threshold`

- `--draw` (there are many possible tasks here, like image to image, text to image, image infilling, control net, lora, unconditional image generation, etc.)
  - `--height`
  - `--width`
  - `--guidance_scale`
  - `--negative_prompt`
  - `--num_inference_steps`
  - `--vae_slicing`
  - `--vae_tiling`
  - `--sequential_cpu_offload`
  - `--model_cpu_offload`
  - `--channels_last` 
  - `--attention_slicing`
  - `--trace`
  - `--sdpa`

- `--speak` or `--tts`
  - [many possible arguments]

## Examples
```bash
clai --transcribe --file audio.mp3 --model openai/whisper-large-v3 --diarize --num_speakers 3 transcription.txt
clai --draw --prompt "man eating a banana" --model stabilityai/stable-diffusion-xl-1.0 --guidance_scale 0.7 --channels_last --attention_slicing --sdpa --trace
```

## Roadmap

1. get huggingface & api based stuff working first (api stuff comes after huggingface probably since it is comparatively easy). first work on transcription since that seems like the easiest task. the hardest one may be the chatting with LLMs one.
2. then work on optimized backends (deepspeed-fastgen, cTranslate2, llama.cpp, whisper.cpp, coqui TTS, etc.)

## Notes
- Use `torch.inference_mode()` instead of `torch.no_grad()` since inference_mode is slightly faster.