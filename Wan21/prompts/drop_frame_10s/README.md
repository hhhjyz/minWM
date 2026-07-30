# 10-second dropped-frame stress set

This set contains 30 prompt/trajectory pairs for inspecting temporal stalls,
duplicated frames, sudden motion jumps, and inconsistent motion cadence.

- Samples 01-10: small scene motion with sparse, countable changes.
- Samples 11-20: medium continuous or periodic motion.
- Samples 21-30: large, dense, or fast motion.
- Every trajectory contains 39 actions and therefore produces 40 latent poses.
- At the default 16 decoded FPS, 40 latent frames produce 157 video frames
  (approximately 9.81 seconds).
- Trajectories are not designed for loop closure or revisit evaluation.

Useful signals for manual inspection include clock hands, blinking lights,
conveyor spacing, wheel rotation, pendulum phase, repeated machinery cycles,
fluid motion, and objects crossing stable high-contrast background edges.

Run all samples with fixed sink only:

```bash
CUDA_VISIBLE_DEVICES=0 \
SINK_SIZE=4 \
bash Wan21/scripts/inference/run_drop_frame_10s_fixed_sink.sh
```

Run one category by line range:

```bash
# Small motion: lines 1-10
PROMPT_START=0 MAX_PROMPTS=10 \
bash Wan21/scripts/inference/run_drop_frame_10s_fixed_sink.sh

# Medium motion: lines 11-20
PROMPT_START=10 MAX_PROMPTS=10 \
bash Wan21/scripts/inference/run_drop_frame_10s_fixed_sink.sh

# Large motion: lines 21-30
PROMPT_START=20 MAX_PROMPTS=10 \
bash Wan21/scripts/inference/run_drop_frame_10s_fixed_sink.sh
```
