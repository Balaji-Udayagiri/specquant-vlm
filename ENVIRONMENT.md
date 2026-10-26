# Run SpecQuant-VLM on RunPod, cloud only

## Actual deployment, Oct 20 2026 (use this, not the generic steps below)

- Pod: dear_amber_mandrill (dzmmoek761ex8m), US-CA-2, RTX 6000 Ada 48GB, $0.99/hr GPU, $1.00/hr total. The 4090 was out of capacity in US-CA-2, the 6000 Ada is the benchmark GPU for the whole project now, do not switch GPUs mid project.
- Volume: 50GB Network Volume (high-performance tier, about $7/month), mounted at /workspace.
- Muse has SSH access via its public key added by Balaji. Connection as of Oct 20: ssh root@38.80.152.248 -p 33592 -i ~/.ssh/id_ed25519 (address/port can change on restart, check Connect panel).
- Setup done Oct 20: code in /workspace/specquant-vlm, venv in /workspace/venv with system site packages, HF_HOME=/workspace/hf-cache, 3 smoke tests passed.
- Generic guide below is kept for reference if the pod ever has to be recreated.

Do not run this project on your local machine. Your GTX 1650 4GB cannot run LLaVA-7B, and this guide assumes every step, editing, training, evaluation, and serving, happens on the RunPod pod.

## 1. What to rent

- Provider: RunPod
- Type: Secure Cloud if available, Community Cloud is fine if Secure has no stock
- GPU: RTX 4090, 24GB, 1x GPU is enough
- Region: any US datacenter. Pick one and use it for both the volume and the pod
- Template: RunPod PyTorch 2.4, CUDA 12.4 (or the newest PyTorch template RunPod offers)
- Network Volume: 50GB, mounted at /workspace
- Container disk: 20GB or more

Typical pricing is about $0.34/hr on Community Cloud and about $0.69/hr on Secure Cloud. Check the exact price on the deploy screen, it changes by datacenter and availability. Budget $15 to $25 total if you stop the pod whenever you are not running code.

## 2. Create the volume first

1. In RunPod, go to Storage, then Network Volumes, then create one.
2. Size: 50GB. Name it specquant-vlm.
3. Pick your US datacenter and remember it. The pod must be created in the same datacenter or it cannot attach the volume.

50GB is not extra. You need room for the code, a 14GB model cache, datasets, quantized checkpoints, and outputs.

## 3. Deploy the pod

1. Go to Pods, then Deploy.
2. Select the RTX 4090.
3. Select the datacenter from step 2 and attach the specquant-vlm volume at /workspace.
4. Select the PyTorch template above.
5. Enable JupyterLab and SSH.
6. Deploy, then wait until the pod shows Running and Connect options appear.

## 4. Set up, all inside RunPod

Open JupyterLab from the pod Connect page, then open a Terminal inside JupyterLab. Run everything there. Nothing runs locally.

```bash
cd /workspace
export HF_HOME=/workspace/hf-cache
mkdir -p $HF_HOME

# Get the code onto the volume. Use one of these:
# Option A: clone from GitHub, if you pushed the scaffold there
git clone <your-github-repo-url> specquant-vlm

# Option B: upload the specquant-vlm folder through JupyterLab upload,
# then unzip it in /workspace

cd /workspace/specquant-vlm
python -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pytest tests/test_smoke.py -v
nvidia-smi
```

`nvidia-smi` must show an RTX 4090 with 24GB. If it does not, stop and fix the pod before continuing.

Add the cache setting so you do not set it every session:

```bash
echo 'export HF_HOME=/workspace/hf-cache' >> ~/.bashrc
echo 'source /workspace/venv/bin/activate' >> ~/.bashrc
```

## 5. Daily workflow on RunPod

Start of session:
1. Start the pod in the RunPod dashboard.
2. Open JupyterLab, open Terminal.
3. `cd /workspace/specquant-vlm && git pull` if you use GitHub.

During the session:
- Edit code in JupyterLab editor, or use the web VS Code if your template provides it.
- Run all commands from the PLAN.md in the JupyterLab terminal.
- First run of any eval: use limit 50 to catch bugs. Use limit 200 to 500 only for real results.

End of session, every time:
1. Make sure your latest results are saved in `benchmarks/results.csv` on /workspace.
2. Commit and push to GitHub from the pod if you set it up, as a second backup.
3. Stop the pod in the dashboard. Do not just close the browser tab. A running pod keeps billing.
4. Do not terminate the pod or delete the volume until the project is finished. Stop preserves the volume, terminate can delete the container disk.

## 6. What runs on the pod, by day

- Day 1 to 2: baseline profiling and POPE eval
- Day 3 to 4: sensitivity sweep and AWQ / bitsandbytes quantization
- Day 5 to 6: vision token pruning ablations
- Day 8 to 9: speculative decoding and draft model runs
- Day 10: combined model eval
- Day 11: FastAPI / vLLM serving test, on the pod, using its public port forwarding if you want a shareable demo link

All of it, on the same pod and volume.

## 7. Troubleshooting

- Out of memory on FP16: close other notebooks, run `nvidia-smi` and check nothing else is using VRAM, restart the Jupyter kernel, and rerun with batch_size 1.
- Model downloads again every session: HF_HOME is not set to /workspace/hf-cache in that terminal. Check with `echo $HF_HOME`.
- Pod is in a different datacenter than the volume: the volume will not attach. Recreate the pod in the volume datacenter.
- No 4090 stock: wait, pick another US datacenter and create a new volume there, or take an RTX 3090 24GB or A40 48GB if the price is similar. Do not drop to a 16GB card for the main runs.
- Session feels slow for editing: that is JupyterLab, not the GPU. Keep edits small, use git commits often.

## 8. When the project is done

1. Download `benchmarks/results.csv`, the final README results table, and any checkpoints you want to keep from /workspace.
2. Push the final code to GitHub from the pod.
3. Only then terminate the pod and delete the Network Volume, to stop the small storage charge.
