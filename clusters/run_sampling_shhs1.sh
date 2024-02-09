TAG=ldm_eeg

specs=("no-spectral" "spectral")
latents=("1" "3")
type_dataset="shhs1"
path_pre_processed="/data/polysomnography/shhs_numpy"

for sp in "${specs[@]}"; do
	for latent in "${latents[@]}"; do
	runai submit \
	  --name  sampling-shhs1-${sp}-${latent} \
	  --image "aicregistry:5000/${USER}:${TAG}" \
	  --backoff-limit 0 \
	  --cpu-limit 25 \
	  --gpu 1 \
	  --node-type "A100" \
	  --large-shm \
	  --host-ipc \
	  --project wds20 \
	  	    --run-as-user \
	    --volume /nfs/home/wds20/bruno/data/SHHS/shhs/:/data/ \
		--volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
	  --command -- bash /project/src/bash/start_training.sh python3 /project/src/sample_trials.py spe=${sp} latent=${latent} type_dataset=${type_dataset} path_pre_processed=${path_pre_processed}
	done
done
