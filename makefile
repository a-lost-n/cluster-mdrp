train:
	sbatch train.sh

clean:
	rm -f models/*model*
	rm -f slurm*
