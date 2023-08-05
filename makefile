train:
	sbatch train.sh

clean:
	rm -f *model*
	rm -f slurm*
